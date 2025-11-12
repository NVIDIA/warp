#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <cassert>

// ============================================================================
// MACROS FOR CHECKING AND ERROR HANDLING
// ============================================================================

// Simple assertion macro for device code - triggers breakpoint on failure
#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond))                                                                               \
            asm("brkpt;");                                                                         \
    } while (0)

// CUDA error checking macro for host-side API calls
#define CUDA_SAFE_CALL(call)                                                                       \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s (error code: %d)\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err), err);                                                 \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Memory layout constants for performance
// These prevent false sharing and page thrashing between kernels
namespace work_stealing_layout {
// Cache line separation (conservative: 512KB to avoid any L2 conflicts)
static constexpr size_t CACHE_LINE_SEPARATION = 512 * 1024;

// Page separation (2MB pages for TLB efficiency)
// TODO: Later we can use CUDA VMM API for explicit page mapping
static constexpr size_t PAGE_SIZE = 2 * 1024 * 1024;

// Calculate offset for a kernel's data within the unified buffer
// Marked constexpr to allow compile-time evaluation when possible
static constexpr inline size_t kernel_data_offset(int kernel_id) { return kernel_id * PAGE_SIZE; }

// Offsets within a kernel's page for front/back/init_tag
// These are pure compile-time constants
__host__ __device__ static constexpr size_t front_offset() { return 0; }
__host__ __device__ static constexpr size_t back_offset() { return CACHE_LINE_SEPARATION; }
__host__ __device__ static constexpr size_t init_tag_offset() { return 2 * CACHE_LINE_SEPARATION; }
} // namespace work_stealing_layout

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

// Ring neighbor calculation (r=0 is self, then +1,-1,+2,-2,... mod k)
// For r=0: me
// For r=1: me+1 mod k
// For r=2: me-1 mod k
// For r=3: me+2 mod k
// For r=4: me-2 mod k, etc.
__device__ inline int neighbor_ring_idx(int me, int r, int k) {
    if (r == 0) {
        return me;
    }

    int offset;
    if (r % 2 == 1) {
        // Odd r: positive offset
        offset = (r + 1) / 2;
    } else {
        // Even r: negative offset
        offset = -(r / 2);
    }

    int neighbor = me + offset;
    // Handle wrap-around
    if (neighbor < 0) {
        neighbor += k;
    } else if (neighbor >= k) {
        neighbor -= k;
    }

    return neighbor;
}

// Per-block state for work stealing sweep through deques
struct ws_sweep_state {
    int next_r;        // Next r position to check in the ring
    int checked_count; // How many deques we've checked without finding work

    __device__ __host__ inline ws_sweep_state() : next_r(0), checked_count(0) {}
};

// Result structure for fetch operations
struct fetch_result {
    int line;           // Which deque the item came from
    int item;           // The item index
    bool ok;            // Whether we successfully got an item
    bool can_terminate; // Whether we can safely terminate (full sweep with no work)
};

// Device-accessible descriptor for work-stealing deques
// This lightweight struct can be passed to CUDA kernels
// IMPORTANT: front/back/init_tag use strided access (PAGE_SIZE stride) to avoid false sharing
struct ws_queues_view {
    char* unified_base; // Base pointer to unified buffer (all counters computed from this)
    int k;
    int epoch;

    // Optional instrumentation: records which kernel processed each work item
    int* instrumentation_buffer; // If non-null, records kernel IDs: [line * m + item] = kernel_id
    int m;                       // Items per deque (buffer capacity per deque)
    int max_work_items;          // Total valid work items (may be < k*m)

    // Device functions to access counters with proper stride
    // __forceinline__ ensures zero overhead from offset computation
    // All offsets are compile-time constants - no runtime overhead!
    // Layout within each kernel's page:
    //   offset 0:              front counter
    //   offset +CACHE_LINE_SEPARATION:   back counter
    //   offset +2*CACHE_LINE_SEPARATION: init_tag counter
    __device__ __forceinline__ int* get_front(int index) const {
        using namespace work_stealing_layout;
        return reinterpret_cast<int*>(unified_base + index * PAGE_SIZE + front_offset());
    }
    __device__ __forceinline__ int* get_back(int index) const {
        using namespace work_stealing_layout;
        return reinterpret_cast<int*>(unified_base + index * PAGE_SIZE + back_offset());
    }
    __device__ __forceinline__ int* get_init_tag(int index) const {
        using namespace work_stealing_layout;
        return reinterpret_cast<int*>(unified_base + index * PAGE_SIZE + init_tag_offset());
    }

    // ============================================================================
    // WORK DEQUE OPERATIONS (Member Functions)
    // ============================================================================

    // Epoch-tag lazy initialization (reentrant; non-blocking)
    // - init_tag[line] == epoch: initialized
    // - init_tag[line] == -epoch: initialization in progress
    // - other: not initialized for this run
    //
    // This function should be called by the owner thread before work begins.
    // It sets front[line] = 0, back[line] = m, and atomically updates init_tag[line] to epoch.
    __device__ inline void lazy_init_line_epoch(int line) const {
        int* front_ptr = get_front(line);
        int* back_ptr = get_back(line);
        int* init_tag_ptr = get_init_tag(line);

        int current_tag = *init_tag_ptr;

        // Already initialized for this epoch
        if (current_tag == epoch) {
            return;
        }

        // Try to claim initialization
        int old = atomicCAS(init_tag_ptr, current_tag, -epoch);
        if (old == current_tag) {
            // We won the race - do the initialization
            // Always initialize to full capacity - application code can skip unwanted items
            // Note: With complex layouts (e.g., CUTE), linear capping doesn't work correctly
            *front_ptr = 0;
            *back_ptr = m;
            __threadfence(); // Ensure stores are visible before marking as initialized
            *init_tag_ptr = epoch;
        } else if (old == -epoch) {
            // Someone else is initializing - spin until done
            while (*init_tag_ptr == -epoch) {
                __threadfence(); // Force memory read - prevent infinite loop from compiler
                                 // optimization
            }
        } else if (old == epoch) {
            // Already initialized by someone else
            return;
        } else {
            // Different epoch or in-progress, retry
            lazy_init_line_epoch(line);
        }
    }

    // CVR (Claim-Validate-Rollback) pop from front
    // Multi-consumer safe; last item allowed from either end
    // Returns true and sets out_idx if successful
    __device__ inline bool pop_front_cvr(int line, int& out_idx) const {
        int* front_ptr = get_front(line);
        int* back_ptr = get_back(line);

        // Claim phase: atomically increment front
        int claimed_front = atomicAdd(front_ptr, 1);

        // Validate phase: check if we're still within bounds
        __threadfence(); // Ensure we see the latest back value
        int current_back = *back_ptr;

        if (claimed_front < current_back) {
            // Success - we have a valid item
            out_idx = claimed_front;
            return true;
        }

        // Rollback phase: we went too far, give back our slot
        atomicAdd(front_ptr, -1);
        return false;
    }

    // CVR (Claim-Validate-Rollback) pop from back
    // Multi-consumer safe; last item allowed from either end
    // Returns true and sets out_idx if successful
    __device__ inline bool pop_back_cvr(int line, int& out_idx) const {
        int* front_ptr = get_front(line);
        int* back_ptr = get_back(line);

        // Claim phase: atomically decrement back
        int claimed_back = atomicAdd(back_ptr, -1);

        // Validate phase: check if we're still within bounds
        // Use >= to allow last item to be taken from either end
        __threadfence(); // Ensure we see the latest front value
        int current_front = *front_ptr;

        if (claimed_back > current_front) {
            // Success - we have a valid item
            out_idx = claimed_back - 1; // Adjust because we decremented first
            return true;
        }

        // Rollback phase: we went too far, give back our slot
        atomicAdd(back_ptr, 1);
        return false;
    }

    // Try to get the next item from the ring of K deques
    // Advances monotonically through the ring - never rechecks exhausted deques
    // Returns can_terminate=true only when all k deques have been exhausted (or local deque if
    // !enable_stealing)
    //
    // enable_stealing: if false, only processes own deque (static partitioning)
    __device__ inline fetch_result try_get_next_item_with_sweep(int me, ws_sweep_state& rs,
                                                                bool enable_stealing = true) const {
        fetch_result result;
        result.ok = false;
        result.can_terminate = false;
        result.line = -1;
        result.item = -1;

        // For static partitioning (no stealing), only check once and terminate if empty
        if (!enable_stealing) {
            if (rs.checked_count > 0) {
                result.can_terminate = true;
                return result;
            }
        } else {
            // If we've already checked all k deques without finding work, we're done
            if (rs.checked_count >= k) {
                result.can_terminate = true;
                return result;
            }
        }

        // Get the current r position to check
        int r = enable_stealing ? rs.next_r : 0; // Static mode: always check own deque (r=0)
        int victim_line = neighbor_ring_idx(me, r, k);

        // Check initialization status
        int* init_tag_ptr = get_init_tag(victim_line);
        int tag = *init_tag_ptr;
        if (tag != epoch) {
            // Not initialized yet for this epoch - initialize it ourselves
            // This allows work stealing to proceed even if the owner hasn't started yet
            lazy_init_line_epoch(victim_line);
        }

        // Try to get an item from this deque
        int item_idx;
        bool success;

        if (r == 0) {
            // r=0 is our own deque - pop from front (LIFO for cache locality)
            success = pop_front_cvr(victim_line, item_idx);
        } else {
            // Steal from other deques from back (less contention with owner)
            success = pop_back_cvr(victim_line, item_idx);
        }

        if (success) {
            // Found work! Reset the checked count
            result.ok = true;
            result.line = victim_line;
            result.item = item_idx;
            rs.checked_count = 0;

            // Optional instrumentation: record which kernel processed this work item
            if (instrumentation_buffer != nullptr) {
                int global_idx = victim_line * m + item_idx;
                instrumentation_buffer[global_idx] = me;
            }

            // Only advance if we're stealing (r != 0)
            // Keep next_r at 0 when draining local deque for cache locality
            if (r != 0) {
                rs.next_r = (r + 1) % k;
            }
            return result;
        }

        // This deque is exhausted - advance to next position
        rs.checked_count++;
        rs.next_r = (r + 1) % k;

        // Check if we've now exhausted all deques
        if (rs.checked_count >= k) {
            result.can_terminate = true;
        }

        return result;
    }

    // High-level wrapper that only returns when:
    // 1. Work is available (res.ok = true), OR
    // 2. All work is done (res.ok = false, can terminate)
    //
    // IMPORTANT: ALL THREADS IN THE BLOCK MUST CALL THIS TOGETHER (collective operation)
    // Thread 0 performs the work stealing, and the result is broadcast to all threads.
    //
    // This hides the retry logic when no work is temporarily available.
    // Use this for most cases. Use try_get_next_item_with_sweep directly only if
    // you need custom behavior when there's temporarily no work.
    //
    // enable_stealing: if false, only processes own deque (static partitioning, no work stealing)
    __device__ inline fetch_result get_next_item(int me, ws_sweep_state& rs,
                                                 bool enable_stealing = true) const {
        __shared__ fetch_result shared_result;

        while (true) {
            // Only thread 0 performs work stealing
            if (threadIdx.x == 0) {
                shared_result = try_get_next_item_with_sweep(me, rs, enable_stealing);
            }
            __syncthreads();

            // All threads read the result
            fetch_result res = shared_result;

            // Return if we got work OR if we can terminate
            if (res.ok || res.can_terminate) {
                return res;
            }

            // Otherwise: no work now, but others may still be working
            // Loop back and try again
            __syncthreads(); // Ensure all threads loop together
        }
    }
};

// ============================================================================
// WORK STEALING STATISTICS
// ============================================================================

// Statistics computed from work-stealing instrumentation data
struct ws_stats {
    // Default constructor for empty/invalid stats
    ws_stats()
        : num_kernels(0), items_per_deque(0), total_capacity(0), total_work_items(0), min_items(0),
          max_items(0), mean_items(0.0f), stddev_items(0.0f), imbalance(0.0f), valid_items(0),
          unprocessed_items(0), invalid_items(0), is_valid(false) {}

    int num_kernels;      // Number of kernels (k)
    int items_per_deque;  // Items per deque (m)
    int total_capacity;   // Total buffer capacity (k * m)
    int total_work_items; // Actual work items to process

    // Per-kernel statistics
    std::vector<int> items_processed;  // Total items processed by each kernel
    std::vector<int> own_items;        // Items from own deque
    std::vector<int> stolen_items;     // Items stolen from other deques
    std::vector<float> own_percentage; // Percentage of own work

    // Work stealing matrix: stealing_matrix[from_deque][by_kernel] = count
    // Shows how many items from deque 'from_deque' were processed by 'by_kernel'
    std::vector<std::vector<int>> stealing_matrix;

    // Load balance metrics
    int min_items;      // Minimum items processed by any kernel
    int max_items;      // Maximum items processed by any kernel
    float mean_items;   // Average items per kernel
    float stddev_items; // Standard deviation
    float imbalance;    // Load imbalance ratio (max/mean)

    // Validation
    int valid_items;       // Items processed correctly
    int unprocessed_items; // Items not processed
    int invalid_items;     // Items with invalid kernel IDs
    bool is_valid;         // True if all work was processed correctly
};

// Compute statistics from instrumentation buffer
// instr_data: instrumentation buffer (k*m elements, value is kernel_id that processed it)
// k: number of kernels/deques
// m: items per deque
// max_work_items: actual number of work items (may be < k*m)
inline ws_stats compute_ws_stats(const std::vector<int>& instr_data, int k, int m,
                                 int max_work_items = -1) {
    ws_stats stats;

    if (max_work_items < 0) {
        max_work_items = k * m;
    }

    stats.num_kernels = k;
    stats.items_per_deque = m;
    stats.total_capacity = k * m;
    stats.total_work_items = max_work_items;

    // Initialize per-kernel vectors
    stats.items_processed.resize(k, 0);
    stats.own_items.resize(k, 0);
    stats.stolen_items.resize(k, 0);
    stats.own_percentage.resize(k, 0.0f);

    // Initialize stealing matrix
    stats.stealing_matrix.resize(k, std::vector<int>(k, 0));

    // Initialize validation counters
    stats.valid_items = 0;
    stats.unprocessed_items = 0;
    stats.invalid_items = 0;

    // Process instrumentation data
    for (int line = 0; line < k; line++) {
        for (int item = 0; item < m; item++) {
            int idx = line * m + item;
            int processed_by = instr_data[idx];

            if (processed_by == -1) {
                // Not processed
                if (idx < max_work_items) {
                    stats.unprocessed_items++;
                }
            } else if (processed_by < 0 || processed_by >= k) {
                // Invalid kernel ID
                stats.invalid_items++;
            } else {
                // Valid processing
                if (idx < max_work_items) {
                    stats.valid_items++;
                    stats.items_processed[processed_by]++;
                    stats.stealing_matrix[line][processed_by]++;

                    if (processed_by == line) {
                        stats.own_items[processed_by]++;
                    } else {
                        stats.stolen_items[processed_by]++;
                    }
                } else {
                    // Item beyond work range was processed (error)
                    stats.invalid_items++;
                }
            }
        }
    }

    // Compute own work percentages
    for (int i = 0; i < k; i++) {
        if (stats.items_processed[i] > 0) {
            stats.own_percentage[i] = 100.0f * stats.own_items[i] / stats.items_processed[i];
        }
    }

    // Compute load balance metrics
    stats.min_items = *std::min_element(stats.items_processed.begin(), stats.items_processed.end());
    stats.max_items = *std::max_element(stats.items_processed.begin(), stats.items_processed.end());

    int total_processed = 0;
    for (int count : stats.items_processed) {
        total_processed += count;
    }
    stats.mean_items = static_cast<float>(total_processed) / k;

    // Compute standard deviation
    float variance = 0.0f;
    for (int count : stats.items_processed) {
        float diff = count - stats.mean_items;
        variance += diff * diff;
    }
    stats.stddev_items = std::sqrt(variance / k);

    // Compute imbalance ratio
    if (stats.mean_items > 0) {
        stats.imbalance = stats.max_items / stats.mean_items;
    } else {
        stats.imbalance = 1.0f;
    }

    // Validation
    stats.is_valid = (stats.valid_items == max_work_items) && (stats.unprocessed_items == 0) &&
                     (stats.invalid_items == 0);

    return stats;
}

// Print comprehensive statistics report
inline void print_ws_stats(const ws_stats& stats, bool verbose = true) {
    printf("\n=== Work Stealing Statistics ===\n");
    printf("Configuration: %d kernels × %d items/deque = %d capacity\n", stats.num_kernels,
           stats.items_per_deque, stats.total_capacity);
    printf("Work items: %d\n", stats.total_work_items);

    // Validation summary
    printf("\nValidation:\n");
    printf("  Processed: %d/%d items %s\n", stats.valid_items, stats.total_work_items,
           stats.valid_items == stats.total_work_items ? "✓" : "✗");
    if (stats.unprocessed_items > 0) {
        printf("  Missing:   %d items ✗\n", stats.unprocessed_items);
    }
    if (stats.invalid_items > 0) {
        printf("  Invalid:   %d items ✗\n", stats.invalid_items);
    }
    printf("  Status:    %s\n", stats.is_valid ? "PASS ✓" : "FAIL ✗");

    // Load balance
    printf("\nLoad Balance:\n");
    printf("  Min items:      %d\n", stats.min_items);
    printf("  Max items:      %d\n", stats.max_items);
    printf("  Mean items:     %.1f\n", stats.mean_items);
    printf("  Std deviation:  %.2f\n", stats.stddev_items);
    printf("  Imbalance:      %.2fx %s\n", stats.imbalance,
           stats.imbalance < 1.2f   ? "(excellent)"
           : stats.imbalance < 1.5f ? "(good)"
           : stats.imbalance < 2.0f ? "(fair)"
                                    : "(poor)");

    // Per-kernel distribution
    printf("\nPer-Kernel Distribution:\n");
    for (int i = 0; i < stats.num_kernels; i++) {
        int total = stats.items_processed[i];
        float pct = stats.total_work_items > 0 ? 100.0f * total / stats.total_work_items : 0.0f;

        printf("  Kernel %d: %4d items (%5.1f%%) - %4d own, %4d stolen (%.1f%% own)\n", i, total,
               pct, stats.own_items[i], stats.stolen_items[i], stats.own_percentage[i]);
    }

    // Work stealing matrix (only in verbose mode)
    if (verbose && stats.num_kernels <= 16) {
        printf("\nWork Stealing Matrix (rows=from deque, cols=processed by):\n");
        printf("     ");
        for (int i = 0; i < stats.num_kernels; i++) {
            printf(" K%-2d", i);
        }
        printf("\n");

        for (int from = 0; from < stats.num_kernels; from++) {
            printf("  K%-2d:", from);
            for (int by = 0; by < stats.num_kernels; by++) {
                int count = stats.stealing_matrix[from][by];
                if (from == by) {
                    // Own work - bold
                    printf(" \033[1m%3d\033[0m", count);
                } else if (count > 0) {
                    // Stolen work
                    printf(" %3d", count);
                } else {
                    // No stealing
                    printf("   ·");
                }
            }
            printf("\n");
        }
        printf("\n  Legend: Bold = own work, numbers = stolen items, · = no stealing\n");
    }
}

// Compact one-line summary
inline std::string ws_stats_summary(const ws_stats& stats) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%d/%d items, imbalance=%.2fx, %d stolen (%.1f%%)",
             stats.valid_items, stats.total_work_items, stats.imbalance,
             stats.stolen_items[0] + stats.stolen_items[1], // Total stolen
             100.0f * (stats.valid_items - stats.own_items[0] - stats.own_items[1]) /
                 stats.valid_items);
    return std::string(buf);
}

// Export statistics to CSV format for analysis
inline void export_ws_stats_csv(const ws_stats& stats, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error: Could not open %s for writing\n", filename);
        return;
    }

    // Per-kernel data
    fprintf(f, "kernel_id,total_items,own_items,stolen_items,own_percentage\n");
    for (int i = 0; i < stats.num_kernels; i++) {
        fprintf(f, "%d,%d,%d,%d,%.2f\n", i, stats.items_processed[i], stats.own_items[i],
                stats.stolen_items[i], stats.own_percentage[i]);
    }

    fclose(f);
    printf("Exported statistics to %s\n", filename);
}

// ============================================================================
// HOST-SIDE DEQUE MANAGER CLASS
// ============================================================================

// Forward declaration of implementation class (PIMPL idiom)
class ws_queues_impl;

// Manages device memory for work-stealing deques with epoch-based reuse
// IMPORTANT: Uses page-aligned layout to avoid false sharing and cache conflicts:
//   - Each kernel's counters (front/back/init_tag) are on a separate 2MB page
//   - Within each page, counters are separated by 512KB to avoid cache line conflicts
//   - Single unified allocation for all data
//
// Uses PIMPL idiom for easy wrapping in Python/other languages
class ws_queues {
  private:
    ws_queues_impl* impl_; // Pointer to implementation (PIMPL idiom)

  public:
    // Simple constructor for Python/external interfaces (PIMPL-friendly)
    // Takes only essential parameters: k, m, and instrumentation flag
    //
    // Parameters:
    //   k: Number of kernels/deques
    //   m: Items per deque (buffer size = k×m slots)
    //   enable_instrumentation: Whether to track which kernel processes each item (default: false)
    ws_queues(int k, int m, bool enable_instrumentation = false);

    // Advanced constructor with explicit max_work_items
    // Use this when total work items don't fill the buffer completely
    //
    // Parameters:
    //   k: Number of kernels/deques
    //   m: Items per deque (buffer size = k×m slots)
    //   max_work_items: Maximum valid work item index (must be <= k×m)
    //   enable_instrumentation: Whether to track which kernel processes each item
    ws_queues(int k, int m, int max_work_items, bool enable_instrumentation);

    // Destructor: frees device memory
    ~ws_queues();

    // Delete copy constructor and assignment operator (prevent double-free)
    ws_queues(const ws_queues&) = delete;
    ws_queues& operator=(const ws_queues&) = delete;

    // Move constructor
    ws_queues(ws_queues&& other) noexcept;

    // Move assignment operator
    ws_queues& operator=(ws_queues&& other) noexcept;

    // Get device pointer to front array
    int* front();
    const int* front() const;

    // Get device pointer to back array
    int* back();
    const int* back() const;

    // Get device pointer to init_tag array
    int* init_tag();
    const int* init_tag() const;

    // Get device pointer to instrumentation buffer (or nullptr if disabled)
    int* instrumentation_buffer();
    const int* instrumentation_buffer() const;

    // Check if instrumentation is enabled
    bool has_instrumentation() const;

    // Get instrumentation buffer (read-only access)
    const int* get_instrumentation_buffer() const;

    // Get items per deque
    int items_per_deque() const;

    // Get current epoch
    int epoch() const;

    // Get number of deques
    int num_deques() const;

    // Advance to next epoch (for reusing the same deques with new work)
    // Automatically resets to epoch 1 if counter gets too large (>= 1 billion)
    void next_epoch();

    // Get a device-accessible view to pass to kernels
    ws_queues_view view() const;

    // Validate that all work items were assigned exactly once
    //
    // This method checks the instrumentation buffer to ensure:
    //   1. All work items (deque slots) were processed
    //   2. Each item was processed by exactly one kernel
    //   3. All kernel IDs are valid (0 to k-1)
    //
    // Requirements:
    //   - Instrumentation must be enabled (pass true to constructor)
    //   - Call this AFTER cudaDeviceSynchronize() to ensure all kernels finished
    //
    // Returns: true if validation passes, false otherwise
    // Side effects: Prints detailed diagnostics to stdout
    //
    // Example usage:
    //   ws_queues deques(4, 100, true);  // 4 kernels, 100 items each, instrumentation ON
    //   // ... launch kernels ...
    //   cudaDeviceSynchronize();
    //   if (!deques.validate_work_assignment()) {
    //       fprintf(stderr, "Work distribution failed validation!\n");
    //       return 1;
    //   }
    bool validate_work_assignment() const;

    // Get a copy of the instrumentation buffer data as a host vector
    //
    // Returns: std::vector<int> containing the instrumentation data (k*m elements)
    //   Each element is the kernel ID that processed that work item (-1 if unprocessed)
    //
    // Requirements:
    //   - Instrumentation must be enabled
    //   - Call this AFTER cudaDeviceSynchronize()
    //
    // Example:
    //   auto instr_data = deques.get_instrumentation_data();
    std::vector<int> get_instrumentation_data() const;

    // Compute work-stealing statistics from instrumentation data
    //
    // Returns: ws_stats structure with comprehensive statistics
    //
    // Requirements:
    //   - Instrumentation must be enabled
    //   - Call this AFTER cudaDeviceSynchronize()
    //
    // Example:
    //   ws_stats stats = deques.compute_stats();
    //   print_ws_stats(stats, true);
    ws_stats compute_stats() const;
};

// ============================================================================
// IMPLEMENTATION CLASS (PIMPL)
// ============================================================================

class ws_queues_impl {
  public:
    char* d_unified_buffer_; // Single unified allocation (page-aligned)
    int* d_front_;           // Pointer to front array (within unified buffer)
    int* d_back_;            // Pointer to back array (within unified buffer)
    int* d_init_tag_;        // Pointer to init_tag array (within unified buffer)
    int k_;                  // Number of deques
    int current_epoch_;      // Current epoch number

    // Optional instrumentation (separate allocation, doesn't need page alignment)
    int* d_instrumentation_buffer_; // Device pointer to instrumentation buffer (or nullptr)
    int m_;                         // Items per deque (for instrumentation indexing)
    int max_work_items_;            // Maximum valid work item index (inclusive upper bound)

    // Constructor implementation
    ws_queues_impl(int k, int m, int max_work_items, bool enable_instrumentation)
        : d_unified_buffer_(nullptr), k_(k), current_epoch_(1), d_instrumentation_buffer_(nullptr),
          m_(m), max_work_items_(max_work_items) {

        // Validate parameters
        if (max_work_items_ > k * m) {
            fprintf(
                stderr,
                "ERROR: max_work_items (%d) exceeds buffer capacity (%d kernels × %d items = %d)\n",
                max_work_items_, k, m, k * m);
            exit(EXIT_FAILURE);
        }

        using namespace work_stealing_layout;

        // Allocate unified buffer: k pages of 2MB each
        size_t total_size = k * PAGE_SIZE;
        CUDA_SAFE_CALL(cudaMallocManaged(&d_unified_buffer_, total_size));

        // Set up pointers to the start of front/back/init_tag regions in unified buffer
        // The arrays will be accessed via front[i], where each element is PAGE_SIZE bytes apart
        d_front_ = reinterpret_cast<int*>(d_unified_buffer_ + front_offset());
        d_back_ = reinterpret_cast<int*>(d_unified_buffer_ + back_offset());
        d_init_tag_ = reinterpret_cast<int*>(d_unified_buffer_ + init_tag_offset());

        // Initialize all counters to 0
        // Each kernel's counters are PAGE_SIZE bytes apart
        for (int i = 0; i < k; i++) {
            char* kernel_page = d_unified_buffer_ + kernel_data_offset(i);
            int* front_ptr = reinterpret_cast<int*>(kernel_page + front_offset());
            int* back_ptr = reinterpret_cast<int*>(kernel_page + back_offset());
            int* init_ptr = reinterpret_cast<int*>(kernel_page + init_tag_offset());

            *front_ptr = 0;
            *back_ptr = 0;
            *init_ptr = 0;
        }

        // Optionally allocate instrumentation buffer (separate, doesn't need page alignment)
        if (enable_instrumentation && m > 0) {
            size_t buffer_size = k * m * sizeof(int);
            CUDA_SAFE_CALL(cudaMallocManaged(&d_instrumentation_buffer_, buffer_size));
            CUDA_SAFE_CALL(
                cudaMemset(d_instrumentation_buffer_, -1, buffer_size)); // -1 = unprocessed
        }
    }

    // Destructor
    ~ws_queues_impl() {
        if (d_unified_buffer_)
            cudaFree(d_unified_buffer_);
        if (d_instrumentation_buffer_)
            cudaFree(d_instrumentation_buffer_);
    }
};

// ============================================================================
// WS_QUEUES METHOD IMPLEMENTATIONS (after ws_queues_impl)
// ============================================================================

// Constructors
inline ws_queues::ws_queues(int k, int m, bool enable_instrumentation)
    : impl_(new ws_queues_impl(k, m, k * m, enable_instrumentation)) {}

inline ws_queues::ws_queues(int k, int m, int max_work_items, bool enable_instrumentation)
    : impl_(new ws_queues_impl(k, m, max_work_items, enable_instrumentation)) {}

// Destructor
inline ws_queues::~ws_queues() { delete impl_; }

// Move constructor
inline ws_queues::ws_queues(ws_queues&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

// Move assignment operator
inline ws_queues& ws_queues::operator=(ws_queues&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

// Accessor methods - inline for zero overhead
inline int* ws_queues::front() { return impl_->d_front_; }
inline const int* ws_queues::front() const { return impl_->d_front_; }

inline int* ws_queues::back() { return impl_->d_back_; }
inline const int* ws_queues::back() const { return impl_->d_back_; }

inline int* ws_queues::init_tag() { return impl_->d_init_tag_; }
inline const int* ws_queues::init_tag() const { return impl_->d_init_tag_; }

inline int* ws_queues::instrumentation_buffer() { return impl_->d_instrumentation_buffer_; }
inline const int* ws_queues::instrumentation_buffer() const {
    return impl_->d_instrumentation_buffer_;
}

inline bool ws_queues::has_instrumentation() const {
    return impl_->d_instrumentation_buffer_ != nullptr;
}

inline const int* ws_queues::get_instrumentation_buffer() const {
    return impl_->d_instrumentation_buffer_;
}

inline int ws_queues::items_per_deque() const { return impl_->m_; }

inline int ws_queues::epoch() const { return impl_->current_epoch_; }

inline int ws_queues::num_deques() const { return impl_->k_; }

inline void ws_queues::next_epoch() {
    impl_->current_epoch_++;

    // Automatically reset epoch if it gets too large (implementation detail)
    // This prevents integer overflow and ensures init_tag logic remains correct
    // Conservative threshold: reset at half of int32 max to avoid any issues
    constexpr int EPOCH_RESET_THRESHOLD = 1000000000; // 1 billion
    if (impl_->current_epoch_ >= EPOCH_RESET_THRESHOLD) {
        impl_->current_epoch_ = 1;
        // Clear init_tag array to avoid conflicts from previous epochs
        CUDA_SAFE_CALL(cudaMemset(impl_->d_init_tag_, 0, impl_->k_ * sizeof(int)));
    }

    // Reset instrumentation buffer if enabled
    if (impl_->d_instrumentation_buffer_ != nullptr) {
        size_t buffer_size = impl_->k_ * impl_->m_ * sizeof(int);
        CUDA_SAFE_CALL(cudaMemset(impl_->d_instrumentation_buffer_, -1, buffer_size));
    }
}

inline ws_queues_view ws_queues::view() const {
    ws_queues_view desc;
    desc.unified_base = impl_->d_unified_buffer_;
    desc.k = impl_->k_;
    desc.epoch = impl_->current_epoch_;
    desc.instrumentation_buffer = impl_->d_instrumentation_buffer_;
    desc.m = impl_->m_;
    desc.max_work_items = impl_->max_work_items_;
    return desc;
}

// validate_work_assignment implementation
inline bool ws_queues::validate_work_assignment() const {
    if (!has_instrumentation()) {
        printf("Warning: Instrumentation not enabled, skipping work assignment validation\n");
        return true;
    }

    int k_ = impl_->k_;
    int m_ = impl_->m_;
    int max_work_items_ = impl_->max_work_items_;
    int* d_instrumentation_buffer_ = impl_->d_instrumentation_buffer_;

    int total_slots = k_ * m_;
    int unused_slots = 0;     // Slots that are -1 (not processed)
    int unprocessed_work = 0; // Work items in [0, max_work_items_) that weren't processed
    int invalid_refs = 0;     // Invalid kernel IDs or out-of-range references
    int valid_work = 0;       // Work items successfully processed

    std::vector<int> items_per_kernel(k_, 0);
    std::vector<std::pair<int, int>> error_slots; // (line, item) of first few errors

    // Scan the instrumentation buffer
    // Buffer format: buffer[line * m + item] = kernel_id that processed this slot
    // Slots beyond max_work_items_ should remain unused (-1)
    for (int line = 0; line < k_; line++) {
        for (int item = 0; item < m_; item++) {
            int slot_idx = line * m_ + item;
            int kernel_id = d_instrumentation_buffer_[slot_idx];

            if (kernel_id == -1) {
                // This buffer slot was never used
                // This is expected if slot_idx >= max_work_items_ (spare capacity)
                // But it's an error if slot_idx < max_work_items_ (work was missed)
                if (slot_idx < max_work_items_) {
                    unprocessed_work++;
                    if (error_slots.size() < 5) {
                        error_slots.push_back({line, item});
                    }
                }
                unused_slots++;
            } else if (kernel_id < 0 || kernel_id >= k_) {
                // Invalid kernel ID (shouldn't happen - corruption?)
                printf("  ERROR: Slot %d (deque %d, item %d) has invalid kernel_id %d\n", slot_idx,
                       line, item, kernel_id);
                invalid_refs++;
            } else if (slot_idx >= max_work_items_) {
                // This slot was processed but it's beyond the valid work range
                printf("  ERROR: Slot %d >= max_work_items %d was processed by kernel %d\n",
                       slot_idx, max_work_items_, kernel_id);
                invalid_refs++;
            } else {
                // Valid: slot < max_work_items and has valid kernel ID
                items_per_kernel[kernel_id]++;
                valid_work++;
            }
        }
    }

    // Print diagnostics
    printf("\n=== Work Assignment Validation ===\n");
    printf("  Buffer capacity: %d slots (%d kernels × %d items)\n", total_slots, k_, m_);
    printf("  Expected work items: %d\n", max_work_items_);
    printf("  Work items processed: %d\n", valid_work);
    printf("  Work items missing: %d\n", unprocessed_work);
    printf("  Invalid references: %d\n", invalid_refs);
    printf("  Unused buffer slots: %d\n", unused_slots);

    if (total_slots > max_work_items_) {
        printf("  Note: %d buffer slots are spare capacity\n", total_slots - max_work_items_);
    }

    if (!error_slots.empty() && unprocessed_work > 0) {
        printf("\n  First %zu missing work item slots:\n", error_slots.size());
        for (const auto& err : error_slots) {
            int slot_idx = err.first * m_ + err.second;
            printf("    - Slot %d (deque %d, item %d) was never processed\n", slot_idx, err.first,
                   err.second);
        }
        if (unprocessed_work > 5) {
            printf("    ... and %d more missing\n", unprocessed_work - 5);
        }
    }

    printf("\n  Work distribution by kernel:\n");
    for (int i = 0; i < k_; i++) {
        float percentage = 100.0f * items_per_kernel[i] / max_work_items_;
        printf("    Kernel %d: %d items (%.1f%%)\n", i, items_per_kernel[i], percentage);
    }

    bool success = (unprocessed_work == 0 && invalid_refs == 0 && valid_work == max_work_items_);
    if (success) {
        printf("  ✓ All %d work items assigned exactly once!\n", max_work_items_);
    } else {
        printf("  ✗ VALIDATION FAILED\n");
        if (total_slots < max_work_items_) {
            printf("  ERROR: Buffer too small! Need %d slots but only have %d\n", max_work_items_,
                   total_slots);
        }
    }

    return success;
}

// Get instrumentation data as a host vector
inline std::vector<int> ws_queues::get_instrumentation_data() const {
    if (!has_instrumentation()) {
        fprintf(stderr, "Error: Instrumentation not enabled\n");
        return std::vector<int>();
    }

    int k_ = impl_->k_;
    int m_ = impl_->m_;
    int* d_instrumentation_buffer_ = impl_->d_instrumentation_buffer_;

    int total_slots = k_ * m_;
    std::vector<int> h_buffer(total_slots);

    // Copy from device to host
    CUDA_SAFE_CALL(cudaMemcpy(h_buffer.data(), d_instrumentation_buffer_, total_slots * sizeof(int),
                              cudaMemcpyDeviceToHost));

    return h_buffer;
}

// Compute statistics from instrumentation data
inline ws_stats ws_queues::compute_stats() const {
    if (!has_instrumentation()) {
        fprintf(stderr, "Error: Instrumentation not enabled, cannot compute statistics\n");
        return ws_stats(); // Return empty stats
    }

    auto instr_data = get_instrumentation_data();
    return compute_ws_stats(instr_data, impl_->k_, impl_->m_, impl_->max_work_items_);
}
