#include "warp.h"
#include "work_stealing.h"
#include <cstdint>
#include <map>

namespace {
    // Global registry mapping IDs to ws_queues instances
    // Follows the same pattern as BVH and Mesh in Warp
    std::map<uint64_t, ws_queues*> g_ws_queues_registry;
    uint64_t g_ws_queues_counter = 1;
}

// C interface for Python ctypes binding
extern "C" {

// Create a new ws_queues object (device-agnostic via unified memory)
// m is set per epoch via wp_ws_queues_next_epoch()
WP_API uint64_t wp_ws_queues_create(int k, int enable_instrumentation) {
    try {
        ws_queues* queues = new ws_queues(k, enable_instrumentation != 0);
        uint64_t id = g_ws_queues_counter++;
        g_ws_queues_registry[id] = queues;
        return id;
    } catch (...) {
        return 0;
    }
}

// Destroy ws_queues object
WP_API void wp_ws_queues_destroy(uint64_t id) {
    auto it = g_ws_queues_registry.find(id);
    if (it != g_ws_queues_registry.end()) {
        delete it->second;
        g_ws_queues_registry.erase(it);
    }
}

// Advance to next epoch with new m parameter
WP_API void wp_ws_queues_next_epoch(uint64_t id, int m, int max_work_items) {
    auto it = g_ws_queues_registry.find(id);
    if (it != g_ws_queues_registry.end()) {
        it->second->next_epoch(m, max_work_items);
    }
}

// Get current epoch
WP_API int wp_ws_queues_get_epoch(uint64_t id) {
    auto it = g_ws_queues_registry.find(id);
    if (it != g_ws_queues_registry.end()) {
        return it->second->epoch();
    }
    return -1;
}

// Get number of deques (k)
WP_API int wp_ws_queues_num_deques(uint64_t id) {
    auto it = g_ws_queues_registry.find(id);
    if (it != g_ws_queues_registry.end()) {
        return it->second->num_deques();
    }
    return -1;
}

// Get view (fills provided ws_queues_view structure)
// Returns 1 on success, 0 on failure
WP_API int wp_ws_queues_get_view(uint64_t id, void* view_out) {
    auto it = g_ws_queues_registry.find(id);
    if (it != g_ws_queues_registry.end() && view_out) {
        ws_queues_view* view = static_cast<ws_queues_view*>(view_out);
        *view = it->second->view();
        return 1;
    }
    return 0;
}

// Validate work assignment (requires instrumentation enabled)
WP_API int wp_ws_queues_validate_work_assignment(uint64_t id) {
    auto it = g_ws_queues_registry.find(id);
    if (it != g_ws_queues_registry.end()) {
        return it->second->validate_work_assignment() ? 1 : 0;
    }
    return 0;
}

// Check if instrumentation is enabled
WP_API int wp_ws_queues_has_instrumentation(uint64_t id) {
    auto it = g_ws_queues_registry.find(id);
    if (it != g_ws_queues_registry.end()) {
        return it->second->has_instrumentation() ? 1 : 0;
    }
    return 0;
}

// Get pointer to instrumentation buffer (device memory)
WP_API void* wp_ws_queues_instrumentation_buffer(uint64_t id) {
    auto it = g_ws_queues_registry.find(id);
    if (it != g_ws_queues_registry.end()) {
        return it->second->instrumentation_buffer();
    }
    return nullptr;
}

} // extern "C"
