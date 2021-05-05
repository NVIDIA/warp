#include "bvh.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

#include <vector>

namespace og
{

/*
g_sort_temp = NULL;
g_sort_temp_size = 0;

void sort_reset()
{
    check_cuda(cudaFree(g_sort_temp));

    g_sort_temp_size = 0;
    g_sort_temp = 0;
}

void sort(int* keys, int* values, int n, int numBits)
{
    cub::DoubleBuffer<int> d_keys(keys, keys + n);
    cub::DoubleBuffer<int> d_values(values, values + n);

    size_t sortTempSize;
    cub::DeviceRadixSort::SortPairs(NULL, sortTempSize, d_keys, d_values, int(n), 0, numBits);

    // first time initialization
    if (sortTempSize > g_sort_temp_size)
    {
        sort_reset(lib);

        check_cuda(cudaMalloc(&g_sort_temp, sortTempSize));

        g_sort_temp_size = sortTempSize;
    }

    cub::DeviceRadixSort::SortPairs(g_sort_temp, sortTempSize, d_keys, d_values, n, 0, numBits);

    if (d_keys.Current() != keys)
        check_cuda(cudaMemcpyAsync(keys, d_keys.Current(), sizeof(int)*n, cudaMemcpyDeviceToDevice));

    if (d_values.Current() != values)
        check_cuda(cudaMemcpyAsync(values, d_values.Current(), sizeof(int)*n, cudaMemcpyDeviceToDevice));

}

void inclusive_scan(const int* input, int* output, int n)
{
    // Declare, allocate, and initialize device-accessible pointers for input and output
    
    void* tempStorage = NULL;
    size_t tempStorageSize = 0;

    cub::DeviceScan::InclusiveSum(tempStorage, tempStorageSize, input, output, n);

    // Determine temporary device storage requirements
    if (tempStorageSize > g_sort_temp_size)
    {
        sort_reset(lib);

        check_cuda(cudaMalloc(&g_sort_temp, tempStorageSize));

        g_sort_temp_size = tempStorageSize;
    }

    // Run exclusive prefix sum
    cub::DeviceScan::InclusiveSum(g_sort_temp, tempStorageSize, input, output, n);

}

*/



CUDA_CALLABLE inline int clz(int x)
{
    int n;
    if (x == 0) return 32;
    for (n = 0; ((x & 0x80000000) == 0); n++, x <<= 1);
    return n;
}

CUDA_CALLABLE inline uint32_t part1by2(uint32_t n)
{
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n <<  8)) & 0x0300f00f;
    n = (n ^ (n <<  4)) & 0x030c30c3;
    n = (n ^ (n <<  2)) & 0x09249249;

    return n;
}

// Takes values in the range [0, 1] and assigns an index based Morton codes of length 3*log2(dim) bits 
template <int dim>
CUDA_CALLABLE inline uint32_t morton3(float x, float y, float z)
{
    uint32_t ux = clamp(int(x*dim), 0, dim-1);
    uint32_t uy = clamp(int(y*dim), 0, dim-1);
    uint32_t uz = clamp(int(z*dim), 0, dim-1);

    return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux);
}

CUDA_CALLABLE inline BVHPackedNodeHalf make_node(const vec3& bound, int child, bool leaf)
{
    BVHPackedNodeHalf n;
    n.x = bound.x;
    n.y = bound.y;
    n.z = bound.z;
    n.i = (unsigned int)child;
    n.b = (unsigned int)(leaf?1:0);

    return n;
}

// variation of make_node through volatile pointers used in BuildHierarchy
CUDA_CALLABLE inline void make_node(volatile BVHPackedNodeHalf* n, const vec3& bound, int child, bool leaf)
{
    n->x = bound.x;
    n->y = bound.y;
    n->z = bound.z;
    n->i = (unsigned int)child;
    n->b = (unsigned int)(leaf?1:0);
}



// 

/////////////////////////////////////////////////////////////////////////////////////////////

class MedianBVHBuilder
{	
public:

    void build(BVH& bvh, const bounds3* items, int n);

private:

    bounds3 calc_bounds(const bounds3* bounds, const int* indices, int start, int end);

    int partition_median(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds);
    int partition_midpoint(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds);

    int build_recursive(BVH& bvh, const bounds3* bounds, int* indices, int start, int end, int depth, int parent);
};

/////////////////////////////////////////////////////////////////////////////////////////////

class LinearBVHBuilderCPU
{
public:
    
    void build(BVH& bvh, const bounds3* items, int n);

private:

    // calculate Morton codes
    struct KeyIndexPair
    {
        uint32_t key;
        int index;

        inline bool operator < (const KeyIndexPair& rhs) const { return key < rhs.key; }
    };	

    bounds3 calc_bounds(const bounds3* bounds, const KeyIndexPair* keys, int start, int end);
    int find_split(const KeyIndexPair* pairs, int start, int end);
    int build_recursive(BVH& bvh, const KeyIndexPair* keys, const bounds3* bounds, int start, int end, int depth);

};

/*
/////////////////////////////////////////////////////////////////////////////////////////////

// Create a linear BVH as described in Fast and Simple Agglomerative LBVH construction
// this is a bottom-up clustering method that outputs one node per-leaf 
//
class LinearBVHBuilderGPU
{
public:

    LinearBVHBuilderGPU();
    ~LinearBVHBuilderGPU();

    // takes a bvh (host ref), and pointers to the GPU lower and upper bounds for each triangle
    // the priorities array allows specifying a 5-bit [0-31] value priority such that lower priority
    // leaves will always be returned first
    void build(NvFlexLibrary* lib, BVH& bvh, const Vec4* lowers, const Vec4* uppers, const int* priorities, int n, bounds3* total_bounds);

private:

    // temporary data used during building
    int* mIndices;
    int* mKeys;
    int* mDeltas;
    int* mRangeLefts;
    int* mRangeRights;
    int* mNumChildren;

    // bounds data when total item bounds built on GPU
    vec3* mTotalLower;
    vec3* mTotalUpper;
    vec3* mTotalInvEdges;

    int mMaxItems;

};
*/

//////////////////////////////////////////////////////////////////////

void MedianBVHBuilder::build(BVH& bvh, const bounds3* items, int n)
{
    memset(&bvh, 0, sizeof(BVH));

    bvh.max_nodes = 2*n;

    bvh.node_lowers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_uppers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_parents = new int[bvh.max_nodes];

    bvh.num_nodes = 0;

    // root is always in first slot for top down builders
    bvh.root = 0;
    
    std::vector<int> indices(n);
    for (int i=0; i < n; ++i)
        indices[i] = i;

    build_recursive(bvh, items, &indices[0], 0, n, 0, -1);
}


bounds3 MedianBVHBuilder::calc_bounds(const bounds3* bounds, const int* indices, int start, int end)
{
    bounds3 u;

    for (int i=start; i < end; ++i)
        u = bounds_union(u, bounds[indices[i]]);

    return u;
}

struct PartitionPredicateMedian
{
    PartitionPredicateMedian(const bounds3* bounds, int a) : bounds(bounds), axis(a) {}

    bool operator()(int a, int b) const
    {
        return bounds[a].center()[axis] < bounds[b].center()[axis];
    }

    const bounds3* bounds;
    int axis;
};


int MedianBVHBuilder::partition_median(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds)
{
    assert(end-start >= 2);

    vec3 edges = range_bounds.edges();

    int axis = longest_axis(edges);

    const int k = (start+end)/2;

    std::nth_element(&indices[start], &indices[k], &indices[end], PartitionPredicateMedian(&bounds[0], axis));

    return k;
}	
    
struct PartitionPredictateMidPoint
{
    PartitionPredictateMidPoint(const bounds3* bounds, int a, float m) : bounds(bounds), axis(a), mid(m) {}

    bool operator()(int index) const 
    {
        return bounds[index].center()[axis] <= mid;
    }

    const bounds3* bounds;
    int axis;
    float mid;
};


int MedianBVHBuilder::partition_midpoint(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds)
{
    assert(end-start >= 2);

    vec3 edges = range_bounds.edges();
    vec3 center = range_bounds.center();

    int axis = longest_axis(edges);
    float mid = center[axis];

    int* upper = std::partition(indices+start, indices+end, PartitionPredictateMidPoint(&bounds[0], axis, mid));

    int k = upper-indices;

    // if we failed to split items then just split in the middle
    if (k == start || k == end)
        k = (start+end)/2;


    return k;
}

int MedianBVHBuilder::build_recursive(BVH& bvh, const bounds3* bounds, int* indices, int start, int end, int depth, int parent)
    {
        assert(start < end);

        const int n = end-start;
        const int node_index = bvh.num_nodes++;

        assert(node_index < bvh.max_nodes);

        if (depth > bvh.max_depth)
            bvh.max_depth = depth;

        bounds3 b = calc_bounds(bounds, indices, start, end);
        
        const int kMaxItemsPerLeaf = 1;

        if (n <= kMaxItemsPerLeaf)
        {
            bvh.node_lowers[node_index] = make_node(b.lower, indices[start], true);
            bvh.node_uppers[node_index] = make_node(b.upper, indices[start], false);
            bvh.node_parents[node_index] = parent;
        }
        else
        {
            //int split = partition_midpoint(bounds, indices, start, end, b);
            int split = partition_median(bounds, indices, start, end, b);
        
            int left_child = build_recursive(bvh, bounds, indices, start, split, depth+1, node_index);
            int right_child = build_recursive(bvh,bounds, indices, split, end, depth+1, node_index);
            
            bvh.node_lowers[node_index] = make_node(b.lower, left_child, false);
            bvh.node_uppers[node_index] = make_node(b.upper, right_child, false);		
            bvh.node_parents[node_index] = parent;
        }

        return node_index;
    }



/////////////////////////////////////////////////////////////////////////////////////////


void LinearBVHBuilderCPU::build(BVH& bvh, const bounds3* items, int n)
{
    memset(&bvh, 0, sizeof(BVH));

    bvh.max_nodes = 2*n;

    bvh.node_lowers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_uppers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.num_nodes = 0;

    // root is always in first slot for top down builders
    bvh.root = 0;

    std::vector<KeyIndexPair> keys;
    keys.reserve(n);

    bounds3 total_bounds;
    for (int i=0; i < n; ++i)
        total_bounds = bounds_union(total_bounds, items[i]);

    // ensure non-zero edge length in all dimensions
    total_bounds.expand(0.001f);

    vec3 edges = total_bounds.edges();
    
    vec3 invEdges;
    invEdges.x = 1.0f/edges.x;
    invEdges.y = 1.0f/edges.y;
    invEdges.z = 1.0f/edges.z;

    for (int i=0; i < n; ++i)
    {
        vec3 center = items[i].center();
        vec3 local = (center-total_bounds.lower)*invEdges;

        KeyIndexPair l;
        l.key = morton3<1024>(local.x, local.y, local.z);
        l.index = i;

        keys.push_back(l);
    }

    // sort by key
    std::sort(keys.begin(), keys.end());

    build_recursive(bvh, &keys[0], items,  0, n, 0);

    printf("Created BVH for %d items with %d nodes, max depth of %d\n", n, bvh.num_nodes, bvh.max_depth);
}


inline bounds3 LinearBVHBuilderCPU::calc_bounds(const bounds3* bounds, const KeyIndexPair* keys, int start, int end)
{
    bounds3 u;

    for (int i=start; i < end; ++i)
        u = bounds_union(u, bounds[keys[i].index]);

    return u;
}

inline int LinearBVHBuilderCPU::find_split(const KeyIndexPair* pairs, int start, int end)
{
    if (pairs[start].key == pairs[end-1].key)
        return (start+end)/2;

    // find split point between keys, xor here means all bits 
    // of the result are zero up until the first differing bit
    int commonPrefix = clz(pairs[start].key ^ pairs[end-1].key);

    // use binary search to find the point at which this bit changes
    // from zero to a 1		
    const int mask = 1 << (31-commonPrefix);

    while (end-start > 0)
    {
        int index = (start+end)/2;

        if (pairs[index].key&mask)
        {
            end = index;
        }
        else
            start = index+1;
    }

    assert(start == end);

    return start;
}

int LinearBVHBuilderCPU::build_recursive(BVH& bvh, const KeyIndexPair* keys, const bounds3* bounds, int start, int end, int depth)
{
    assert(start < end);

    const int n = end-start;
    const int node_index = bvh.num_nodes++;

    assert(node_index < bvh.max_nodes);

    if (depth > bvh.max_depth)
        bvh.max_depth = depth;

    bounds3 b = calc_bounds(bounds, keys, start, end);
        
    const int kMaxItemsPerLeaf = 1;

    if (n <= kMaxItemsPerLeaf)
    {
        bvh.node_lowers[node_index] = make_node(b.lower, keys[start].index, true);
        bvh.node_uppers[node_index] = make_node(b.upper, keys[start].index, false);
    }
    else
    {
        int split = find_split(keys, start, end);
        
        int left_child = build_recursive(bvh, keys, bounds, start, split, depth+1);
        int right_child = build_recursive(bvh, keys, bounds, split, end, depth+1);
            
        bvh.node_lowers[node_index] = make_node(b.lower, left_child, false);
        bvh.node_uppers[node_index] = make_node(b.upper, right_child, false);		
    }

    return node_index;
}


////////////////////////////////////////////////////////



/*
// build kernels
__global__ void CalculateTrianglebounds3(const vec3* __restrict__ vertices, const int* __restrict__ indices, int numTris, Vec4* __restrict__ lowers, Vec4* __restrict__ uppers)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < numTris)
    {
        vec3 p = vec3(vertices[indices[index * 3 + 0]]);
        vec3 q = vec3(vertices[indices[index * 3 + 1]]);
        vec3 r = vec3(vertices[indices[index * 3 + 2]]);

        vec3 lower = min(p, min(q, r));
        vec3 upper = max(p, max(q, r));

        // use vec4 type for 16-byte stores
        ((vec4*)lowers)[index] = make_vec4(lower.x, lower.y, lower.z, 0.0f);
        ((vec4*)uppers)[index] = make_vec4(upper.x, upper.y, upper.z, 0.0f);
    }
}


__global__ void CalculateMortonCodes(const Vec4* __restrict__ itemLowers, const Vec4* __restrict__ itemUppers, const int* __restrict__ itemPriorities, int n, const vec3* gridLower, const vec3* gridInvEdges, int* __restrict__ indices, int* __restrict__ keys)
{
    const int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        vec3 lower = vec3(itemLowers[index]);
        vec3 upper = vec3(itemUppers[index]);

        vec3 center = 0.5f*(lower+upper);

        vec3 local = (center-gridLower[0])*gridInvEdges[0];
        
        // 9-bit Morton codes stored in lower 27bits (512^3 effective resolution)
        // 5-bit priority code stored in the upper 5-bits
        int key = morton3<512>(local.x, local.y, local.z);

        // we invert priorities (so that higher priority items appear first in sorted order)
        if (itemPriorities)
            key |= (~itemPriorities[index])<<27;

        indices[index] = index;
        keys[index] = key;
    }
}

// calculate the index of the first differing bit between two adjacent Morton keys
__global__ void CalculateKeyDeltas(const int* __restrict__ keys, int* __restrict__ deltas, int n)
{
    const int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        int a = keys[index];
        int b = keys[index+1];

        int x = a^b;
        
        deltas[index] = x;// __clz(x);
    }
}

__global__ void BuildLeaves(const Vec4* __restrict__ itemLowers, const Vec4* __restrict__ itemUppers, int n, const int* __restrict__ indices, int* __restrict__ rangeLefts, int* __restrict__ rangeRights, BVHPackedNodeHalf* __restrict__ lowers, BVHPackedNodeHalf* __restrict__ uppers)
{
    const int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        const int item = indices[index];

        const vec3 lower = vec3(itemLowers[item]);
        const vec3 upper = vec3(itemUppers[item]);

        // write leaf nodes 
        lowers[index] = make_node(lower, item, true);
        uppers[index] = make_node(upper, item, false);

        // write leaf key ranges
        rangeLefts[index] = index;
        rangeRights[index] = index;
    }
}

// this bottom-up process assigns left and right children and combines bounds to form internal nodes
// there is one thread launched per-leaf node, each thread calculates it's parent node and assigns
// itself to either the left or right parent slot, the last child to complete the parent and moves
// up the hierarchy
__global__ void BuildHierarchy(int n, int* root, const int* __restrict__ deltas,  int* __restrict__ numChildren, volatile int* __restrict__ rangeLefts, volatile int* __restrict__ rangeRights, volatile BVHPackedNodeHalf* __restrict__ lowers, volatile BVHPackedNodeHalf* __restrict__ uppers)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        const int internalOffset = n;

        for (;;)
        {
            int left = rangeLefts[index];
            int right = rangeRights[index];

            // check if we are the root node, if so then store out our index and terminate
            if (left == 0 && right == n-1)
            {					
                *root = index;
                break;
            }

            int childCount = 0;

            int parent;

            if (left == 0 || (right != n-1 && deltas[right] < deltas[left-1]))
            {
                parent = right + internalOffset;

                // set parent left child
                lowers[parent].i = index;				
                rangeLefts[parent] = left;

                // ensure above writes are visible to all threads
                __threadfence();
                
                childCount = atomicAdd(&numChildren[parent], 1);
            }
            else
            {
                parent = left + internalOffset - 1;
                
                // set parent right child
                uppers[parent].i = index;
                rangeRights[parent] = right;

                // ensure above writes are visible to all threads
                __threadfence();
                
                childCount = atomicAdd(&numChildren[parent], 1);
            }

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the the next parent in the hierarchy
            if (childCount == 1)
            {
                const int left_child = lowers[parent].i;
                const int right_child = uppers[parent].i;

                vec3 left_lower = vec3(lowers[left_child].x,
                                      lowers[left_child].y, 
                                      lowers[left_child].z);

                vec3 left_upper = vec3(uppers[left_child].x,
                                      uppers[left_child].y, 
                                      uppers[left_child].z);

                vec3 right_lower = vec3(lowers[right_child].x,
                                       lowers[right_child].y,
                                       lowers[right_child].z);


                vec3 right_upper = vec3(uppers[right_child].x, 
                                       uppers[right_child].y, 
                                       uppers[right_child].z);

                // bounds_union of child bounds
                vec3 lower = Min(left_lower, right_lower);
                vec3 upper = Max(left_upper, right_upper);
                
                // write new BVH nodes
                make_node(lowers+parent, lower, left_child, false);
                make_node(uppers+parent, upper, right_child, false);

                // move onto processing the parent
                index = parent;
            }
            else
            {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }		
    }
}


__global__ void Computetotal_bounds(const Vec4* itemLowers, const Vec4* itemUppers, vec3* totalLower, vec3* totalUpper, int numItems)
{
     typedef cub::BlockReduce<vec3, kNumThreadsPerBlock> BlockReduce;

     __shared__ typename BlockReduce::TempStorage temp_storage;

     const int blockStart = blockDim.x*blockIdx.x;
     const int numValid = min(numItems-blockStart, blockDim.x);

     const int tid = blockStart + threadIdx.x;

     if (tid < numItems)
     {
         vec3 lower = vec3(itemLowers[tid]);
         vec3 upper = vec3(itemUppers[tid]);

         vec3 blockUpper = BlockReduce(temp_storage).Reduce(upper, vec3Max(), numValid);

         // sync threads because second reduce uses same temp storage as first
         __syncthreads();

         vec3 blockLower = BlockReduce(temp_storage).Reduce(lower, vec3Min(), numValid);

         if (threadIdx.x == 0)
         {
             // write out block results, expanded by the radius
             AtomicMaxvec3(totalUpper, blockUpper);
             AtomicMinvec3(totalLower, blockLower);
         }	 
    }
}

// compute inverse edge length, this is just done on the GPU to avoid a CPU->GPU sync point
__global__ void ComputeTotalInvEdges(const vec3* totalLower, const vec3* totalUpper, vec3* totalInvEdges)
{
    vec3 edges = (totalUpper[0]-totalLower[0]);
    edges += vec3(0.0001f);

    totalInvEdges[0] = vec3(1.0f/edges.x, 1.0f/edges.y, 1.0f/edges.z);
}



LinearBVHBuilderGPU::LinearBVHBuilderGPU() 
    : mIndices(NULL)
    , mKeys(NULL)
    , mDeltas(NULL)
    , mRangeLefts(NULL)
    , mRangeRights(NULL)
    , mNumChildren(NULL)
    , mMaxItems(0)
    , mTotalLower(NULL)
    , mTotalUpper(NULL)
    , mTotalInvEdges(NULL)
{
    alloc_device(&mTotalLower, sizeof(vec3));
    alloc_device(&mTotalUpper, sizeof(vec3));
    alloc_device(&mTotalInvEdges, sizeof(vec3));
}

LinearBVHBuilderGPU::~LinearBVHBuilderGPU()
{
    free_device(mIndices);
    free_device(mKeys);
    free_device(mDeltas);

    free_device(mRangeLefts);
    free_device(mRangeRights);
    free_device(mNumChildren);

    free_device(mTotalLower);
    free_device(mTotalUpper);
    free_device(mTotalInvEdges);

}


void LinearBVHBuilderGPU::build(NvFlexLibrary* lib, BVH& bvh, const Vec4* itemLowers, const Vec4* itemUppers, const int* itemPriorities, int numItems, bounds3* total_bounds)
{
    const int maxNodes = 2*numItems;

    bvh_resize(bvh, maxNodes);

    if (numItems > mMaxItems)
    {
        const int itemsToAlloc = (numItems*3)/2;
        const int nodesToAlloc = (maxNodes*3)/2;

        // reallocate temporary storage if necessary
        free_device(mIndices);
        free_device(mKeys);
        free_device(mDeltas);
        free_device(mRangeLefts);
        free_device(mRangeRights);
        free_device(mNumChildren);

        alloc_device(&mIndices, sizeof(int)*itemsToAlloc*2);	// *2 for radix sort
        alloc_device(&mKeys, sizeof(int)*itemsToAlloc*2);	// *2 for radix sort
        alloc_device(&mDeltas, sizeof(int)*itemsToAlloc);	// highest differenting bit between keys for item i and i+1
        alloc_device(&mRangeLefts, sizeof(int)*nodesToAlloc);
        alloc_device(&mRangeRights, sizeof(int)*nodesToAlloc);
        alloc_device(&mNumChildren, sizeof(int)*nodesToAlloc);

        mMaxItems = itemsToAlloc;
    }

    const int kNumThreadsPerBlock = 256;
    const int kNumBlocks = (numItems+kNumThreadsPerBlock-1)/kNumThreadsPerBlock;

    // if total bounds supplied by the host then we just 
    // compute our edge length and upload it to the GPU directly
    if (total_bounds)
    {
        // calculate Morton codes
        vec3 edges = (*total_bounds).edges();
        edges += vec3(0.0001f);

        vec3 invEdges = vec3(1.0f/edges.x, 1.0f/edges.y, 1.0f/edges.z);
        
        check_cuda(cudaMemcpyAsync(mTotalLower, &total_bounds->lower.x, sizeof(vec3), cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpyAsync(mTotalUpper, &total_bounds->upper.x, sizeof(vec3), cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpyAsync(mTotalInvEdges, &invEdges.x, sizeof(vec3), cudaMemcpyHostToDevice));
    }
    else
    {
        static const vec3 upper(-FLT_MAX);
        static const vec3 lower(FLT_MAX);

        check_cuda(cudaMemcpyAsync(mTotalLower, &lower, sizeof(lower), cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpyAsync(mTotalUpper, &upper, sizeof(upper), cudaMemcpyHostToDevice));

        // compute the bounds bounds_union on the GPU
        Computetotal_bounds<<<kNumBlocks, kNumThreadsPerBlock>>>(itemLowers, itemUppers, mTotalLower, mTotalUpper, numItems);

        // compute the total edge length
        ComputeTotalInvEdges<<<1, 1>>>(mTotalLower, mTotalUpper, mTotalInvEdges);
    }

    // assign 30-bit Morton code based on the centroid of each triangle and bounds for each leaf
    CalculateMortonCodes<<<kNumBlocks, kNumThreadsPerBlock>>>(itemLowers, itemUppers, itemPriorities, numItems, mTotalLower, mTotalInvEdges, mIndices, mKeys);
    
    // sort items based on Morton key (note the 32-bit sort key corresponds to the template parameter to morton3, i.e. 3x9 bit keys combined)
    sort(lib, mKeys, mIndices, numItems, 32);

    // calculate deltas between adjacent keys
    CalculateKeyDeltas<<<kNumBlocks, kNumThreadsPerBlock>>>(mKeys, mDeltas, numItems-1);

    // initialize leaf nodes
    BuildLeaves<<<kNumBlocks, kNumThreadsPerBlock>>>(itemLowers, itemUppers, numItems, mIndices, mRangeLefts, mRangeRights, bvh.node_lowers, bvh.node_uppers);

#if 1
    
    // reset children count, this is our atomic counter so we know when an internal node is complete, only used during building
    check_cuda(cudaMemsetAsync(mNumChildren, 0, sizeof(int)*maxNodes));

    // build the tree and internal node bounds
    BuildHierarchy<<<kNumBlocks, kNumThreadsPerBlock>>>(numItems, bvh.root, mDeltas, mNumChildren, mRangeLefts, mRangeRights, bvh.node_lowers, bvh.node_uppers);

#else

    // Reference CPU implementation of BuildHierarchy()

    std::vector<int> indices(numItems);
    std::vector<int> keys(numItems);
    std::vector<int> deltas(numItems-1);
    std::vector<BVHPackedNodeHalf> lowers(bvh.max_nodes);
    std::vector<BVHPackedNodeHalf> uppers(bvh.max_nodes);

    std::vector<int> rangeLefts(bvh.max_nodes);
    std::vector<int> rangeRights(bvh.max_nodes);
    std::vector<int> numChildren(bvh.max_nodes);

    check_cuda(cudaMemcpy(&indices[0], mIndices, sizeof(int)*numItems, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(&keys[0], mKeys, sizeof(int)*numItems, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(&deltas[0], mDeltas, sizeof(int)*(numItems-1), cudaMemcpyDeviceToHost));

    check_cuda(cudaMemcpy(&lowers[0], bvh.node_lowers, bvh.max_nodes*sizeof(BVHPackedNodeHalf), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(&uppers[0], bvh.node_uppers, bvh.max_nodes*sizeof(BVHPackedNodeHalf), cudaMemcpyDeviceToHost));

    // zero child pointers for internal nodes so we can do checking
    for (int i=numItems; i < bvh.max_nodes; ++i)
    {
        lowers[i].i = 0;
        uppers[i].i = 0;
    }

    const int internalOffset = numItems;
    int rootNode = -1;

    // initialize leaf ranges [left, right]
    for (int i=0; i < numItems; ++i)
    {
        rangeLefts[i] = i;
        rangeRights[i] = i;
    }

    for (int i=numItems; i < maxNodes;++i)
    {
        rangeLefts[i] = -1;
        rangeRights[i] = -1;
    }
    
    for (int i=0; i < numItems && rootNode == -1; ++i)
    {
        int index = i;

        for (;;)
        {
            int left = rangeLefts[index];
            int right = rangeRights[index];

            int parent;

            if (left == 0 || (right != numItems-1 && deltas[right] < deltas[left-1]))
            {
                parent = right + internalOffset;

                // check that no other node has assigned itself to the parent
                assert(lowers[parent].i == 0);
                assert(rangeLefts[parent] == -1);
                
                // set parent's left child
                lowers[parent].i = index;
                
                rangeLefts[parent] = left;
                numChildren[parent]++;
            }
            else
            {
                parent = left + internalOffset - 1;
                
                // check that no other node has assigned itself to the parent
                assert(uppers[parent].i == 0);
                assert(rangeRights[parent] == -1);

                // set parent's right child
                uppers[parent].i = index;

                rangeRights[parent] = right;
                numChildren[parent]++;
            }

            assert(numChildren[parent] <= 2);

            // assign bounds to parent
            if (numChildren[parent] == 2)
            {
                const int left_child = lowers[parent].i;
                const int right_child = uppers[parent].i;

                vec3 lower = Min(vec3(&lowers[left_child].x), vec3(&lowers[right_child].x));
                vec3 upper = Max(vec3(&uppers[left_child].x), vec3(&uppers[right_child].x));
                
                lowers[parent] = { lower.x, lower.y, lower.z, (unsigned int)left_child, 0 };
                uppers[parent] = { upper.x, upper.y, upper.z, (unsigned int)right_child, 0 };

                // store index of root node
                if (rangeLefts[parent] == 0 && rangeRights[parent] == numItems-1 && numChildren[parent] == 2)
                {					
                    rootNode = parent;
                    break;
                }

                index = parent;
            }
            else
            {
                // parent not ready, terminate
                break;
            }
        }
    }

    for (int i=numItems; i < maxNodes-1; ++i)
    {
        assert(lowers[i].i != rootNode);
        assert(uppers[i].i != rootNode);

        assert(numChildren[i] == 2);

    }

    check_cuda(cudaMemcpy(bvh.node_lowers, &lowers[0], sizeof(BVHPackedNodeHalf)*bvh.max_nodes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(bvh.node_uppers, &uppers[0], sizeof(BVHPackedNodeHalf)*bvh.max_nodes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(bvh.root, &rootNode, sizeof(int), cudaMemcpyHostToDevice));

#endif

}
*/



// create only happens on host currently, use bvh_clone() to transfer BVH To device
BVH bvh_create(const bounds3* bounds, int num_bounds)
{
    BVH bvh;
    memset(&bvh, 0, sizeof(bvh));

   // bvh.node_lowers = (BVHPackedNodeHalf*)alloc_host(sizeof(BVHPackedNodeHalf)*num_bounds*2);
   // bvh.node_uppers = (BVHPackedNodeHalf*)alloc_host(sizeof(BVHPackedNodeHalf)*num_bounds*2);
   // bvh.num_nodes = num_bounds;

    MedianBVHBuilder builder;
    builder.build(bvh, bounds, num_bounds);

    return bvh;
}

void bvh_destroy_host(BVH& bvh)
{
    delete[] bvh.node_lowers;
    delete[] bvh.node_uppers;
    delete[] bvh.node_parents;

    bvh.node_lowers = 0;
    bvh.node_uppers = 0;
    bvh.max_nodes = 0;
    bvh.num_nodes = 0;
}

void bvh_destroy_device(BVH& bvh)
{
    free_device(bvh.node_lowers); bvh.node_lowers = NULL;
    free_device(bvh.node_uppers); bvh.node_uppers = NULL;
    free_device(bvh.node_parents); bvh.node_parents = NULL;
    free_device(bvh.node_counts); bvh.node_counts = NULL;
}


BVH bvh_clone(const BVH& bvh_host)
{
    BVH bvh_device = bvh_host;

    bvh_device.node_lowers = (BVHPackedNodeHalf*)alloc_device(sizeof(BVHPackedNodeHalf)*bvh_host.max_nodes);
    bvh_device.node_uppers = (BVHPackedNodeHalf*)alloc_device(sizeof(BVHPackedNodeHalf)*bvh_host.max_nodes);
    bvh_device.node_parents = (int*)alloc_device(sizeof(int)*bvh_host.max_nodes);
    bvh_device.node_counts = (int*)alloc_device(sizeof(int)*bvh_host.max_nodes);

    // copy host data to device
    memcpy_h2d(bvh_device.node_lowers, bvh_host.node_lowers, sizeof(BVHPackedNodeHalf)*bvh_host.max_nodes);
    memcpy_h2d(bvh_device.node_uppers, bvh_host.node_uppers, sizeof(BVHPackedNodeHalf)*bvh_host.max_nodes);
    memcpy_h2d(bvh_device.node_parents, bvh_host.node_parents, sizeof(int)*bvh_host.max_nodes);

    return bvh_device;
}

void bvh_refit_recursive(BVH& bvh, int index, const bounds3* bounds)
{
    BVHPackedNodeHalf& lower = bvh.node_lowers[index];
    BVHPackedNodeHalf& upper = bvh.node_uppers[index];

    if (lower.b)
    {
        const int leaf_index = lower.i;

        (vec3&)lower = bounds[leaf_index].lower;
        (vec3&)upper = bounds[leaf_index].upper;
    }
    else
    {
        int left_index = lower.i;
        int right_index = upper.i;

        bvh_refit_recursive(bvh, left_index, bounds);
        bvh_refit_recursive(bvh, right_index, bounds);

        // compute union of children
        const vec3& left_lower = (vec3&)bvh.node_lowers[left_index];
        const vec3& left_upper = (vec3&)bvh.node_uppers[left_index];

        const vec3& right_lower = (vec3&)bvh.node_lowers[right_index];
        const vec3& right_upper = (vec3&)bvh.node_uppers[right_index];

        // union of child bounds
        vec3 new_lower = min(left_lower, right_lower);
        vec3 new_upper = max(left_upper, right_upper);
        
        // write new BVH nodes
        (vec3&)lower = new_lower;
        (vec3&)upper = new_upper;
    }
}

void bvh_refit_host(BVH& bvh, const bounds3* b)
{
    bvh_refit_recursive(bvh, 0, b);
}


//__global__ void bvh_refit_kernel(int n, const int* __restrict__ parents, volatile int* __restrict__ child_count, volatile BVHPackedNodeHalf* __restrict__ lowers, volatile BVHPackedNodeHalf* __restrict__ uppers, const bounds3* bounds)
__global__ void bvh_refit_kernel(int n, const int* __restrict__ parents, int* __restrict__ child_count, BVHPackedNodeHalf* __restrict__ lowers, BVHPackedNodeHalf* __restrict__ uppers, const bounds3* bounds)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        bool leaf = lowers[index].b;

        if (leaf)
        {
            // update the leaf node
            const int leaf_index = lowers[index].i;
            const bounds3& b = bounds[leaf_index];

            make_node(lowers+index, b.lower, leaf_index, true);
            make_node(uppers+index, b.upper, 0, false);
        }
        else
        {
            // only keep leaf threads
            return;
        }

        // update hierarchy
        for (;;)
        {
            int parent = parents[index];
            
            // reached root
            if (parent == -1)
                return;

            // ensure all writes are visible
            __threadfence();
         
            int finished = atomicAdd(&child_count[parent], 1);

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the the next parent in the hierarchy
            if (finished == 1)
            {
                const int left_child = lowers[parent].i;
                const int right_child = uppers[parent].i;

                vec3 left_lower = vec3(lowers[left_child].x,
                                       lowers[left_child].y, 
                                       lowers[left_child].z);

                vec3 left_upper = vec3(uppers[left_child].x,
                                       uppers[left_child].y, 
                                       uppers[left_child].z);

                vec3 right_lower = vec3(lowers[right_child].x,
                                       lowers[right_child].y,
                                       lowers[right_child].z);


                vec3 right_upper = vec3(uppers[right_child].x, 
                                       uppers[right_child].y, 
                                       uppers[right_child].z);

                // union of child bounds
                vec3 lower = min(left_lower, right_lower);
                vec3 upper = max(left_upper, right_upper);
                
                // write new BVH nodes
                make_node(lowers+parent, lower, left_child, false);
                make_node(uppers+parent, upper, right_child, false);

                // move onto processing the parent
                index = parent;
            }
            else
            {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }		
    }
}


void bvh_refit_device(BVH& bvh, const bounds3* b)
{
    // clear child counters
    memset_device(bvh.node_counts, 0, sizeof(int)*bvh.max_nodes);

    const int num_threads_per_block = 256;
    const int num_blocks = (bvh.max_nodes + num_threads_per_block - 1)/num_threads_per_block;
  
    bvh_refit_kernel<<<num_blocks, num_threads_per_block>>>(bvh.max_nodes, bvh.node_parents, bvh.node_counts, bvh.node_lowers, bvh.node_uppers, b);
}


} // namespace og



