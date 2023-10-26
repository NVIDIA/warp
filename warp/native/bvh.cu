/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "warp.h"
#include "cuda_util.h"
#include "bvh.h"
#include "sort.h"

#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/cub.cuh>


namespace wp
{

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
    ContextGuard guard(bvh.context);

    // clear child counters
    memset_device(WP_CURRENT_CONTEXT, bvh.node_counts, 0, sizeof(int)*bvh.max_nodes);

    wp_launch_device(WP_CURRENT_CONTEXT, bvh_refit_kernel, bvh.num_bounds, (bvh.num_bounds, bvh.node_parents, bvh.node_counts, bvh.node_lowers, bvh.node_uppers, b));
}

__global__ void set_bounds_from_lowers_and_uppers(int n, bounds3* b, const vec3* lowers, const vec3* uppers)
{
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < n)
    {
        b[tid] = bounds3(lowers[tid], uppers[tid]);
    }
}


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
    void build(BVH& bvh, const vec3* lowers, const vec3* uppers, const int* priorities, int n, bounds3* total_bounds);

private:

    // temporary data used during building
    int* indices;
    int* keys;
    int* deltas;
    int* range_lefts;
    int* range_rights;
    int* num_children;

    // bounds data when total item bounds built on GPU
    vec3* total_lower;
    vec3* total_upper;
    vec3* total_inv_edges;

    int max_items;

};

////////////////////////////////////////////////////////



__global__ void compute_morton_codes(const vec3* __restrict__ item_lowers, const vec3* __restrict__ item_uppers, int n, const vec3* grid_lower, const vec3* grid_inv_edges, int* __restrict__ indices, int* __restrict__ keys)
{
    const int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        vec3 lower = item_lowers[index];
        vec3 upper = item_uppers[index];

        vec3 center = 0.5f*(lower+upper);

        vec3 local = cw_mul((center-grid_lower[0]), grid_inv_edges[0]);
        
        // 10-bit Morton codes stored in lower 30bits (1024^3 effective resolution)
        int key = morton3<1024>(local[0], local[1], local[2]);

        indices[index] = index;
        keys[index] = key;
    }
}

// calculate the index of the first differing bit between two adjacent Morton keys
__global__ void compute_key_deltas(const int* __restrict__ keys, int* __restrict__ deltas, int n)
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

__global__ void build_leaves(const vec3* __restrict__ item_lowers, const vec3* __restrict__ item_uppers, int n, const int* __restrict__ indices, int* __restrict__ range_lefts, int* __restrict__ range_rights, BVHPackedNodeHalf* __restrict__ lowers, BVHPackedNodeHalf* __restrict__ uppers)
{
    const int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        const int item = indices[index];

        vec3 lower = item_lowers[item];
        vec3 upper = item_uppers[item];

        // write leaf nodes 
        lowers[index] = make_node(lower, item, true);
        uppers[index] = make_node(upper, item, false);

        // write leaf key ranges
        range_lefts[index] = index;
        range_rights[index] = index;
    }
}

// this bottom-up process assigns left and right children and combines bounds to form internal nodes
// there is one thread launched per-leaf node, each thread calculates it's parent node and assigns
// itself to either the left or right parent slot, the last child to complete the parent and moves
// up the hierarchy
__global__ void build_hierarchy(int n, int* root, const int* __restrict__ deltas,  int* __restrict__ num_children, volatile int* __restrict__ range_lefts, volatile int* __restrict__ range_rights, volatile int* __restrict__ parents, volatile BVHPackedNodeHalf* __restrict__ lowers, volatile BVHPackedNodeHalf* __restrict__ uppers)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        const int internal_offset = n;

        for (;;)
        {
            int left = range_lefts[index];
            int right = range_rights[index];

            // check if we are the root node, if so then store out our index and terminate
            if (left == 0 && right == n-1)
            {					
                *root = index;
                parents[index] = -1;

                break;
            }

            int childCount = 0;

            int parent;

            if (left == 0 || (right != n-1 && deltas[right] < deltas[left-1]))
            {
                parent = right + internal_offset;

                // set parent left child
                parents[index] = parent;
                lowers[parent].i = index;				
                range_lefts[parent] = left;

                // ensure above writes are visible to all threads
                __threadfence();
                
                childCount = atomicAdd(&num_children[parent], 1);
            }
            else
            {
                parent = left + internal_offset - 1;
                
                // set parent right child
                parents[index] = parent;
                uppers[parent].i = index;
                range_rights[parent] = right;

                // ensure above writes are visible to all threads
                __threadfence();
                
                childCount = atomicAdd(&num_children[parent], 1);
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

CUDA_CALLABLE inline vec3 Vec3Max(const vec3& a, const vec3& b) { return wp::max(a, b); }
CUDA_CALLABLE inline vec3 Vec3Min(const vec3& a, const vec3& b) { return wp::min(a, b); }

__global__ void compute_total_bounds(const vec3* item_lowers, const vec3* item_uppers, vec3* total_lower, vec3* total_upper, int num_items)
{
     typedef cub::BlockReduce<vec3, 256> BlockReduce;

     __shared__ typename BlockReduce::TempStorage temp_storage;

     const int blockStart = blockDim.x*blockIdx.x;
     const int numValid = ::min(num_items-blockStart, blockDim.x);

     const int tid = blockStart + threadIdx.x;

     if (tid < num_items)
     {
        vec3 lower = item_lowers[tid];
        vec3 upper = item_uppers[tid];

         vec3 block_upper = BlockReduce(temp_storage).Reduce(upper, Vec3Max, numValid);

         // sync threads because second reduce uses same temp storage as first
         __syncthreads();

         vec3 block_lower = BlockReduce(temp_storage).Reduce(lower, Vec3Min, numValid);

         if (threadIdx.x == 0)
         {
             // write out block results, expanded by the radius
             atomic_max(total_upper, block_upper);
             atomic_min(total_lower, block_lower);
         }	 
    }
}

// compute inverse edge length, this is just done on the GPU to avoid a CPU->GPU sync point
__global__ void compute_total_inv_edges(const vec3* total_lower, const vec3* total_upper, vec3* total_inv_edges)
{
    vec3 edges = (total_upper[0]-total_lower[0]);
    edges += vec3(0.0001f);

    total_inv_edges[0] = vec3(1.0f/edges[0], 1.0f/edges[1], 1.0f/edges[2]);
}



LinearBVHBuilderGPU::LinearBVHBuilderGPU() 
    : indices(NULL)
    , keys(NULL)
    , deltas(NULL)
    , range_lefts(NULL)
    , range_rights(NULL)
    , num_children(NULL)
    , max_items(0)
    , total_lower(NULL)
    , total_upper(NULL)
    , total_inv_edges(NULL)
{
    total_lower = (vec3*)alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3));
    total_upper = (vec3*)alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3));
    total_inv_edges = (vec3*)alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3));
}

LinearBVHBuilderGPU::~LinearBVHBuilderGPU()
{
    free_device(WP_CURRENT_CONTEXT, indices);
    free_device(WP_CURRENT_CONTEXT, keys);
    free_device(WP_CURRENT_CONTEXT, deltas);

    free_device(WP_CURRENT_CONTEXT, range_lefts);
    free_device(WP_CURRENT_CONTEXT, range_rights);
    free_device(WP_CURRENT_CONTEXT, num_children);

    free_device(WP_CURRENT_CONTEXT, total_lower);
    free_device(WP_CURRENT_CONTEXT, total_upper);
    free_device(WP_CURRENT_CONTEXT, total_inv_edges);

}



void LinearBVHBuilderGPU::build(BVH& bvh, const vec3* item_lowers, const vec3* item_uppers, const int* item_priorities, int num_items, bounds3* total_bounds)
{
    if (num_items > max_items)
    {
        const int items_to_alloc = (num_items*3)/2;
        const int nodes_to_alloc = (bvh.max_nodes*3)/2;

        // reallocate temporary storage if necessary
        free_device(WP_CURRENT_CONTEXT, indices);
        free_device(WP_CURRENT_CONTEXT, keys);
        free_device(WP_CURRENT_CONTEXT, deltas);
        free_device(WP_CURRENT_CONTEXT, range_lefts);
        free_device(WP_CURRENT_CONTEXT, range_rights);
        free_device(WP_CURRENT_CONTEXT, num_children);

        indices = (int*)alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*items_to_alloc*2); 	// *2 for radix sort
        keys = (int*)alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*items_to_alloc*2);	    // *2 for radix sort
        deltas = (int*)alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*items_to_alloc);    	// highest differenting bit between keys for item i and i+1
        range_lefts = (int*)alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*nodes_to_alloc);
        range_rights = (int*)alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*nodes_to_alloc);
        num_children = (int*)alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*nodes_to_alloc);

        max_items = items_to_alloc;
    }

    // if total bounds supplied by the host then we just 
    // compute our edge length and upload it to the GPU directly
    if (total_bounds)
    {
        // calculate Morton codes
        vec3 edges = (*total_bounds).edges();
        edges += vec3(0.0001f);

        vec3 inv_edges = vec3(1.0f/edges[0], 1.0f/edges[1], 1.0f/edges[2]);
        
        memcpy_h2d(WP_CURRENT_CONTEXT, total_lower, &total_bounds->lower[0], sizeof(vec3));
        memcpy_h2d(WP_CURRENT_CONTEXT, total_upper, &total_bounds->upper[0], sizeof(vec3));
        memcpy_h2d(WP_CURRENT_CONTEXT, total_inv_edges, &inv_edges[0], sizeof(vec3));
    }
    else
    {
        static vec3 upper(-FLT_MAX);
        static vec3 lower(FLT_MAX);

        memcpy_h2d(WP_CURRENT_CONTEXT, total_lower, &lower, sizeof(lower));
        memcpy_h2d(WP_CURRENT_CONTEXT, total_upper, &upper, sizeof(upper));

        // compute the total bounds on the GPU
        wp_launch_device(WP_CURRENT_CONTEXT, compute_total_bounds, num_items, (item_lowers, item_uppers, total_lower, total_upper, num_items));

        // compute the total edge length
        wp_launch_device(WP_CURRENT_CONTEXT, compute_total_inv_edges, 1, (total_lower, total_upper, total_inv_edges));
    }

    // assign 30-bit Morton code based on the centroid of each triangle and bounds for each leaf
    wp_launch_device(WP_CURRENT_CONTEXT, compute_morton_codes, num_items, (item_lowers, item_uppers, num_items, total_lower, total_inv_edges, indices, keys));
    
    // sort items based on Morton key (note the 32-bit sort key corresponds to the template parameter to morton3, i.e. 3x9 bit keys combined)
    radix_sort_pairs_device(WP_CURRENT_CONTEXT, keys, indices, num_items);

    // calculate deltas between adjacent keys
    wp_launch_device(WP_CURRENT_CONTEXT, compute_key_deltas, num_items, (keys, deltas, num_items-1));

    // initialize leaf nodes
    wp_launch_device(WP_CURRENT_CONTEXT, build_leaves, num_items, (item_lowers, item_uppers, num_items, indices, range_lefts, range_rights, bvh.node_lowers, bvh.node_uppers));

#if 1
    
    // reset children count, this is our atomic counter so we know when an internal node is complete, only used during building
    memset_device(WP_CURRENT_CONTEXT, num_children, 0, sizeof(int)*bvh.max_nodes);

    // build the tree and internal node bounds
    wp_launch_device(WP_CURRENT_CONTEXT, build_hierarchy, num_items, (num_items, bvh.root, deltas, num_children, range_lefts, range_rights, bvh.node_parents, bvh.node_lowers, bvh.node_uppers));

#else

    // Reference CPU implementation of build_hierarchy()

    std::vector<int> indices(num_items);
    std::vector<int> keys(num_items);
    std::vector<int> deltas(num_items-1);
    std::vector<BVHPackedNodeHalf> lowers(bvh.max_nodes);
    std::vector<BVHPackedNodeHalf> uppers(bvh.max_nodes);

    std::vector<int> range_lefts(bvh.max_nodes);
    std::vector<int> range_rights(bvh.max_nodes);
    std::vector<int> num_children(bvh.max_nodes);

    check_cuda(cudaMemcpy(&indices[0], mIndices, sizeof(int)*num_items, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(&keys[0], mKeys, sizeof(int)*num_items, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(&deltas[0], mDeltas, sizeof(int)*(num_items-1), cudaMemcpyDeviceToHost));

    check_cuda(cudaMemcpy(&lowers[0], bvh.node_lowers, bvh.max_nodes*sizeof(BVHPackedNodeHalf), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(&uppers[0], bvh.node_uppers, bvh.max_nodes*sizeof(BVHPackedNodeHalf), cudaMemcpyDeviceToHost));

    // zero child pointers for internal nodes so we can do checking
    for (int i=num_items; i < bvh.max_nodes; ++i)
    {
        lowers[i].i = 0;
        uppers[i].i = 0;
    }

    const int internal_offset = num_items;
    int rootNode = -1;

    // initialize leaf ranges [left, right]
    for (int i=0; i < num_items; ++i)
    {
        range_lefts[i] = i;
        range_rights[i] = i;
    }

    for (int i=num_items; i < max_nodes;++i)
    {
        range_lefts[i] = -1;
        range_rights[i] = -1;
    }
    
    for (int i=0; i < num_items && rootNode == -1; ++i)
    {
        int index = i;

        for (;;)
        {
            int left = range_lefts[index];
            int right = range_rights[index];

            int parent;

            if (left == 0 || (right != num_items-1 && deltas[right] < deltas[left-1]))
            {
                parent = right + internal_offset;

                // check that no other node has assigned itself to the parent
                assert(lowers[parent].i == 0);
                assert(range_lefts[parent] == -1);
                
                // set parent's left child
                lowers[parent].i = index;
                
                range_lefts[parent] = left;
                num_children[parent]++;
            }
            else
            {
                parent = left + internal_offset - 1;
                
                // check that no other node has assigned itself to the parent
                assert(uppers[parent].i == 0);
                assert(range_rights[parent] == -1);

                // set parent's right child
                uppers[parent].i = index;

                range_rights[parent] = right;
                num_children[parent]++;
            }

            assert(num_children[parent] <= 2);

            // assign bounds to parent
            if (num_children[parent] == 2)
            {
                const int left_child = lowers[parent].i;
                const int right_child = uppers[parent].i;

                vec3 lower = Min(vec3(&lowers[left_child][0]), vec3(&lowers[right_child][0]));
                vec3 upper = Max(vec3(&uppers[left_child][0]), vec3(&uppers[right_child][0]));
                
                lowers[parent] = { lower[0], lower[1], lower[2], (unsigned int)left_child, 0 };
                uppers[parent] = { upper[0], upper[1], upper[2], (unsigned int)right_child, 0 };

                // store index of root node
                if (range_lefts[parent] == 0 && range_rights[parent] == num_items-1 && num_children[parent] == 2)
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

    for (int i=num_items; i < max_nodes-1; ++i)
    {
        assert(lowers[i].i != rootNode);
        assert(uppers[i].i != rootNode);

        assert(num_children[i] == 2);

    }

    check_cuda(cudaMemcpy(bvh.node_lowers, &lowers[0], sizeof(BVHPackedNodeHalf)*bvh.max_nodes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(bvh.node_uppers, &uppers[0], sizeof(BVHPackedNodeHalf)*bvh.max_nodes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(bvh.root, &rootNode, sizeof(int), cudaMemcpyHostToDevice));

#endif

}

void bvh_destroy_device(wp::BVH& bvh)
{
    ContextGuard guard(bvh.context);

    free_device(WP_CURRENT_CONTEXT, bvh.node_lowers); bvh.node_lowers = NULL;
    free_device(WP_CURRENT_CONTEXT, bvh.node_uppers); bvh.node_uppers = NULL;
    free_device(WP_CURRENT_CONTEXT, bvh.node_parents); bvh.node_parents = NULL;
    free_device(WP_CURRENT_CONTEXT, bvh.node_counts); bvh.node_counts = NULL;
    free_device(WP_CURRENT_CONTEXT, bvh.bounds); bvh.bounds = NULL;
    free_device(WP_CURRENT_CONTEXT, bvh.root); bvh.root = NULL;
}

} // namespace wp




void bvh_refit_device(uint64_t id)
{
    wp::BVH bvh;
    if (bvh_get_descriptor(id, bvh))
    {
        ContextGuard guard(bvh.context);
        wp_launch_device(WP_CURRENT_CONTEXT, wp::set_bounds_from_lowers_and_uppers, bvh.num_bounds, (bvh.num_bounds, bvh.bounds, bvh.lowers, bvh.uppers));

        bvh_refit_device(bvh, bvh.bounds);
    }

}

uint64_t bvh_create_device(void* context, wp::vec3* lowers, wp::vec3* uppers, int num_items)
{
    ContextGuard guard(context);

    // // todo: BVH creation only on CPU at the moment so temporarily bring all the data back to host
    // vec3* lowers_host = (vec3*)alloc_host(sizeof(vec3)*num_bounds);
    // vec3* uppers_host = (vec3*)alloc_host(sizeof(vec3)*num_bounds);
    // bounds3* bounds_host = (bounds3*)alloc_host(sizeof(bounds3)*num_bounds);

    // memcpy_d2h(WP_CURRENT_CONTEXT, lowers_host, lowers, sizeof(vec3)*num_bounds);
    // memcpy_d2h(WP_CURRENT_CONTEXT, uppers_host, uppers, sizeof(vec3)*num_bounds);
    // cuda_context_synchronize(WP_CURRENT_CONTEXT);

    // for (int i=0; i < num_bounds; ++i)
    // {
    //     bounds_host[i] = bounds3();
    //     bounds_host[i].lower = lowers_host[i];
    //     bounds_host[i].upper = uppers_host[i];
    // }

    // BVH bvh_host = bvh_create(bounds_host, num_bounds);
    // bvh_host.context = context ? context : cuda_context_get_current();
	// bvh_host.bounds = bounds_host;
    // bvh_host.num_bounds = num_bounds;        
    // BVH bvh_device_clone = bvh_clone(WP_CURRENT_CONTEXT, bvh_host);

	// bvh_device_clone.lowers = lowers;		// managed by the user
	// bvh_device_clone.uppers = uppers;		// managed by the user


    wp::BVH bvh_host;

    bvh_host.num_bounds = num_items;
    bvh_host.max_nodes = 2*num_items;
    bvh_host.node_lowers = (wp::BVHPackedNodeHalf*)alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::BVHPackedNodeHalf)*bvh_host.max_nodes);
    bvh_host.node_uppers = (wp::BVHPackedNodeHalf*)alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::BVHPackedNodeHalf)*bvh_host.max_nodes);
    bvh_host.node_parents = (int*)alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*bvh_host.max_nodes);
    bvh_host.node_counts = (int*)alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*bvh_host.max_nodes);
	bvh_host.bounds = (wp::bounds3*)alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::bounds3)*num_items);
    bvh_host.root = (int*)alloc_device(WP_CURRENT_CONTEXT, sizeof(int));

    bvh_host.context = context ? context : cuda_context_get_current();

    static wp::LinearBVHBuilderGPU* builder = new wp::LinearBVHBuilderGPU();
    builder->build(bvh_host, lowers, uppers, NULL, num_items, NULL);

    // wp::LinearBVHBuilderGPU builder;
    // builder.build(bvh_host, lowers, uppers, NULL, num_items, NULL);

    // create device-side BVH descriptor
    wp::BVH* bvh_device = (wp::BVH*)alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::BVH));
    memcpy_h2d(WP_CURRENT_CONTEXT, bvh_device, &bvh_host, sizeof(wp::BVH));
        
    uint64_t bvh_id = (uint64_t)bvh_device;
    wp::bvh_add_descriptor(bvh_id, bvh_host);

    return bvh_id;
}


void bvh_destroy_device(uint64_t id)
{
    wp::BVH bvh;
    if (wp::bvh_get_descriptor(id, bvh))
    {
        wp::bvh_destroy_device(bvh);
        
        wp::bvh_rem_descriptor(id);
    }
}
