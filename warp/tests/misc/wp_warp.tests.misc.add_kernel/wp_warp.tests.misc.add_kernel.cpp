
#define WP_TILE_BLOCK_DIM 1
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(task_index, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, task_index, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, task_index, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, task_index, dim)

extern "C" {
}


void add_kernel_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp::array_t<wp::int32> var_a,
    wp::array_t<wp::int32> var_b,
    wp::array_t<wp::int32> var_res)
{
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32* var_1;
    wp::int32* var_2;
    wp::int32 var_3;
    wp::int32 var_4;
    wp::int32 var_5;
    //---------
    // forward
    // def add_kernel(a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32), res: wp.array(dtype=wp.int32)):       <L 4>
    // i = wp.tid()                                                                           <L 6>
    var_0 = builtin_tid1d();
    // res[i] = a[i] + b[i]                                                                   <L 7>
    var_1 = wp::address(var_a, var_0);
    var_2 = wp::address(var_b, var_0);
    var_4 = wp::load(var_1);
    var_5 = wp::load(var_2);
    var_3 = wp::add(var_4, var_5);
    wp::array_store(var_res, var_0, var_3);
}



void add_kernel_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp::array_t<wp::int32> var_a,
    wp::array_t<wp::int32> var_b,
    wp::array_t<wp::int32> var_res,
    wp::array_t<wp::int32> adj_a,
    wp::array_t<wp::int32> adj_b,
    wp::array_t<wp::int32> adj_res)
{
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32* var_1;
    wp::int32* var_2;
    wp::int32 var_3;
    wp::int32 var_4;
    wp::int32 var_5;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::int32 adj_5 = {};
    //---------
    // forward
    // def add_kernel(a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32), res: wp.array(dtype=wp.int32)):       <L 4>
    // i = wp.tid()                                                                           <L 6>
    var_0 = builtin_tid1d();
    // res[i] = a[i] + b[i]                                                                   <L 7>
    var_1 = wp::address(var_a, var_0);
    var_2 = wp::address(var_b, var_0);
    var_4 = wp::load(var_1);
    var_5 = wp::load(var_2);
    var_3 = wp::add(var_4, var_5);
    // wp::array_store(var_res, var_0, var_3);
    //---------
    // reverse
    wp::adj_array_store(var_res, var_0, var_3, adj_res, adj_0, adj_3);
    wp::adj_add(var_4, var_5, adj_1, adj_2, adj_3);
    wp::adj_load(var_2, adj_2, adj_5);
    wp::adj_load(var_1, adj_1, adj_4);
    wp::adj_address(var_b, var_0, adj_b, adj_0, adj_2);
    wp::adj_address(var_a, var_0, adj_a, adj_0, adj_1);
    // adj: res[i] = a[i] + b[i]                                                              <L 7>
    // adj: i = wp.tid()                                                                      <L 6>
    // adj: def add_kernel(a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32), res: wp.array(dtype=wp.int32)):  <L 4>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void add_kernel_cpu_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_a,
    wp::array_t<wp::int32> var_b,
    wp::array_t<wp::int32> var_res)
{
for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        // init shared memory allocator
        wp::tile_alloc_shared(0, true);

        add_kernel_cpu_kernel_forward(
            dim,
            task_index,
            var_a,
            var_b,
            var_res);

        // check shared memory allocator
        wp::tile_alloc_shared(0, false, true);

    }
}

} // extern C



extern "C" {

WP_API void add_kernel_cpu_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_a,
    wp::array_t<wp::int32> var_b,
    wp::array_t<wp::int32> var_res,
    wp::array_t<wp::int32> adj_a,
    wp::array_t<wp::int32> adj_b,
    wp::array_t<wp::int32> adj_res)
{
    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        // initialize shared memory allocator
        wp::tile_alloc_shared(0, true);

        add_kernel_cpu_kernel_backward(
            dim,
            task_index,
            var_a,
            var_b,
            var_res,
            adj_a,
            adj_b,
            adj_res);

        // check shared memory allocator
        wp::tile_alloc_shared(0, false, true);
    }
}

} // extern C

