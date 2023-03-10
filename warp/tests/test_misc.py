import warp as wp

vec5d = wp.vector_t(length=5, dtype=wp.float64)
spatial_matrix = wp.spatial_matrix_t(dtype=wp.float16)

@wp.kernel
def inc(a: wp.array(dtype=vec5d)):

    # q = wp.quat(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
    # q = quatf(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))






    m = spatial_matrix()

    # U = wp.mat33()

    # v5d_decl = vec5d(wp.float64(1.0), wp.float64(2.0), wp.float64(3.0), wp.float64(4.0), wp.float64(5.0))
    # v5d_decl = vec5d(wp.float64(1.0))
    # print(v5d_decl)

    # a[0] = v5d_decl

    # v5d_const = wp.vec(wp.float64(1.0), wp.float64(2.0), wp.float64(3.0), wp.float64(4.0), wp.float64(5.0))
    # print(v5d_const)

    # v5d_ones = wp.vec(wp.float64(1.0), length=5)
    # print(v5d_ones)

    # v5d_zeros = wp.vec(dtype=wp.float64, length=5)
    # print(v5d_zeros)

wp.init()

a = wp.zeros(shape=(3,), dtype=vec5d)
wp.launch(inc, dim=1, inputs=[a])
wp.synchronize()

wp.force_load()