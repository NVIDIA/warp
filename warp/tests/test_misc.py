import warp as wp

vec5d = wp.types.vector(length=5, dtype=wp.float64)

# v3f = wp.types.vector(length=3, dtype=float)
# qf = wp.types.quaternion(dtype=float)


# # declare a new vector type
# vec5d = wp.types.vector(length=5, dtype=wp.float64)

# # construct a new vector inside a kernel from arguments
# wp.vector(length=5)
# wp.matrix(shape=(3,2), dtype=wp.float16)

# #wp.quaternion()
# #wp.transformation()


@wp.kernel
def inc(a: wp.array(dtype=vec5d)):

    q = wp.quaternion(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
    q = wp.quatf(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))

    print(1.0)

    #m = spatial_matrix()

    # U = wp.mat33()

    

    v5d_decl = vec5d(wp.float64(1.0), wp.float64(2.0), wp.float64(3.0), wp.float64(4.0), wp.float64(5.0))
    v5d_decl = vec5d(wp.float64(1.0))
    print(v5d_decl)




    a[0] = v5d_decl

    v5d_const = wp.vector(wp.float64(1.0), wp.float64(2.0), wp.float64(3.0), wp.float64(4.0), wp.float64(5.0))
    print(v5d_const)

    v5d_ones = wp.vector(wp.float64(1.0), length=5)
    print(v5d_ones)

    v5d_zeros = wp.vector(dtype=wp.float64, length=5)
    print(v5d_zeros)

wp.init()

a = wp.zeros(shape=(3,), dtype=vec5d)
wp.launch(inc, dim=1, inputs=[a])
wp.synchronize()

wp.force_load()