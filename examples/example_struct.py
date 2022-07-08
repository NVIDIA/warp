import warp as wp

wp.init()


@wp.struct
class State:
    test: float
    particle_q: wp.array(dtype=wp.vec3)
    particle_qd: wp.array(dtype=wp.vec3)


arr = wp.zeros(10, dtype=wp.vec3, device="cuda")
state = State()
state.test = 0.01
state.particle_q = arr


@wp.kernel
def test_print(state: State, arr: wp.array(dtype=wp.vec3)):
    state.particle_q[0] = wp.vec3(1.0, 1.0, 1.0)
    print(state.test)


wp.launch(kernel=test_print, dim=1, inputs=[state, arr], outputs=[], device="cuda")
wp.synchronize()
print(state.particle_q)
