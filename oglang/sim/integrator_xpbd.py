import oglang as og


@og.kernel
def integrate_particles(x: og.array(dtype=og.vec3),
                        v: og.array(dtype=og.vec3),
                        f: og.array(dtype=og.vec3),
                        w: og.array(dtype=float),
                        gravity: og.vec3,
                        dt: float,
                        x_new: og.array(dtype=og.vec3),
                        v_new: og.array(dtype=og.vec3)):

    tid = og.tid()

    x0 = og.load(x, tid)
    v0 = og.load(v, tid)
    f0 = og.load(f, tid)
    inv_mass = og.load(w, tid)

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + gravity * og.step(0.0 - inv_mass)) * dt
    x1 = x0 + v1 * dt

    og.store(x_new, tid, x1)
    og.store(v_new, tid, v1)


@og.kernel
def solve_springs(x: og.array(dtype=og.vec3),
                 v: og.array(dtype=og.vec3),
                 invmass: og.array(dtype=float),
                 spring_indices: og.array(dtype=int),
                 spring_rest_lengths: og.array(dtype=float),
                 spring_stiffness: og.array(dtype=float),
                 spring_damping: og.array(dtype=float),
                 dt: float,
                 delta: og.array(dtype=og.vec3)):

    tid = og.tid()

    i = og.load(spring_indices, tid * 2 + 0)
    j = og.load(spring_indices, tid * 2 + 1)

    ke = og.load(spring_stiffness, tid)
    kd = og.load(spring_damping, tid)
    rest = og.load(spring_rest_lengths, tid)

    xi = og.load(x, i)
    xj = og.load(x, j)

    vi = og.load(v, i)
    vj = og.load(v, j)

    xij = xi - xj
    vij = vi - vj

    l = length(xij)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = dot(dir, vij)

    # damping based on relative velocity.
    #fs = dir * (ke * c + kd * dcdt)

    wi = og.load(invmass, i)
    wj = og.load(invmass, j)

    denom = wi + wj
    alpha = 1.0/(ke*dt*dt)

    multiplier = c / (denom)# + alpha)

    xd = dir*multiplier

    og.atomic_sub(delta, i, xd*wi)
    og.atomic_add(delta, j, xd*wj)



@og.kernel
def solve_tetrahedra(x: og.array(dtype=og.vec3),
                     v: og.array(dtype=og.vec3),
                     inv_mass: og.array(dtype=float),
                     indices: og.array(dtype=int),
                     pose: og.array(dtype=og.mat33),
                     activation: og.array(dtype=float),
                     materials: og.array(dtype=float),
                     dt: float,
                     relaxation: float,
                     delta: og.array(dtype=og.vec3)):

    tid = og.tid()

    i = og.load(indices, tid * 4 + 0)
    j = og.load(indices, tid * 4 + 1)
    k = og.load(indices, tid * 4 + 2)
    l = og.load(indices, tid * 4 + 3)

    act = og.load(activation, tid)

    k_mu = og.load(materials, tid * 3 + 0)
    k_lambda = og.load(materials, tid * 3 + 1)
    k_damp = og.load(materials, tid * 3 + 2)

    x0 = og.load(x, i)
    x1 = og.load(x, j)
    x2 = og.load(x, k)
    x3 = og.load(x, l)

    v0 = og.load(v, i)
    v1 = og.load(v, j)
    v2 = og.load(v, k)
    v3 = og.load(v, l)

    w0 = og.load(inv_mass, i)
    w1 = og.load(inv_mass, j)
    w2 = og.load(inv_mass, k)
    w3 = og.load(inv_mass, l)

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0

    Ds = og.mat33(x10, x20, x30)
    Dm = og.load(pose, tid)

    inv_rest_volume = og.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm

    f1 = og.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = og.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = og.vec3(F[0, 2], F[1, 2], F[2, 2])

    r_s = og.sqrt(dot(f1, f1) + dot(f2, f2) + dot(f3, f3))
    r_s_inv = 1.0/r_s

    C = r_s - og.sqrt(3.0) 
    dCdx = F*og.transpose(Dm)*r_s_inv

    grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    denom = dot(grad0,grad0)*w0 + dot(grad1,grad1)*w1 + dot(grad2,grad2)*w2 + dot(grad3,grad3)*w3
    multiplier = C/(denom + 1.0/(k_mu*dt*dt*rest_volume))

    delta0 = grad0*multiplier
    delta1 = grad1*multiplier
    delta2 = grad2*multiplier
    delta3 = grad3*multiplier

       
    # r_s = og.sqrt(og.abs(dot(f1, f1) + dot(f2, f2) + dot(f3, f3) - 3.0))

    # grad0 = og.vec3(0.0, 0.0, 0.0)
    # grad1 = og.vec3(0.0, 0.0, 0.0)
    # grad2 = og.vec3(0.0, 0.0, 0.0)
    # grad3 = og.vec3(0.0, 0.0, 0.0)
    # multiplier = 0.0

    # if (r_s > 0.0):
    #     r_s_inv = 1.0/r_s

    #     C = r_s 
    #     dCdx = F*og.transpose(Dm)*r_s_inv*og.sign(r_s)

    #     grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    #     grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    #     grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    #     grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    #     denom = dot(grad0,grad0)*w0 + dot(grad1,grad1)*w1 + dot(grad2,grad2)*w2 + dot(grad3,grad3)*w3 
    #     multiplier = C/(denom + 1.0/(k_mu*dt*dt*rest_volume))

    # delta0 = grad0*multiplier
    # delta1 = grad1*multiplier
    # delta2 = grad2*multiplier
    # delta3 = grad3*multiplier

    # hydrostatic part
    J = og.determinant(F)

    C_vol = J - 1.0
    # dCdx = og.mat33(cross(f2, f3), cross(f3, f1), cross(f1, f2))*og.transpose(Dm)

    # grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    # grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    s = inv_rest_volume / 6.0
    grad1 = og.cross(x20, x30) * s
    grad2 = og.cross(x30, x10) * s
    grad3 = og.cross(x10, x20) * s
    grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    denom = dot(grad0, grad0)*w0 + dot(grad1, grad1)*w1 + dot(grad2, grad2)*w2 + dot(grad3, grad3)*w3
    multiplier = C_vol/(denom + 1.0/(k_lambda*dt*dt*rest_volume))

    delta0 = delta0 + grad0 * multiplier
    delta1 = delta1 + grad1 * multiplier
    delta2 = delta2 + grad2 * multiplier
    delta3 = delta3 + grad3 * multiplier

    # apply forces
    og.atomic_sub(delta, i, delta0*w0*relaxation)
    og.atomic_sub(delta, j, delta1*w1*relaxation)
    og.atomic_sub(delta, k, delta2*w2*relaxation)
    og.atomic_sub(delta, l, delta3*w3*relaxation)


@og.kernel
def apply_deltas(x_orig: og.array(dtype=og.vec3),
                 v_orig: og.array(dtype=og.vec3),
                 x_pred: og.array(dtype=og.vec3),
                 delta: og.array(dtype=og.vec3),
                 dt: float,
                 x_out: og.array(dtype=og.vec3),
                 v_out: og.array(dtype=og.vec3)):

    tid = og.tid()

    x0 = og.load(x_orig, tid)
    xp = og.load(x_pred, tid)

    # constraint deltas
    d = og.load(delta, tid)

    x_new = xp + d
    v_new = (x_new - x0)/dt

    og.store(x_out, tid, x_new)
    og.store(v_out, tid, v_new)

    # clear forces
    og.store(delta, tid, vec3(0.0, 0.0, 0.0))


class XPBDIntegrator:
    """A implicit integrator using XPBD

    After constructing `Model` and `State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Semi-implicit time integration is a variational integrator that 
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    Example:

        >>> integrator = og.SemiImplicitIntegrator()
        >>>
        >>> # simulation loop
        >>> for i in range(100):
        >>>     state = integrator.forward(model, state, dt)

    """

    def __init__(self, iterations):
        
        self.iterations = iterations


    def simulate(self, model, state_in, state_out, dt):

        with og.util.ScopedTimer("simulate", False):

            q_pred = torch.zeros_like(state_in.particle_q)
            qd_pred = torch.zeros_like(state_in.particle_qd)

            # alloc particle force buffer
            if (model.particle_count):
                state_out.particle_f.zero_()


            #----------------------------
            # integrate particles

            if (model.particle_count):
                og.launch(kernel=integrate_particles,
                            dim=model.particle_count,
                            inputs=[state_in.particle_q, state_in.particle_qd, state_out.particle_f, model.particle_inv_mass, model.gravity, dt],
                            outputs=[q_pred, qd_pred],
                            device=model.device)


            for i in range(self.iterations):

                # damped springs
                if (model.spring_count):

                    og.launch(kernel=solve_springs,
                                dim=model.spring_count,
                                inputs=[state_in.particle_q, state_in.particle_qd, model.particle_inv_mass, model.spring_indices, model.spring_rest_length, model.spring_stiffness, model.spring_damping, dt],
                                outputs=[state_out.particle_f],
                                device=model.device)
               
                # tetrahedral FEM
                if (model.tet_count):

                    og.launch(kernel=solve_tetrahedra,
                                dim=model.tet_count,
                                inputs=[q_pred, qd_pred, model.particle_inv_mass, model.tet_indices, model.tet_poses, model.tet_activations, model.tet_materials, dt, model.relaxation],
                                outputs=[state_out.particle_f],
                                device=model.device)

                # apply updates
                og.launch(kernel=apply_deltas,
                            dim=model.particle_count,
                            inputs=[state_in.particle_q,
                                    state_in.particle_qd,
                                    q_pred,
                                    state_out.particle_f,
                                    dt],
                            outputs=[q_pred,
                                     qd_pred],
                            device=model.device)

            state_out.particle_q = q_pred
            state_out.particle_qd = qd_pred

            return state_out

