import warp as wp

@wp.kernel
def eval_springs(x: wp.array(dtype=wp.vec3),
                 v: wp.array(dtype=wp.vec3),
                 spring_indices: wp.array(dtype=int),
                 spring_rest_lengths: wp.array(dtype=float),
                 spring_stiffness: wp.array(dtype=float),
                 spring_damping: wp.array(dtype=float),
                 f: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    i = wp.load(spring_indices, tid * 2 + 0)
    j = wp.load(spring_indices, tid * 2 + 1)

    ke = wp.load(spring_stiffness, tid)
    kd = wp.load(spring_damping, tid)
    rest = wp.load(spring_rest_lengths, tid)

    xi = wp.load(x, i)
    xj = wp.load(x, j)

    vi = wp.load(v, i)
    vj = wp.load(v, j)

    xij = xi - xj
    vij = vi - vj

    l = wp.length(xij)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = dot(dir, vij)

    # damping based on relative velocity.
    fs = dir * (ke * c + kd * dcdt)

    wp.atomic_sub(f, i, fs)
    wp.atomic_add(f, j, fs)


@wp.kernel
def integrate_particles(x: wp.array(dtype=wp.vec3),
                        v: wp.array(dtype=wp.vec3),
                        f: wp.array(dtype=wp.vec3),
                        w: wp.array(dtype=float),
                        dt: float):

    tid = wp.tid()

    x0 = wp.load(x, tid)
    v0 = wp.load(v, tid)
    f0 = wp.load(f, tid)
    inv_mass = wp.load(w, tid)

    g = wp.vec3(0.0, 0.0, 0.0)

    # treat particles with inv_mass == 0 as kinematic
    if (inv_mass > 0.0):
        g = wp.vec3(0.0, 0.0 - 9.81, 0.0)

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + g) * dt
    x1 = x0 + v1 * dt

    wp.store(x, tid, x1)
    wp.store(v, tid, v1)

    # clear forces
    wp.store(f, tid, wp.vec3(0.0, 0.0, 0.0))


class WpIntegrator:

    def __init__(self, cloth, device):

        self.device = device

        self.positions = wp.from_numpy(cloth.positions, dtype=wp.vec3, device=device)
        self.positions_host = wp.from_numpy(cloth.positions, dtype=wp.vec3, device="cpu")
        self.invmass = wp.from_numpy(cloth.inv_masses, dtype=float, device=device)

        self.velocities = wp.zeros(cloth.num_particles, dtype=wp.vec3, device=device)
        self.forces = wp.zeros(cloth.num_particles, dtype=wp.vec3, device=device)

        self.spring_indices = wp.from_numpy(cloth.spring_indices, dtype=int, device=device)
        self.spring_lengths = wp.from_numpy(cloth.spring_lengths, dtype=float, device=device)
        self.spring_stiffness = wp.from_numpy(cloth.spring_stiffness, dtype=float, device=device)
        self.spring_damping = wp.from_numpy(cloth.spring_damping, dtype=float, device=device)

        self.cloth = cloth


    def simulate(self, dt, substeps):

        sim_dt = dt/substeps
        
        for s in range(substeps):

            wp.launch(
                kernel=eval_springs, 
                dim=self.cloth.num_springs, 
                inputs=[self.positions, 
                        self.velocities,
                        self.spring_indices,
                        self.spring_lengths,
                        self.spring_stiffness,
                        self.spring_damping,
                        self.forces], 
                outputs=[],
                device=self.device)

            # integrate 
            wp.launch(
                kernel=integrate_particles, 
                dim=self.cloth.num_particles, 
                inputs=[self.positions,
                        self.velocities,
                        self.forces,
                        self.invmass,
                        sim_dt], 
                outputs=[],
                device=self.device)


        # copy data back to host
        if (self.device == "cuda"):
            wp.copy(self.positions_host, self.positions)
            wp.synchronize()
    
            return self.positions_host.numpy()
        
        else:
            return self.positions.numpy()