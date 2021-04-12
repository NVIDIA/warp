import oglang as og

@og.kernel
def eval_springs(x: og.array(og.float3),
                 v: og.array(og.float3),
                 spring_indices: og.array(int),
                 spring_rest_lengths: og.array(float),
                 spring_stiffness: og.array(float),
                 spring_damping: og.array(float),
                 f: og.array(og.float3)):

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

    l = og.length(xij)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = dot(dir, vij)

    # damping based on relative velocity.
    fs = dir * (ke * c + kd * dcdt)

    og.atomic_sub(f, i, fs)
    og.atomic_add(f, j, fs)


@og.kernel
def integrate_particles(x: og.array(og.float3),
                        v: og.array(og.float3),
                        f: og.array(og.float3),
                        w: og.array(float),
                        dt: float):

    tid = og.tid()

    x0 = og.load(x, tid)
    v0 = og.load(v, tid)
    f0 = og.load(f, tid)
    inv_mass = og.load(w, tid)

    g = og.float3(0.0, 0.0 - 9.8, 0.0)

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + g * og.step(0.0 - inv_mass)) * dt
    x1 = x0 + v1 * dt

    og.store(x, tid, x1)
    og.store(v, tid, v1)

    # clear forces
    og.store(f, tid, og.float3(0.0, 0.0, 0.0))


class OgIntegrator:

    def __init__(self, cloth):


        self.positions = og.from_numpy(cloth.positions, dtype=og.float3, device="cuda")
        self.positions_host = og.from_numpy(cloth.positions, dtype=og.float3, device="cpu")
        self.invmass = og.from_numpy(cloth.inv_masses, dtype=float, device="cuda")

        self.velocities = og.zeros(cloth.num_particles, dtype=og.float3, device="cuda")
        self.forces = og.zeros(cloth.num_particles, dtype=og.float3, device="cuda")

        self.spring_indices = og.from_numpy(cloth.spring_indices, dtype=int, device="cuda")
        self.spring_lengths = og.from_numpy(cloth.spring_lengths, dtype=float, device="cuda")
        self.spring_stiffness = og.from_numpy(cloth.spring_stiffness, dtype=float, device="cuda")
        self.spring_damping = og.from_numpy(cloth.spring_damping, dtype=float, device="cuda")

        self.cloth = cloth


    def simulate(self, dt, substeps):

        sim_dt = dt/substeps
        
        for s in range(substeps):

            og.launch(
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
                device="cuda")

            # integrate 
            og.launch(
                kernel=integrate_particles, 
                dim=self.cloth.num_particles, 
                inputs=[self.positions,
                        self.velocities,
                        self.forces,
                        self.invmass,
                        sim_dt], 
                outputs=[],
                device="cuda")


        # copy data back to host
        og.copy(self.positions, self.positions_host)
        og.synchronize()

        return self.positions_host.numpy()