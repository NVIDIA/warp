# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample node that simulates flocking behaviors by animating prim attributes."""

import math
import traceback

import carb.settings
import numpy as np
import omni.graph.core as og
import omni.kit.app
import omni.usd
import omni.warp.nodes
import usdrt
from omni.warp.nodes.ogn.OgnSamplePrimFlockingDatabase import OgnSamplePrimFlockingDatabase

import warp as wp

# device used for flocking simulation
MAIN_DEVICE = "cuda:0"

# device used for updating colors
COLOR_DEVICE = "cpu"


#   Kernels
# -----------------------------------------------------------------------------


@wp.struct
class Boid:
    vel: wp.vec3f
    wander_angles: wp.vec2f
    mass: float
    group: int


@wp.struct
class Obstacle:
    pos: wp.vec3f
    radius: float


@wp.struct
class World:
    lower: wp.vec3f
    upper: wp.vec3f
    grid: wp.uint64
    seed: int
    biases: wp.mat33f
    obstacles: wp.array(dtype=Obstacle)


@wp.kernel(enable_backward=False)
def copy_positions(dst: wp.array(dtype=wp.vec3f), src: wp.fabricarray(dtype=wp.vec3d)):
    tid = wp.tid()
    pos = src[tid]
    dst[tid] = wp.vec3f(float(pos[0]), float(pos[1]), float(pos[2]))


@wp.kernel(enable_backward=False)
def assign_colors(
    glows: wp.array(dtype=float),
    groups: wp.array(dtype=int),
    color_ramps: wp.array2d(dtype=wp.vec3f),
    colors: wp.fabricarrayarray(dtype=wp.vec3f),
):
    tid = wp.tid()

    glow = glows[tid]
    group = groups[tid]

    if glow < 0.4:
        alpha = glow / 0.4
        colors[tid][0] = (1.0 - alpha) * color_ramps[group, 0] + alpha * color_ramps[group, 1]
    elif glow < 0.8:
        alpha = (glow - 0.4) / 0.4
        colors[tid][0] = (1.0 - alpha) * color_ramps[group, 1] + alpha * color_ramps[group, 2]
    else:
        alpha = (glow - 0.8) / 0.2
        colors[tid][0] = (1.0 - alpha) * color_ramps[group, 2] + alpha * color_ramps[group, 3]


@wp.func
def intersect_ray_sphere(origin: wp.vec3f, dir: wp.vec3f, center: wp.vec3f, radius: float):
    to_sphere = center - origin

    tc = wp.dot(to_sphere, dir)

    if tc < 0.0:
        return tc

    d = wp.sqrt(wp.length_sq(to_sphere) - tc * tc)
    if d < 0.0:
        return -999999.0

    ts = wp.sqrt(radius * radius - d * d)

    return tc - ts


@wp.kernel(enable_backward=False)
def boids(
    boids: wp.array(dtype=Boid),
    world: World,
    dt: float,
    positions: wp.fabricarray(dtype=wp.vec3d),
    orientations: wp.fabricarray(dtype=wp.quatf),
    glows: wp.array(dtype=float),
):
    tid = wp.tid()

    boid = boids[tid]

    old_pos = positions[tid]
    old_rot = orientations[tid]

    pos = wp.vec3(float(old_pos[0]), float(old_pos[1]), float(old_pos[2]))
    vel = boid.vel

    forward = wp.quat_rotate(old_rot, wp.vec3f(1.0, 0.0, 0.0))

    force = wp.vec3f(0.0)

    # obstacle avoidance
    depenetration_force = 100.0
    avoidance_dist = 20.0
    avoidance_force = 20.0
    obstacles = world.obstacles
    num_obstacles = obstacles.shape[0]
    for i in range(num_obstacles):
        obstacle = obstacles[i]
        to_obstacle = obstacle.pos - pos
        # use padded radius
        radius = obstacle.radius + 2.0
        if wp.length(to_obstacle) < radius:
            # depenetration
            force += depenetration_force * wp.normalize(-to_obstacle)
        else:
            # avoidance
            t = intersect_ray_sphere(pos, forward, obstacle.pos, radius)
            if t > 0.0 and t < avoidance_dist:
                intersection_point = pos + t * forward
                out = intersection_point - obstacle.pos
                force += avoidance_force * (1.0 - t / avoidance_dist) * wp.normalize(out)

    # wander
    r = 10.0
    s0 = wp.sin(boid.wander_angles[0])
    c0 = wp.cos(boid.wander_angles[0])
    s1 = wp.sin(boid.wander_angles[1])
    c1 = wp.cos(boid.wander_angles[1])
    p = wp.vec3f(r * s0 * s1, r * s0 * c1, r * c0)
    offset = r + 1.0
    target = pos + wp.quat_rotate(old_rot, wp.vec3f(offset, 0.0, 0.0) + p)

    wander_force = 7.0
    force += wander_force * wp.normalize(target - pos)

    state = wp.rand_init(world.seed, tid)

    angle0 = boid.wander_angles[0] + wp.pi * (0.1 - 0.2 * wp.randf(state))
    angle1 = boid.wander_angles[1] + wp.pi * (0.1 - 0.2 * wp.randf(state))
    boid.wander_angles = wp.vec2f(angle0, angle1)

    cohesion_radius = 15.0
    cohesion_force = 20.0

    separation_radius = 10.0
    separation_force = 100.0

    # flocking
    query = wp.hash_grid_query(world.grid, pos, cohesion_radius)
    num_neighbors = int(0)
    num_align_neighbors = int(0)
    num_cohesion_neighbors = float(0)
    num_decohesion_neighbors = float(0)
    cohesion_pos_sum = wp.vec3f(0.0)
    decohesion_pos_sum = wp.vec3f(0.0)
    vel_sum = wp.vec3f(0.0)
    for index in query:
        if index != tid:
            other = boids[index]
            other_pos64 = positions[index]
            other_pos = wp.vec3f(float(other_pos64[0]), float(other_pos64[1]), float(other_pos64[2]))
            dist = wp.length(pos - other_pos)

            if dist < cohesion_radius:
                to_other = wp.normalize(other_pos - pos)
                # separation
                if dist < separation_radius:
                    force -= separation_force * (1.0 - dist / separation_radius) * to_other
                # cohesion
                bias = world.biases[boid.group, other.group]
                if bias > 0.0:
                    cohesion_pos_sum += bias * other_pos
                    num_cohesion_neighbors += bias
                else:
                    decohesion_pos_sum -= bias * other_pos
                    num_decohesion_neighbors -= bias
                # alignment
                if other.group == boid.group:
                    vel_sum += bias * other.vel
                    num_align_neighbors += 1
                num_neighbors += 1

    # align
    if num_align_neighbors > 0:
        vel_avg = vel_sum / float(num_align_neighbors)
        force += vel_avg - vel

    # cohere
    if num_cohesion_neighbors > 0.0:
        cohesion_pos_avg = cohesion_pos_sum / float(num_cohesion_neighbors)
        force += cohesion_force * wp.normalize(cohesion_pos_avg - pos)

    # decohere (group separation)
    if num_decohesion_neighbors > 0.0:
        decohesion_pos_avg = decohesion_pos_sum / float(num_decohesion_neighbors)
        force += cohesion_force * wp.normalize(pos - decohesion_pos_avg)

    # boundaries
    boundary_force = 20.0
    if pos[0] >= world.upper[0]:
        force += wp.vec3f(-boundary_force, 0.0, 0.0)
    if pos[0] <= world.lower[0]:
        force += wp.vec3f(boundary_force, 0.0, 0.0)
    if pos[1] >= world.upper[1]:
        force += wp.vec3f(0.0, -0.5 * boundary_force, 0.0)
    if pos[1] <= world.lower[1]:
        force += wp.vec3f(0.0, 5.0 * boundary_force, 0.0)
    if pos[2] >= world.upper[2]:
        force += wp.vec3f(0.0, 0.0, -boundary_force)
    if pos[2] <= world.lower[2]:
        force += wp.vec3f(0.0, 0.0, boundary_force)

    vel += dt * force / boid.mass

    # clamp speed
    max_speed = 15.0
    speed_sq = wp.length_sq(vel)
    if speed_sq > max_speed * max_speed:
        vel = max_speed * wp.normalize(vel)

    # update position
    pos += dt * vel
    positions[tid] = wp.vec3d(wp.float64(pos[0]), wp.float64(pos[1]), wp.float64(pos[2]))

    # update orientation
    dq = wp.quat_between_vectors(forward, vel)
    orientations[tid] = wp.normalize(dq * orientations[tid])

    # save velocity
    boid.vel = vel
    boids[tid] = boid

    # update glow as an exponentially weighted moving average to keep it smooth
    glow = wp.min(1.0, float(num_neighbors) / 40.0)
    glow_alpha = 0.5
    glows[tid] = glow_alpha * glow + (1.0 - glow_alpha) * glows[tid]


#   Internal State
# ------------------------------------------------------------------------------


class InternalState:
    """Internal state for the node."""

    def __init__(self) -> None:
        self.initialized = False

    def initialize(self, device):
        self.rng = np.random.default_rng(42)

        # requirement checks
        ext_mgr = omni.kit.app.get_app().get_extension_manager()

        # make sure USDRT is enabled
        usdrt_ext_name = "usdrt.scenegraph"
        if not ext_mgr.is_extension_enabled(usdrt_ext_name):
            raise RuntimeError(f"This sample requires the '{usdrt_ext_name}' extension to be enabled")

        # check USDRT version to make sure we have a working SelectPrims()
        usdrt_ext_id = ext_mgr.get_enabled_extension_id(usdrt_ext_name)
        usdrt_version_string = ext_mgr.get_extension_dict(usdrt_ext_id)["package"]["version"].split("-")[0]
        usdrt_version = tuple(int(v) for v in usdrt_version_string.split("."))
        if usdrt_version < (7, 3, 0):
            raise RuntimeError(
                f"USDRT version 7.3.0 is required, found {usdrt_version_string}.  Please update to a newer version of Kit to run this sample."
            )

        # check if FSD is enabled
        settings = carb.settings.get_settings()
        is_fsd_enabled = settings.get_as_bool("/app/useFabricSceneDelegate")
        if not is_fsd_enabled:
            print("***")
            print("*** Flocking demo warning: The Fabric Scene Delegate is not enabled.")
            print("*** Some features, like color animation, may not work.")
            print("*** You can enable FSD in Preferences->Rendering.")
            print("***")

        stage_id = omni.usd.get_context().get_stage_id()

        usdrt_stage = usdrt.Usd.Stage.Attach(stage_id)

        # import to Fabric
        for _prim in usdrt_stage.Traverse():
            pass

        # set up for Fabric interop
        boid_root = usdrt_stage.GetPrimAtPath(usdrt.Sdf.Path("/World/Boids"))
        boid_prims = boid_root.GetChildren()
        for prim in boid_prims:
            pos = prim.GetAttribute("xformOp:translate").Get()
            prim.CreateAttribute("_worldPosition", usdrt.Sdf.ValueTypeNames.Double3, True).Set(pos)
            prim.CreateAttribute("_worldOrientation", usdrt.Sdf.ValueTypeNames.Quatf, True).Set(
                usdrt.Gf.Quatf(1, 0, 0, 0)
            )
            prim.CreateAttribute("_worldScale", usdrt.Sdf.ValueTypeNames.Float3, True).Set(usdrt.Gf.Vec3f(1, 1, 1))
            prim.CreateAttribute("primvars:_emissive", usdrt.Sdf.ValueTypeNames.Float3Array, True).Set(
                usdrt.Vt.Vec3fArray([usdrt.Gf.Vec3f(1, 0, 1)])
            )

            # create a custom tag for the boids (could use applied schema too)
            prim.CreateAttribute("BoidTag", usdrt.Sdf.ValueTypeNames.AppliedSchemaTypeTag, True)

        num_boids = len(boid_prims)

        self.stage = usdrt_stage

        self.require_schemas = ["BoidTag"]

        self.transform_attrs = [
            (usdrt.Sdf.ValueTypeNames.Double3, "_worldPosition", usdrt.Usd.Access.ReadWrite),
            (usdrt.Sdf.ValueTypeNames.Quatf, "_worldOrientation", usdrt.Usd.Access.ReadWrite),
        ]

        self.color_attrs = [
            (usdrt.Sdf.ValueTypeNames.Float3Array, "primvars:_emissive", usdrt.Usd.Access.ReadWrite),
        ]

        npboids = np.zeros(num_boids, dtype=Boid.numpy_dtype())

        angles = math.pi - 2 * math.pi * self.rng.random(size=num_boids)
        vx = 20 * np.sin(angles)
        vz = 20 * np.cos(angles)
        npboids["vel"][:, 0] = vx
        npboids["vel"][:, 2] = vz

        npboids["wander_angles"][:, 0] = math.pi * self.rng.random(size=num_boids)
        npboids["wander_angles"][:, 1] = 2 * math.pi * self.rng.random(size=num_boids)

        min_mass = 1.0
        max_mass = 2.0
        npboids["mass"][:] = min_mass + (max_mass - min_mass) * self.rng.random(size=num_boids)

        # we can have up to 2 groups currently, but that can be easily extended
        self.num_groups = 2
        npboids["group"] = (self.rng.random(size=num_boids) * self.num_groups).astype(np.int32)

        num_obstacles = 3
        npobstacles = np.zeros(num_obstacles, dtype=Obstacle.numpy_dtype())
        npobstacles["pos"][0] = (-20, 30, -40)
        npobstacles["radius"][0] = 40
        npobstacles["pos"][1] = (90, 30, 30)
        npobstacles["radius"][1] = 30
        npobstacles["pos"][2] = (-100, 30, 60)
        npobstacles["radius"][2] = 25

        self.grid = wp.HashGrid(dim_x=32, dim_y=32, dim_z=32, device=device)

        biases = wp.mat33f(-1.0)
        for i in range(self.num_groups):
            biases[i, i] = 1.0

        world = World()
        world.lower = (-120, 10, -90)
        world.upper = (120, 40, 90)
        world.grid = self.grid.id
        world.seed = 0
        world.biases = biases
        world.obstacles = wp.array(npobstacles, dtype=Obstacle, device=device)
        self.world = world

        self.num_boids = num_boids
        self.boids = wp.array(npboids, dtype=Boid, device=device)

        # color ramps per group
        color_ramps = [
            [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.5]],
            [[0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.5, 1.0], [0.5, 1.0, 1.0]],
        ]

        # copy of positions used for updating the spatial grid
        self.positions = wp.zeros(num_boids, dtype=wp.vec3f, device=device)

        # color ramps are only used on the COLOR_DEVICE
        self.color_ramps_c = wp.array(color_ramps, dtype=wp.vec3f, device=COLOR_DEVICE)

        # keep a copy of group assignments on the COLOR_DEVICE
        self.groups_c = wp.array(npboids["group"], device=COLOR_DEVICE)

        # if we use different devices, the glow array must be copied on each update
        if COLOR_DEVICE == device:
            # use the same glow array on each device, no copying needed
            self.glows_c = wp.zeros(num_boids, dtype=float, device=device)
            self.glows_m = self.glows_c
        elif COLOR_DEVICE == "cpu" or device == "cpu":
            # use a pinned host array for async copying glows between devices
            glows_h = wp.zeros(num_boids, dtype=float, device="cpu", pinned=True)
            if COLOR_DEVICE == "cpu":
                self.glows_c = glows_h
                self.glows_m = wp.zeros_like(glows_h, device=device)
            else:
                self.glows_c = wp.zeros_like(glows_h, device=COLOR_DEVICE)
                self.glows_m = glows_h
        else:
            # two different CUDA devices
            self.glows_c = wp.zeros(num_boids, dtype=float, device=COLOR_DEVICE)
            self.glows_m = wp.zeros(num_boids, dtype=float, device=device)

            # ...but that's currently not supported in Kit
            raise ValueError("Multiple GPUs not supported yet")

        self.time = 0.0

        self.min_group_think = 3.0
        self.max_group_think = 10.0
        self.next_group_think = self.min_group_think + (self.max_group_think - self.min_group_think) * self.rng.random()

        self.frameno = 0

        self.initialized = True


#   Compute
# ------------------------------------------------------------------------------


def compute(db: OgnSamplePrimFlockingDatabase) -> None:
    """Evaluates the node."""

    state = db.per_instance_state

    device = wp.get_device()

    if not state.initialized:
        state.initialize(device)

    state.frameno += 1

    # get transform attributes
    selection = state.stage.SelectPrims(
        require_applied_schemas=state.require_schemas, require_attrs=state.transform_attrs, device=str(device)
    )

    fpos = wp.fabricarray(data=selection, attrib="_worldPosition")
    frot = wp.fabricarray(data=selection, attrib="_worldOrientation")

    # use fixed dt for stability
    dt = 1.0 / 60.0

    state.time += dt

    # copy positions to a contiguous array and convert to vec3f so they can be used to update the spatial grid
    wp.launch(copy_positions, dim=state.num_boids, inputs=[state.positions, fpos])

    # grid cell radius should be a bit bigger than query radius
    cell_radius = 20.0
    state.grid.build(state.positions, cell_radius)

    state.world.seed = state.frameno

    # step the flocking simulation
    wp.launch(boids, dim=state.num_boids, inputs=[state.boids, state.world, dt, fpos, frot, state.glows_m])

    # async copy from main device and remember the stream so we can sync later
    if COLOR_DEVICE != device:
        if device.is_cuda:
            work_stream = device.stream
        else:
            work_stream = wp.get_stream(COLOR_DEVICE)
        wp.copy(state.glows_c, state.glows_m, stream=work_stream)
    else:
        work_stream = None

    # get color attributes
    color_selection = state.stage.SelectPrims(
        require_applied_schemas=state.require_schemas, require_attrs=state.color_attrs, device=COLOR_DEVICE
    )

    fcolor = wp.fabricarray(data=color_selection, attrib="primvars:_emissive")

    # occasionally update group biases (whether they are attracted or repelled from each other)
    if state.num_groups > 1 and state.time >= state.next_group_think:
        # pick two random groups
        group0 = int(state.rng.random() * state.num_groups)
        group1 = int(state.rng.random() * state.num_groups)
        while group0 == group1:
            group1 = int(state.rng.random() * state.num_groups)

        # bias towards intra-group separation, but also allow attraction
        state.world.biases[group0, group1] = 1.0 - 5.0 * state.rng.random()
        state.world.biases[group1, group0] = 1.0 - 5.0 * state.rng.random()

        state.next_group_think += (
            state.min_group_think + (state.max_group_think - state.min_group_think) * state.rng.random()
        )

    if work_stream is not None:
        # wait for async GPU work to complete
        wp.synchronize_stream(work_stream)

    # update colors
    wp.launch(
        assign_colors,
        dim=state.num_boids,
        inputs=[state.glows_c, state.groups_c, state.color_ramps_c, fcolor],
        device=COLOR_DEVICE,
    )


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnSamplePrimFlocking:
    """Node."""

    @staticmethod
    def internal_state() -> InternalState:
        return InternalState()

    @staticmethod
    def compute(db: OgnSamplePrimFlockingDatabase) -> None:
        device = wp.get_device(MAIN_DEVICE)

        try:
            with wp.ScopedDevice(device):
                compute(db)
        except Exception:
            db.log_error(traceback.format_exc())
            return

        # Fire the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
