# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from warp.utils import warn

from .articulation import eval_fk, eval_ik
from .collide import collide
from .import_mjcf import parse_mjcf
from .import_snu import parse_snu
from .import_urdf import parse_urdf
from .import_usd import parse_usd, resolve_usd_from_url
from .inertia import transform_inertia
from .integrator import Integrator, integrate_bodies, integrate_particles
from .integrator_euler import SemiImplicitIntegrator
from .integrator_featherstone import FeatherstoneIntegrator
from .integrator_vbd import VBDIntegrator
from .integrator_xpbd import XPBDIntegrator
from .model import (
    GEO_BOX,
    GEO_CAPSULE,
    GEO_CONE,
    GEO_CYLINDER,
    GEO_MESH,
    GEO_NONE,
    GEO_PLANE,
    GEO_SDF,
    GEO_SPHERE,
    JOINT_BALL,
    JOINT_COMPOUND,
    JOINT_D6,
    JOINT_DISTANCE,
    JOINT_FIXED,
    JOINT_FREE,
    JOINT_MODE_FORCE,
    JOINT_MODE_TARGET_POSITION,
    JOINT_MODE_TARGET_VELOCITY,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    JOINT_UNIVERSAL,
    SDF,
    Control,
    JointAxis,
    Mesh,
    Model,
    ModelBuilder,
    ModelShapeGeometry,
    ModelShapeMaterials,
    State,
)
from .utils import (
    load_mesh,
    quat_from_euler,
    quat_to_euler,
    velocity_at_point,
)

warn(
    "The `warp.sim` module is deprecated and will be removed in v1.10. "
    "Please transition to using the forthcoming Newton library instead.",
    DeprecationWarning,
    stacklevel=2,
)
