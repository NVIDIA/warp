# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Quaternion Shape Alignment (one query vs a dataset)
#
# Matches one small query molecule against a dataset of candidate molecules
# by finding, for each candidate, the rigid rotation that best overlays the
# query onto it, then ranking candidates by the aligned residual.
#
# The overlay is scored with a Cauchy robust loss on the per-atom distances,
#
#     rho(s) = 0.5 * delta^2 * log(1 + s / delta^2).
#
# This is the point of the example: a plain sum-of-squared-distances fit is the
# Wahba/Procrustes problem, which has a closed-form solution from the SVD of a
# 3x3 cross-covariance -- no Newton needed. The robust loss is nonconvex, so
# there is no such direct eigen solution; the rotation must be found iteratively,
# and each step needs the exact Hessian.
#
# Translation is removed by centering each molecule, leaving a pure
# orientation fit. The orientation is a unit quaternion, optimized
# intrinsically on the rotation manifold: each Newton step works in a
# 3-vector tangent chart
#
#     q(dtheta) = exp_map(dtheta) * q,
#
# so the local problem is a 3x3 dense solve. The tangent gradient and the
# 3x3 Hessian come from a single forward pass of a second-order quaternion
# jet (wp.JetSpace2) -- no tape, no hand-derivatives, no reverse sweep.
#
# The whole dataset is fit in parallel: one thread per candidate runs its own
# Newton loop, with the exact Hessian's eigenvalues replaced by their absolute
# values (it can be indefinite far from the optimum) and an Armijo line search.
#
###########################################################################

import numpy as np

import warp as wp

# This example is pure forward-mode (second-order jets, no tape / reverse pass),
# so skip generating adjoint code for its kernels -- it is unused here and, on
# the CUDA (NVRTC) backend, dominates compile time for the long jet chain.
wp.set_module_options({"enable_backward": False})

# One vec3 slot: the SO(3) tangent chart, width 3. One forward pass yields the 3x3 Hessian.
J = wp.JetSpace2((wp.vec3d,))

DTHETA_CLAMP = wp.float64(0.5)  # cap on a single Newton step, in radians
EIG_FLOOR = wp.float64(1.0e-6)  # floor for the absolute Hessian eigenvalues
ARMIJO_C = wp.float64(1.0e-4)  # sufficient-decrease constant for the line search

# Cauchy robust-loss scale. A finite delta makes the per-atom loss nonconvex, so
# the fit is not a linear least-squares problem and has no closed-form
# Procrustes/SVD solution -- which is exactly what motivates an iterative Newton
# solve and the exact 3x3 Hessian computed here.
DELTA = wp.float64(0.5)


@wp.func
def align_energy(
    dtheta: J.vec3,
    q0: wp.quatd,
    query: wp.array[wp.vec3d],
    cand: wp.array2d[wp.vec3d],
    c: int,
    n: int,
) -> J.scalar:
    """Cauchy robust alignment energy as a second-order jet.

    sum_a rho(s_a), s_a = || R(q) query_a - cand[c, a] ||^2,
    rho(s) = 0.5 * delta^2 * log(1 + s / delta^2).

    Nonconvex in the rotation, so it is not a linear least-squares (Wahba)
    problem and cannot be solved by a direct Procrustes/SVD fit; the exact
    Hessian drives an intrinsic Newton solve instead.
    """
    q = J.exp_map(dtheta) * q0
    d2 = DELTA * DELTA
    e = J.constant(wp.float64(0.0))
    for a in range(n):
        r = wp.quat_rotate(q, query[a]) - cand[c, a]
        s = wp.dot(r, r)
        e = e + wp.float64(0.5) * d2 * wp.log(wp.float64(1.0) + s / d2)
    return e


@wp.func
def rotvec_to_quat(v: wp.vec3d) -> wp.quatd:
    """Rotation-vector exp map for the (plain, non-jet) pose update."""
    a = wp.length(v)
    if a < wp.float64(1.0e-12):
        return wp.quatd(0.0, 0.0, 0.0, 1.0)
    return wp.quat_from_axis_angle(v / a, a)


@wp.func
def energy_value(
    q: wp.quatd,
    query: wp.array[wp.vec3d],
    cand: wp.array2d[wp.vec3d],
    c: int,
    n: int,
) -> wp.float64:
    """Plain (non-jet) Cauchy energy, used to accept or reject a trial step."""
    d2 = DELTA * DELTA
    e = wp.float64(0.0)
    for a in range(n):
        r = wp.quat_rotate(q, query[a]) - cand[c, a]
        s = wp.dot(r, r)
        e += wp.float64(0.5) * d2 * wp.log(wp.float64(1.0) + s / d2)
    return e


@wp.struct
class StepResult:
    q: wp.quatd
    improved: wp.int32


@wp.func
def newton_step(
    q: wp.quatd,
    query: wp.array[wp.vec3d],
    cand: wp.array2d[wp.vec3d],
    c: int,
    n: int,
) -> StepResult:
    """One eigenvalue-clamped Newton step with an Armijo line search."""
    dtheta = J.seed(wp.vec3d(0.0, 0.0, 0.0))
    e = align_energy(dtheta, q, query, cand, c, n)
    g = e.grad
    f0 = e.value

    # Positive-definite modified Newton direction: the robust Hessian can be
    # indefinite away from the optimum, so symmetrize it, replace its eigenvalues
    # by their absolute values -- flipping negative-curvature directions rather
    # than inflating the whole diagonal the way Levenberg-Marquardt would -- and
    # solve in the eigenbasis. Flooring the eigenvalues at a small positive value
    # instead would turn a large negative eigenvalue into a near-zero one and take
    # a huge step along it, which only suits nearly convex problems.
    hs = wp.float64(0.5) * (e.hess + wp.transpose(e.hess))
    eig_q, eig_d = wp.eig3(hs)
    gq = wp.transpose(eig_q) * g
    y = wp.vec3d(
        gq[0] / wp.max(wp.abs(eig_d[0]), EIG_FLOOR),
        gq[1] / wp.max(wp.abs(eig_d[1]), EIG_FLOOR),
        gq[2] / wp.max(wp.abs(eig_d[2]), EIG_FLOOR),
    )
    step = -(eig_q * y)

    sn = wp.length(step)
    if sn > DTHETA_CLAMP:
        step = step * (DTHETA_CLAMP / sn)

    # Armijo backtracking along the clamped Newton direction.
    gts = wp.dot(g, step)
    alpha = wp.float64(1.0)
    for _ls in range(30):
        # rotvec_to_quat(...) and q are unit, so the product is unit in exact
        # arithmetic; normalize only mops up floating-point drift.
        qn = wp.normalize(rotvec_to_quat(alpha * step) * q)
        if energy_value(qn, query, cand, c, n) <= f0 + ARMIJO_C * alpha * gts:
            return StepResult(qn, wp.int32(1))
        alpha = wp.float64(0.5) * alpha

    return StepResult(q, wp.int32(0))


@wp.func
def rmsd(q: wp.quatd, query: wp.array[wp.vec3d], cand: wp.array2d[wp.vec3d], c: int, n: int) -> wp.float64:
    """Plain geometric RMSD (not the robust objective), an interpretable score."""
    ssd = wp.float64(0.0)
    for a in range(n):
        r = wp.quat_rotate(q, query[a]) - cand[c, a]
        ssd += wp.dot(r, r)
    return wp.sqrt(ssd / wp.float64(n))


@wp.kernel
def fit_dataset(
    query: wp.array[wp.vec3d],
    cand: wp.array2d[wp.vec3d],
    n: int,
    iters: int,
    out_q: wp.array[wp.quatd],
    out_rmsd: wp.array[wp.float64],
):
    c = wp.tid()  # one thread per candidate
    q = wp.quatd(0.0, 0.0, 0.0, 1.0)

    for _it in range(iters):
        res = newton_step(q, query, cand, c, n)
        q = res.q
        if res.improved == 0:
            break

    out_rmsd[c] = rmsd(q, query, cand, c, n)
    out_q[c] = q


# --------------------------------------------------------------------------
# Dataset generation.
# --------------------------------------------------------------------------


def _rand_rotation(rng, max_angle=np.pi):
    v = rng.standard_normal(3)
    ang = rng.uniform(0.0, max_angle)
    axis = v / np.linalg.norm(v)
    return np.array([*(np.sin(ang / 2) * axis), np.cos(ang / 2)])  # [x, y, z, w]


def _qrot(q, X):
    x, y, z, w = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return (R @ X.T).T


def make_dataset(n_atoms=12, n_candidates=64, n_matches=4, seed=0):
    """One query molecule and a dataset of candidate molecules.

    Molecules are generated as small self-avoiding 3D chains with approximately
    fixed bond lengths and tetrahedral bond angles. This gives much more
    molecule-like ball-and-stick geometry than independent Gaussian points,
    while keeping the example deliberately simple.

    ``n_matches`` candidates are rotated, slightly distorted copies of the query;
    the remaining candidates are independently generated decoys.

    Everything is centered, so the fitting problem remains rotation-only.
    """
    rng = np.random.default_rng(seed)

    def random_molecule():
        # These are not intended as chemically exact units, just pleasant
        # ball-and-stick proportions.
        bond_length = 0.75
        bond_jitter = 0.05

        # For three consecutive atoms A-B-C, tetrahedral chemistry gives
        # angle ABC ~= 109.5 degrees. Since `prev` points A->B and the new
        # direction points B->C, their vector angle is 180 - 109.5 = 70.5 deg.
        turn_angle = np.deg2rad(70.5)
        angle_jitter = np.deg2rad(8.0)

        # Keep non-neighboring atoms from collapsing onto one another.
        min_separation = 0.62 * bond_length

        X = np.zeros((n_atoms, 3), dtype=np.float64)

        if n_atoms <= 1:
            return X

        # Arbitrary first bond; the entire molecule will subsequently be
        # subjected to a random rotation anyway.
        X[1] = np.array([bond_length, 0.0, 0.0])

        for i in range(2, n_atoms):
            prev = X[i - 1] - X[i - 2]
            prev /= np.linalg.norm(prev)

            accepted = False
            for _attempt in range(64):
                # Random unit direction perpendicular to the previous bond.
                u = rng.standard_normal(3)
                u -= np.dot(u, prev) * prev
                un = np.linalg.norm(u)
                if un < 1.0e-10:
                    continue
                u /= un

                theta = turn_angle + rng.normal(scale=angle_jitter)
                length = bond_length * (1.0 + rng.normal(scale=bond_jitter))

                direction = np.cos(theta) * prev + np.sin(theta) * u
                p = X[i - 1] + length * direction

                # Adjacent atoms are supposed to be close; only test against
                # atoms before the immediate predecessor.
                if i <= 2:
                    accepted = True
                else:
                    d = np.linalg.norm(X[: i - 1] - p, axis=1)
                    accepted = np.all(d > min_separation)

                if accepted:
                    X[i] = p
                    break

            if not accepted:
                # Extremely unlikely fallback: accept the last proposal rather
                # than complicating this toy generator with backtracking.
                X[i] = p

        # Make the generated molecule itself randomly oriented. This prevents
        # the query/decoys from sharing an accidental preferred construction
        # direction.
        X = _qrot(_rand_rotation(rng), X)

        X -= X.mean(axis=0)
        return X

    # ----------------------------------------------------------------------
    # Query molecule
    # ----------------------------------------------------------------------

    query = random_molecule()

    # ----------------------------------------------------------------------
    # Candidate dataset
    # ----------------------------------------------------------------------

    cands = np.empty((n_candidates, n_atoms, 3), dtype=np.float64)
    is_match = np.zeros(n_candidates, dtype=bool)

    n_matches = min(n_matches, n_candidates)
    match_ids = set(rng.choice(n_candidates, size=n_matches, replace=False))

    for c in range(n_candidates):
        if c in match_ids:
            # A true match has the same underlying geometry, with small
            # per-atom structural distortion. Subtracting the mean again
            # removes the tiny translation introduced by the noise.
            noise = 0.025 * rng.standard_normal((n_atoms, 3))
            base = query + noise
            base -= base.mean(axis=0)

            is_match[c] = True

            # Keep planted matches within a reasonably large but reliable
            # Newton basin from the identity initialization.
            rot = _rand_rotation(rng, max_angle=np.deg2rad(50.0))

        else:
            # Independently generated molecule: similar geometric statistics
            # and scale, but genuinely different shape.
            base = random_molecule()
            rot = _rand_rotation(rng)

        cands[c] = _qrot(rot, base)

    return query, cands, is_match


class Example:
    def __init__(self, n_atoms=12, n_candidates=64, n_matches=4, iters=25, seed=0):
        self.iters = iters
        query, cands, is_match = make_dataset(n_atoms, n_candidates, n_matches, seed)
        self.n_atoms = n_atoms
        self.n_candidates = n_candidates
        self.is_match = is_match
        self.query_np = query
        self.cands_np = cands

        self.query = wp.array(query, dtype=wp.vec3d)
        self.cands = wp.array(cands, dtype=wp.vec3d)

    def fit(self):
        """Fit every candidate to the query and return the rotations and aligned RMSDs."""
        out_q = wp.zeros(self.n_candidates, dtype=wp.quatd)
        out_rmsd = wp.zeros(self.n_candidates, dtype=wp.float64)
        wp.launch(
            fit_dataset,
            dim=self.n_candidates,
            inputs=[self.query, self.cands, self.n_atoms, self.iters],
            outputs=[out_q, out_rmsd],
        )
        return out_q.numpy(), out_rmsd.numpy()

    def report(self, rmsd, top=6):
        """Print the best matches and return whether every planted match ranks above every decoy."""
        order = np.argsort(rmsd)
        print(f"\nbest {top} matches of {self.n_candidates} candidates (aligned RMSD):")
        print("  rank  candidate   RMSD      true match?")
        for rank, c in enumerate(order[:top]):
            print(f"  {rank + 1:>4}  {c:>9}   {rmsd[c]:.5f}   {'yes' if self.is_match[c] else 'no'}")

        # A clean separation: every planted match should rank above every decoy.
        match_rmsd = rmsd[self.is_match]
        decoy_rmsd = rmsd[~self.is_match]
        print(f"\nplanted matches: max RMSD {match_rmsd.max():.5f}")
        print(f"decoys:          min RMSD {decoy_rmsd.min():.5f}")
        separated = match_rmsd.max() < decoy_rmsd.min()
        print(f"matches separated from decoys: {separated}")
        return separated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example()
        _, rmsd = example.fit()
        if not example.report(rmsd):
            raise RuntimeError("The planted matches did not separate from the decoys")
