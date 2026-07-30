# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""weldpath - nearest-segment assignment on a folded path, and the scatter
that consumes the returned index.

`nearest_segment()` returns (squared distance, segment index). The distance is
what the caller reads. The **index** is not part of the public return - it is
fed straight into `scatter_to_nodes()`, which pushes the update onto whichever
end of the chosen segment the closest point actually landed on. So the index is
not a free implementation detail: it decides where the downstream update goes.

Adjacent segments share an endpoint, so a point nearest to that shared endpoint
is *exactly* equidistant to both, and two implementations may break the tie
differently. Those ties are common and harmless. A second, much rarer kind of
tie - between two segments that share no node - is not harmless at all.

Whether any of this matters depends on the level you compare at and on whether
the downstream artifact actually diverges. Run `python weldpath.py --report`.
"""

import argparse

import numpy as np

GAP = 1.0
HEIGHT = 3.0


def make_comb(teeth=16, gap=GAP, height=HEIGHT, duplicate_tooth=None):
    """A zigzag comb: tall vertical teeth joined alternately at top and bottom.

    Teeth are `gap` apart, so the midline between two teeth is exactly
    equidistant to two vertical segments that share no node.

    `duplicate_tooth` additionally appends a segment that is geometrically
    COINCIDENT with tooth `duplicate_tooth` but built from its own nodes -
    the "duplicate primitive" case. Points near it tie exactly and
    *structurally*: unlike a midline tie, the tie survives perturbation,
    which is what makes it show up in a trajectory rather than only in a
    static comparison.
    """
    pts = []
    for k in range(teeth):
        x = k * gap
        pts += [(x, 0.0), (x, height)] if k % 2 == 0 else [(x, height), (x, 0.0)]
    nodes = np.array(pts, dtype=float)
    segs = np.stack([np.arange(len(nodes) - 1), np.arange(1, len(nodes))], axis=1)
    if duplicate_tooth is not None:
        a, b = segs[2 * duplicate_tooth]         # a vertical tooth
        extra = np.array([nodes[a], nodes[b]])
        segs = np.concatenate([segs, [[len(nodes), len(nodes) + 1]]], 0)
        nodes = np.concatenate([nodes, extra], 0)
    return nodes, segs


def sample_points(nodes, segs, n_on_path=6000, n_mid=200, seed=0, noise=0.03,
                  teeth=16, gap=GAP, height=HEIGHT):
    """Targets sampled ON the path plus noise (so a fit can actually converge),
    plus explicitly placed midline points that tie between segments sharing no
    node."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, len(segs), size=n_on_path)
    t = rng.random(n_on_path)[:, None]
    a, b = nodes[segs[s, 0]], nodes[segs[s, 1]]
    on = a + t * (b - a) + rng.normal(0.0, noise, size=(n_on_path, 2))
    k = rng.integers(0, teeth - 1, size=n_mid)
    x = k * gap + gap / 2.0                      # exactly between two teeth
    y = rng.uniform(0.25 * height, 0.75 * height, size=n_mid)
    mid = np.stack([x, y], axis=1)
    return np.concatenate([on, mid], 0), np.arange(n_on_path, n_on_path + n_mid)


def _proj(p, a, b):
    ab = b - a
    t = np.einsum("...i,...i->...", p - a, ab)
    den = np.einsum("...i,...i->...", ab, ab)
    t = np.clip(np.where(den < 1e-12, 0.0, t / np.maximum(den, 1e-12)), 0.0, 1.0)
    d = p - (a + t[..., None] * ab)
    return t, np.einsum("...i,...i->...", d, d)


def nearest_segment(points, nodes, segs, tie="first"):
    """tie='first' keeps the lowest index on an exact tie, 'last' the highest.
    Both are valid; neither is documented."""
    a, b = nodes[segs[:, 0]][None], nodes[segs[:, 1]][None]
    _, d2 = _proj(points[:, None, :], a, b)
    if tie == "first":
        idx = np.argmin(d2, axis=1)
    else:
        idx = d2.shape[1] - 1 - np.argmin(d2[:, ::-1], axis=1)
    return d2[np.arange(len(points)), idx], idx


def scatter_to_nodes(points, idx, nodes, segs, n_nodes):
    """Downstream consumer: the update lands on the CLOSEST FEATURE - split
    between the segment's endpoints by the projection parameter."""
    a, b = nodes[segs[idx, 0]], nodes[segs[idx, 1]]
    t, _ = _proj(points, a, b)
    out = np.zeros(n_nodes)
    np.add.at(out, segs[idx, 0], 1.0 - t)
    np.add.at(out, segs[idx, 1], t)
    return out


def scatter_to_segments(idx, n_segs):
    out = np.zeros(n_segs)
    np.add.at(out, idx, 1.0)
    return out


def relax(nodes, segs, points, tie, steps=200, lr=0.6, jitter=0.25, seed=11):
    """Pull a jittered path back onto the points. Returns the loss history.

    Starting jittered makes the fit non-vacuous: the loss has real distance to
    travel, and - as in a real fit - exact ties reappear as it converges.
    """
    rng = np.random.default_rng(seed)
    nd = nodes + rng.normal(0.0, jitter, size=nodes.shape)
    hist = []
    for _ in range(steps):
        d2, idx = nearest_segment(points, nd, segs, tie=tie)
        hist.append(float(d2.mean()))
        a, b = nd[segs[idx, 0]], nd[segs[idx, 1]]
        t, _ = _proj(points, a, b)
        resid = points - (a + t[:, None] * (b - a))
        num = np.zeros_like(nd)
        den = np.zeros(len(nd))
        np.add.at(num, segs[idx, 0], resid * (1.0 - t)[:, None])
        np.add.at(den, segs[idx, 0], 1.0 - t)
        np.add.at(num, segs[idx, 1], resid * t[:, None])
        np.add.at(den, segs[idx, 1], t)
        nd = nd + lr * num / np.maximum(den, 1e-9)[:, None]
    return hist


def report():
    nodes, segs = make_comb()
    pts, mid_ids = sample_points(nodes, segs, seed=3)
    d_a, i_a = nearest_segment(pts, nodes, segs, tie="first")
    d_b, i_b = nearest_segment(pts, nodes, segs, tie="last")

    print(f"path: {len(nodes)} nodes, {len(segs)} segments; {len(pts)} points "
          f"({len(mid_ids)} of them placed on exact midlines)\n")
    print(f"distances identical              : {np.array_equal(d_a, d_b)} "
          f"(max abs diff {np.abs(d_a - d_b).max():.3e})")
    tie = i_a != i_b
    print(f"returned INDEX differs           : {tie.sum()}/{len(pts)} "
          f"({100 * tie.mean():.1f}%)")
    sa, sb = segs[i_a[tie]], segs[i_b[tie]]
    shares = np.array([len(set(x) & set(y)) > 0 for x, y in zip(sa, sb)])
    print(f"  tied segments SHARE a node     : {shares.sum()} "
          f"({100 * shares.mean():.1f}%)  -> downstream-equivalent")
    print(f"  tied segments share NO node    : {(~shares).sum()} "
          f"({100 * (~shares).mean():.1f}%)  -> downstream-divergent")

    seg_a = scatter_to_segments(i_a, len(segs))
    seg_b = scatter_to_segments(i_b, len(segs))
    nod_a = scatter_to_nodes(pts, i_a, nodes, segs, len(nodes))
    nod_b = scatter_to_nodes(pts, i_b, nodes, segs, len(nodes))
    nod_d = np.abs(nod_a - nod_b)
    print(f"\ncompared at the INDEX-SHAPED level (per segment, the array the "
          f"index has\nthe shape of - NOT what any caller reads):")
    print(f"  max abs diff {np.abs(seg_a - seg_b).max():.1f} over "
          f"{int((np.abs(seg_a - seg_b) > 0).sum())} segments  <- looks fatal")
    print(f"compared at the OBSERVABLE level (per node, what the caller reads):")
    print(f"  max abs diff {nod_d.max():.4f} over "
          f"{int((nod_d > 1e-9).sum())} nodes  <- the number that counts")

    # which points actually drive the observable difference?
    only_shared = ~np.isin(np.arange(len(pts)), mid_ids)
    nA = scatter_to_nodes(pts[only_shared], i_a[only_shared], nodes, segs, len(nodes))
    nB = scatter_to_nodes(pts[only_shared], i_b[only_shared], nodes, segs, len(nodes))
    print(f"  with the midline points removed: max abs diff "
          f"{np.abs(nA - nB).max():.3e}  <- shared-node ties are harmless")

    print("\ndownstream trajectory (200 relaxation steps), two regimes:")
    for tag, jit in (("perturbed start (jitter=0.25)", 0.25),
                     ("exact start   (jitter=0.0)", 0.0)):
        ha = relax(nodes, segs, pts, "first", jitter=jit)
        hb = relax(nodes, segs, pts, "last", jitter=jit)
        red = ha[0] / max(ha[-1], 1e-30)
        rel = [abs(x - y) / max(abs(x), 1e-30) for x, y in zip(ha, hb)]
        print(f"  {tag}")
        print(f"    loss {ha[0]:.6f} -> {ha[-1]:.6f}  ({red:.1f}x reduction, "
              f"{'non-vacuous' if red > 1.5 else 'VACUOUS - proves nothing'})")
        print(f"    max rel divergence between tie-breaks: {max(rel):.3e}")
    print("\nThe two regimes disagree, and that is the point: the perturbed run "
          "moves the\ngeometry enough to break the *positional* ties, so the two "
          "tie-breaks track each\nother exactly and a trajectory test run that "
          "way proves nothing about ties. A\nstatic tie count is not a "
          "downstream failure, and a downstream test only counts\nif the "
          "condition under test survives into it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        nodes, segs = make_comb()
        pts, _ = sample_points(nodes, segs)
        d, i = nearest_segment(pts, nodes, segs)
        print(f"mean d2 = {d.mean():.6f}")
