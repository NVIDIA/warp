# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example cuTile Neural SDF
#
# Trains a DeepSDF-style neural signed-distance field of the Stanford
# bunny with a fused Warp + cuTile kernel. Per block of sample points,
# Warp's signed closest-point mesh query (BVH traversal) produces the
# ground-truth distances through the foreign-function bridge, a
# positional-encoding MLP forward pass runs as cuTile tensor-core
# matmuls, and the training loss is reduced in the same kernel. A second
# set of cuTile kernels applies the chain rule to the stored activations
# and Warp's Adam optimizer updates the weights, so training never leaves
# the GPU. The learned surface is extracted with marching cubes and
# ray-traced next to the ground-truth mesh.
#
# References:
#   Jeong Joon Park et al. 2019. DeepSDF: learning continuous signed
#   distance functions for shape representation. CVPR 2019.
#   https://arxiv.org/abs/1901.05103
#
###########################################################################

import os

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.experimental.cutile as wpct
import warp.geometry
import warp.optim

try:
    import cuda.tile as ct
except ImportError as err:
    raise ImportError('This example requires cuTile Python. Install it with: pip install "warp-lang[cutile]"') from err

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("This example requires the Pillow package. Please install it with 'pip install Pillow'.") from err

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

TILE = 128  # tile length per call
PAD = 4  # xyz padded to a power of two for the query tile
NUM_FREQ = 4  # positional-encoding octaves (finest wavelength ~0.25)
DIM_FEAT = 32  # 3 + 3*2*NUM_FREQ = 27, padded to a power of two
HIDDEN = 128  # MLP width: DIM_FEAT -> 3x HIDDEN -> 1
BATCH = 4096  # sample points per training iteration
NUM_BLOCKS = BATCH // TILE  # tile blocks per launch
CLAMP = 0.1  # DeepSDF-style SDF clamp: focus capacity near the surface
EVAL_CHUNK = 1 << 18  # grid points evaluated per host MLP pass when meshing the field

CAM_RADIUS = 2.2
CAM_HEIGHT = 0.35
CAM_POS = wp.vec3(0.0, CAM_HEIGHT, CAM_RADIUS)
CAM_TARGET = wp.vec3(0.0, 0.05, 0.0)
LIGHT_DIR = wp.normalize(wp.vec3(0.4, 0.8, 0.6))


@wp.func
def mesh_sdf(mesh_id: wp.handle, p: wp.vec3) -> float:
    """Signed distance from p to the mesh (BVH closest-point query)."""
    q = wp.mesh_query_point_sign_normal(mesh_id, p, 1.0e6)
    cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
    return q.sign * wp.length(p - cp)


@wp.kernel
def shade(mesh_id: wp.handle, cam_pos: wp.vec3, width: int, height: int, pixels: wp.array[wp.vec3]):
    """Normal-shaded pinhole ray-cast of a mesh with a look-at camera."""
    tid = wp.tid()
    x = tid % width
    y = tid // width
    sx = (2.0 * float(x) - float(width)) / float(height)
    sy = 2.0 * float(y) / float(height) - 1.0
    forward = wp.normalize(CAM_TARGET - cam_pos)
    right = wp.normalize(wp.cross(forward, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, forward)
    rd = wp.normalize(right * sx + up * sy + forward * 1.7)
    color = wp.vec3(1.0, 1.0, 1.0)
    query = wp.mesh_query_ray(mesh_id, cam_pos, rd, 1.0e6)
    if query.result:
        n = wp.normalize(query.normal)
        lite = wp.clamp(wp.dot(n, LIGHT_DIR), 0.0, 1.0)
        color = (n * 0.5 + wp.vec3(0.5, 0.5, 0.5)) * (0.35 + 0.65 * lite)
    pixels[tid] = color


@wp.kernel
def sum_partials(partials: wp.array2d[float], out: wp.array[float]):
    """Sum per-block partial gradients (one row per block) into one flat gradient."""
    i = wp.tid()
    total = float(0.0)
    for k in range(partials.shape[0]):
        total += partials[k, i]
    out[i] = total


def fourier_features(p):
    """Encode (n, 3) points into (n, DIM_FEAT) padded positional-encoding features."""
    feats = [p]
    for k in range(NUM_FREQ):
        w = (2.0**k) * np.pi
        feats += [np.sin(w * p), np.cos(w * p)]
    f = np.concatenate(feats, axis=1).astype(np.float32)
    out = np.zeros((len(p), DIM_FEAT), dtype=np.float32)
    out[:, : f.shape[1]] = f
    return out


class Example:
    def __init__(self, train_iters=8000, grid=128, resolution=512):
        self.train_iters = train_iters
        self.grid = grid
        self.resolution = resolution
        self.iteration = 0
        self.loss = None
        self.rng = np.random.default_rng(0)

        self._load_bunny()
        self._init_mlp()
        self._build_kernels()

        # per-iteration kernel inputs and outputs, allocated once
        self.points = wp.zeros((BATCH, PAD), dtype=wp.float32)
        self.feats = wp.zeros((BATCH, DIM_FEAT), dtype=wp.float32)
        self.sdf_true = wp.zeros(BATCH, dtype=wp.float32)
        self.sdf_pred = wp.zeros(BATCH, dtype=wp.float32)
        self.acts = [wp.zeros((BATCH, HIDDEN), dtype=wp.float32) for _ in range(3)]
        self.loss_out = wp.zeros(NUM_BLOCKS, dtype=wp.float32)
        # backward: loss gradient, pre-activation gradients, and per-block
        # partial weight gradients that a reduction kernel sums
        self.dpred = wp.zeros(BATCH, dtype=wp.float32)
        self.dzs = [wp.zeros((BATCH, HIDDEN), dtype=wp.float32) for _ in range(3)]
        self.grads = [wp.zeros_like(p) for p in self.params]
        self.partials = [wp.zeros((NUM_BLOCKS, *p.shape), dtype=wp.float32) for p in self.params]
        # flat views of the same memory for the reduction and the optimizer
        self.flat_grads = [g.flatten() for g in self.grads]
        self.flat_partials = [
            part.reshape((NUM_BLOCKS, g.size)) for part, g in zip(self.partials, self.grads, strict=True)
        ]

        self.learned_mesh = None  # extracted from the trained field by render()
        self.pixels = None  # side-by-side render, filled by render()

    def _load_bunny(self):
        """Load the bunny asset, normalize into [-0.85, 0.85]^3, build the Warp mesh."""
        stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bunny"))
        points = np.array(geom.GetPointsAttr().Get(), dtype=np.float32)
        indices = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        center = 0.5 * (points.min(axis=0) + points.max(axis=0))
        scale = 1.7 / (points.max(axis=0) - points.min(axis=0)).max()
        points = (points - center) * scale
        self.mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3),
            indices=wp.array(indices, dtype=int),
        )

        # area-weighted surface sampling setup
        faces = indices.reshape(-1, 3)
        self.tri_pts = points[faces]
        areas = 0.5 * np.linalg.norm(
            np.cross(self.tri_pts[:, 1] - self.tri_pts[:, 0], self.tri_pts[:, 2] - self.tri_pts[:, 0]), axis=1
        )
        self.tri_cdf = np.cumsum(areas) / areas.sum()

    def _init_mlp(self):
        """Initialize MLP weights (He init) and Adam optimizer state."""

        def he(shape):
            """He-scaled Gaussian initialization for one weight matrix."""
            return (self.rng.standard_normal(shape) * np.sqrt(2.0 / shape[0])).astype(np.float32)

        host_params = [
            he((DIM_FEAT, HIDDEN)),
            np.zeros(HIDDEN, np.float32),
            he((HIDDEN, HIDDEN)),
            np.zeros(HIDDEN, np.float32),
            he((HIDDEN, HIDDEN)),
            np.zeros(HIDDEN, np.float32),
            he((HIDDEN, 1)),
            np.zeros(1, np.float32),
        ]
        self.params = [wp.array(p, dtype=wp.float32) for p in host_params]
        # Adam works on flat arrays; these are views of the same memory
        self.optimizer = warp.optim.Adam([p.flatten() for p in self.params], lr=1e-3)

    def _build_kernels(self):
        """Bridge the Warp query into cuTile and build the fused forward and the backward kernels."""
        wp_sdf = wpct.export_func(mesh_sdf)

        @ct.kernel(num_worker_warps=TILE // 32)
        def fused_kernel(
            mesh_id: ct.Constant[int],
            points,  # (BATCH, PAD) raw xyz for the geometry query
            feats,  # (BATCH, DIM_FEAT) positional-encoding features for the MLP
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            w4,
            b4,
            sdf_true_out,
            sdf_pred_out,
            h1_out,
            h2_out,
            h3_out,
            loss_out,
        ):
            """Produce labels (Warp mesh query), MLP forward, activations, and loss in one launch."""
            bid = ct.bid(0)

            # Warp: ground-truth signed distance, one BVH query per tile element
            p = ct.load(points, (bid, 0), shape=(TILE, PAD))
            sdf_true = wp_sdf(mesh_id, ct.reshape(p, (TILE * PAD,)))

            # cuTile: MLP forward pass on tensor-core matmuls
            x = ct.load(feats, (bid, 0), shape=(TILE, DIM_FEAT))
            h1 = ct.maximum(
                ct.matmul(x, ct.load(w1, (0, 0), shape=(DIM_FEAT, HIDDEN))) + ct.load(b1, (0,), shape=(HIDDEN,)), 0.0
            )
            h2 = ct.maximum(
                ct.matmul(h1, ct.load(w2, (0, 0), shape=(HIDDEN, HIDDEN))) + ct.load(b2, (0,), shape=(HIDDEN,)), 0.0
            )
            h3 = ct.maximum(
                ct.matmul(h2, ct.load(w3, (0, 0), shape=(HIDDEN, HIDDEN))) + ct.load(b3, (0,), shape=(HIDDEN,)), 0.0
            )
            y = ct.matmul(h3, ct.load(w4, (0, 0), shape=(HIDDEN, 1)))
            sdf_pred = ct.reshape(y, (TILE,)) + ct.load(b4, (0,), shape=(1,))

            # fused supervision: per-block MSE against the clamped ground truth,
            # plus the hidden activations so the backward kernels differentiate
            # this forward pass instead of recomputing one
            target = ct.minimum(ct.maximum(sdf_true, -CLAMP), CLAMP)
            err = sdf_pred - target
            ct.store(sdf_true_out, (bid,), target)
            ct.store(sdf_pred_out, (bid,), sdf_pred)
            ct.store(h1_out, (bid, 0), h1)
            ct.store(h2_out, (bid, 0), h2)
            ct.store(h3_out, (bid, 0), h3)
            ct.store(loss_out, (bid,), ct.reshape(ct.sum(err * err) / float(TILE), (1,)))

        self.fused_kernel = fused_kernel

        @ct.kernel(num_worker_warps=TILE // 32)
        def backward_kernel(sdf_pred, sdf_true, h1, h2, h3, w2, w3, w4, dpred_out, dz1_out, dz2_out, dz3_out):
            """Back-propagate the MSE through the MLP, per block, from the forward kernel's stored activations."""
            bid = ct.bid(0)
            pred = ct.load(sdf_pred, (bid,), shape=(TILE,))
            target = ct.load(sdf_true, (bid,), shape=(TILE,))
            a1 = ct.load(h1, (bid, 0), shape=(TILE, HIDDEN))
            a2 = ct.load(h2, (bid, 0), shape=(TILE, HIDDEN))
            a3 = ct.load(h3, (bid, 0), shape=(TILE, HIDDEN))
            zeros = ct.zeros((TILE, HIDDEN), dtype=ct.float32)

            # d(mean squared error)/d(prediction), then the ReLU masks come from
            # the stored activations (a > 0 is equivalent to z > 0)
            dpred = (pred - target) * (2.0 / BATCH)
            dz3 = ct.reshape(dpred, (TILE, 1)) * ct.transpose(ct.load(w4, (0, 0), shape=(HIDDEN, 1)))
            dz3 = ct.where(a3 > 0.0, dz3, zeros)
            dz2 = ct.matmul(dz3, ct.transpose(ct.load(w3, (0, 0), shape=(HIDDEN, HIDDEN))))
            dz2 = ct.where(a2 > 0.0, dz2, zeros)
            dz1 = ct.matmul(dz2, ct.transpose(ct.load(w2, (0, 0), shape=(HIDDEN, HIDDEN))))
            dz1 = ct.where(a1 > 0.0, dz1, zeros)
            ct.store(dpred_out, (bid,), dpred)
            ct.store(dz1_out, (bid, 0), dz1)
            ct.store(dz2_out, (bid, 0), dz2)
            ct.store(dz3_out, (bid, 0), dz3)

        self.backward_kernel = backward_kernel

        def make_weight_grad(dim_in):
            """Build the kernel computing one block's partial weight and bias gradients for a hidden layer."""

            @ct.kernel(num_worker_warps=TILE // 32)
            def weight_grad(a, dz, gw_part, gb_part):
                """Store a^T dz and the column sums of dz for this block."""
                k = ct.bid(0)
                a_blk = ct.load(a, (k, 0), shape=(TILE, dim_in))
                dz_blk = ct.load(dz, (k, 0), shape=(TILE, HIDDEN))
                ct.store(gw_part, (k, 0, 0), ct.reshape(ct.matmul(ct.transpose(a_blk), dz_blk), (1, dim_in, HIDDEN)))
                ct.store(gb_part, (k, 0), ct.reshape(ct.sum(dz_blk, axis=0), (1, HIDDEN)))

            return weight_grad

        self.weight_grad_kernels = [make_weight_grad(DIM_FEAT), make_weight_grad(HIDDEN), make_weight_grad(HIDDEN)]

        @ct.kernel(num_worker_warps=TILE // 32)
        def output_grad(a3, dpred, gw_part, gb_part):
            """Store this block's partial gradient of the single-output layer."""
            k = ct.bid(0)
            a_blk = ct.load(a3, (k, 0), shape=(TILE, HIDDEN))
            d_blk = ct.reshape(ct.load(dpred, (k,), shape=(TILE,)), (TILE, 1))
            ct.store(gw_part, (k, 0, 0), ct.reshape(ct.sum(a_blk * d_blk, axis=0), (1, HIDDEN, 1)))
            ct.store(gb_part, (k, 0), ct.reshape(ct.sum(d_blk), (1, 1)))

        self.output_grad_kernel = output_grad

    def _sample_batch(self):
        """Half uniform box samples, half surface samples with two offset scales."""
        rng = self.rng
        n = BATCH // 2
        uniform = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
        face = np.minimum(np.searchsorted(self.tri_cdf, rng.random(n)), len(self.tri_cdf) - 1)
        r1, r2 = np.sqrt(rng.random((n, 1))), rng.random((n, 1))
        a, b, c = self.tri_pts[:, 0][face], self.tri_pts[:, 1][face], self.tri_pts[:, 2][face]
        surf = (1 - r1) * a + r1 * (1 - r2) * b + r1 * r2 * c
        sigma = np.where(rng.random((n, 1)) < 0.5, 0.01, 0.08).astype(np.float32)
        surf += (sigma * rng.standard_normal((n, 3))).astype(np.float32)
        return np.concatenate([uniform, surf.astype(np.float32)], axis=0)

    def step(self):
        """Run one training iteration: the fused kernel supplies labels, forward pass, and loss."""
        self.iteration += 1
        it = self.iteration
        if it in (self.train_iters // 2, 3 * self.train_iters // 4, 7 * self.train_iters // 8):
            self.optimizer.lr *= 0.3

        batch3 = self._sample_batch()
        batch_pad = np.zeros((BATCH, PAD), dtype=np.float32)
        batch_pad[:, :3] = batch3
        self.points.assign(batch_pad)
        self.feats.assign(fourier_features(batch3))

        stream = wp.get_stream().cuda_stream
        grid = (NUM_BLOCKS, 1, 1)
        ct.launch(
            stream,
            grid,
            self.fused_kernel,
            (
                self.mesh.id,
                self.points,
                self.feats,
                *self.params,
                self.sdf_true,
                self.sdf_pred,
                *self.acts,
                self.loss_out,
            ),
        )

        # backward differentiates the kernel's own forward: prediction and
        # activations come from the launch above, labels from the fused query
        ct.launch(
            stream,
            grid,
            self.backward_kernel,
            (
                self.sdf_pred,
                self.sdf_true,
                *self.acts,
                self.params[2],
                self.params[4],
                self.params[6],
                self.dpred,
                *self.dzs,
            ),
        )
        layer_inputs = [self.feats, *self.acts]
        for layer in range(3):
            gw_part, gb_part = self.partials[2 * layer], self.partials[2 * layer + 1]
            ct.launch(
                stream, grid, self.weight_grad_kernels[layer], (layer_inputs[layer], self.dzs[layer], gw_part, gb_part)
            )
        ct.launch(stream, grid, self.output_grad_kernel, (self.acts[2], self.dpred, self.partials[6], self.partials[7]))
        # Warp reduces the per-block partials into the gradients Adam consumes
        for part, grad in zip(self.flat_partials, self.flat_grads, strict=True):
            wp.launch(sum_partials, dim=grad.shape[0], inputs=[part, grad])
        self.optimizer.step(self.flat_grads)

        if it == 1 or it % 500 == 0 or it == self.train_iters:
            # the only host read of the iteration; .numpy() synchronizes
            self.loss = float(self.loss_out.numpy().mean())
            print(f"Iter: {it} Loss: {self.loss:.6f}")

    def _extract_surface(self):
        """Mesh the learned SDF's zero level set with marching cubes, then lightly smooth it for display."""
        grid = self.grid
        axis = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
        gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
        grid_pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
        sdf_grid = np.empty(len(grid_pts), dtype=np.float32)
        # The field is evaluated once here with NumPy: the training kernel
        # fuses the MLP forward with the mesh query and the loss, so there is
        # no standalone device forward pass to reuse for arbitrary points.
        p = [param.numpy() for param in self.params]
        for s in range(0, len(grid_pts), EVAL_CHUNK):
            f = fourier_features(grid_pts[s : s + EVAL_CHUNK])
            h = np.maximum(f @ p[0] + p[1], 0.0)
            h = np.maximum(h @ p[2] + p[3], 0.0)
            h = np.maximum(h @ p[4] + p[5], 0.0)
            sdf_grid[s : s + EVAL_CHUNK] = (h @ p[6]).reshape(-1) + p[7][0]

        field = wp.array(sdf_grid.reshape(grid, grid, grid), dtype=wp.float32)
        verts_dev, tris_dev = wp.geometry.IsoSurfaceMarchingCubes.extract(
            field, 0.0, lower=wp.vec3(-1.0, -1.0, -1.0), upper=wp.vec3(1.0, 1.0, 1.0)
        )
        verts = verts_dev.numpy()
        tris = tris_dev.numpy().reshape(-1, 3)

        # light Laplacian smoothing, for the render only: the short training
        # schedule leaves high-frequency ripple on the extracted surface
        for _ in range(3):
            acc = np.zeros_like(verts)
            cnt = np.zeros(len(verts), dtype=np.int32)
            for e0, e1 in ((0, 1), (1, 2), (2, 0)):
                np.add.at(acc, tris[:, e0], verts[tris[:, e1]])
                np.add.at(acc, tris[:, e1], verts[tris[:, e0]])
                np.add.at(cnt, tris[:, e0], 1)
                np.add.at(cnt, tris[:, e1], 1)
            mask = cnt > 0
            verts[mask] += 0.5 * (acc[mask] / cnt[mask, None] - verts[mask])

        return wp.Mesh(points=wp.array(verts, dtype=wp.vec3), indices=wp.array(tris.flatten(), dtype=int))

    def render(self, cam_pos=None):
        """Ray-trace the ground-truth and learned surfaces side by side into self.pixels."""
        if self.learned_mesh is None:
            self.learned_mesh = self._extract_surface()

        img = self.resolution
        cam = cam_pos if cam_pos is not None else CAM_POS
        panels = []
        for m in (self.mesh, self.learned_mesh):
            pixels = wp.zeros(img * img, dtype=wp.vec3)
            wp.launch(shade, dim=img * img, inputs=[m.id, cam, img, img, pixels])
            frame = (np.clip(pixels.numpy().reshape(img, img, 3), 0.0, 1.0) * 255).astype(np.uint8)
            panels.append(frame[::-1])
        self.pixels = np.concatenate(panels, axis=1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--train-iters", type=int, default=8000, help="Total number of training iterations.")
    parser.add_argument("--grid", type=int, default=128, help="Marching-cubes grid resolution.")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution per panel.")
    parser.add_argument("--turntable", type=int, default=0, help="Also save a turntable GIF with this many frames.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the saving of output images and the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]
    if args.train_iters < 1:
        parser.error("--train-iters must be at least 1")
    if args.grid < 2:
        parser.error("--grid must be at least 2 (marching cubes needs one cell per axis)")
    if args.resolution < 1:
        parser.error("--resolution must be at least 1")

    with wp.ScopedDevice(args.device):
        example = Example(train_iters=args.train_iters, grid=args.grid, resolution=args.resolution)

        for _ in range(args.train_iters):
            example.step()

        if not args.headless:
            example.render()
            out_path = "example_cutile_neural_sdf.png"
            Image.fromarray(example.pixels).save(out_path)
            print(f"Saved {out_path} (left: ground-truth mesh | right: learned SDF reconstruction)")

            if args.turntable > 0:
                frames = []
                for k in range(args.turntable):
                    a = 2.0 * np.pi * k / args.turntable
                    example.render(cam_pos=wp.vec3(CAM_RADIUS * np.sin(a), CAM_HEIGHT, CAM_RADIUS * np.cos(a)))
                    frames.append(Image.fromarray(example.pixels))
                gif_path = "example_cutile_neural_sdf_turntable.gif"
                frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=70, loop=0)
                print(f"Saved {gif_path} ({args.turntable} frames)")

            if MATPLOTLIB_AVAILABLE:
                plt.imshow(example.pixels, interpolation="antialiased")
                plt.axis("off")
                plt.show()
            else:
                print("matplotlib is not installed; skipping the interactive display.")
