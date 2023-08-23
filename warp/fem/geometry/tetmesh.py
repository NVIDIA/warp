import warp as wp

from warp.fem.types import ElementIndex, Coords, vec2i, vec3i, Sample
from warp.fem.types import NULL_ELEMENT_INDEX, OUTSIDE, NULL_DOF_INDEX, NULL_QP_INDEX

from .geometry import Geometry
from .element import Triangle, Tetrahedron
from .closest_point import project_on_tet_at_origin


@wp.struct
class TetmeshArg:
    tet_vertex_indices: wp.array2d(dtype=int)
    positions: wp.array(dtype=wp.vec3)

    vertex_tet_offsets: wp.array(dtype=int)
    vertex_tet_indices: wp.array(dtype=int)

    face_vertex_indices: wp.array(dtype=vec3i)
    face_tet_indices: wp.array(dtype=vec2i)


class Tetmesh(Geometry):
    """Tetrahedral mesh geometry"""

    def __init__(self, tet_vertex_indices: wp.array, positions: wp.array):
        """
        Constructs a tetrahedral mesh.

        Args:
            tet_vertex_indices: warp array of shape (num_tets, 4) containing vertex indices for each tet
            positions: warp array of shape (num_vertices, 3) containing 3d position for each vertex

        """
        self.dimension = 3

        self.tet_vertex_indices = tet_vertex_indices
        self.positions = positions

        self._face_vertex_indices: wp.array = None
        self._face_tet_indices: wp.array = None
        self._vertex_tet_offsets: wp.array = None
        self._vertex_tet_indices: wp.array = None

        self._build_topology()

    def cell_count(self):
        return self.tet_vertex_indices.shape[0]

    def vertex_count(self):
        return self.positions.shape[0]

    def side_count(self):
        return self._face_vertex_indices.shape[0]

    def boundary_side_count(self):
        return self._boundary_face_indices.shape[0]

    def reference_cell(self) -> Triangle:
        return Tetrahedron()

    def reference_side(self) -> Triangle:
        return Triangle()

    CellArg = TetmeshArg
    SideArg = TetmeshArg

    @wp.struct
    class SideIndexArg:
        boundary_face_indices: wp.array(dtype=int)

    # Geometry device interface

    def cell_arg_value(self, device) -> CellArg:
        args = self.CellArg()

        args.tet_vertex_indices = self.tet_vertex_indices.to(device)
        args.positions = self.positions.to(device)
        args.face_vertex_indices = self._face_vertex_indices.to(device)
        args.face_tet_indices = self._face_tet_indices.to(device)
        args.vertex_tet_offsets = self._vertex_tet_offsets.to(device)
        args.vertex_tet_indices = self._vertex_tet_indices.to(device)

        return args

    @wp.func
    def cell_position(args: CellArg, s: Sample):
        tet_idx = args.tet_vertex_indices[s.element_index]
        w0 = 1.0 - s.element_coords[0] - s.element_coords[1] - s.element_coords[2]
        return (
            w0 * args.positions[tet_idx[0]]
            + s.element_coords[0] * args.positions[tet_idx[1]]
            + s.element_coords[1] * args.positions[tet_idx[2]]
            + s.element_coords[2] * args.positions[tet_idx[3]]
        )

    @wp.func
    def _project_on_tet(args: CellArg, pos: wp.vec3, tet_index: int):
        p0 = args.positions[args.tet_vertex_indices[tet_index, 0]]

        q = pos - p0
        e1 = args.positions[args.tet_vertex_indices[tet_index, 1]] - p0
        e2 = args.positions[args.tet_vertex_indices[tet_index, 2]] - p0
        e3 = args.positions[args.tet_vertex_indices[tet_index, 3]] - p0

        dist, coords = project_on_tet_at_origin(q, e1, e2, e3)
        return dist, coords

    @wp.func
    def cell_lookup(args: CellArg, pos: wp.vec3, guess: Sample):
        closest_tet = int(NULL_ELEMENT_INDEX)
        closest_coords = Coords(OUTSIDE)
        closest_dist = float(1.0e8)

        for v in range(4):
            vtx = args.tet_vertex_indices[guess.element_index, v]
            tet_beg = args.vertex_tet_offsets[vtx]
            tet_end = args.vertex_tet_offsets[vtx + 1]

            for t in range(tet_beg, tet_end):
                tet = args.vertex_tet_indices[t]
                dist, coords = Tetmesh._project_on_tet(args, pos, tet)
                if dist <= closest_dist:
                    closest_dist = dist
                    closest_tet = tet
                    closest_coords = coords

        return Sample(closest_tet, closest_coords, NULL_QP_INDEX, 0.0, NULL_DOF_INDEX, NULL_DOF_INDEX)

    @wp.func
    def cell_measure(args: CellArg, cell_index: ElementIndex, coords: Coords):
        tet_idx = args.tet_vertex_indices[cell_index]

        v0 = args.positions[tet_idx[0]]
        v1 = args.positions[tet_idx[1]]
        v2 = args.positions[tet_idx[2]]
        v3 = args.positions[tet_idx[3]]

        mat = wp.mat33(
            v1 - v0,
            v2 - v0,
            v3 - v0,
        )

        return wp.abs(wp.determinant(mat)) / 6.0

    @wp.func
    def cell_measure(args: CellArg, s: Sample):
        return Tetmesh.cell_measure(args, s.element_index, s.element_coords)

    @wp.func
    def cell_measure_ratio(args: CellArg, s: Sample):
        return 1.0

    @wp.func
    def cell_normal(args: CellArg, s: Sample):
        return wp.vec3(0.0)

    def side_arg_value(self, device) -> SideArg:
        return self.cell_arg_value(device)

    def side_index_arg_value(self, device) -> SideIndexArg:
        args = self.SideIndexArg()

        args.boundary_face_indices = self._boundary_face_indices.to(device)

        return args

    @wp.func
    def boundary_side_index(args: SideIndexArg, boundary_side_index: int):
        """Boundary side to side index"""

        return args.boundary_face_indices[boundary_side_index]

    @wp.func
    def side_position(args: SideArg, s: Sample):
        face_idx = args.face_vertex_indices[s.element_index]
        return (
            s.element_coords[0] * args.positions[face_idx[0]]
            + s.element_coords[1] * args.positions[face_idx[1]]
            + s.element_coords[2] * args.positions[face_idx[2]]
        )

    @wp.func
    def side_measure(args: SideArg, side_index: ElementIndex, coords: Coords):
        face_idx = args.face_vertex_indices[side_index]
        v0 = args.positions[face_idx[0]]
        v1 = args.positions[face_idx[1]]
        v2 = args.positions[face_idx[2]]

        return 0.5 * wp.length(wp.cross(v1 - v0, v2 - v0))

    @wp.func
    def side_measure(args: SideArg, s: Sample):
        return Tetmesh.side_measure(args, s.element_index, s.element_coords)

    @wp.func
    def side_measure_ratio(args: SideArg, s: Sample):
        inner = Tetmesh.side_inner_cell_index(args, s.element_index)
        outer = Tetmesh.side_outer_cell_index(args, s.element_index)
        return Tetmesh.side_measure(args, s) / wp.min(
            Tetmesh.cell_measure(args, inner, Coords()),
            Tetmesh.cell_measure(args, outer, Coords()),
        )

    @wp.func
    def side_normal(args: SideArg, s: Sample):
        face_idx = args.face_vertex_indices[s.element_index]
        v0 = args.positions[face_idx[0]]
        v1 = args.positions[face_idx[1]]
        v2 = args.positions[face_idx[2]]

        return wp.normalize(wp.cross(v1 - v0, v2 - v0))

    @wp.func
    def face_to_tet_coords(args: SideArg, side_index: ElementIndex, tet_index: ElementIndex, side_coords: Coords):
        fvi = args.face_vertex_indices[side_index]

        tv1 = args.tet_vertex_indices[tet_index, 1]
        tv2 = args.tet_vertex_indices[tet_index, 2]
        tv3 = args.tet_vertex_indices[tet_index, 3]

        c1 = float(0.0)
        c2 = float(0.0)
        c3 = float(0.0)

        for k in range(3):
            if tv1 == fvi[k]:
                c1 = side_coords[k]
            elif tv2 == fvi[k]:
                c2 = side_coords[k]
            elif tv3 == fvi[k]:
                c3 = side_coords[k]

        return Coords(c1, c2, c3)

    @wp.func
    def tet_to_face_coords(args: SideArg, side_index: ElementIndex, tet_index: ElementIndex, tet_coords: Coords):
        fvi = args.face_vertex_indices[side_index]

        tv1 = args.tet_vertex_indices[tet_index, 1]
        tv2 = args.tet_vertex_indices[tet_index, 2]
        tv3 = args.tet_vertex_indices[tet_index, 3]

        if tv1 == fvi[0]:
            c0 = tet_coords[0]
        elif tv2 == fvi[0]:
            c0 = tet_coords[1]
        elif tv3 == fvi[0]:
            c0 = tet_coords[2]
        else:
            c0 = 1.0 - tet_coords[0] - tet_coords[1] - tet_coords[2]

        if tv1 == fvi[1]:
            c1 = tet_coords[0]
        elif tv2 == fvi[1]:
            c1 = tet_coords[1]
        elif tv3 == fvi[1]:
            c1 = tet_coords[2]
        else:
            c1 = 1.0 - tet_coords[0] - tet_coords[1] - tet_coords[2]

        if tv1 == fvi[2]:
            c2 = tet_coords[0]
        elif tv2 == fvi[2]:
            c2 = tet_coords[1]
        elif tv3 == fvi[2]:
            c2 = tet_coords[2]
        else:
            c2 = 1.0 - tet_coords[0] - tet_coords[1] - tet_coords[2]

        return wp.select(c0 + c1 + c2 > 0.999, Coords(OUTSIDE), Coords(c0, c1, c2))

    @wp.func
    def side_inner_cell_index(arg: SideArg, side_index: ElementIndex):
        return arg.face_tet_indices[side_index][0]

    @wp.func
    def side_outer_cell_index(arg: SideArg, side_index: ElementIndex):
        return arg.face_tet_indices[side_index][1]

    def _build_topology(self):
        from warp.fem.utils import compress_node_indices, masked_indices, _get_pinned_temp_count_buffer
        from warp.utils import array_scan

        device = self.tet_vertex_indices.device

        self._vertex_tet_offsets, self._vertex_tet_indices, _, __ = compress_node_indices(
            self.vertex_count(), self.tet_vertex_indices
        )

        vertex_start_face_count = wp.zeros(dtype=int, device=device, shape=self.vertex_count())
        vertex_start_face_offsets = wp.empty_like(vertex_start_face_count)

        vertex_face_other_vs = wp.empty(dtype=vec2i, device=device, shape=(4 * self.cell_count()))
        vertex_face_tets = wp.empty(dtype=int, device=device, shape=(4 * self.cell_count(), 2))

        # Count face edges starting at each vertex
        wp.launch(
            kernel=Tetmesh._count_starting_faces_kernel,
            device=device,
            dim=self.cell_count(),
            inputs=[self.tet_vertex_indices, vertex_start_face_count],
        )

        array_scan(in_array=vertex_start_face_count, out_array=vertex_start_face_offsets, inclusive=False)

        # Count number of unique edges (deduplicate across faces)
        vertex_unique_face_count = vertex_start_face_count
        wp.launch(
            kernel=Tetmesh._count_unique_starting_faces_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                self._vertex_tet_offsets,
                self._vertex_tet_indices,
                self.tet_vertex_indices,
                vertex_start_face_offsets,
                vertex_unique_face_count,
                vertex_face_other_vs,
                vertex_face_tets,
            ],
        )

        vertex_unique_face_offsets = wp.empty_like(vertex_start_face_offsets)
        array_scan(in_array=vertex_start_face_count, out_array=vertex_unique_face_offsets, inclusive=False)

        # Get back edge count to host
        if device.is_cuda:
            face_count = _get_pinned_temp_count_buffer(device)
            # Last vertex will not own any edge, so its count will be zero; just fetching last prefix count is ok
            wp.copy(dest=face_count, src=vertex_unique_face_offsets, src_offset=self.vertex_count() - 1, count=1)
            wp.synchronize_stream(wp.get_stream())
            face_count = int(face_count.numpy()[0])
        else:
            face_count = int(vertex_unique_face_offsets.numpy()[self.vertex_count() - 1])

        self._face_vertex_indices = wp.empty(shape=(face_count,), dtype=vec3i, device=device)
        self._face_tet_indices = wp.empty(shape=(face_count,), dtype=vec2i, device=device)

        boundary_mask = wp.empty(shape=(face_count,), dtype=int, device=device)

        # Compress edge data
        wp.launch(
            kernel=Tetmesh._compress_faces_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                vertex_start_face_offsets,
                vertex_unique_face_offsets,
                vertex_unique_face_count,
                vertex_face_other_vs,
                vertex_face_tets,
                self._face_vertex_indices,
                self._face_tet_indices,
                boundary_mask,
            ],
        )

        # Flip normals if necessary
        wp.launch(
            kernel=Tetmesh._flip_face_normals,
            device=device,
            dim=self.side_count(),
            inputs=[self._face_vertex_indices, self._face_tet_indices, self.tet_vertex_indices, self.positions],
        )

        self._boundary_face_indices, _ = masked_indices(boundary_mask)

    @wp.kernel
    def _count_starting_faces_kernel(
        tet_vertex_indices: wp.array2d(dtype=int), vertex_start_face_count: wp.array(dtype=int)
    ):
        t = wp.tid()
        for k in range(4):
            vi = vec3i(tet_vertex_indices[t, k], tet_vertex_indices[t, (k + 1) % 4], tet_vertex_indices[t, (k + 2) % 4])
            vm = wp.min(vi)

            for i in range(3):
                if vm == vi[i]:
                    wp.atomic_add(vertex_start_face_count, vm, 1)

    @wp.func
    def _find(
        needle: vec2i,
        values: wp.array(dtype=vec2i),
        beg: int,
        end: int,
    ):
        for i in range(beg, end):
            if values[i] == needle:
                return i

        return -1

    @wp.kernel
    def _count_unique_starting_faces_kernel(
        vertex_tet_offsets: wp.array(dtype=int),
        vertex_tet_indices: wp.array(dtype=int),
        tet_vertex_indices: wp.array2d(dtype=int),
        vertex_start_face_offsets: wp.array(dtype=int),
        vertex_start_face_count: wp.array(dtype=int),
        face_other_vs: wp.array(dtype=vec2i),
        face_tets: wp.array2d(dtype=int),
    ):
        v = wp.tid()

        face_beg = vertex_start_face_offsets[v]

        tet_beg = vertex_tet_offsets[v]
        tet_end = vertex_tet_offsets[v + 1]

        face_cur = face_beg

        for tet in range(tet_beg, tet_end):
            t = vertex_tet_indices[tet]

            for k in range(4):
                vi = vec3i(
                    tet_vertex_indices[t, k], tet_vertex_indices[t, (k + 1) % 4], tet_vertex_indices[t, (k + 2) % 4]
                )
                min_v = wp.min(vi)

                if v == min_v:
                    max_v = wp.max(vi)
                    mid_v = vi[0] + vi[1] + vi[2] - min_v - max_v
                    other_v = vec2i(mid_v, max_v)

                    # Check if other_v has been seen
                    seen_idx = Tetmesh._find(other_v, face_other_vs, face_beg, face_cur)

                    if seen_idx == -1:
                        face_other_vs[face_cur] = other_v
                        face_tets[face_cur, 0] = t
                        face_tets[face_cur, 1] = t
                        face_cur += 1
                    else:
                        face_tets[seen_idx, 1] = t

        vertex_start_face_count[v] = face_cur - face_beg

    @wp.kernel
    def _compress_faces_kernel(
        vertex_start_face_offsets: wp.array(dtype=int),
        vertex_unique_face_offsets: wp.array(dtype=int),
        vertex_unique_face_count: wp.array(dtype=int),
        uncompressed_face_other_vs: wp.array(dtype=vec2i),
        uncompressed_face_tets: wp.array2d(dtype=int),
        face_vertex_indices: wp.array(dtype=vec3i),
        face_tet_indices: wp.array(dtype=vec2i),
        boundary_mask: wp.array(dtype=int),
    ):
        v = wp.tid()

        start_beg = vertex_start_face_offsets[v]
        unique_beg = vertex_unique_face_offsets[v]
        unique_count = vertex_unique_face_count[v]

        for f in range(unique_count):
            src_index = start_beg + f
            face_index = unique_beg + f

            face_vertex_indices[face_index] = vec3i(
                v,
                uncompressed_face_other_vs[src_index][0],
                uncompressed_face_other_vs[src_index][1],
            )

            t0 = uncompressed_face_tets[src_index, 0]
            t1 = uncompressed_face_tets[src_index, 1]
            face_tet_indices[face_index] = vec2i(t0, t1)
            if t0 == t1:
                boundary_mask[face_index] = 1
            else:
                boundary_mask[face_index] = 0

    @wp.kernel
    def _flip_face_normals(
        face_vertex_indices: wp.array(dtype=vec3i),
        face_tet_indices: wp.array(dtype=vec2i),
        tet_vertex_indices: wp.array2d(dtype=int),
        positions: wp.array(dtype=wp.vec3),
    ):
        e = wp.tid()

        tet = face_tet_indices[e][0]

        tet_vidx = tet_vertex_indices[tet]
        face_vidx = face_vertex_indices[e]

        tet_centroid = (
            positions[tet_vidx[0]] + positions[tet_vidx[1]] + positions[tet_vidx[2]] + positions[tet_vidx[3]]
        ) / 4.0

        v0 = positions[face_vidx[0]]
        v1 = positions[face_vidx[1]]
        v2 = positions[face_vidx[2]]

        face_center = (v1 + v0 + v2) / 3.0
        face_normal = wp.cross(v1 - v0, v2 - v0)

        # if face normal points toward first tet centroid, flip indices
        if wp.dot(tet_centroid - face_center, face_normal) > 0.0:
            face_vertex_indices[e] = vec3i(face_vidx[0], face_vidx[2], face_vidx[1])
