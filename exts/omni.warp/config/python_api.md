
# Public API for module omni.warp.nodes:

## Classes

- class AttrTracking
  - def __init__(self, names: Sequence[str])
  - def have_attrs_changed(self, db: og.Database) -> bool
  - def update_state(self, db: og.Database)

- class NodeTimer
  - def __init__(self, name: str, db: Any, active: bool = False)

## Functions

- def basis_curves_copy_bundle(dst_bundle: og.BundleContents, src_bundle: og.BundleContents, deep_copy: bool = False, child_idx: int = 0)
- def basis_curves_create_bundle(dst_bundle: og.BundleContents, point_count: int, curve_count: int, type: Optional[str] = None, basis: Optional[str] = None, wrap: Optional[str] = None, xform: Optional[np.ndarray] = None, create_display_color: bool = False, create_widths: bool = False, child_idx: int = 0)
- def basis_curves_get_curve_count(bundle: og.BundleContents, child_idx: int = 0) -> int
- def basis_curves_get_curve_vertex_counts(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=int)
- def basis_curves_get_display_color(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=wp.vec3)
- def basis_curves_get_local_extent(bundle: og.BundleContents, child_idx: int = 0) -> np.ndarray
- def basis_curves_get_point_count(bundle: og.BundleContents, child_idx: int = 0) -> int
- def basis_curves_get_points(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=wp.vec3)
- def basis_curves_get_widths(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=float)
- def basis_curves_get_world_extent(bundle: og.BundleContents, axis_aligned: bool = False, child_idx: int = 0) -> np.ndarray
- def bundle_get_attr(bundle: og.BundleContents, name: str, child_idx: int = 0) -> Optional[og.AttributeData]
- def bundle_get_child_count(bundle: og.BundleContents) -> int
- def bundle_get_prim_type(bundle: og.BundleContents, child_idx: int = 0) -> str
- def bundle_get_world_xform(bundle: og.BundleContents, child_idx: int = 0) -> np.ndarray
- def bundle_has_changed(bundle: og.BundleContents, child_idx: int = 0) -> bool
- def bundle_have_attrs_changed(bundle: og.BundleContents, attr_names: Sequence[str], child_idx: int = 0) -> bool
- def device_get_cuda_compute() -> wp.context.Device
- def from_omni_graph_ptr(ptr, shape, dtype = None, device = None)
- def from_omni_graph(value: Union[np.ndarray, og.DataWrapper, og.AttributeData, og.DynamicAttributeAccess], dtype: Optional[type] = None, shape: Optional[Sequence[int]] = None, device: Optional[wp.context.Device] = None) -> wp.array
- def mesh_create_bundle(dst_bundle: og.BundleContents, point_count: int, vertex_count: int, face_count: int, xform: Optional[np.ndarray] = None, create_display_color: bool = False, create_normals: bool = False, create_uvs: bool = False, child_idx: int = 0)
- def mesh_copy_bundle(dst_bundle: og.BundleContents, src_bundle: og.BundleContents, deep_copy: bool = False, child_idx: int = 0)
- def mesh_get_display_color(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=wp.vec3)
- def mesh_get_face_count(bundle: og.BundleContents, child_idx: int = 0) -> int
- def mesh_get_face_vertex_counts(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=int)
- def mesh_get_face_vertex_indices(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=int)
- def mesh_get_local_extent(bundle: og.BundleContents, child_idx: int = 0) -> np.ndarray
- def mesh_get_normals(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=wp.vec3)
- def mesh_get_point_count(bundle: og.BundleContents, child_idx: int = 0) -> int
- def mesh_get_points(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=wp.vec3)
- def mesh_triangulate(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=int)
- def mesh_get_uvs(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=wp.vec2)
- def mesh_get_velocities(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=wp.vec3)
- def mesh_get_vertex_count(bundle: og.BundleContents, child_idx: int = 0) -> int
- def mesh_get_world_extent(bundle: og.BundleContents, axis_aligned: bool = False, child_idx: int = 0) -> np.ndarray
- def points_create_bundle(dst_bundle: og.BundleContents, point_count: int, xform: Optional[np.ndarray] = None, create_display_color: bool = False, create_masses: bool = False, create_velocities: bool = False, create_widths: bool = False, child_idx: int = 0)
- def points_copy_bundle(dst_bundle: og.BundleContents, src_bundle: og.BundleContents, deep_copy: bool = False, child_idx: int = 0)
- def points_get_display_color(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=wp.vec3)
- def points_get_local_extent(bundle: og.BundleContents, child_idx: int = 0) -> np.ndarray
- def points_get_masses(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=float)
- def points_get_point_count(bundle: og.BundleContents, child_idx: int = 0) -> int
- def points_get_points(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=wp.vec3)
- def points_get_velocities(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=wp.vec3)
- def points_get_widths(bundle: og.BundleContents, child_idx: int = 0) -> wp.array(dtype=float)
- def points_get_world_extent(bundle: og.BundleContents, axis_aligned: bool = False, child_idx: int = 0) -> np.ndarray
- def type_convert_og_to_warp(og_type: og.Type, dim_count: Optional[int] = None, as_str: bool = False, str_namespace: Optional[str] = 'wp') -> Union[Any, str]
- def type_convert_sdf_name_to_warp(sdf_type_name: str, dim_count: Optional[int] = None, as_str: bool = False, str_namespace: Optional[str] = 'wp') -> Union[Any, str]
- def type_convert_sdf_name_to_og(sdf_type_name: str, is_array: Optional[bool] = None) -> og.Type
