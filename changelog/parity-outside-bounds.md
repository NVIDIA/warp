Fix an incorrect inside sign from `wp.mesh_query_point_sign_parity()` for points strictly outside a watertight mesh's
current bounding box. When `n_sample > 0`, the query uses the box to determine the outside sign (`+1`) directly. This
skips ray casting for those points and prevents rounding near triangle edges from turning a gap into an apparent penetration.
