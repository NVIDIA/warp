WP_API void builtin_min_int32_int32(int32 x, int32 y, int* ret) { *ret = wp::min(x, y); }
WP_API void builtin_min_float32_float32(float32 x, float32 y, float* ret) { *ret = wp::min(x, y); }
WP_API void builtin_max_int32_int32(int32 x, int32 y, int* ret) { *ret = wp::max(x, y); }
WP_API void builtin_max_float32_float32(float32 x, float32 y, float* ret) { *ret = wp::max(x, y); }
WP_API void builtin_clamp_int32_int32_int32(int32 x, int32 a, int32 b, int* ret) { *ret = wp::clamp(x, a, b); }
WP_API void builtin_clamp_float32_float32_float32(float32 x, float32 a, float32 b, float* ret) { *ret = wp::clamp(x, a, b); }
WP_API void builtin_abs_int32(int32 x, int* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_float32(float32 x, float* ret) { *ret = wp::abs(x); }
WP_API void builtin_sign_int32(int32 x, int* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_float32(float32 x, float* ret) { *ret = wp::sign(x); }
WP_API void builtin_step_float32(float32 x, float* ret) { *ret = wp::step(x); }
WP_API void builtin_nonzero_float32(float32 x, float* ret) { *ret = wp::nonzero(x); }
WP_API void builtin_sin_float32(float32 x, float* ret) { *ret = wp::sin(x); }
WP_API void builtin_cos_float32(float32 x, float* ret) { *ret = wp::cos(x); }
WP_API void builtin_acos_float32(float32 x, float* ret) { *ret = wp::acos(x); }
WP_API void builtin_asin_float32(float32 x, float* ret) { *ret = wp::asin(x); }
WP_API void builtin_sqrt_float32(float32 x, float* ret) { *ret = wp::sqrt(x); }
WP_API void builtin_tan_float32(float32 x, float* ret) { *ret = wp::tan(x); }
WP_API void builtin_atan_float32(float32 x, float* ret) { *ret = wp::atan(x); }
WP_API void builtin_atan2_float32_float32(float32 y, float32 x, float* ret) { *ret = wp::atan2(y, x); }
WP_API void builtin_sinh_float32(float32 x, float* ret) { *ret = wp::sinh(x); }
WP_API void builtin_cosh_float32(float32 x, float* ret) { *ret = wp::cosh(x); }
WP_API void builtin_tanh_float32(float32 x, float* ret) { *ret = wp::tanh(x); }
WP_API void builtin_log_float32(float32 x, float* ret) { *ret = wp::log(x); }
WP_API void builtin_log2_float32(float32 x, float* ret) { *ret = wp::log2(x); }
WP_API void builtin_log10_float32(float32 x, float* ret) { *ret = wp::log10(x); }
WP_API void builtin_exp_float32(float32 x, float* ret) { *ret = wp::exp(x); }
WP_API void builtin_pow_float32_float32(float32 x, float32 y, float* ret) { *ret = wp::pow(x, y); }
WP_API void builtin_round_float32(float32 x, float* ret) { *ret = wp::round(x); }
WP_API void builtin_rint_float32(float32 x, float* ret) { *ret = wp::rint(x); }
WP_API void builtin_trunc_float32(float32 x, float* ret) { *ret = wp::trunc(x); }
WP_API void builtin_floor_float32(float32 x, float* ret) { *ret = wp::floor(x); }
WP_API void builtin_ceil_float32(float32 x, float* ret) { *ret = wp::ceil(x); }
WP_API void builtin_dot_vec2_vec2(vec2 x, vec2 y, float* ret) { *ret = wp::dot(x, y); }
WP_API void builtin_dot_vec3_vec3(vec3 x, vec3 y, float* ret) { *ret = wp::dot(x, y); }
WP_API void builtin_dot_vec4_vec4(vec4 x, vec4 y, float* ret) { *ret = wp::dot(x, y); }
WP_API void builtin_dot_quat_quat(quat x, quat y, float* ret) { *ret = wp::dot(x, y); }
WP_API void builtin_outer_vec2_vec2(vec2 x, vec2 y, mat22* ret) { *ret = wp::outer(x, y); }
WP_API void builtin_outer_vec3_vec3(vec3 x, vec3 y, mat33* ret) { *ret = wp::outer(x, y); }
WP_API void builtin_cross_vec3_vec3(vec3 x, vec3 y, vec3* ret) { *ret = wp::cross(x, y); }
WP_API void builtin_skew_vec3(vec3 x, mat33* ret) { *ret = wp::skew(x); }
WP_API void builtin_length_vec2(vec2 x, float* ret) { *ret = wp::length(x); }
WP_API void builtin_length_vec3(vec3 x, float* ret) { *ret = wp::length(x); }
WP_API void builtin_length_vec4(vec4 x, float* ret) { *ret = wp::length(x); }
WP_API void builtin_normalize_vec2(vec2 x, vec2* ret) { *ret = wp::normalize(x); }
WP_API void builtin_normalize_vec3(vec3 x, vec3* ret) { *ret = wp::normalize(x); }
WP_API void builtin_normalize_vec4(vec4 x, vec4* ret) { *ret = wp::normalize(x); }
WP_API void builtin_normalize_quat(quat x, quat* ret) { *ret = wp::normalize(x); }
WP_API void builtin_transpose_mat22(mat22 m, mat22* ret) { *ret = wp::transpose(m); }
WP_API void builtin_transpose_mat33(mat33 m, mat33* ret) { *ret = wp::transpose(m); }
WP_API void builtin_transpose_mat44(mat44 m, mat44* ret) { *ret = wp::transpose(m); }
WP_API void builtin_transpose_spatial_matrix(spatial_matrix m, spatial_matrix* ret) { *ret = wp::transpose(m); }
WP_API void builtin_inverse_mat22(mat22 m, mat22* ret) { *ret = wp::inverse(m); }
WP_API void builtin_inverse_mat33(mat33 m, mat33* ret) { *ret = wp::inverse(m); }
WP_API void builtin_inverse_mat44(mat44 m, mat44* ret) { *ret = wp::inverse(m); }
WP_API void builtin_determinant_mat22(mat22 m, float* ret) { *ret = wp::determinant(m); }
WP_API void builtin_determinant_mat33(mat33 m, float* ret) { *ret = wp::determinant(m); }
WP_API void builtin_determinant_mat44(mat44 m, float* ret) { *ret = wp::determinant(m); }
WP_API void builtin_diag_vec2(vec2 d, mat22* ret) { *ret = wp::diag(d); }
WP_API void builtin_diag_vec3(vec3 d, mat33* ret) { *ret = wp::diag(d); }
WP_API void builtin_diag_vec4(vec4 d, mat44* ret) { *ret = wp::diag(d); }
WP_API void builtin_cw_mul_vec2_vec2(vec2 x, vec2 y, vec2* ret) { *ret = wp::cw_mul(x, y); }
WP_API void builtin_cw_mul_vec3_vec3(vec3 x, vec3 y, vec3* ret) { *ret = wp::cw_mul(x, y); }
WP_API void builtin_cw_mul_vec4_vec4(vec4 x, vec4 y, vec4* ret) { *ret = wp::cw_mul(x, y); }
WP_API void builtin_cw_div_vec2_vec2(vec2 x, vec2 y, vec2* ret) { *ret = wp::cw_div(x, y); }
WP_API void builtin_cw_div_vec3_vec3(vec3 x, vec3 y, vec3* ret) { *ret = wp::cw_div(x, y); }
WP_API void builtin_cw_div_vec4_vec4(vec4 x, vec4 y, vec4* ret) { *ret = wp::cw_div(x, y); }
WP_API void builtin_svd3_mat33_mat33_vec3_mat33(mat33 A, mat33 U, vec3 sigma, mat33 V) { wp::svd3(A, U, sigma, V); }
WP_API void builtin_quat_identity(quat* ret) { *ret = wp::quat_identity(); }
WP_API void builtin_quat_from_axis_angle_vec3_float32(vec3 axis, float32 angle, quat* ret) { *ret = wp::quat_from_axis_angle(axis, angle); }
WP_API void builtin_quat_from_matrix_mat33(mat33 m, quat* ret) { *ret = wp::quat_from_matrix(m); }
WP_API void builtin_quat_rpy_float32_float32_float32(float32 roll, float32 pitch, float32 yaw, quat* ret) { *ret = wp::quat_rpy(roll, pitch, yaw); }
WP_API void builtin_quat_inverse_quat(quat q, quat* ret) { *ret = wp::quat_inverse(q); }
WP_API void builtin_quat_rotate_quat_vec3(quat q, vec3 p, vec3* ret) { *ret = wp::quat_rotate(q, p); }
WP_API void builtin_quat_rotate_inv_quat_vec3(quat q, vec3 p, vec3* ret) { *ret = wp::quat_rotate_inv(q, p); }
WP_API void builtin_quat_to_matrix_quat(quat q, mat33* ret) { *ret = wp::quat_to_matrix(q); }
WP_API void builtin_transform_identity(transform* ret) { *ret = wp::transform_identity(); }
WP_API void builtin_transform_get_translation_transform(transform t, vec3* ret) { *ret = wp::transform_get_translation(t); }
WP_API void builtin_transform_get_rotation_transform(transform t, quat* ret) { *ret = wp::transform_get_rotation(t); }
WP_API void builtin_transform_multiply_transform_transform(transform a, transform b, transform* ret) { *ret = wp::transform_multiply(a, b); }
WP_API void builtin_transform_point_transform_vec3(transform t, vec3 p, vec3* ret) { *ret = wp::transform_point(t, p); }
WP_API void builtin_transform_point_mat44_vec3(mat44 m, vec3 p, vec3* ret) { *ret = wp::transform_point(m, p); }
WP_API void builtin_transform_vector_transform_vec3(transform t, vec3 v, vec3* ret) { *ret = wp::transform_vector(t, v); }
WP_API void builtin_transform_vector_mat44_vec3(mat44 m, vec3 v, vec3* ret) { *ret = wp::transform_vector(m, v); }
WP_API void builtin_transform_inverse_transform(transform t, transform* ret) { *ret = wp::transform_inverse(t); }
WP_API void builtin_spatial_dot_spatial_vector_spatial_vector(spatial_vector a, spatial_vector b, float* ret) { *ret = wp::spatial_dot(a, b); }
WP_API void builtin_spatial_cross_spatial_vector_spatial_vector(spatial_vector a, spatial_vector b, spatial_vector* ret) { *ret = wp::spatial_cross(a, b); }
WP_API void builtin_spatial_cross_dual_spatial_vector_spatial_vector(spatial_vector a, spatial_vector b, spatial_vector* ret) { *ret = wp::spatial_cross_dual(a, b); }
WP_API void builtin_spatial_top_spatial_vector(spatial_vector a, vec3* ret) { *ret = wp::spatial_top(a); }
WP_API void builtin_spatial_bottom_spatial_vector(spatial_vector a, vec3* ret) { *ret = wp::spatial_bottom(a); }
WP_API void builtin_mesh_query_point_uint64_vec3_float32_float32_int32_float32_float32(uint64 id, vec3 point, float32 max_dist, float32 inside, int32 face, float32 bary_u, float32 bary_v, bool* ret) { *ret = wp::mesh_query_point(id, point, max_dist, inside, face, bary_u, bary_v); }
WP_API void builtin_mesh_query_ray_uint64_vec3_vec3_float32_float32_float32_float32_float32_vec3_int32(uint64 id, vec3 start, vec3 dir, float32 max_t, float32 t, float32 bary_u, float32 bary_v, float32 sign, vec3 normal, int32 face, bool* ret) { *ret = wp::mesh_query_ray(id, start, dir, max_t, t, bary_u, bary_v, sign, normal, face); }
WP_API void builtin_mesh_query_aabb_uint64_vec3_vec3(uint64 id, vec3 lower, vec3 upper, mesh_query_aabb_t* ret) { *ret = wp::mesh_query_aabb(id, lower, upper); }
WP_API void builtin_mesh_query_aabb_next_mesh_query_aabb_t_int32(mesh_query_aabb_t query, int32 index, bool* ret) { *ret = wp::mesh_query_aabb_next(query, index); }
WP_API void builtin_mesh_eval_position_uint64_int32_float32_float32(uint64 id, int32 face, float32 bary_u, float32 bary_v, vec3* ret) { *ret = wp::mesh_eval_position(id, face, bary_u, bary_v); }
WP_API void builtin_mesh_eval_velocity_uint64_int32_float32_float32(uint64 id, int32 face, float32 bary_u, float32 bary_v, vec3* ret) { *ret = wp::mesh_eval_velocity(id, face, bary_u, bary_v); }
WP_API void builtin_hash_grid_query_uint64_vec3_float32(uint64 id, vec3 point, float32 max_dist, hash_grid_query_t* ret) { *ret = wp::hash_grid_query(id, point, max_dist); }
WP_API void builtin_hash_grid_query_next_hash_grid_query_t_int32(hash_grid_query_t query, int32 index, bool* ret) { *ret = wp::hash_grid_query_next(query, index); }
WP_API void builtin_hash_grid_point_id_uint64_int32(uint64 id, int32 index, int* ret) { *ret = wp::hash_grid_point_id(id, index); }
WP_API void builtin_intersect_tri_tri_vec3_vec3_vec3_vec3_vec3_vec3(vec3 v0, vec3 v1, vec3 v2, vec3 u0, vec3 u1, vec3 u2, int* ret) { *ret = wp::intersect_tri_tri(v0, v1, v2, u0, u1, u2); }
WP_API void builtin_mesh_eval_face_normal_uint64_int32(uint64 id, int32 face, vec3* ret) { *ret = wp::mesh_eval_face_normal(id, face); }
WP_API void builtin_mesh_get_point_uint64_int32(uint64 id, int32 index, vec3* ret) { *ret = wp::mesh_get_point(id, index); }
WP_API void builtin_mesh_get_velocity_uint64_int32(uint64 id, int32 index, vec3* ret) { *ret = wp::mesh_get_velocity(id, index); }
WP_API void builtin_mesh_get_index_uint64_int32(uint64 id, int32 index, int* ret) { *ret = wp::mesh_get_index(id, index); }
WP_API void builtin_iter_next_range_t(range_t range, int* ret) { *ret = wp::iter_next(range); }
WP_API void builtin_iter_next_hash_grid_query_t(hash_grid_query_t query, int* ret) { *ret = wp::iter_next(query); }
WP_API void builtin_iter_next_mesh_query_aabb_t(mesh_query_aabb_t query, int* ret) { *ret = wp::iter_next(query); }
WP_API void builtin_volume_sample_f_uint64_vec3_int32(uint64 id, vec3 uvw, int32 sampling_mode, float* ret) { *ret = wp::volume_sample_f(id, uvw, sampling_mode); }
WP_API void builtin_volume_lookup_f_uint64_int32_int32_int32(uint64 id, int32 i, int32 j, int32 k, float* ret) { *ret = wp::volume_lookup_f(id, i, j, k); }
WP_API void builtin_volume_sample_v_uint64_vec3_int32(uint64 id, vec3 uvw, int32 sampling_mode, vec3* ret) { *ret = wp::volume_sample_v(id, uvw, sampling_mode); }
WP_API void builtin_volume_lookup_v_uint64_int32_int32_int32(uint64 id, int32 i, int32 j, int32 k, vec3* ret) { *ret = wp::volume_lookup_v(id, i, j, k); }
WP_API void builtin_volume_sample_i_uint64_vec3(uint64 id, vec3 uvw, int* ret) { *ret = wp::volume_sample_i(id, uvw); }
WP_API void builtin_volume_lookup_i_uint64_int32_int32_int32(uint64 id, int32 i, int32 j, int32 k, int* ret) { *ret = wp::volume_lookup_i(id, i, j, k); }
WP_API void builtin_volume_index_to_world_uint64_vec3(uint64 id, vec3 uvw, vec3* ret) { *ret = wp::volume_index_to_world(id, uvw); }
WP_API void builtin_volume_world_to_index_uint64_vec3(uint64 id, vec3 xyz, vec3* ret) { *ret = wp::volume_world_to_index(id, xyz); }
WP_API void builtin_volume_index_to_world_dir_uint64_vec3(uint64 id, vec3 uvw, vec3* ret) { *ret = wp::volume_index_to_world_dir(id, uvw); }
WP_API void builtin_volume_world_to_index_dir_uint64_vec3(uint64 id, vec3 xyz, vec3* ret) { *ret = wp::volume_world_to_index_dir(id, xyz); }
WP_API void builtin_rand_init_int32(int32 seed, uint32* ret) { *ret = wp::rand_init(seed); }
WP_API void builtin_rand_init_int32_int32(int32 seed, int32 offset, uint32* ret) { *ret = wp::rand_init(seed, offset); }
WP_API void builtin_randi_uint32(uint32 state, int* ret) { *ret = wp::randi(state); }
WP_API void builtin_randi_uint32_int32_int32(uint32 state, int32 min, int32 max, int* ret) { *ret = wp::randi(state, min, max); }
WP_API void builtin_randf_uint32(uint32 state, float* ret) { *ret = wp::randf(state); }
WP_API void builtin_randf_uint32_float32_float32(uint32 state, float32 min, float32 max, float* ret) { *ret = wp::randf(state, min, max); }
WP_API void builtin_randn_uint32(uint32 state, float* ret) { *ret = wp::randn(state); }
WP_API void builtin_noise_uint32_float32(uint32 state, float32 x, float* ret) { *ret = wp::noise(state, x); }
WP_API void builtin_noise_uint32_vec2(uint32 state, vec2 xy, float* ret) { *ret = wp::noise(state, xy); }
WP_API void builtin_noise_uint32_vec3(uint32 state, vec3 xyz, float* ret) { *ret = wp::noise(state, xyz); }
WP_API void builtin_noise_uint32_vec4(uint32 state, vec4 xyzt, float* ret) { *ret = wp::noise(state, xyzt); }
WP_API void builtin_pnoise_uint32_float32_int32(uint32 state, float32 x, int32 px, float* ret) { *ret = wp::pnoise(state, x, px); }
WP_API void builtin_pnoise_uint32_vec2_int32_int32(uint32 state, vec2 xy, int32 px, int32 py, float* ret) { *ret = wp::pnoise(state, xy, px, py); }
WP_API void builtin_pnoise_uint32_vec3_int32_int32_int32(uint32 state, vec3 xyz, int32 px, int32 py, int32 pz, float* ret) { *ret = wp::pnoise(state, xyz, px, py, pz); }
WP_API void builtin_pnoise_uint32_vec4_int32_int32_int32_int32(uint32 state, vec4 xyzt, int32 px, int32 py, int32 pz, int32 pt, float* ret) { *ret = wp::pnoise(state, xyzt, px, py, pz, pt); }
WP_API void builtin_curlnoise_uint32_vec2(uint32 state, vec2 xy, vec2* ret) { *ret = wp::curlnoise(state, xy); }
WP_API void builtin_curlnoise_uint32_vec3(uint32 state, vec3 xyz, vec3* ret) { *ret = wp::curlnoise(state, xyz); }
WP_API void builtin_curlnoise_uint32_vec4(uint32 state, vec4 xyzt, vec3* ret) { *ret = wp::curlnoise(state, xyzt); }
WP_API void builtin_tid(int* ret) { *ret = wp::tid(); }
WP_API void builtin_index_vec2_int32(vec2 a, int32 i, float* ret) { *ret = wp::index(a, i); }
WP_API void builtin_index_vec3_int32(vec3 a, int32 i, float* ret) { *ret = wp::index(a, i); }
WP_API void builtin_index_vec4_int32(vec4 a, int32 i, float* ret) { *ret = wp::index(a, i); }
WP_API void builtin_index_mat22_int32(mat22 a, int32 i, vec2* ret) { *ret = wp::index(a, i); }
WP_API void builtin_index_mat22_int32_int32(mat22 a, int32 i, int32 j, float* ret) { *ret = wp::index(a, i, j); }
WP_API void builtin_index_mat33_int32(mat33 a, int32 i, vec3* ret) { *ret = wp::index(a, i); }
WP_API void builtin_index_mat33_int32_int32(mat33 a, int32 i, int32 j, float* ret) { *ret = wp::index(a, i, j); }
WP_API void builtin_index_mat44_int32(mat44 a, int32 i, vec4* ret) { *ret = wp::index(a, i); }
WP_API void builtin_index_mat44_int32_int32(mat44 a, int32 i, int32 j, float* ret) { *ret = wp::index(a, i, j); }
WP_API void builtin_expect_eq_int8_int8(int8 arg1, int8 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_uint8_uint8(uint8 arg1, uint8 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_int16_int16(int16 arg1, int16 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_uint16_uint16(uint16 arg1, uint16 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_int32_int32(int32 arg1, int32 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_uint32_uint32(uint32 arg1, uint32 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_int64_int64(int64 arg1, int64 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_uint64_uint64(uint64 arg1, uint64 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_float16_float16(float16 arg1, float16 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_float32_float32(float32 arg1, float32 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_float64_float64(float64 arg1, float64 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_vec2_vec2(vec2 arg1, vec2 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_vec3_vec3(vec3 arg1, vec3 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_vec4_vec4(vec4 arg1, vec4 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_mat22_mat22(mat22 arg1, mat22 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_mat33_mat33(mat33 arg1, mat33 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_mat44_mat44(mat44 arg1, mat44 arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_quat_quat(quat arg1, quat arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_transform_transform(transform arg1, transform arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_spatial_vector_spatial_vector(spatial_vector arg1, spatial_vector arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_expect_eq_spatial_matrix_spatial_matrix(spatial_matrix arg1, spatial_matrix arg2) { wp::expect_eq(arg1, arg2); }
WP_API void builtin_lerp_float16_float16_float32(float16 a, float16 b, float32 t, float16* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_float32_float32_float32(float32 a, float32 b, float32 t, float32* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_float64_float64_float32(float64 a, float64 b, float32 t, float64* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_vec2_vec2_float32(vec2 a, vec2 b, float32 t, vec2* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_vec3_vec3_float32(vec3 a, vec3 b, float32 t, vec3* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_vec4_vec4_float32(vec4 a, vec4 b, float32 t, vec4* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_mat22_mat22_float32(mat22 a, mat22 b, float32 t, mat22* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_mat33_mat33_float32(mat33 a, mat33 b, float32 t, mat33* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_mat44_mat44_float32(mat44 a, mat44 b, float32 t, mat44* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_quat_quat_float32(quat a, quat b, float32 t, quat* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_transform_transform_float32(transform a, transform b, float32 t, transform* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_spatial_vector_spatial_vector_float32(spatial_vector a, spatial_vector b, float32 t, spatial_vector* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_spatial_matrix_spatial_matrix_float32(spatial_matrix a, spatial_matrix b, float32 t, spatial_matrix* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_expect_near_float32_float32_float32(float32 arg1, float32 arg2, float32 tolerance) { wp::expect_near(arg1, arg2, tolerance); }
WP_API void builtin_expect_near_vec3_vec3_float32(vec3 arg1, vec3 arg2, float32 tolerance) { wp::expect_near(arg1, arg2, tolerance); }
WP_API void builtin_add_int32_int32(int32 x, int32 y, int* ret) { *ret = wp::add(x, y); }
WP_API void builtin_add_float32_float32(float32 x, float32 y, float* ret) { *ret = wp::add(x, y); }
WP_API void builtin_add_vec2_vec2(vec2 x, vec2 y, vec2* ret) { *ret = wp::add(x, y); }
WP_API void builtin_add_vec3_vec3(vec3 x, vec3 y, vec3* ret) { *ret = wp::add(x, y); }
WP_API void builtin_add_vec4_vec4(vec4 x, vec4 y, vec4* ret) { *ret = wp::add(x, y); }
WP_API void builtin_add_quat_quat(quat x, quat y, quat* ret) { *ret = wp::add(x, y); }
WP_API void builtin_add_mat22_mat22(mat22 x, mat22 y, mat22* ret) { *ret = wp::add(x, y); }
WP_API void builtin_add_mat33_mat33(mat33 x, mat33 y, mat33* ret) { *ret = wp::add(x, y); }
WP_API void builtin_add_mat44_mat44(mat44 x, mat44 y, mat44* ret) { *ret = wp::add(x, y); }
WP_API void builtin_add_spatial_vector_spatial_vector(spatial_vector x, spatial_vector y, spatial_vector* ret) { *ret = wp::add(x, y); }
WP_API void builtin_add_spatial_matrix_spatial_matrix(spatial_matrix x, spatial_matrix y, spatial_matrix* ret) { *ret = wp::add(x, y); }
WP_API void builtin_sub_int32_int32(int32 x, int32 y, int* ret) { *ret = wp::sub(x, y); }
WP_API void builtin_sub_float32_float32(float32 x, float32 y, float* ret) { *ret = wp::sub(x, y); }
WP_API void builtin_sub_vec2_vec2(vec2 x, vec2 y, vec2* ret) { *ret = wp::sub(x, y); }
WP_API void builtin_sub_vec3_vec3(vec3 x, vec3 y, vec3* ret) { *ret = wp::sub(x, y); }
WP_API void builtin_sub_vec4_vec4(vec4 x, vec4 y, vec4* ret) { *ret = wp::sub(x, y); }
WP_API void builtin_sub_mat22_mat22(mat22 x, mat22 y, mat22* ret) { *ret = wp::sub(x, y); }
WP_API void builtin_sub_mat33_mat33(mat33 x, mat33 y, mat33* ret) { *ret = wp::sub(x, y); }
WP_API void builtin_sub_mat44_mat44(mat44 x, mat44 y, mat44* ret) { *ret = wp::sub(x, y); }
WP_API void builtin_sub_spatial_vector_spatial_vector(spatial_vector x, spatial_vector y, spatial_vector* ret) { *ret = wp::sub(x, y); }
WP_API void builtin_sub_spatial_matrix_spatial_matrix(spatial_matrix x, spatial_matrix y, spatial_matrix* ret) { *ret = wp::sub(x, y); }
WP_API void builtin_mul_int32_int32(int32 x, int32 y, int* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_float32_float32(float32 x, float32 y, float* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_float32_vec2(float32 x, vec2 y, vec2* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_float32_vec3(float32 x, vec3 y, vec3* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_float32_vec4(float32 x, vec4 y, vec4* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_float32_quat(float32 x, quat y, quat* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_float32_mat22(float32 x, mat22 y, mat22* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_float32_mat33(float32 x, mat33 y, mat33* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_float32_mat44(float32 x, mat44 y, mat44* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_vec2_float32(vec2 x, float32 y, vec2* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_vec3_float32(vec3 x, float32 y, vec3* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_vec4_float32(vec4 x, float32 y, vec4* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_quat_float32(quat x, float32 y, quat* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_quat_quat(quat x, quat y, quat* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_mat22_float32(mat22 x, float32 y, mat22* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_mat22_vec2(mat22 x, vec2 y, vec2* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_mat22_mat22(mat22 x, mat22 y, mat22* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_mat33_float32(mat33 x, float32 y, mat33* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_mat33_vec3(mat33 x, vec3 y, vec3* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_mat33_mat33(mat33 x, mat33 y, mat33* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_mat44_float32(mat44 x, float32 y, mat44* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_mat44_vec4(mat44 x, vec4 y, vec4* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_mat44_mat44(mat44 x, mat44 y, mat44* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_spatial_vector_float32(spatial_vector x, float32 y, spatial_vector* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_spatial_matrix_spatial_matrix(spatial_matrix x, spatial_matrix y, spatial_matrix* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_spatial_matrix_spatial_vector(spatial_matrix x, spatial_vector y, spatial_vector* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mul_transform_transform(transform x, transform y, transform* ret) { *ret = wp::mul(x, y); }
WP_API void builtin_mod_int32_int32(int32 x, int32 y, int* ret) { *ret = wp::mod(x, y); }
WP_API void builtin_mod_float32_float32(float32 x, float32 y, float* ret) { *ret = wp::mod(x, y); }
WP_API void builtin_div_int32_int32(int32 x, int32 y, int* ret) { *ret = wp::div(x, y); }
WP_API void builtin_div_float32_float32(float32 x, float32 y, float* ret) { *ret = wp::div(x, y); }
WP_API void builtin_div_vec2_float32(vec2 x, float32 y, vec2* ret) { *ret = wp::div(x, y); }
WP_API void builtin_div_vec3_float32(vec3 x, float32 y, vec3* ret) { *ret = wp::div(x, y); }
WP_API void builtin_div_vec4_float32(vec4 x, float32 y, vec4* ret) { *ret = wp::div(x, y); }
WP_API void builtin_floordiv_int32_int32(int32 x, int32 y, int* ret) { *ret = wp::floordiv(x, y); }
WP_API void builtin_floordiv_float32_float32(float32 x, float32 y, float* ret) { *ret = wp::floordiv(x, y); }
WP_API void builtin_neg_int32(int32 x, int* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_float32(float32 x, float* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec2(vec2 x, vec2* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec3(vec3 x, vec3* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec4(vec4 x, vec4* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_quat(quat x, quat* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_mat33(mat33 x, mat33* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_mat44(mat44 x, mat44* ret) { *ret = wp::neg(x); }
WP_API void builtin_unot_bool(bool b, bool* ret) { *ret = wp::unot(b); }
