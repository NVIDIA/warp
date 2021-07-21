Warp Function Reference
=======================

Operators
---------------
.. function:: add(x: int, y: int)

   :return: int

.. function:: add(x: float, y: float)

   :return: float

.. function:: add(x: vec3, y: vec3)

   :return: vec3

.. function:: add(x: vec4, y: vec4)

   :return: vec4

.. function:: add(x: quat, y: quat)

   :return: quat

.. function:: add(x: mat22, y: mat22)

   :return: mat22

.. function:: add(x: mat33, y: mat33)

   :return: mat33

.. function:: add(x: mat44, y: mat44)

   :return: mat44

.. function:: add(x: spatial_vector, y: spatial_vector)

   :return: spatial_vector

.. function:: add(x: spatial_matrix, y: spatial_matrix)

   :return: spatial_matrix

.. function:: sub(x: int, y: int)

   :return: int

.. function:: sub(x: float, y: float)

   :return: float

.. function:: sub(x: vec3, y: vec3)

   :return: vec3

.. function:: sub(x: vec4, y: vec4)

   :return: vec4

.. function:: sub(x: mat22, y: mat22)

   :return: mat22

.. function:: sub(x: mat33, y: mat33)

   :return: mat33

.. function:: sub(x: mat44, y: mat44)

   :return: mat44

.. function:: sub(x: spatial_vector, y: spatial_vector)

   :return: spatial_vector

.. function:: sub(x: spatial_matrix, y: spatial_matrix)

   :return: spatial_matrix

.. function:: mul(x: int, y: int)

   :return: int

.. function:: mul(x: float, y: float)

   :return: float

.. function:: mul(x: float, y: vec3)

   :return: vec3

.. function:: mul(x: float, y: vec4)

   :return: vec4

.. function:: mul(x: vec3, y: float)

   :return: vec3

.. function:: mul(x: vec4, y: float)

   :return: vec4

.. function:: mul(x: quat, y: float)

   :return: quat

.. function:: mul(x: quat, y: quat)

   :return: quat

.. function:: mul(x: mat22, y: float)

   :return: mat22

.. function:: mul(x: mat33, y: float)

   :return: mat33

.. function:: mul(x: mat33, y: vec3)

   :return: vec3

.. function:: mul(x: mat33, y: mat33)

   :return: mat33

.. function:: mul(x: mat44, y: float)

   :return: mat44

.. function:: mul(x: mat44, y: vec4)

   :return: vec4

.. function:: mul(x: mat44, y: mat44)

   :return: mat44

.. function:: mul(x: spatial_vector, y: float)

   :return: spatial_vector

.. function:: mul(x: spatial_matrix, y: spatial_matrix)

   :return: spatial_matrix

.. function:: mul(x: spatial_matrix, y: spatial_vector)

   :return: spatial_vector

.. function:: mod(x: int, y: int)

   :return: int

.. function:: mod(x: float, y: float)

   :return: float

.. function:: div(x: int, y: int)

   :return: int

.. function:: div(x: float, y: float)

   :return: float

.. function:: div(x: vec3, y: float)

   :return: vec3

.. function:: neg(x: int)

   :return: int

.. function:: neg(x: float)

   :return: float

.. function:: neg(x: vec3)

   :return: vec3

.. function:: neg(x: vec4)

   :return: vec4

.. function:: neg(x: quat)

   :return: quat

.. function:: neg(x: mat33)

   :return: mat33

.. function:: neg(x: mat44)

   :return: mat44



Scalar Math
---------------
.. function:: min(x: int, y: int)

   :return: int

.. function:: min(x: float, y: float)

   :return: float

.. function:: max(x: int, y: int)

   :return: int

.. function:: max(x: float, y: float)

   :return: float

.. function:: clamp(x: float, a: float, b: float)

   :return: float

.. function:: clamp(x: int, a: int, b: int)

   :return: int

.. function:: step(x: float)

   :return: float

.. function:: nonzero(x: float)

   :return: float

.. function:: sign(x: float)

   :return: float

.. function:: abs(x: float)

   :return: float

.. function:: sin(x: float)

   :return: float

.. function:: cos(x: float)

   :return: float

.. function:: acos(x: float)

   :return: float

.. function:: sqrt(x: float)

   :return: float

.. function:: int(x: int)

   :return: int

.. function:: int(x: float)

   :return: int

.. function:: float(x: int)

   :return: float

.. function:: float(x: float)

   :return: float



Vector Math
---------------
.. function:: dot(x: vec3, y: vec3)

   :return: float

.. function:: dot(x: vec4, y: vec4)

   :return: float

.. function:: dot(x: quat, y: quat)

   :return: float

.. function:: cross(x: vec3, y: vec3)

   :return: vec3

.. function:: skew(x: vec3)

   :return: mat33

.. function:: length(x: vec3)

   :return: float

.. function:: normalize(x: vec3)

   :return: vec3

.. function:: normalize(x: vec4)

   :return: vec4

.. function:: normalize(x: quat)

   :return: quat

.. function:: rotate(q: quat, p: vec3)

   :return: vec3

.. function:: rotate_inv(q: quat, p: vec3)

   :return: vec3

.. function:: determinant(m: mat22)

   :return: float

.. function:: determinant(m: mat33)

   :return: float

.. function:: transpose(m: mat22)

   :return: mat22

.. function:: transpose(m: mat33)

   :return: mat33

.. function:: transpose(m: mat44)

   :return: mat44

.. function:: transpose(m: spatial_matrix)

   :return: spatial_matrix

.. function:: vec3()

   :return: vec3

.. function:: vec3(x: float, y: float, z: float)

   :return: vec3

.. function:: vec3(s: float)

   :return: vec3

.. function:: vec4()

   :return: vec4

.. function:: vec4(x: float, y: float, z: float, w: float)

   :return: vec4

.. function:: vec4(s: float)

   :return: vec4

.. function:: mat22(m00: float, m01: float, m10: float, m11: float)

   :return: mat22

.. function:: mat33(c0: vec3, c1: vec3, c2: vec3)

   :return: mat33

.. function:: mat44(c0: vec4, c1: vec4, c2: vec4, c3: vec4)

   :return: mat44

.. function:: transform_point(m: mat44, p: vec3)

   :return: vec3

.. function:: transform_vector(m: mat44, p: vec3)

   :return: vec3



Quaternion Math
---------------
.. function:: quat()

   :return: quat

.. function:: quat(x: float, y: float, z: float, w: float)

   :return: quat

.. function:: quat(i: vec3, r: float)

   :return: quat

.. function:: quat_identity()

   :return: quat

.. function:: quat_from_axis_angle(axis: vec3, angle: float)

   :return: quat

.. function:: quat_inverse(q: quat)

   :return: quat



Spatial Math
---------------
.. function:: spatial_vector()

   :return: spatial_vector

.. function:: spatial_vector(a: float, b: float, c: float, d: float, e: float, f: float)

   :return: spatial_vector

.. function:: spatial_vector(w: vec3, v: vec3)

   :return: spatial_vector

.. function:: spatial_vector(s: float)

   :return: spatial_vector

.. function:: spatial_transform(p: vec3, q: quat)

   :return: spatial_transform

.. function:: spatial_transform_identity()

   :return: spatial_transform

.. function:: spatial_transform_get_translation(t: spatial_transform)

   :return: vec3

.. function:: spatial_transform_get_rotation(t: spatial_transform)

   :return: quat

.. function:: spatial_transform_multiply(a: spatial_transform, b: spatial_transform)

   :return: spatial_transform

.. function:: spatial_adjoint(r: mat33, s: mat33)

   :return: spatial_matrix

.. function:: spatial_dot(a: spatial_vector, b: spatial_vector)

   :return: float

.. function:: spatial_cross(a: spatial_vector, b: spatial_vector)

   :return: spatial_vector

.. function:: spatial_cross_dual(a: spatial_vector, b: spatial_vector)

   :return: spatial_vector

.. function:: spatial_transform_point(t: spatial_transform, p: vec3)

   :return: vec3

.. function:: spatial_transform_vector(t: spatial_transform, p: vec3)

   :return: vec3

.. function:: spatial_top(a: spatial_vector)

   :return: vec3

.. function:: spatial_bottom(a: spatial_vector)

   :return: vec3

.. function:: spatial_jacobian(S: array(spatial_vector), joint_parents: array(int32), joint_qd_start: array(int32), joint_start: int, joint_count: int, J_start: int, J_out: array(float32))

   :return: Input dependent

.. function:: spatial_mass(I_s: array(spatial_matrix), joint_start: int, joint_count: int, M_start: int, M: array(float32))

   :return: Input dependent



Linear Algebra
---------------
.. function:: dense_gemm(m: int, n: int, p: int, t1: int, t2: int, A: array(float32), B: array(float32), C: array(float32))

   :return: Input dependent

.. function:: dense_gemm_batched(m: array(int32), n: array(int32), p: array(int32), t1: int, t2: int, A_start: array(int32), B_start: array(int32), C_start: array(int32), A: array(float32), B: array(float32), C: array(float32))

   :return: Input dependent

.. function:: dense_chol(n: int, A: array(float32), regularization: float, L: array(float32))

   :return: Input dependent

.. function:: dense_chol_batched(A_start: array(int32), A_dim: array(int32), A: array(float32), regularization: float, L: array(float32))

   :return: Input dependent

.. function:: dense_subs(n: int, L: array(float32), b: array(float32), x: array(float32))

   :return: Input dependent

.. function:: dense_solve(n: int, A: array(float32), L: array(float32), b: array(float32), x: array(float32))

   :return: Input dependent

.. function:: dense_solve_batched(b_start: array(int32), A_start: array(int32), A_dim: array(int32), A: array(float32), L: array(float32), b: array(float32), x: array(float32))

   :return: Input dependent



Geometry
---------------
.. function:: mesh_query_point(id: uint64, point: vec3, max_dist: float, inside: float, face: int, bary_u: float, bary_v: float)

   :return: bool

.. function:: mesh_query_ray(id: uint64, start: vec3, dir: vec3, max_t: float, t: float, bary_u: float, bary_v: float, sign: float, normal: vec3, face: int)

   :return: bool

.. function:: mesh_eval_position(id: uint64, face: int, bary_u: float, bary_v: float)

   :return: vec3

.. function:: mesh_eval_velocity(id: uint64, face: int, bary_u: float, bary_v: float)

   :return: vec3



Utility
---------------
.. function:: tid()

   :return: int


Other
---------------

.. function:: select()

   :return: Input dependent

.. function:: copy()

   :return: Input dependent

.. function:: load()

   :return: Input dependent

.. function:: store()

   :return: Input dependent

.. function:: atomic_add()

   :return: Input dependent

.. function:: atomic_sub()

   :return: Input dependent

.. function:: index()

   :return: float

.. function:: print()

   :return: Input dependent

.. function:: expect_eq()

   :return: Input dependent