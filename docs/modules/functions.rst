Language Reference
==================


Operators
---------------
.. function:: add(x: int, y: int) -> int


.. function:: add(x: float, y: float) -> float


.. function:: add(x: vec2, y: vec2) -> vec2


.. function:: add(x: vec3, y: vec3) -> vec3


.. function:: add(x: vec4, y: vec4) -> vec4


.. function:: add(x: quat, y: quat) -> quat


.. function:: add(x: mat22, y: mat22) -> mat22


.. function:: add(x: mat33, y: mat33) -> mat33


.. function:: add(x: mat44, y: mat44) -> mat44


.. function:: add(x: spatial_vector, y: spatial_vector) -> spatial_vector


.. function:: add(x: spatial_matrix, y: spatial_matrix) -> spatial_matrix


.. function:: sub(x: int, y: int) -> int


.. function:: sub(x: float, y: float) -> float


.. function:: sub(x: vec2, y: vec2) -> vec2


.. function:: sub(x: vec3, y: vec3) -> vec3


.. function:: sub(x: vec4, y: vec4) -> vec4


.. function:: sub(x: mat22, y: mat22) -> mat22


.. function:: sub(x: mat33, y: mat33) -> mat33


.. function:: sub(x: mat44, y: mat44) -> mat44


.. function:: sub(x: spatial_vector, y: spatial_vector) -> spatial_vector


.. function:: sub(x: spatial_matrix, y: spatial_matrix) -> spatial_matrix


.. function:: mul(x: int, y: int) -> int


.. function:: mul(x: float, y: float) -> float


.. function:: mul(x: float, y: vec2) -> vec2


.. function:: mul(x: float, y: vec3) -> vec3


.. function:: mul(x: float, y: vec4) -> vec4


.. function:: mul(x: vec2, y: float) -> vec2


.. function:: mul(x: vec3, y: float) -> vec3


.. function:: mul(x: vec4, y: float) -> vec4


.. function:: mul(x: quat, y: float) -> quat


.. function:: mul(x: quat, y: quat) -> quat


.. function:: mul(x: mat22, y: float) -> mat22


.. function:: mul(x: mat22, y: vec2) -> vec2


.. function:: mul(x: mat33, y: float) -> mat33


.. function:: mul(x: mat33, y: vec3) -> vec3


.. function:: mul(x: mat33, y: mat33) -> mat33


.. function:: mul(x: mat44, y: float) -> mat44


.. function:: mul(x: mat44, y: vec4) -> vec4


.. function:: mul(x: mat44, y: mat44) -> mat44


.. function:: mul(x: spatial_vector, y: float) -> spatial_vector


.. function:: mul(x: spatial_matrix, y: spatial_matrix) -> spatial_matrix


.. function:: mul(x: spatial_matrix, y: spatial_vector) -> spatial_vector


.. function:: mul(x: transform, y: transform) -> transform


.. function:: mod(x: int, y: int) -> int


.. function:: mod(x: float, y: float) -> float


.. function:: div(x: int, y: int) -> int


.. function:: div(x: float, y: float) -> float


.. function:: div(x: vec2, y: float) -> vec2


.. function:: div(x: vec3, y: float) -> vec3


.. function:: neg(x: int) -> int


.. function:: neg(x: float) -> float


.. function:: neg(x: vec2) -> vec2


.. function:: neg(x: vec3) -> vec3


.. function:: neg(x: vec4) -> vec4


.. function:: neg(x: quat) -> quat


.. function:: neg(x: mat33) -> mat33


.. function:: neg(x: mat44) -> mat44


.. function:: unot(b: bool) -> bool




Scalar Math
---------------
.. function:: min(x: int, y: int) -> int


.. function:: min(x: float, y: float) -> float


.. function:: max(x: int, y: int) -> int


.. function:: max(x: float, y: float) -> float


.. function:: clamp(x: float, a: float, b: float) -> float


.. function:: clamp(x: int, a: int, b: int) -> int


.. function:: step(x: float) -> float


.. function:: nonzero(x: float) -> float


.. function:: sign(x: float) -> float


.. function:: abs(x: float) -> float


.. function:: sin(x: float) -> float


.. function:: cos(x: float) -> float


.. function:: acos(x: float) -> float


.. function:: asin(x: float) -> float


.. function:: sqrt(x: float) -> float


.. function:: tan(x: float) -> float


.. function:: atan(x: float) -> float


.. function:: atan2(y: float, x: float) -> float


.. function:: log(x: float) -> float


.. function:: log(x: vec3) -> vec3


.. function:: exp(x: float) -> float


.. function:: exp(x: vec3) -> vec3


.. function:: pow(x: float, y: float) -> float


.. function:: pow(x: vec3, y: float) -> vec3


.. function:: round(x: float) -> float

   Calculate the nearest integer value, rounding halfway cases away from zero.
   This is the most intuitive form of rounding in the colloquial sense, but can be slower than other options like ``warp.rint()``.
   Differs from ``numpy.round()``, which behaves the same way as ``numpy.rint()``.


.. function:: rint(x: float) -> float

   Calculate the nearest integer value, rounding halfway cases to nearest even integer.
   It is generally faster than ``warp.round()``.
   Equivalent to ``numpy.rint()``.


.. function:: trunc(x: float) -> float

   Calculate the nearest integer that is closer to zero than x.
   In other words, it discards the fractional part of x.
   It is similar to casting ``float(int(x))``, but preserves the negative sign when x is in the range [-0.0, -1.0).
   Equivalent to ``numpy.trunc()`` and ``numpy.fix()``.


.. function:: floor(x: float) -> float

   Calculate the largest integer that is less than or equal to x.


.. function:: ceil(x: float) -> float

   Calculate the smallest integer that is greater than or equal to x.


.. function:: rand_init(seed: int) -> uint32

   Initialize a new random number generator given a user-defined seed. Returns a 32-bit integer representing the RNG state.


.. function:: rand_init(seed: int, offset: int) -> uint32

   Initialize a new random number generator given a user-defined seed and an offset. 
   This alternative constructor can be useful in parallel programs, where a kernel as a whole should share a seed,
   but each thread should generate uncorrelated values. In this case usage should be ``r = rand_init(seed, tid)``


.. function:: randi(state: uint32) -> int

   Return a random integer between [0, 2^32)


.. function:: randi(state: uint32, min: int, max: int) -> int

   Return a random integer between [min, max)


.. function:: randf(state: uint32) -> float

   Return a random float between [0.0, 1.0)


.. function:: randf(state: uint32, min: float, max: float) -> float

   Return a random float between [min, max)


.. function:: noise(seed: uint32, x: float) -> float

   Non-periodic Perlin-style noise in 1d.


.. function:: noise(seed: uint32, xy: vec2) -> float

   Non-periodic Perlin-style noise in 2d.


.. function:: noise(seed: uint32, xyz: vec3) -> float

   Non-periodic Perlin-style noise in 3d.


.. function:: noise(seed: uint32, xyzt: vec4) -> float

   Non-periodic Perlin-style noise in 4d.


.. function:: pnoise(seed: uint32, x: float, px: int) -> float

   Periodic Perlin-style noise in 1d.


.. function:: pnoise(seed: uint32, xy: vec2, px: int, py: int) -> float

   Periodic Perlin-style noise in 2d.


.. function:: pnoise(seed: uint32, xyz: vec3, px: int, py: int, pz: int) -> float

   Periodic Perlin-style noise in 3d.


.. function:: pnoise(seed: uint32, xyzt: vec4, px: int, py: int, pz: int, pt: int) -> float

   Periodic Perlin-style noise in 4d.


.. function:: curlnoise(seed: uint32, xy: vec2) -> vec2

   Divergence-free vector field based on the gradient of a Perlin noise function.


.. function:: curlnoise(seed: uint32, xyz: vec3) -> vec3

   Divergence-free vector field based on the curl of three Perlin noise functions.


.. function:: curlnoise(seed: uint32, xyzt: vec4) -> vec3

   Divergence-free vector field based on the curl of three Perlin noise functions.


.. function:: int(x: int) -> int


.. function:: int(x: float) -> int


.. function:: float(x: int) -> float


.. function:: float(x: float) -> float




Vector Math
---------------
.. function:: cw_mul(x: vec2, y: vec2) -> vec2


.. function:: cw_mul(x: vec3, y: vec3) -> vec3


.. function:: cw_mul(x: vec4, y: vec4) -> vec4


.. function:: cw_div(x: vec2, y: vec2) -> vec2


.. function:: cw_div(x: vec3, y: vec3) -> vec3


.. function:: cw_div(x: vec4, y: vec4) -> vec3


.. function:: dot(x: vec2, y: vec2) -> float

   Compute the dot product between two 2d vectors.


.. function:: dot(x: vec3, y: vec3) -> float

   Compute the dot product between two 3d vectors.


.. function:: dot(x: vec4, y: vec4) -> float

   Compute the dot product between two 4d vectors.


.. function:: dot(x: quat, y: quat) -> float

   Compute the dot product between two quaternions.


.. function:: outer(x: vec2, y: vec2) -> mat22

   Compute the outer product x*y^T for two vec2 objects.


.. function:: outer(x: vec3, y: vec3) -> mat33

   Compute the outer product x*y^T for two vec3 objects.


.. function:: cross(x: vec3, y: vec3) -> vec3

   Compute the cross product of two 3d vectors.


.. function:: skew(x: vec3) -> mat33

   Compute the skew symmetric matrix for a 3d vector.


.. function:: length(x: vec2) -> float

   Compute the length of a 2d vector.


.. function:: length(x: vec3) -> float

   Compute the length of a 3d vector.


.. function:: length(x: vec4) -> float

   Compute the length of a 4d vector.


.. function:: normalize(x: vec2) -> vec2

   Compute the normalized value of x, if length(x) is 0 then the zero vector is returned.


.. function:: normalize(x: vec3) -> vec3

   Compute the normalized value of x, if length(x) is 0 then the zero vector is returned.


.. function:: normalize(x: vec4) -> vec4

   Compute the normalized value of x, if length(x) is 0 then the zero vector is returned.


.. function:: normalize(x: quat) -> quat

   Compute the normalized value of x, if length(x) is 0 then the zero quat is returned.


.. function:: determinant(m: mat22) -> float

   Compute the determinant of a 2x2 matrix.


.. function:: determinant(m: mat33) -> float

   Compute the determinant of a 3x3 matrix.


.. function:: transpose(m: mat22) -> mat22

   Return the transpose of the matrix m


.. function:: transpose(m: mat33) -> mat33

   Return the transpose of the matrix m


.. function:: transpose(m: mat44) -> mat44

   Return the transpose of the matrix m


.. function:: transpose(m: spatial_matrix) -> spatial_matrix

   Return the transpose of the matrix m


.. function:: diag(d: vec2) -> mat22

   Returns a matrix with the components of the vector d on the diagonal


.. function:: diag(d: vec3) -> mat33

   Returns a matrix with the components of the vector d on the diagonal


.. function:: diag(d: vec4) -> mat44

   Returns a matrix with the components of the vector d on the diagonal


.. function:: vec2() -> vec2


.. function:: vec2(x: float, y: float) -> vec2


.. function:: vec2(s: float) -> vec2


.. function:: vec3() -> vec3


.. function:: vec3(x: float, y: float, z: float) -> vec3


.. function:: vec3(s: float) -> vec3


.. function:: vec4() -> vec4


.. function:: vec4(x: float, y: float, z: float, w: float) -> vec4


.. function:: vec4(s: float) -> vec4


.. function:: mat22(c0: vec2, c1: vec2) -> mat22


.. function:: mat22(m00: float, m01: float, m10: float, m11: float) -> mat22


.. function:: mat33(c0: vec3, c1: vec3, c2: vec3) -> mat33


.. function:: mat33(m00: float, m01: float, m02: float, m10: float, m11: float, m12: float, m20: float, m21: float, m22: float) -> mat33


.. function:: mat44(c0: vec4, c1: vec4, c2: vec4, c3: vec4) -> mat44


.. function:: mat44(m00: float, m01: float, m02: float, m03: float, m10: float, m11: float, m12: float, m13: float, m20: float, m21: float, m22: float, m23: float, m30: float, m31: float, m32: float, m33: float) -> mat44


.. function:: svd3(A: mat33, U: mat33, sigma: vec3, V: mat33) -> None

   Compute the SVD of a 3x3 matrix. The singular values are returned in sigma, 
   while the left and right basis vectors are returned in U and V




Quaternion Math
---------------
.. function:: quat() -> quat

   Construct a zero-initialized quaternion, quaternions are laid out as
   [ix, iy, iz, r], where ix, iy, iz are the imaginary part, and r the real part.


.. function:: quat(x: float, y: float, z: float, w: float) -> quat

   Construct a quarternion from its components x, y, z are the imaginary parts, w is the real part.


.. function:: quat(i: vec3, r: float) -> quat

   Construct a quaternion from it's imaginary components i, and real part r


.. function:: quat_identity() -> quat

   Construct an identity quaternion with zero imaginary part and real part of 1.0


.. function:: quat_from_axis_angle(axis: vec3, angle: float) -> quat

   Construct a quaternion representing a rotation of angle radians around the given axis.


.. function:: quat_inverse(q: quat) -> quat

   Compute quaternion conjugate.


.. function:: quat_rotate(q: quat, p: vec3) -> vec3

   Rotate a vector by a quaternion.


.. function:: quat_rotate_inv(q: quat, p: vec3) -> vec3

   Rotate a vector the inverse of a quaternion.




Transformations
---------------
.. function:: transform(p: vec3, q: quat) -> transform

   Construct a rigid body transformation with translation part p and rotation q.


.. function:: transform_identity() -> transform

   Construct an identity transform with zero translation and identity rotation.


.. function:: transform_get_translation(t: transform) -> vec3

   Return the translational part of a transform.


.. function:: transform_get_rotation(t: transform) -> quat

   Return the rotational part of a transform.


.. function:: transform_multiply(a: transform, b: transform) -> transform

   Multiply two rigid body transformations together.


.. function:: transform_point(t: transform, p: vec3) -> vec3

   Apply the transform to a point p treating the homogenous coordinate as w=1 (translation and rotation).


.. function:: transform_point(m: mat44, p: vec3) -> vec3

   Apply the transform to a point p treating the homogenous coordinate as w=1 (translation and rotation)


.. function:: transform_vector(t: transform, v: vec3) -> vec3

   Apply the transform to a vector v treating the homogenous coordinate as w=0 (rotation only).


.. function:: transform_vector(m: mat44, v: vec3) -> vec3

   Apply the transform to a vector v treating the homogenous coordinate as w=0 (rotation only).




Spatial Math
---------------
.. function:: spatial_vector() -> spatial_vector

   Construct a zero-initialized 6d screw vector. Screw vectors may be used to represent rigid body wrenches and twists (velocites).


.. function:: spatial_vector(a: float, b: float, c: float, d: float, e: float, f: float) -> spatial_vector

   Construct a 6d screw vector from it's components.


.. function:: spatial_vector(w: vec3, v: vec3) -> spatial_vector

   Construct a 6d screw vector from two 3d vectors.


.. function:: spatial_vector(s: float) -> spatial_vector

   Construct a 6d screw vector with all components set to s


.. function:: spatial_adjoint(r: mat33, s: mat33) -> spatial_matrix

   Constructs a 6x6 spatial inertial matrix from two 3x3 diagonal blocks.


.. function:: spatial_dot(a: spatial_vector, b: spatial_vector) -> float

   Compute the dot product of two 6d screw vectors.


.. function:: spatial_cross(a: spatial_vector, b: spatial_vector) -> spatial_vector

   Compute the cross-product of two 6d screw vectors.


.. function:: spatial_cross_dual(a: spatial_vector, b: spatial_vector) -> spatial_vector

   Compute the dual cross-product of two 6d screw vectors.


.. function:: spatial_top(a: spatial_vector) -> vec3

   Return the top (first) part of a 6d screw vector.


.. function:: spatial_bottom(a: spatial_vector) -> vec3

   Return the bottom (second) part of a 6d screw vector.


.. function:: spatial_jacobian(S: array(spatial_vector), joint_parents: array(int32), joint_qd_start: array(int32), joint_start: int, joint_count: int, J_start: int, J_out: array(float32)) -> None


.. function:: spatial_mass(I_s: array(spatial_matrix), joint_start: int, joint_count: int, M_start: int, M: array(float32)) -> None




Linear Algebra
---------------
.. function:: dense_gemm(m: int, n: int, p: int, t1: int, t2: int, A: array(float32), B: array(float32), C: array(float32)) -> None


.. function:: dense_gemm_batched(m: array(int32), n: array(int32), p: array(int32), t1: int, t2: int, A_start: array(int32), B_start: array(int32), C_start: array(int32), A: array(float32), B: array(float32), C: array(float32)) -> None


.. function:: dense_chol(n: int, A: array(float32), regularization: float, L: array(float32)) -> None


.. function:: dense_chol_batched(A_start: array(int32), A_dim: array(int32), A: array(float32), regularization: float, L: array(float32)) -> None


.. function:: dense_subs(n: int, L: array(float32), b: array(float32), x: array(float32)) -> None


.. function:: dense_solve(n: int, A: array(float32), L: array(float32), b: array(float32), x: array(float32)) -> None


.. function:: dense_solve_batched(b_start: array(int32), A_start: array(int32), A_dim: array(int32), A: array(float32), L: array(float32), b: array(float32), x: array(float32)) -> None




Geometry
---------------
.. function:: mesh_query_point(id: uint64, point: vec3, max_dist: float, inside: float, face: int, bary_u: float, bary_v: float) -> bool

   Computes the closest point on the mesh with identifier `id` to the given point in space.

   :param id: The mesh identifier
   :param point: The point in space to query
   :param max_dist: Mesh faces above this distance will not be considered by the query
   :param inside: Returns a value < 0 if query point is inside the mesh, >=0 otherwise. Note that mesh must be watertight for this to be robust


.. function:: mesh_query_ray(id: uint64, start: vec3, dir: vec3, max_t: float, t: float, bary_u: float, bary_v: float, sign: float, normal: vec3, face: int) -> bool


.. function:: mesh_query_aabb(id: uint64, lower: vec3, upper: vec3) -> mesh_query_aabb_t


.. function:: mesh_query_aabb_next(id: mesh_query_aabb_t, index: int) -> bool


.. function:: mesh_eval_position(id: uint64, face: int, bary_u: float, bary_v: float) -> vec3


.. function:: mesh_eval_velocity(id: uint64, face: int, bary_u: float, bary_v: float) -> vec3


.. function:: hash_grid_query(id: uint64, point: vec3, max_dist: float) -> hash_grid_query_t


.. function:: hash_grid_query_next(id: hash_grid_query_t, index: int) -> bool


.. function:: hash_grid_point_id(id: uint64, index: int) -> int




Utility
---------------
.. function:: printf() -> None


.. function:: tid() -> int

   Return the current thread id.





---------------
.. function:: select()


.. function:: copy() -> None


.. function:: load()


.. function:: store()


.. function:: atomic_add()


.. function:: atomic_sub()


.. function:: index() -> float


.. function:: print() -> None


.. function:: expect_eq() -> None


