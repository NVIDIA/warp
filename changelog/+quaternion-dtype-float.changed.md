Reject non-floating-point `dtype` arguments when constructing a quaternion from its components. Quaternions only
exist in `float16`, `float32`, and `float64` flavors, but `wp.quaternion(1.0, 2.0, 3.0, 4.0, dtype=wp.int32)`
previously built an integer quaternion that no quaternion operation supports; it is now rejected at compile time.
Pass a floating-point `dtype` instead. Constructing from runtime variables was already rejected.
