"""Minimal reproduction of the uint8 type closure conversion bug."""

import warp as wp

wp.init()


def create_type_closure_scalar(scalar_type):
    @wp.kernel
    def k(input: float, expected: float):
        x = scalar_type(input)
        wp.expect_eq(float(x), expected)

    return k


# These work fine (int, float closures)
type_closure_kernel_int = create_type_closure_scalar(int)
type_closure_kernel_float = create_type_closure_scalar(float)

# This is the broken one
type_closure_kernel_uint8 = create_type_closure_scalar(wp.uint8)

print("Testing int closure...")
wp.launch(type_closure_kernel_int, dim=1, inputs=[-1.5, -1.0], device="cpu")
wp.synchronize()
print("  PASSED")

print("Testing float closure...")
wp.launch(type_closure_kernel_float, dim=1, inputs=[-1.5, -1.5], device="cpu")
wp.synchronize()
print("  PASSED")

print("Testing uint8 closure...")
try:
    wp.launch(type_closure_kernel_uint8, dim=1, inputs=[-1.5, 255.0], device="cpu")
    wp.synchronize()
    print("  PASSED")
except Exception as e:
    print(f"  FAILED with exception: {type(e).__name__}: {e}")
