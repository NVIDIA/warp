Fix `wp.grad()` calls to functions with custom `@wp.func_grad` implementations failing to compile when the custom
gradient was not otherwise referenced, and under-reserving forward shared memory when the custom gradient used tile
operations.
