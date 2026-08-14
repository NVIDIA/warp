Add `wp.graph_set_conditional()` and the `wp.graph_cond_handle` annotation
type, exposing device-side control of CUDA conditional graph nodes
(CUDA 12.4+) for foreign scopes where the caller owns the condition handle,
such as conditional nodes created by another framework or by direct CUDA
Graph API use. `wp.capture_while()`/`wp.capture_if()` scopes keep their
condition-array contract.
