# Scoring plug-in boundary

The scoring function runs in a short-lived, isolated worker process. The host
application sends one point and at most 192 line segments through shared host
memory, then receives one JSON scalar. Production invokes the plug-in no more
than three times per minute.

Process isolation and the public plug-in ABI are fixed requirements. The host
application cannot hand a device allocation to the worker, batch requests, keep
worker state alive, or consume a device-resident result. Any GPU implementation
must therefore initialize its runtime and copy the inputs to the device and the
result back for every call.

The existing NumPy implementation takes 45–80 microseconds per production call.
The product has no memory- or capacity-related objective for this stage.
