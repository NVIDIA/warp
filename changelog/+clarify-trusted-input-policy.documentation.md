Clarify Warp's trusted-input policy in `SECURITY.md`.
Applications must validate array contents and other in-memory data from untrusted sources before passing them to Warp,
while `wp.Volume.load_from_nvdb()` validates serialized NanoVDB data itself.
The policy also defines when invalid input is a robustness issue rather than a vulnerability
and adds guidance for using Warp with untrusted data.
