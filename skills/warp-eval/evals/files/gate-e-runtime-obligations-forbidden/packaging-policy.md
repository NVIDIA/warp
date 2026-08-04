# Binding appliance packaging policy

The collision package ships in an audited, read-only appliance image. Its
release policy requires all of the following:

- the package and every optional extra must remain pure Python;
- compiled dependencies and platform-specific binaries are prohibited;
- runtime code generation and compiler or toolchain components are prohibited;
- the application has no writable location for generated code or a kernel
  cache; and
- an accelerator-specific optional backend is not an approved exception.

These are product requirements, not observations about the current dependency
list. The policy owner has rejected changes to these constraints for this
release line.

A dependency-free CPU broad phase, such as a uniform-grid cell list followed by
the existing exact overlap test, is compatible with the policy.
