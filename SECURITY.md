# Security Policy: NVIDIA Warp

NVIDIA is dedicated to the security and trust of our software products and
services, including source code repositories managed through our organization.

If you believe you have found a security vulnerability in NVIDIA Warp, report it
privately. **Do not open a public GitHub or GitLab issue, pull request,
discussion, or merge request for security vulnerabilities.**

## Reporting Potential Security Vulnerabilities

To report a potential security vulnerability in Warp:

- **Web (preferred):** Use the
  [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/report-vulnerability/).
- **Email:** Send details to [psirt@nvidia.com](mailto:psirt@nvidia.com).
  NVIDIA encourages using the
  [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key)
  for secure email communication.
- **Repository private reporting:** Use this repository's **Security** tab to
  submit the report privately.

Please include the following information:

- Product/project name and affected version, branch, or commit.
- Vulnerability type, such as code execution, denial of service, memory
  corruption, privilege escalation, or information disclosure.
- Step-by-step reproduction instructions.
- Proof-of-concept code or exploit details, if available.
- Potential impact, including how an attacker could exploit the vulnerability.
- Relevant platform details, including operating system, Python version, CUDA
  Toolkit and driver version, GPU model, compiler/toolchain version, and whether
  the issue affects CPU, CUDA, or both.

NVIDIA PSIRT will coordinate intake, validation, severity assessment,
remediation, and disclosure. NVIDIA strives to follow coordinated vulnerability
disclosure and may publish security bulletins, remediations, and
acknowledgements as appropriate. For current program details, see the
[NVIDIA Product Security](https://www.nvidia.com/en-us/security/) portal.

## Security Architecture & Context

Warp is a Python framework and native runtime for GPU-accelerated simulation,
robotics, geometry processing, graphics, and machine-learning workloads. The
published Python package is `warp-lang`; the core runtime combines Python APIs,
ctypes bindings, native C++/CUDA libraries, and JIT compilation for user-defined
Warp kernels.

Warp operates at the library/SDK level inside the caller's Python or C++ process.
It is not a network service, authentication layer, database, or sandbox. The
tracked code does not expose HTTP routes, gRPC services, network listeners,
credential stores, or TLS configuration. Applications and services embedding
Warp are responsible for their own authentication, authorization, request
validation, tenant isolation, transport security, logging policy, and secrets
handling.

The main security boundaries in this repository are:

- **Native runtime boundary:** `warp/_src/context.py` loads native libraries from
  `warp/bin/` such as `warp.so`, `warp.dll`, `libwarp.dylib`, and the optional
  `warp-clang` backend through ctypes. These libraries implement memory
  allocation, CUDA interaction, CPU/CUDA kernel launch, graph capture, sparse
  data structures, geometry queries, and other performance-critical operations.
- **JIT compiler boundary:** `warp/_src/codegen.py`, `warp/_src/build.py`, and
  `warp/_src/context.py` translate trusted Python kernel definitions into C++ or
  CUDA source, compile them with LLVM/Clang, NVRTC, nvJitLink, and optional
  libmathdx components, then load the resulting CPU object files, PTX, or CUBIN
  modules into the current process or CUDA context.
- **Kernel cache boundary:** compiled sources, object files, PTX/CUBIN files,
  LTO artifacts, and metadata are cached under `warp.config.kernel_cache_dir`.
  The default location is a per-user cache directory. Callers may override it
  with `warp.config.kernel_cache_dir` or `WARP_CACHE_PATH`.
- **External memory boundary:** Warp accepts and exports arrays through NumPy,
  PyTorch, JAX, Paddle, DLPack, `__array_interface__`, and
  `__cuda_array_interface__`. Many conversions are zero-copy, so Warp may launch
  kernels against memory allocated and owned by another framework.
- **Serialized-data and executable-artifact boundary:** public APIs and examples
  load local assets such as NanoVDB buffers through
  `warp.Volume.load_from_nvdb()`, APIC graph bundles through
  `warp.capture_load()`, USD stages in examples, and image files in examples. An
  APIC graph bundle consists of a `.wrp` file and its companion `_modules/`
  directory, which contains compiled CPU or CUDA modules. Loading the bundle
  reconstructs recorded operations and loads executable code, so the complete
  bundle is treated as a trusted executable artifact rather than an untrusted
  data-interchange format.
- **Build and release boundary:** source builds and CI use `build_lib.py`,
  `setup.py`, `pyproject.toml`, Dockerfiles, Packman manifests, CUDA Toolkit
  paths, LLVM/Clang paths, and optional dependency extras to build Python wheels
  and native libraries.

Warp's primary security responsibility is to keep its public APIs, native
runtime, compiler pipeline, cache loading, memory management, and supported file
loaders robust when used according to these trust boundaries. Warp does not make
untrusted Python code, untrusted native snippets, untrusted compiled artifacts,
or untrusted local users safe to execute in the same process.

## Threat Model

The following scenarios represent the primary security concerns for Warp:

1. **Untrusted kernel or native snippet execution:** Warp compiles user-defined
   `@wp.kernel` functions and `@wp.func_native` C++/CUDA snippets into executable
   CPU or GPU code. If an application accepts kernel definitions, native snippets,
   generated Python modules, or C++ example inputs from an untrusted party and
   compiles them with Warp, that party can execute code with the privileges of
   the hosting process and GPU context.

2. **Kernel cache or compiled artifact tampering:** Warp loads cached object
   files, PTX/CUBIN modules, LTO artifacts, and `.meta` files from the configured
   kernel cache. If `WARP_CACHE_PATH` or `warp.config.kernel_cache_dir` points to
   a shared or writable-by-untrusted-users location, another local user or
   compromised process could replace cached artifacts and affect later kernel
   loads.

3. **Native memory corruption through malformed inputs or pointer metadata:**
   Warp passes ctypes structures, raw pointers, external-array metadata, device
   pointers, strides, shapes, and kernel launch parameters into native C++/CUDA
   code. Incorrect device selection, malformed `__cuda_array_interface__` or
   DLPack producers, stale external buffers, or validation gaps in native
   bindings can cause process crashes, data corruption, GPU faults, or memory
   disclosure within the caller's process boundary.

4. **Loading serialized data and executable graph artifacts:**
   `warp.capture_load()` reads an APIC `.wrp` file and loads compiled modules
   from its companion `_modules/` directory. APIC bundles are trusted executable
   artifacts; Warp does not sandbox them or guarantee safe processing of
   intentionally malicious bundles. Other serialized inputs retain their
   existing trust assumptions; this APIC-specific classification does not
   reclassify them. Corrupt or incompatible artifacts can still expose parser,
   resource-exhaustion, object-loading, or replay defects. Warp therefore
   applies defense-in-depth validation where practical, but such validation
   does not establish an adversarial-input security boundary.

5. **Toolchain and dependency substitution during source builds:** `build_lib.py`
   discovers CUDA through `WARP_CUDA_PATH`, `CUDA_HOME`, `CUDA_PATH`, and `nvcc`,
   can use `LIBMATHDX_HOME` or `--libmathdx-path`, and can consume existing or
   built LLVM/Clang installations. Build and CI tooling also downloads packages
   through Packman, uv, PyPI, and CUDA-related channels. A compromised toolchain,
   dependency source, or environment variable can compromise generated native
   libraries and wheels.

6. **Cross-framework memory and stream misuse:** PyTorch, JAX, Paddle, DLPack,
   CUDA array interface, CUDA graph capture, peer access, memory pools, and
   custom allocators share memory and stream ownership across libraries. A
   producer that misreports metadata or violates lifetime and synchronization
   rules can make Warp kernels access stale, inaccessible, or incorrectly typed
   memory. The default launch array access mode is permissive for performance;
   callers can enable stricter diagnostics with
   `warp.config.launch_array_access_mode`.

7. **Auxiliary example and visualization inputs:** Warp examples load USD, NVDB,
   image, CUBIN, and APIC graph artifacts, and C++ examples may dynamically load
   `warp-clang` or link against `warp.so` from local paths. These examples are
   intended for local developer use, not for processing attacker-supplied assets
   in a privileged service.

## Critical Security Assumptions

- The Python process executing Warp code is trusted. Warp does not isolate
  tenants, authenticate callers, authorize actions, or sandbox untrusted Python,
  C++/CUDA snippets, kernels, or compiled artifacts.
- The host operating system enforces process isolation, filesystem permissions,
  dynamic loader policy, and user separation. Warp assumes native libraries under
  `warp/bin/` and configured build toolchain paths are controlled by trusted
  users.
- The CUDA driver, GPU hardware, CUDA memory manager, MMU/UVA behavior, peer
  access configuration, and memory pool implementation correctly enforce device
  memory access rules.
- The kernel cache directory is per-user or otherwise protected from untrusted
  writers. Cached CPU object, PTX, CUBIN, and LTO files are trusted executable
  artifacts; their metadata is trusted as well. Shared cache directories must
  be explicitly hardened by the embedding application or deployment
  environment.
- External array producers correctly report pointer, dtype, shape, stride,
  device, stream, ownership, and lifetime metadata. Warp cannot fully verify
  every custom allocator or externally wrapped allocation.
- An APIC `.wrp` file and its companion `_modules/` directory are one trusted
  executable-artifact bundle. Callers must load bundles only from trusted
  sources. `warp.capture_load()` is not a sandbox or security boundary for
  attacker-controlled artifacts.
- Warp assumes that other serialized inputs, including `.nvdb`, USD, and image
  files, come from trusted sources. Applications accepting untrusted inputs must
  supply the validation or sandboxing required by their threat model.
- Services that expose Warp functionality to remote users must validate and
  constrain inputs before invoking Warp. Network transport security, rate
  limiting, authentication, authorization, request logging, and secrets handling
  are outside Warp's built-in responsibilities.
- Source builds trust the selected package indexes, Packman configuration,
  CUDA Toolkit, LLVM/Clang installation, compilers, linkers, environment
  variables, and CI runners used to produce native libraries and wheels.

## Scope

Security reports are in scope when they affect Warp's tracked source, packaged
Python APIs, native libraries, JIT compilation pipeline, kernel cache behavior,
supported interop APIs, APIC graph loading, Volume/NanoVDB loading, memory
management, release artifacts, or build/release tooling.

Reports involving APIC graph loading are evaluated against the trusted
executable-artifact boundary. Defects are security-relevant when they occur with
a valid bundle produced by supported Warp workflows, cross the stated trust
boundary, or arise through a Warp-supported path that accepts
attacker-controlled artifacts.

Defects that require an intentionally malicious APIC bundle, without a trust
boundary bypass, are generally treated as robustness or defense-in-depth issues
rather than vulnerabilities in Warp's supported security model. Warp may still
accept fixes for such defects, especially when they address memory safety with
focused, low-risk validation.

The following are generally out of scope unless they demonstrate a vulnerability
inside Warp itself:

- Expected code execution from intentionally running untrusted Python code,
  untrusted `@wp.kernel` definitions, untrusted `@wp.func_native` snippets, or
  untrusted compiled object/PTX/CUBIN artifacts.
- Denial of service caused only by intentionally launching extremely large
  kernels, allocating excessive memory, or running examples with unrealistic
  local inputs.
- Misconfigured deployments where a service exposes Warp compilation or asset
  loading to unauthenticated users without upstream validation.
- Issues in third-party frameworks or drivers that can be reproduced without
  Warp and do not involve a Warp-specific integration path.
- Public disclosure through issues, pull requests, discussions, or merge
  requests before NVIDIA PSIRT has had an opportunity to coordinate remediation.

## Dependency And Lockfile Security

Warp declares its Python packaging contract in `pyproject.toml` and maintains
`uv.lock` for repository development, CI, documentation, examples, optional
extras, and release workflows. A vulnerability reported only from `uv.lock`
should identify whether the affected package is used by the default runtime
install, an optional extra such as a framework integration, docs or test tooling,
CI, or release-build infrastructure.

Maintainers triage lockfile findings based on reachability, affected artifact,
and whether the dependency is declared by published wheel metadata, used to
build or validate release artifacts, or only used during development. For
example, a scanner finding against an optional PyTorch, JAX, Paddle, docs, or
examples dependency may have a different impact than a finding against a
dependency required by the default `warp-lang` install.

Warp dependency minimums are compatibility requirements, not a complete security
blocklist for historical third-party releases. Maintainers do not raise package
minimums solely because an older allowed version of a dependency has a CVE.
Dependency CVEs are triaged based on whether the vulnerable behavior is
reachable through Warp, whether the affected version is selected by Warp's normal
development or release workflows, and whether the dependency is part of the
default runtime contract or an optional extra. Users and downstream distributors
remain responsible for applying dependency constraints when their environments
require patched third-party versions.

## Security Update Process

Security fixes may be delivered through repository commits, release branches,
Python wheels, documentation updates, GitHub or GitLab advisories, CVEs, and
NVIDIA security bulletins, depending on severity and affected artifacts.
Consumers should keep Warp, CUDA drivers, CUDA Toolkit components, Python
dependencies, and downstream framework integrations current according to their
deployment requirements.

For source builds, use trusted toolchain paths and package sources, avoid
world-writable build and cache directories, and review build logs when overriding
CUDA, LLVM/Clang, libmathdx, or package index settings.
