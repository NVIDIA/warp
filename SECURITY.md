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

If a potential vulnerability is reported through a public channel, maintainers
may limit public discussion and redirect the reporter to a private channel. The
reporting channel does not determine the report's technical scope or validity.

## Security Architecture & Context

Warp is an actively maintained, production/stable Python framework and native
runtime for GPU-accelerated simulation, robotics, geometry processing,
graphics, and machine-learning workloads. The publicly distributed Python
package is `warp-lang`; the core runtime combines Python APIs, ctypes bindings,
native C++/CUDA libraries, and JIT compilation for user-defined Warp kernels.

**Repository Exposure Classification:** Public. Basis: the canonical source is
publicly readable at `github.com/NVIDIA/warp`.

**Service Exposure Classification:** External / Regulated (high confidence).
Basis: Warp is an externally distributed library/SDK published through PyPI.
This classification describes distribution and governance context, not the
severity of any vulnerability.

Warp operates at the library/SDK level inside the caller's Python or C++
process. It has no standalone deployment or network identity and is not a
network service, authentication layer, database, or sandbox. The tracked code
does not expose HTTP routes, gRPC services, network listeners, credential
stores, or TLS configuration. Applications and services embedding Warp
determine who can reach it and what data it processes, and are responsible for
authentication, authorization, request validation, tenant isolation, transport
security, logging policy, and secrets handling.

The main security boundaries in this repository are:

- **Native runtime boundary:** `warp/_src/context.py` loads native libraries from
  `warp/bin/` such as `warp.so`, `warp.dll`, `libwarp.dylib`, and the optional
  `warp-clang` backend through ctypes. These libraries implement memory
  allocation, CUDA interaction, CPU/CUDA kernel launch, graph capture, sparse
  data structures, geometry queries, and other performance-critical operations.
- **JIT and external-compilation boundary:** `warp/_src/codegen.py`,
  `warp/_src/build.py`, and `warp/_src/context.py` translate trusted Python
  kernel definitions and `@wp.func_native` snippets into C++ or CUDA source.
  `wp.ModuleBuildOptions` can add CPU/CUDA preambles, include directories, and
  dependency files; `warp.build_experimental` can register external native
  types and functions; and AOT APIs can emit compiled artifacts for external
  runtimes. Warp compiles these inputs with LLVM/Clang, NVRTC, nvJitLink, and
  optional libmathdx components, then loads or returns the resulting CPU object
  files, PTX, or CUBIN modules.
- **Kernel cache boundary:** compiled sources, object files, PTX/CUBIN files,
  LTO artifacts, and metadata are cached under `warp.config.kernel_cache_dir`.
  The default location is a per-user cache directory. Callers may override it
  with `warp.config.kernel_cache_dir` or `WARP_CACHE_PATH`.
- **External memory boundary:** Warp accepts and exports arrays through NumPy,
  PyTorch, JAX, Paddle, DLPack, `__array_interface__`, and
  `__cuda_array_interface__`. Many conversions are zero-copy, so Warp may launch
  kernels against memory allocated and owned by another framework.
- **Serialized-data boundary:** `warp.Volume.load_from_nvdb()` parses serialized
  NanoVDB grid data supplied through a public Warp API. USD stages and images
  are loaded primarily by examples through optional third-party libraries.
  NanoVDB parsing defects in Warp and defects in Warp-specific integration code
  are distinct from vulnerabilities in those third-party parsers.
- **Executable-artifact boundary:** an APIC graph bundle consists of a `.wrp`
  file and its companion `_modules/` directory, which contains compiled CPU or
  CUDA modules. `warp.capture_load()` reconstructs recorded operations and
  loads this executable code. APIC bundles, AOT output, and other CPU object,
  PTX, or CUBIN modules are therefore trusted executable artifacts rather than
  untrusted data-interchange formats.
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

1. **Native memory corruption through pointer metadata and bindings:** Warp
   passes ctypes structures, raw pointers, external-array metadata, device
   pointers, strides, shapes, and launch parameters into native C++/CUDA code.
   Incorrect device selection, stale buffers, or validation gaps can cause
   process termination, data corruption, GPU faults, or memory disclosure.
   **Security stance:** Failures in Warp's validation or memory handling through
   a supported API are in scope. Callers remain responsible for the validity and
   lifetime of raw addresses and externally owned allocations that Warp cannot
   independently verify.

2. **Malformed NanoVDB data:** `warp.Volume.load_from_nvdb()` parses file
   headers, metadata, compression records, and grid buffers before making them
   available to native volume operations. Malformed data can target bounds and
   size calculations, decompression, resource consumption, and native memory
   access. **Security stance:** Native process termination, GPU faults, memory
   corruption, information disclosure, or disproportionate resource exhaustion
   caused by malformed data through this supported API are in scope for
   assessment. A handled validation or unsupported-format error is not a
   vulnerability, nor is an application failure caused only by not handling
   such an error.

3. **Kernel cache or compiled artifact tampering:** Warp loads cached object
   files, PTX/CUBIN modules, LTO artifacts, and `.meta` files from the configured
   kernel cache. Another local user or compromised process could replace cached
   artifacts in a shared or unprotected location. **Security stance:** Warp uses
   a per-user cache by default. Applications that configure a shared cache are
   responsible for preventing untrusted writes; defects that compromise the
   isolation or integrity of a properly protected cache remain in scope.

4. **Cross-framework memory and stream misuse:** PyTorch, JAX, Paddle, DLPack,
   CUDA array interface, CUDA graph capture, peer access, memory pools, and
   custom allocators share memory and stream ownership across libraries. A
   producer that misreports metadata or violates lifetime and synchronization
   rules can make Warp kernels access stale, inaccessible, or incorrectly typed
   memory. **Security stance:** Warp-owned violations of supported interop
   contracts are in scope. Custom producers remain responsible for accurate
   metadata, ownership, lifetime, and synchronization. Callers can enable
   stricter diagnostics with `warp.config.launch_array_access_mode`.

5. **Trusted native code and external build inputs:** Warp compiles
   `@wp.kernel` definitions, `@wp.func_native` snippets, CPU/CUDA preambles and
   headers supplied through `wp.ModuleBuildOptions`, and native types and
   functions registered through `warp.build_experimental`. These inputs can
   affect executable CPU or GPU code and native ABI boundaries. **Security
   stance:** Kernels, addons, preambles, headers, native registrations, ABI
   descriptions, and other build inputs must come from trusted sources. Their
   expected execution is not a sandbox escape. External files whose contents
   affect compilation must be listed in `extra_build_dependencies` so changes
   participate in module cache invalidation.

6. **Loading APIC and other executable artifacts:** `warp.capture_load()` reads
   an APIC `.wrp` file and loads compiled modules from its companion `_modules/`
   directory. AOT output and other CPU object, PTX, or CUBIN modules likewise
   contain executable code. **Security stance:** These inputs are trusted
   executable artifacts; Warp does not sandbox intentionally malicious modules.
   Defects that cause Warp to load more than the caller explicitly trusted,
   cross another trust boundary, or occur with a valid artifact produced by a
   supported Warp workflow remain in scope. Other malformed-artifact failures
   may still warrant defense-in-depth hardening.

7. **Toolchain and dependency substitution during source builds:**
   `build_lib.py` discovers CUDA through `WARP_CUDA_PATH`, `CUDA_HOME`,
   `CUDA_PATH`, and `nvcc`, can use `LIBMATHDX_HOME` or `--libmathdx-path`, and
   can consume existing or built LLVM/Clang installations. Build and CI tooling
   also downloads packages through Packman, uv, PyPI, GitHub Releases, and
   CUDA-related channels. **Security stance:** Source builds trust the selected
   toolchains, package sources, environment, and runner. Defects that compromise
   official Warp build or release artifacts remain in scope; substituting an
   untrusted local toolchain is outside the boundary.

## Critical Security Assumptions

- The Python process executing Warp code is trusted. Warp does not isolate
  tenants, authenticate callers, authorize actions, or sandbox untrusted Python
  code, kernels, C++/CUDA snippets, external-compilation addons, headers,
  preambles, native registrations, or compiled artifacts.
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
- External native type descriptions match the actual C++ ABI, and addons list
  every file whose contents affect compilation in `extra_build_dependencies`.
- An APIC `.wrp` file and its companion `_modules/` directory are one trusted
  executable-artifact bundle. Callers must load bundles only from trusted
  sources. `warp.capture_load()` is not a sandbox or security boundary for
  attacker-controlled artifacts.
- Applications that accept NanoVDB data from untrusted sources bound input and
  decompressed sizes according to their threat model. Warp aims to fail safely
  on malformed NanoVDB data, but does not provide application-level quotas or a
  general-purpose parser sandbox.
- Examples and their optional third-party USD and image parsers are local
  developer workflows, not hardened ingestion services for attacker-controlled
  assets.
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
external-compilation interfaces, supported interop APIs, APIC graph loading,
Volume/NanoVDB loading, memory management, release artifacts, or build/release
tooling.

Reports involving malformed NanoVDB data are in scope for assessment when a
supported Warp API causes native process termination, GPU faults, memory
corruption, information disclosure, or disproportionate resource exhaustion.
A handled validation or unsupported-format error is not a vulnerability.
Resource exhaustion caused only by intentionally excessive input is generally
out of scope unless a comparatively small input causes disproportionate
consumption.

USD and image parsing in examples is primarily implemented by optional
third-party libraries. Defects that reproduce in those parsers without Warp are
out of scope for Warp. Defects caused or amplified by Warp-specific buffer,
lifetime, or integration behavior remain in scope. Invalid input merely causing
a local example to fail is generally a robustness issue unless it exposes a
Warp-owned memory-safety defect or crosses another security boundary.

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
  untrusted `@wp.kernel` definitions, untrusted `@wp.func_native` snippets,
  untrusted external-compilation inputs, or untrusted APIC, object, PTX, or
  CUBIN artifacts.
- Denial of service caused only by intentionally launching extremely large
  kernels, allocating excessive memory, or running examples with unrealistic
  local inputs.
- Misconfigured deployments where a service exposes Warp compilation or asset
  loading to unauthenticated users without upstream validation.
- Issues in third-party frameworks or drivers that can be reproduced without
  Warp and do not involve a Warp-specific integration path.

## Dependency And Lockfile Security

Warp declares its published Python packaging contract in `pyproject.toml`.
`uv.lock` defines the repository's development, testing, documentation, and CI
environment; it is not installed with `warp-lang` and does not define
dependencies for downstream runtime deployments. Findings based solely on a
package or version appearing in `uv.lock` are out of scope.

A dependency report must demonstrate reachable impact on Warp's published
artifacts or an actual build or release path; presence in the development
lockfile alone is insufficient. Maintainers consider whether the affected
dependency is required by the default runtime, used through an optional feature
such as PyTorch, JAX, Paddle, docs, or examples, or involved in producing a
release artifact.

Dependency minimums in `pyproject.toml` express compatibility, not a security
recommendation for every historical compatible release. Dependency CVEs are
triaged based on reachable behavior, the affected artifact or workflow, and
whether Warp selects or distributes the affected version. Users and downstream
distributors remain responsible for applying dependency constraints required by
their environments.

## Supported Versions

Only the latest Warp feature-release line is actively maintained and eligible
for bugfix or security releases. Earlier feature-release lines do not normally
receive backports, although critical fixes may be backported in exceptional
cases. Users should upgrade to the latest feature release. See the
[release support policy](https://nvidia.github.io/warp/stable/user_guide/compatibility.html#release-support-policy)
for authoritative guidance.

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
