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
may limit public discussion and redirect the reporter to a private channel.

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
- **Input-data boundary:** Warp APIs interpret array contents as offsets,
  indices, counts, and sizes, for example in sparse matrix topology, mesh and
  BVH indices, and indexed arrays. Warp trusts this data: it must come from a
  trusted source or be validated by the application before it reaches a Warp
  API.
- **Serialized-data boundary:** `warp.Volume.load_from_nvdb()` parses
  serialized NanoVDB grid data from a file or an in-memory buffer and validates
  it before creating a volume. Examples load USD stages and images primarily
  through optional third-party libraries. Defects in third-party parsers and
  decompressors are distinct from defects in Warp.
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
runtime, compiler pipeline, cache loading, and memory management robust when
used with valid input according to these trust boundaries, and to validate
serialized NanoVDB data loaded through `warp.Volume.load_from_nvdb()`. Warp does
not make untrusted Python code, untrusted native snippets, untrusted compiled
artifacts, or untrusted local users safe to execute in the same process.

Warp is a programming framework for building applications. Applications control
their input sources and decide whether, when, and how to validate data based on
their trust model and performance requirements. Validating contents on every
operation adds computational cost. For GPU-resident data, validation that needs
host-side results also requires device-to-host readback and synchronization,
which can prevent CUDA graph capture. Warp therefore leaves comprehensive
content validation to the application.

Warp trusts the data passed to its APIs. Array contents and data-structure
topology, such as sparse offsets and column indices, must come from trusted
sources or be validated by the application before use. Warp may detect some
invalid input and raise an error, but these checks are best-effort diagnostics,
not a security boundary. Undefined behavior caused by invalid input, including
out-of-bounds memory access, is a robustness issue rather than a vulnerability
unless Warp itself produced the invalid data from valid inputs or the defect
also occurs with valid input.

`warp.Volume.load_from_nvdb()` is the exception. A NanoVDB file contains only
data that Warp's own code interprets, and checking the grid structure outside
Warp would require a separate NanoVDB implementation. Warp validates every grid
on the host before creating the volume, so the check avoids the per-operation
cost and CUDA graph capture limits described above. APIC bundles and other
executable artifacts stay trusted. They contain compiled code, and a `.wrp` file
also supplies kernel arguments and initial memory contents, so no file-level
check can make loading an untrusted bundle safe.

## Threat Model

The following scenarios represent the primary security concerns for Warp:

1. **Native memory corruption through pointer metadata and bindings:** Warp
   passes ctypes structures, raw pointers, external-array metadata, device
   pointers, strides, shapes, and launch parameters into native C++/CUDA code.
   Incorrect device selection, stale buffers, or validation gaps can cause
   process termination, data corruption, GPU faults, or memory disclosure.
   **Security stance:** Failures in Warp's memory handling through a supported
   API with valid input are in scope. Callers remain responsible for the
   validity and lifetime of raw addresses and externally owned allocations that
   Warp cannot independently verify, and for the contents of the arrays and data
   structures they pass to Warp, as described in the next item.

2. **Invalid array contents and data structures:** Many Warp APIs interpret
   array contents as offsets, indices, counts, or sizes, for example sparse
   matrix topology, mesh and BVH indices, and indexed arrays. Invalid values can
   cause out-of-bounds memory access, GPU faults, process termination, or
   excessive resource consumption. **Security stance:** Warp does not impose
   content validation on every operation; applications control validation
   according to their trust model and performance requirements. Applications
   that obtain data from untrusted sources must validate it before passing it to
   Warp. Checks that Warp performs, such as array bounds checking in debug mode,
   are best-effort diagnostics and not a security boundary. Defects are in scope
   when they occur with valid input or when Warp itself produces invalid data
   from valid inputs. See
   [Using Warp with Untrusted Data](#using-warp-with-untrusted-data) for the
   example APIs and the contents they trust.

3. **Malformed NanoVDB data:** `warp.Volume.load_from_nvdb()` parses file
   headers, metadata, compression records, and grid buffers before creating a
   volume that native volume operations read. Malformed data can target bounds
   and size calculations, decompression, and native memory access. **Security
   stance:** Warp treats serialized NanoVDB data as untrusted. Before creating
   the volume, it validates the header, metadata, tree layout, node ranges, and
   child references of each grid on the host after decompression. Validation
   defects that let malformed data cause native process termination, GPU faults,
   memory corruption, or information disclosure are in scope for assessment.
   Warp does not limit file or decompressed sizes. Defects in the zlib and blosc
   decompressors are third-party defects. Volumes created from in-memory grid
   buffers with `wp.Volume(array)` or `Volume.load_from_address()` follow the
   trusted-input rule in the previous item.

4. **Kernel cache or compiled artifact tampering:** Warp loads cached object
   files, PTX/CUBIN modules, LTO artifacts, and `.meta` files from the configured
   kernel cache. Another local user or compromised process could replace cached
   artifacts in a shared or unprotected location. **Security stance:** Warp uses
   a per-user cache by default. Applications that configure a shared cache are
   responsible for preventing untrusted writes; defects that compromise the
   isolation or integrity of a properly protected cache remain in scope.

5. **Cross-framework memory and stream misuse:** PyTorch, JAX, Paddle, DLPack,
   CUDA array interface, CUDA graph capture, peer access, memory pools, and
   custom allocators share memory and stream ownership across libraries. A
   producer that misreports metadata or violates lifetime and synchronization
   rules can make Warp kernels access stale, inaccessible, or incorrectly typed
   memory. **Security stance:** Warp-owned violations of supported interop
   contracts are in scope. Custom producers remain responsible for accurate
   metadata, ownership, lifetime, and synchronization. Callers can enable
   stricter diagnostics with `warp.config.launch_array_access_mode`.

6. **Trusted native code and external build inputs:** Warp compiles
   `@wp.kernel` definitions, `@wp.func_native` snippets, CPU/CUDA preambles and
   headers supplied through `wp.ModuleBuildOptions`, and native types and
   functions registered through `warp.build_experimental`. These inputs can
   affect executable CPU or GPU code and native ABI boundaries. **Security
   stance:** Kernels, addons, preambles, headers, native registrations, ABI
   descriptions, and other build inputs must come from trusted sources. Their
   expected execution is not a sandbox escape. External files whose contents
   affect compilation must be listed in `extra_build_dependencies` so changes
   participate in module cache invalidation.

7. **Loading APIC and other executable artifacts:** `warp.capture_load()` reads
   an APIC `.wrp` file and loads compiled modules from its companion `_modules/`
   directory. AOT output and other CPU object, PTX, or CUBIN modules likewise
   contain executable code. **Security stance:** These inputs are trusted
   executable artifacts; Warp does not sandbox intentionally malicious modules.
   Defects that cause Warp to load more than the caller explicitly trusted,
   cross another trust boundary, or occur with a valid artifact produced by a
   supported Warp workflow remain in scope.

8. **Toolchain and dependency substitution during source builds:**
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
- Data passed to Warp APIs is trusted or has been validated by the
  application. This includes array contents that encode topology, indexing,
  counts, or sizes. Warp does not provide input sanitization or
  application-level quotas for attacker-controlled data.
- Applications that accept NanoVDB data from untrusted sources load it with
  `warp.Volume.load_from_nvdb()` and bound memory and CPU use themselves, for
  example with input size limits or process resource limits. Warp validates
  grid structure but does not limit decompressed sizes or provide a
  general-purpose parser sandbox.
- Examples and their optional third-party USD and image parsers are local
  developer workflows, not hardened ingestion services for attacker-controlled
  assets.
- Applications and services that accept data from remote users, files, or
  other untrusted sources must validate and constrain it before invoking Warp.
  Network transport security, rate limiting, authentication, authorization,
  request logging, and secrets handling are outside Warp's built-in
  responsibilities.
- Source builds trust the selected package indexes, Packman configuration,
  CUDA Toolkit, LLVM/Clang installation, compilers, linkers, environment
  variables, and CI runners used to produce native libraries and wheels.

## Using Warp with Untrusted Data

The trusted-data rule applies to every Warp API except
`warp.Volume.load_from_nvdb()`. Any other API may use array contents as memory
offsets, indices, counts, or sizes, and invalid contents are undefined behavior
that can cause out-of-bounds memory access, GPU faults, or process termination.
The table below lists common examples. It is not exhaustive, and an API's
absence from it does not mean Warp validates that API's input contents.

| Example API                                      | Contents Warp trusts               | What Warp checks                                |
| ------------------------------------------------ | ---------------------------------- | ----------------------------------------------- |
| `warp.sparse` operations on a `BsrMatrix`        | `offsets`, `row_counts`, `columns` | Array shapes, dtypes, and devices               |
| `wp.Mesh`                                        | Vertex `indices` into `points`     | Array shapes, dtypes, devices, and contiguity   |
| `wp.indexedarray`                                | `indices`                          | Index array type, dimension, dtype, and device  |
| `wp.Volume(array)`, `Volume.load_from_address()` | NanoVDB node and child offsets     | Buffer size, magic, and grid and blind metadata |
| `warp.utils.graph_coloring_get_groups()`         | `node_colors`                      | Array dtype, dimension, and device              |

Undefined behavior from invalid contents passed to these APIs, or to other APIs
not listed above, is a property of Warp's trusted-input model, not a
vulnerability in Warp.

For NanoVDB data from untrusted sources, use `warp.Volume.load_from_nvdb()`,
which accepts a file object or an in-memory buffer in `.nvdb` format and
validates each grid before creating the volume. `wp.Volume(array)` and
`Volume.load_from_address()` run only the metadata checks in the table above
and trust the rest of the grid.

## Scope

Security reports are in scope when they affect Warp's tracked source, packaged
Python APIs, native libraries, JIT compilation pipeline, kernel cache behavior,
external-compilation interfaces, supported interop APIs, APIC graph loading,
Volume/NanoVDB loading, memory management, release artifacts, or build/release
tooling.

Listing a class of reports as in scope for assessment does not mean that every
instance is a security vulnerability or establish its severity. Classification
depends on the affected trust boundary and the demonstrated confidentiality,
integrity, and availability impact.

Reports that depend on invalid array contents or data structures, such as
malformed sparse matrix topology or out-of-range indices, are treated as
robustness or defense-in-depth issues rather than vulnerabilities, regardless
of how the input is delivered. They are security-relevant only when the defect
also occurs with valid input or when Warp itself produced the invalid data from
valid inputs. Applications remain responsible for validating untrusted data,
deciding who may submit it, and enforcing authentication, authorization, and
resource quotas.

Malformed NanoVDB data is in scope for assessment when a validation defect in
`warp.Volume.load_from_nvdb()` lets it cause native process termination, GPU
faults, memory corruption, or information disclosure. The input's delivery path
does not change Warp's responsibility for memory safety. Bounding memory and CPU
use for large or highly compressed NanoVDB input remains the application's
responsibility.

USD and image parsing in examples is primarily implemented by optional
third-party libraries. Defects that reproduce in those parsers without Warp are
out of scope for Warp. Defects caused or amplified by Warp-specific buffer,
lifetime, or integration behavior with valid input remain in scope. Invalid
input causing a local example to fail is a robustness issue unless it crosses
another security boundary.

Reports involving APIC graph loading are evaluated against the trusted
executable-artifact boundary. Defects are security-relevant when they occur with
a valid bundle produced by supported Warp workflows, cross the stated trust
boundary, or arise through a Warp-supported path that accepts
attacker-controlled artifacts.

Defects that require an intentionally malicious APIC bundle, without a trust
boundary bypass, are generally treated as robustness or defense-in-depth issues
rather than vulnerabilities in Warp's supported security model.

The following paths contain vendored third-party or separately maintained
component code:

- `tools/packman/` (NVIDIA Packman)
- `warp/_src/thirdparty/` (multiple upstream projects; see the source
  attribution in each file)
- `warp/native/cuBQL/` (NVIDIA cuBQL)
- `warp/native/nanovdb/` (OpenVDB NanoVDB)

Vulnerabilities confined to code under these paths and reproducible in the
corresponding upstream project or component are outside the scope of Warp's
security policy. Report them through the upstream project's security policy or
vulnerability-reporting channel. For Packman vulnerabilities, use the NVIDIA
PSIRT reporting channels above and identify Packman as the affected component;
these reports are routed separately from Warp vulnerabilities.

Vulnerabilities caused by Warp-specific modifications, configuration, packaging,
or integration of vendored code remain in scope. If it is unclear whether a
vulnerability originates in Warp or the upstream project, report it privately
through one of Warp's reporting channels above; Warp maintainers will triage it
and coordinate with the upstream project when appropriate.

The following are generally out of scope unless they demonstrate a vulnerability
inside Warp itself:

- Expected code execution from intentionally running untrusted Python code,
  untrusted `@wp.kernel` definitions, untrusted `@wp.func_native` snippets,
  untrusted external-compilation inputs, or untrusted APIC, object, PTX, or
  CUBIN artifacts.
- Out-of-bounds access, crashes, resource exhaustion, or other undefined
  behavior that requires passing invalid array contents or data structures to a
  Warp API, unless Warp produced the invalid data from valid inputs.
- Denial of service caused only by intentionally launching extremely large
  kernels, allocating excessive memory, loading very large or highly compressed
  NanoVDB data, or running examples with unrealistic local inputs.
- Missing authentication, authorization, rate limiting, quotas, or other
  controls in an application that embeds Warp. These controls are the
  application's responsibility. Remote reachability and attacker-visible
  information disclosure are assessed from the actual deployment and data
  flow.
- Issues in third-party frameworks or drivers that can be reproduced without
  Warp and do not involve a Warp-specific integration path.

## Dependency and Lockfile Security

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
