# Windows import libraries

**Status**: Proposed

**Issue**: [GH-1888](https://github.com/NVIDIA/warp/issues/1888)

## Motivation

Warp's Windows builds produce `warp.dll` and `warp-clang.dll`. The Microsoft
linker also creates the corresponding `warp.lib` and `warp-clang.lib` import
libraries, but Windows wheels leave the `.lib` files out. Native applications
that use Warp from a wheel must therefore open each DLL and resolve every entry
point with `LoadLibrary()` and `GetProcAddress()`.

The C++ controller runtime proposed in
[Newton PR #4068](https://github.com/newton-physics/newton/pull/4068) shows the
cost of that omission. It has platform-specific library helpers, two function
pointer tables, and a lookup for every Warp entry point it calls. If Warp ships
the import libraries, the controller can link to those functions directly and
drop most of that code.

An import library does not put Warp's implementation into the application. It
records the DLL and symbol names for the Windows loader. This is more convenient
for callers, but it does not make the native ABI stable. Removing or renaming a
symbol can stop the process before `main()`. Changing a function signature
without changing its C symbol name is worse: the process may start, then behave
incorrectly or crash when the application calls the function. For now, native
linking requires an exact version match.

Linux and macOS do not need companion import libraries because their linkers
consume `.so` and `.dylib` files directly. The header and version rules in this
design apply on all platforms. Only the new packaged binaries are specific to
Windows.

## Requirements

| ID | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | Ship `warp.lib` and `warp-clang.lib` alongside their DLLs in Windows wheels. | Must | Applies to every Windows architecture for which a wheel is constructed. |
| R2 | Make Windows import-library output paths deterministic in the native build. | Must | The import libraries remain beside the DLLs under `warp/bin`. |
| R3 | Provide correct export annotations for native-library builds and import annotations for consumers. | Must | Generated JIT modules must retain their existing export behavior. |
| R4 | Provide declarations for the `warp-clang` CPU-module entry points needed by APIC consumers. | Must | Keep the header narrow and mark the interface experimental. |
| R5 | Require headers, import libraries, DLLs, and APIC artifacts from the same Warp release. | Must | No backward or forward native ABI guarantee is introduced. |
| R6 | Fail Windows wheel construction when any required DLL or import library is absent. | Must | Prevent a partial package from silently reaching a release. |
| R7 | Compile, link, and run a native consumer against both import libraries in Windows CI. | Must | The smoke test must not require a GPU. |
| R8 | Preserve existing Python native-library loading and non-Windows packaging. | Must | Python continues loading the shared libraries by explicit path. |
| R9 | Update the C++ APIC examples to demonstrate direct native linking and version checks. | Must | The examples remain self-contained rather than depending on a packaged CMake configuration. |

**Non-goals**:

- Defining a backward- or forward-compatible native ABI.
- Defining cross-version compatibility for `.wrp` files or their companion
  module artifacts.
- Adding a stable entry-point query function or a generated function table.
- Adding versioned DLL filenames.
- Adding a packaged CMake configuration or official umbrella CMake target.
- Adding delay-load behavior for `warp-clang.dll`.
- Updating downstream consumers such as Newton in the Warp change.

## Design

### Approach

Warp packages conventional Microsoft import libraries and treats native
linking as an exact-version workflow. The build already creates the files. The
remaining work is to give them predictable output paths, put them in Windows
wheels, declare the small `warp-clang` surface needed by APIC consumers, and
use the result in Warp's C++ APIC examples and a native smoke test.

Applications check versions before making other native calls. Those checks can
catch a mismatched DLL only after the Windows loader has started the process.
They cannot recover from a missing load-time import.

### Native API visibility

`warp/native/api.h` becomes the single home for `WP_API`, which is currently
defined in more than one native header. `warp.h`, `crt.h`, `apic.h`, and the new
`warp_clang.h` use that definition.

Keeping the visibility macro in a small standalone header lets each native API
header declare imported or exported symbols without including unrelated Warp
runtime, APIC, or CRT declarations. It also keeps the producer-versus-consumer
rule consistent across those independently usable headers.

On Windows:

- A native DLL build defines `WP_BUILD_DLL`, so `WP_API` expands to
  `__declspec(dllexport)`.
- A generated JIT module defines `WP_NO_CRT` and keeps the export annotation
  used today for JIT-visible entry points.
- A normal consumer gets `__declspec(dllimport)`.
- CUDA device code gets no DLL annotation.

On Linux and macOS, `WP_API` keeps default symbol visibility. The
`build_lib.py` and top-level CMake builds define `WP_BUILD_DLL` for both `warp`
and `warp-clang`. Treating `WP_NO_CRT` as an export case preserves the current
behavior of generated CPU and CUDA modules.

### `warp-clang` header

`warp/native/warp_clang.h` declares the four CPU-module functions needed by a
standalone APIC consumer:

- `wp_load_obj()`
- `wp_unload_obj()`
- `wp_lookup()`
- `wp_warp_clang_version()`

The header uses the C ABI and includes both the common visibility definition
and the generated Warp version. Its comments identify the interface as
experimental and require an exactly matching Warp release. Internal compiler
entry points such as `wp_compile_cpp()` and `wp_compile_cuda()` remain out of
the consumer header.

This is enough for Newton PR #4068. `warp.h` and `apic.h` already declare every
core and APIC function used by the proposed controller runtime. The controller
copies only `wp_load_obj()` and `wp_lookup()` from
`warp/native/clang/clang.cpp`.

### Downstream fit: Newton

The proposed `FindWarp.cmake` in draft Newton PR #4068 locates `warp.dll` and
`warp-clang.dll`, then passes both absolute paths to `controller.cpp`. It does
not create linkable targets. The controller opens the libraries and resolves
their entry points at runtime.

Once the wheel contains import libraries, Newton can keep its discovery module
and create ordinary CMake imported targets. On Windows, each target sets the
DLL as `IMPORTED_LOCATION` and its `.lib` as `IMPORTED_IMPLIB`. Because
`newton_controllers` is a static library, it links those targets publicly so
the final executable inherits the native dependencies. Newton may link the
shared libraries directly on Linux and macOS or keep its current runtime loader
there. Nothing in this Warp change forces that choice.

Direct linking removes `dynamic_library.cpp`, `dynamic_library.hpp`, the
absolute-path compile definitions, both function pointer tables, the symbol
resolver, and the copied `warp-clang` declarations from the proposed Newton
runtime. Its CUDA and CPU graph-loading paths stay as they are. The same is true
of CPU module registration; neither part exists to work around Windows DLL
loading.

Newton currently calls `wp_init(nullptr)`, which disables Warp's core-library
version check. It should call `wp_init(WP_VERSION_STRING)` instead. Before the
first CPU-module operation, it should also compare
`wp_warp_clang_version()` with `WP_VERSION_STRING`.

### C++ reference examples

The existing CUDA and CPU APIC examples demonstrate the supported native
linking pattern without introducing a packaged CMake configuration. Their
CMake files define local imported shared targets. On Windows, each target uses
the DLL as `IMPORTED_LOCATION` and the matching import library as
`IMPORTED_IMPLIB`; a post-build command copies the required DLLs beside the
executable. Linux and macOS link their shared libraries directly.

The CUDA example links `Warp::runtime` and passes `WP_VERSION_STRING` to
`wp_init()`. The CPU example also links `Warp::clang`, includes
`warp_clang.h`, and calls `wp_load_obj()` and `wp_lookup()` directly. This
replaces its platform-specific `LoadLibrary()`/`GetProcAddress()` and
`dlopen()`/`dlsym()` code. It checks both the core runtime and `warp-clang`
versions before loading a graph or its CPU modules.

The CPU example's Makefile follows the same dependency model. It links both
shared libraries on Linux and macOS, links both import libraries on Windows,
and stages both Windows DLLs beside the executable.

### Build artifacts

The Windows `link.exe` command passes an explicit `/IMPLIB` path based on the
DLL output path. Building `warp.dll` writes `warp.lib`, and building
`warp-clang.dll` writes `warp-clang.lib`. Both files go in `warp/bin` beside the
DLLs. The linker-generated `.exp` files stay as build intermediates and are not
distributed.

The top-level CMake build already sends archive outputs to `warp/bin`, so its
Windows output remains consistent with `build_lib.py`.

### Wheel packaging

The platform metadata in `setup.py` accepts more than one native-library
extension for a platform. Windows recognizes `.dll` and `.lib`. Linux and
macOS continue to recognize only `.so` and `.dylib`.

A Windows wheel requires all four files:

- `warp.dll`
- `warp.lib`
- `warp-clang.dll`
- `warp-clang.lib`

Wheel construction fails if any one is missing. GitLab's Windows wheel-staging
helpers move the complete set into the platform directory before building the
wheel. The existing `native/*.h` package-data rule already covers `api.h` and
`warp_clang.h`.

### Consumer compatibility contract

Native consumers compile and deploy one matched set of Warp artifacts. On
Windows, that set contains the headers, import libraries, and DLLs from the
same release. A standalone APIC application also uses `.wrp` files and module
artifacts produced by that release.

Before other native calls, the application:

1. Calls `wp_init(WP_VERSION_STRING)` instead of passing a null expected
   version.
2. Compares `wp_warp_clang_version()` with `WP_VERSION_STRING` before loading
   CPU modules.

These checks produce a controlled error when the loader finds a mismatched
library that still exports the requested symbols. They cannot help when a DLL
or symbol is missing, since Windows may reject the process before `main()`.
They also cannot make an incompatible function signature safe. The
documentation states those limits and tells applications to put the matching
DLLs beside the executable or configure the DLL search path deliberately.

A normal link to `warp-clang.lib` makes `warp-clang.dll` a startup dependency,
even when an application eventually uses only CUDA APIC graphs. Both DLLs are
already part of Warp's Windows wheels. The first version of this feature
accepts that dependency instead of adding MSVC delay-load configuration and
failure hooks.

### Continuous integration

`tools/ci/windows_import_library_smoke/` contains a `CMakeLists.txt` and one
C++ source file. The project:

- Finds the native headers and both import libraries under a supplied Warp
  package root.
- Includes `warp.h` and `warp_clang.h`.
- Calls `wp_version()` and `wp_warp_clang_version()` directly so the executable
  imports symbols from both libraries.
- Compares both results with `WP_VERSION_STRING`, using distinct nonzero exit
  codes for mismatches.
- Copies both DLLs beside the executable and registers it with CTest.

The program does not call `wp_init()`, so it does not initialize CUDA or need a
GPU. After the GitHub `build-warp` matrix uploads its artifacts, the
`build-windows-wheels` matrix downloads each Windows artifact, builds a wheel,
checks its platform tag, and uploads it separately. A downstream Windows job
installs each wheel into a clean directory, confirms that Python can initialize
Warp with matching native-library versions and architecture, then configures,
builds, and runs the CMake smoke project against that installed package. The
matrices cover the Windows x86-64 CUDA 12 and CUDA 13 artifacts and the Windows
ARM64 CPU artifact.

Keeping wheel construction separate from consumer validation ensures a CMake,
Python, or smoke-test failure cannot suppress the wheel artifact. A link or
runtime failure still fails CI. These checks belong to build and packaging
validation, not the core Python test suite, so they add no test under
`warp/tests`. The package-time file check separately protects GitLab wheel
builds.

### Alternatives considered

#### Stable entry-point query

Warp could export one stable function that resolves the rest of the API by
name. Applications could then diagnose missing optional functions and choose
their own compatibility behavior. They would still need function tables and
per-symbol resolution, which is most of the code GH-1888 aims to remove. A
query API can be considered later, after Warp defines its native ABI goals.

#### Versioned DLL filenames

Putting an API or release version in each DLL name would stop an application
from loading a differently named release by accident. It would also require
changes to Python loading, native builds, wheels, examples, deployment, and
downstream build systems. A changed signature behind an unchanged C symbol
would remain incompatible.

#### Continue explicit runtime loading

`LoadLibrary()` and `GetProcAddress()` let applications report their own loader
errors and load `warp-clang` only when needed. The cost is repeated platform
code, function pointer tables, and symbol resolution in every consumer. That
is the code GH-1888 removes.

#### Package a CMake configuration

Official imported targets could hide filenames and usage requirements from
consumers such as Newton. They would also create a broader build-system API
than this fix needs. A CMake package can be evaluated separately once native
linking has real downstream use.

## Testing strategy

Verification covers each affected layer:

1. Rebuild Warp after changing native visibility and linker options.
2. Run targeted formatting and static checks on the changed files.
3. Build Windows x86-64 and ARM64 wheels and check their platform tags.
4. Install each wheel into a clean directory and confirm that Python
   initialization reports matching `warp` and `warp-clang` versions and the
   expected machine architecture.
5. Build and run the CMake smoke consumer against each installed wheel.

The smoke program only calls the two version functions. If the process starts,
Windows found both DLLs and resolved the imports. Matching return values show
that both libraries agree with the headers used to compile the program.
Existing APIC tests continue to cover APIC behavior.
