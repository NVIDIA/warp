# Feature investigation protocol

Apply this protocol to the lead feature and to every `### …` block planned in
Phase 4. Skipping it produces release notes that read like "what was added" and
miss "what does it actually do, what doesn't work yet, and where can I see it
running." Concrete agent reviews of past Warp releases consistently surfaced
the same gaps; this protocol exists to catch them in the first pass.

For each feature/symbol you plan to write about, walk through these passes in order:

## 1. Locate the implementation

Resolve the named symbol(s) to actual source files at the head ref:

- Python-scope APIs (`wp.X`, `wp.X.Y`): trace re-exports through `warp/__init__.py` to the real module under `warp/_src/`. Use `git show <head-ref>:<path>`.
- Kernel-scope builtins: search `warp/_src/builtins.py` for the `add_builtin("<name>", ...)` registration.
- C++ / native: paths under `warp/native/` (e.g. `warp/native/apic.cu`, `warp/native/mesh.cu`).
- Examples / tests: globs `warp/examples/**`, `warp/tests/**`.

Read the implementation. The CHANGELOG bullet describes intent; the source describes contract.

## 1.5. Verify symbol references

Before quoting any "Equivalent to", "See also", "Use instead", or other
cross-reference from a docstring, CHANGELOG bullet, or design doc, verify the
referenced symbol actually exists and is reachable from where the user expects.

For each `wp.<symbol>` mentioned in prose you plan to lift:

- Confirm it's a public Python API. Both checks below use already-allowed
  shell commands:
  - Builtin registration: `git show <head-ref>:warp/_src/builtins.py |
    grep -nE 'add_builtin\(\s*"<symbol>"'`. A match means the symbol is
    callable inside `@wp.kernel`.
  - Module re-export: `git show <head-ref>:warp/__init__.py | grep -nE
    '\b<symbol>\b'`. A match means it's also reachable as `warp.<symbol>`
    at Python scope.

If the symbol is C++ only (a `template<...> CUDA_CALLABLE` function in
`warp/native/*.h`), do not reference it in release notes. The user can't call
it from Python. As an example of the failure mode this catches: an earlier
`wp.tile_dot` docstring referenced `wp.tensordot` as an "equivalent form,"
but `wp.tensordot` was never registered as a Python builtin (only the C++
template `tensordot` in `warp/native/quat.h` exists, used internally).
Lifting that string verbatim into the release notes would have invented a
public symbol. The docstring was later corrected; if a similar mismatch
exists today, this pass should catch it.

For each implementation claim ("uses fp32 round-trip", "uses native
intrinsics", "skips runtime checks"):

- Grep `__CUDA_ARCH__` gates in the implementation file. The same operation
  often dispatches to different code paths on different arches (native PTX on
  `sm_90+`, intrinsic emulation on `sm_80`-`sm_89`, fp32 round-trip on older
  arches and CPU).
- Check **both** member-operator overloads (defined inside the struct/class)
  AND standalone overloads (defined at file scope). They can dispatch to
  different paths, so a claim derived from one form of the operator may not
  hold for the other.

For each capability claim about "now supports N-D / multi-X":

- Check what the previous form already supported. "Now supports N-D for
  N >= 2" is often just lifting a rank cap, not adding a wholly new
  capability. The user-facing framing differs ("can now operate on rank-3+
  tiles" vs. "drops the explicit batching loop you used to need").

Skipping this pass produces release notes that make claims that aren't true in
the code as-shipped, even when the source-of-truth doc says them.

## 2. Surface caveats and limits

The single biggest review gap on past releases was "feature ships, but the
limits are buried." Before writing the section, grep the implementation and
nearby files for known patterns:

- `not yet supported`, `not supported`, `unsupported`, `TODO`, `FIXME`, `XXX`
- `experimental`, `unstable`, `subject to change`
- `requires`, `must be`, `currently only`, `assumes`
- Raised exceptions naming the limitation (`raise NotImplementedError`, `raise ValueError("... not supported")`)
- README files and docstring `Notes:` / `Limitations:` sections in the same directory

For every match relevant to the user-facing surface, ask: would a user discover
this only after their code fails? If yes, name the limit in the release notes,
either in prose under the feature description or in a "Known limitations" sub-bullet.

Examples of limits past releases shipped without surfacing:
- `.wrp` graph-capture: Volume and BVH serialization not yet supported (only `wp.Mesh` handles remap).
- `.wrp` graph-capture: CPU `.wrp` load still requires a CUDA-built Warp library.
- `.wrp` graph-capture: saved CUBINs are pinned to one compute capability.
- `cuBQL` BVH backend: only supports `mesh_query_ray`; point/AABB/winding queries unimplemented.

## 3. Identify artifacts beyond the obvious file

For features that produce a new on-disk artifact, wire format, cross-process
boundary, or any other persistent thing, do not stop at the headline file. Read
the save / write / serialize code and answer:

- **Sidecar files or directories.** Does writing `name.wrp` also create a
  `name_modules/` directory? A `.meta` file? A lock file? Users shipping the
  artifact need to know what to ship.
- **Architecture or version pinning.** Is the artifact tied to one compute
  capability, one Warp version, one Python version? If yes, say so.
- **Reverse direction.** What does *load* require that *save* didn't? CUDA
  context? `wp_init` ordering? A specific build flavor?

State this in the section either as a short "What gets written" prose
paragraph or as a literal directory-tree code block. Do not let the user infer.

## 4. Find existing in-tree examples to link

For every lead feature and every `### …` block describing a substantial
capability, search the repo for working examples to link. Past reviews have
flagged "you ship a real C++ example for this and don't even mention it":

```bash
# Cross-language features (C++ from Python)
git ls-tree -r --name-only <head-ref> -- 'warp/examples/cpp/' | grep -i '<feature-keyword>'

# Python examples (kernels, demos)
git ls-tree -r --name-only <head-ref> -- 'warp/examples/' | grep -i '<feature-keyword>'

# Tests as a usage reference
git ls-tree -r --name-only <head-ref> -- 'warp/tests/' | grep -i '<feature-keyword>'
```

If a matching example exists, link it as a relative path on the tag URL:
`[example_name](https://github.com/NVIDIA/warp/blob/v<version>/warp/examples/<path>)`.
Never link by SHA; never link by `main` (the file may move). Link by the tag
that will exist when the release is published.

If no example exists but the feature has reference apps shipping with it
(typical for cross-language features), name them prominently. The user reading
the release notes should never have to grep the repo themselves to find a
working starting point.

## 5. Cross-language requirements

Any feature whose claim crosses a language boundary (Python → C++, Python →
JAX, Python → standalone runtime) needs a code example **on each side**, not
just Python. The pattern that fails reviews: "exposes from C++" with one Python
snippet and zero C++.

When the feature crosses boundaries:
- Show the Python authoring side (one snippet, working).
- Show the consumer-side snippet (C++ headers, function calls; JAX `jax_kernel`
  binding; etc.). Distill from the in-tree example if one exists.
- Name the consumer-side header file or import path explicitly.
- State runtime prerequisites the consumer side has (CUDA context, JAX 0.8+,
  Python interpreter required or not).

### When the consumer-side example must elide boilerplate

The full consumer-side example is sometimes too long to embed (50+ lines:
walking module directories, loading object files, registering kernel symbols,
setting up a CUDA context). When it is, do **all** of the following:

1. **Verify each API call signature shown matches the relevant header.** Grep
   `warp/native/<feature>.h` for every named function in the snippet
   (`wp_apic_load_graph`, `wp_apic_set_param`, etc.) and confirm the parameter
   list and types match what the snippet shows. The release notes' code is
   the only contact most readers will have with the API; signature drift
   silently breaks copy-paste.
2. **State the elision explicitly in prose.** Don't let a reader assume the
   snippet is runnable as-is. Example:

   > The full example also walks the `_modules/` directory, loads each `.o`
   > via `wp_load_obj`, resolves kernel symbols, and registers them with
   > `wp_apic_register_loaded_cpu_kernel` before the first replay. The
   > snippet below elides that boilerplate.

3. **Add an inline placeholder comment in the code** where the elided step
   would go, so a reader scanning only the code block still sees the gap:

   ```cpp
   // (Walk demo_modules/, load each .o, and register kernels. See linked example.)
   ```

4. **Link to the full in-tree example with the release tag URL:**
   `[<example-path>](https://github.com/NVIDIA/warp/blob/v<version>/<path>)`.
   Never link by `main` (the file may move); never link by SHA.

Do not silently elide. A reader who copy-pastes the snippet must know they're
getting an API-shape sketch, not a runnable program.

## 6. Choose caveats depth based on docs

For each `### …` block describing an experimental feature or one with
significant gaps, decide where the limitations live:

1. **Check whether the linked Sphinx docs already enumerate limitations
   comprehensively.** Look at `docs/user_guide/*.rst` and
   `docs/api_reference/*.rst` for the linked feature. Search for
   headings/admonitions like "Current limitations", "Known limitations",
   "Caveats", "Notes:".
2. **If the docs enumerate comprehensively:** summarize 1-2 of the most
   critical user-impact caveats inline (the ones a user would hit on first
   use), and link to the docs section for the full list. This avoids
   duplicating a long enumeration that the docs already maintain.
3. **If the docs do NOT enumerate clearly:** enumerate the full set of
   caveats inline in the release notes, structured as a labeled bullet list
   under `**Known limitations:**`.

Skipping caveats for an experimental feature is a review-fail pattern: readers
discover the limit only after their code breaks. But duplicating a
comprehensive doc enumeration in the release notes is also a problem: it
bloats the section and creates two sources of truth that can drift between
the release-notes draft and the next docs update.

## What not to do

- Do not invent limits. If the source doesn't show one, don't manufacture one.
- Do not fabricate sidecar files. If `save` writes one file, don't claim two.
- Do not link an example you have not opened. If the link doesn't resolve, drop it.
- Do not paraphrase a docstring's "Notes" into a contradiction. When in doubt,
  quote.
