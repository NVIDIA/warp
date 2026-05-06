# Feature investigation protocol

Apply this protocol to the lead feature and to every `### …` block planned in
Phase 4. Skipping it produces release notes that read like "what was added" and
miss "what does it actually do, what doesn't work yet, and where can I see it
running." Concrete agent reviews of past Warp releases consistently surfaced
the same gaps; this protocol exists to catch them in the first pass.

For each feature/symbol you plan to write about, do these four passes:

## 1. Locate the implementation

Resolve the named symbol(s) to actual source files at the head ref:

- Python-scope APIs (`wp.X`, `wp.X.Y`): trace re-exports through `warp/__init__.py` to the real module under `warp/_src/`. Use `git show <head-ref>:<path>`.
- Kernel-scope builtins: search `warp/_src/builtins.py` for the `add_builtin("<name>", ...)` registration.
- C++ / native: paths under `warp/native/` (e.g. `warp/native/apic.cu`, `warp/native/mesh.cu`).
- Examples / tests: globs `warp/examples/**`, `warp/tests/**`.

Read the implementation. The CHANGELOG bullet describes intent; the source describes contract.

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

## What not to do

- Do not invent limits. If the source doesn't show one, don't manufacture one.
- Do not fabricate sidecar files. If `save` writes one file, don't claim two.
- Do not link an example you have not opened. If the link doesn't resolve, drop it.
- Do not paraphrase a docstring's "Notes" into a contradiction. When in doubt,
  quote.
