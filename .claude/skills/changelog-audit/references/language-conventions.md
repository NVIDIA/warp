# CHANGELOG Language Conventions

Loaded by Phase 5. Defines what belongs in CHANGELOG.md and what doesn't, the symbol formatting conventions, the markdown-vs-RST backtick rule, the experimental-feature flag convention, the imperative-mood rule, hyphenation guidance, the GH-link tail position, the internal-jargon flag list, the user-perspective subagent prompt, and worked rewrite examples.

## What belongs in CHANGELOG

- New, removed, deprecated, or changed user-facing APIs (functions, classes, decorators, scalar types, kernel builtins, configuration flags).
- Breaking changes (always flagged with `**Breaking:**`).
- Bug fixes for behavior users would observe.
- New top-level documentation: a brand-new user-guide page, a new example, or a new doctest-driven reference. (A typo fix in an existing page does NOT belong.)
- Performance changes that cross a user-visible threshold (default optimization level, default codegen path, default alignment behavior).
- Platform / Python / dependency support changes (Python 3.9 dropped, new minimum CUDA Toolkit, new minimum driver).

## What does NOT belong

- Test additions, test refactors, test runner improvements (anything with no footprint outside `warp/tests/**` or `warp/_src/tests/**`).
- Internal refactors with no user-observable effect ("Reorganize private helpers", "Move foo from bar to baz", any "tidy up X" with no API delta).
- CI configuration changes (`.gitlab-ci.yml`, `.github/workflows/**`).
- Build-system polish that doesn't change packaging or runtime.
- Single-entry doctest fixes, single-sentence doc reword, error-message wording tweaks (unless the message wording itself is part of a documented contract).

If unsure, default to **drop**. Release notes and `git log` capture the long tail. The CHANGELOG is the curated user-visible manifest.

## Internal-jargon flag list

Flag entry text containing any of:

- **Internal module paths**: `warp._src.*`, anything under the underscore-prefixed package. Also `warp.context.*` (the un-prefixed module is a backwards-compat shim, not the public API).
- **C++/CUDA internal type names**: `launch_bounds_t`, `tile_register_t`, `exec_mode_t`, anything ending `_t` that names a private C++ type.
- **Private identifiers**: anything with a leading underscore (`_compile`, `_resolve`, `Module._foo`).
- **Implementation-only verbs in user-facing prose**: "refactor internal", "reorganize private", "move helper", "split class". A user has no way to act on these.

A flagged entry is **rewritten** if there is a real user-observable effect (e.g., the refactor fixed a sporadic bug, or changed a default). It is **deleted** if there is no user-observable effect.

## Symbol formatting conventions

Apply these consistently when writing or rewriting entries. Inconsistent formatting confuses users skimming the CHANGELOG out of repo context.

- **Always module-qualify Python symbols.** Write `wp.tile_load()`, never bare `tile_load()`. Write `warp.fem.IntegrationOrder`, never bare `IntegrationOrder`. The CHANGELOG is read out of repo context — an unqualified name doesn't tell a user where the symbol lives.
- **Always include parentheses on functions and methods.** Write `wp.tile_dot()`, not `wp.tile_dot`. The parens signal "this is callable" and disambiguate function references from class/type references. Even if the entry doesn't list args, write `()`.
- **Inline a short signature when it adds value.** `wp.tile_load(addr, shape, ..., aligned=True)` is more useful than `wp.tile_load()` when the entry is announcing a new parameter. Don't render a full signature when the entry already names every relevant parameter in prose. Use `...` (literal three dots) to elide unrelated args.
- **Module-relative qualification for submodule symbols.** Write `warp.fem.X` (full dotted path), not `wp.X`. The `wp.` short form is reserved for top-level public API in the `warp` namespace; submodule symbols should be reachable from their full module path.
- **Decorators get the `@` prefix.** Write `@wp.kernel`, `@wp.struct`, `@wp.func`. Never write the bare name (`wp.kernel`) when the symbol is a decorator.
- **Type names are unparenthesized when referencing the type itself.** `wp.array`, `wp.bfloat16`, `wp.float64` (no parens). When the type is being called as a constructor (`wp.array(data, dtype=wp.float32)`), parens are correct because it's a call expression.
- **Consistency check during review.** When a single entry references multiple symbols, all of them follow the same convention. An entry that says "Add `wp.tile_dot()` and improve performance of `tile_load`" should be rewritten to "Add `wp.tile_dot()` and improve performance of `wp.tile_load()`".

These conventions apply to entries you write *or rewrite*. They do NOT mandate retroactively rewriting every existing entry in the upcoming-release section purely for formatting — only fix formatting when an entry is being edited for another reason (Phase 3 verification revision, Phase 5 jargon rewrite). Phase 4's impact sort doesn't touch prose, so it doesn't trigger formatting fixes either. The one exception is the markdown-backtick fix below, which is purely mechanical and applies across the upcoming-release section.

## Markdown backticks, not RST double-backticks

Warp docstrings use RST and require **double** backticks for inline code (per AGENTS.md: ``` ``data`` ```, ``` ``.nvdb`` ```). The CHANGELOG is **markdown**, where **single** backticks (`` `foo` ``) are the convention — double backticks only have a special meaning when the inline code itself contains a literal `` ` ``. Contributors routinely carry the docstring habit into CHANGELOG entries; the rendered output then shows literal backticks around the symbol on github.com/NVIDIA/warp/blob/main/CHANGELOG.md.

Live examples in the current `[Unreleased]` section (as of writing): `` ``wp.tile_scatter_add()`` ``, `` ``atomic`` ``, `` ``True`` ``, `` ``wp.tile_axpy(alpha, src, dest)`` `` — all should be single-backticked.

**Detection rule (purely mechanical, no per-entry prompt):**

- Match `` ``X`` `` in any entry of the upcoming-release section where `X` contains **no** backtick. Rewrite to `` `X` ``.
- Do **NOT** touch `` `` `…` `` `` (a double-backtick run delimiting code that itself contains a single backtick) — that is the legitimate markdown form for code containing a literal `` ` ``.
- Do **NOT** touch fenced code blocks (the triple-backtick fence ``` ``` ```) — different syntax entirely.
- Do **NOT** sweep already-released sections. Same scope rule as the rest of Phase 5: upcoming-release only. A one-shot historical cleanup is the user's call, not the skill's.

This fix does not need user confirmation per occurrence — it is a syntax correction, not a rewrite. Apply across the upcoming-release section in one pass, surface the count in the final report, and let the consolidated diff in Phase 6 carry it.

## Experimental-feature flag convention

Warp marks experimental features with a literal `**Experimental**:` prefix (or `**Experimental:**`) at the start of the entry. The canonical exemplar is the cuBQL BVH backend: `**Experimental**: Add cuBQL BVH backend for \`wp.Mesh\`, selectable via \`bvh_constructor="cubql"\`.` That bold marker is the *only* consistent way to signal "this surface may change before stabilizing" — informal hedging in prose blurs the bar and gets missed by users skimming for stable changes.

**Flag entries that hedge with informal language instead of using the marker.** Hedge words to detect:

- `preliminary` — "preliminary support for X", "preliminary X API"
- `early` / `early access` / `early support`
- `alpha` / `beta` (when qualifying the feature, not as a version number like `1.0.0-beta`)
- `tentative`
- `WIP` / `work-in-progress`
- `draft` (when qualifying a feature, not in the literal sense of unfinished docs)
- `provisional`
- `experimental` (lowercase, or capitalized without the `**…**` bold marker — e.g., `Experimental JAX kernel callback support` or trailing `(experimental)`)

A grep-able starting point is `\b(preliminary|early access|tentative|provisional|WIP|work-in-progress|draft)\b` plus `\bexperimental\b` *not preceded by* `\*\*` — but read each match in context; a "preliminary results" reference inside a perf claim or a `--draft` CLI flag is not a hedge word.

**Rewrite to the canonical form.** Prefix the entry with `**Experimental**:` and drop the hedge word from the prose:

| Original | Result |
|---|---|
| Add preliminary graph capture support for serialization (APIC) and CPU replay. | **Experimental**: Add graph capture support for serialization (APIC) and CPU replay. |
| Experimental JAX kernel callback support | **Experimental**: Add JAX kernel callback support. |
| Add `wp.foo()` (early access) | **Experimental**: Add `wp.foo()`. |

**Confirm with the user before staging the rewrite.** Adding the `**Experimental**:` marker is a public stability signal — it tells users "this may change." Don't apply it unilaterally; show the original and the proposed rewrite and let the user confirm that the feature really is experimental rather than just described with hedge-y prose.

**Do NOT mass-rewrite already-released sections** purely to add the marker. Only apply this rewrite to entries in the upcoming-release section, the same scope as the rest of Phase 5.

## Imperative mood

CHANGELOG entries start with an imperative-mood verb. Consistent voice helps users skim the file as a list of actions taken in this release.

✅ "Add `wp.tile_dot()` for tile dot products."
✅ "Fix crash in `wp.tile_load()` for non-contiguous arrays."
✅ "Switch CPU JIT linker from RTDyld to JITLink."
✅ "Reduce kernel register pressure for low-dimensional launches."

❌ "Added `wp.tile_dot()` ..." — past tense, rewrite to "Add `wp.tile_dot()` ...".
❌ "Adds support for ..." — present indicative, rewrite to "Add support for ..." or "Support ...".
❌ "`wp.foo()` now does X" — declarative, rewrite to "Make `wp.foo()` do X" or "Update `wp.foo()` to do X".
❌ "X is now experimental" — passive, rewrite to "Mark X as experimental".
❌ "X has been deprecated" — passive past, rewrite to "Deprecate X".

Common imperative-mood verbs by section:

- **Added**: Add, Allow, Support, Expose, Re-export, Re-enable.
- **Removed**: Remove, Drop, Retire, Discontinue.
- **Deprecated**: Deprecate, Mark.
- **Changed**: Change, Update, Switch, Replace, Promote, Move, Rename, Reduce, Improve, Restore, Pin, Make, Apply, Require.
- **Fixed**: Fix, Correct, Handle, Resolve, Address.
- **Documentation**: Document, Add, Update, Polish.

For entries that lead with a `**Breaking:**` or `**Experimental**:` marker, the imperative verb follows the marker: "**Breaking:** Change `wp.foo()` ...". The marker counts as a prefix; the imperative-mood rule still applies to the verb that follows it.

## Hyphenation

Apply standard English compound-modifier hyphenation. The CHANGELOG is read by users skimming for impact; a missing or extra hyphen disrupts the read.

**Use a hyphen** when two or more words form a compound modifier *before* a noun:

- "user-facing API"
- "16-byte alignment"
- "thread-block stack"
- "low-dimensional kernels"
- "per-thread cooperative add"
- "non-trivial change" (the `non-` prefix is hyphenated in technical prose)
- "out-of-place factorization", "in-place update"

**Do NOT use a hyphen** when the same words appear in predicate position (after a copular verb like "is" / "are" / "was"):

- "the API is user facing" (no hyphen)
- "the value is 16 bytes" (no hyphen, plural)
- "the operation is out of place" (no hyphen in predicate)

**Do NOT use a hyphen** in established open compounds:

- "machine learning model" (no hyphen, but "machine-learning model" is also acceptable in technical prose; pick one and apply consistently within the section)
- "data structure", "host memory", "device memory"

**Common Warp-specific cases**:

- "GPU-side", "CPU-side", "host-side", "device-side" — hyphenated when used as modifiers.
- "16-bit", "32-bit", "64-bit", "128-bit" — always hyphenated for type widths.
- "JIT-compiled" — hyphenated.
- "ahead-of-time" (compiled) — hyphenated.
- "N-D" / "2-D" / "3-D" — hyphenated when used as a modifier ("N-D tiles").

Judgment cases: when in doubt, follow the dominant pattern in the rest of the upcoming-release section. Do not invent rare hyphenations.

## GH link position

Place every `[GH-NNNN](https://github.com/NVIDIA/warp/issues/NNNN)` link at the **end** of the entry, in parentheses, just before the closing period.

Canonical form (single ref):

> Add `wp.tile_dot()` to compute the dot product of two tiles ([GH-1364](https://github.com/NVIDIA/warp/issues/1364)).

Canonical form (multiple refs):

> Add `wp.tile_axpy(alpha, src, dest)` for fused in-place tile updates ([GH-1363](https://github.com/NVIDIA/warp/issues/1363), [GH-1364](https://github.com/NVIDIA/warp/issues/1364)).

If the link appears mid-sentence in an existing entry, move it to the tail. Preserve the markdown link form (`[GH-NNNN](url)`); never collapse to plain text `GH-NNNN` or to a bare parenthesized number `(GH-NNNN)` without the markdown link. If the entry has no closing period, add one.

For multi-line entries (after Phase 6 line-wrap), the link still goes at the end of the prose, on whichever line ends the entry.

## Worked rewrites

| Original | Action | Result |
|---|---|---|
| Refactor `warp._src.codegen` to share emit path. | Delete | No user-observable effect. |
| Update `launch_bounds_t<N>` template to reduce register pressure. | Rewrite | "Reduce kernel register pressure for low-dimensional launches." |
| Move private `_resolve_dtype` from `warp._src.context` to `warp._src.types`. | Delete | Internal symbol, not exported. |
| Switch CPU JIT linker from RTDyld to JITLink, fixing sporadic access violations. | Keep | Already user-facing — names the symptom, not the implementation. |
| Add helper `_validate_launch_dims` for kernel launch checks. | Delete | Private helper. |
| Move `wp.utils.RmmAllocator` to `wp.utils.AllocatorRmm`. | Keep (this is the rename) | Public symbol; rename is user-visible. |
| Add preliminary graph capture support for serialization (APIC) and CPU replay. | Rewrite | "**Experimental**: Add graph capture support for serialization (APIC) and CPU replay." (hedge word → canonical marker) |

## Carry GH refs through rewrites

When rewriting, preserve every `[GH-NNNN](https://github.com/NVIDIA/warp/issues/NNNN)` link from the original verbatim. A user reading the rewritten entry should still be able to click through to the issue without you having to chase the ref through history.

## User-perspective subagent prompt (canonical)

Dispatch one subagent per ambiguous entry. Use this prompt verbatim:

```
You are a typical Warp user. You write CUDA-style kernels in Python via `@wp.kernel`,
allocate `wp.array`s, and read the CHANGELOG when a new release lands to decide whether
your code is affected. You have not read the Warp source code.

Reading ONLY this entry, with no other context:

<verbatim entry text>

Answer in 3 short bullets:
1. What does this change mean for code I have written today?
2. What words or phrases in this entry are unclear, or assume knowledge I don't have?
3. Does the entry mention internal implementation details that a user shouldn't need to
   understand?

Be honest: if the entry is opaque, say so plainly.
```

Use the response to decide:

- All 3 bullets indicate the user understands and the impact is clear → **keep as-is**.
- Bullet 2 or 3 flags jargon or unclear framing → **rewrite** (you draft; the subagent does NOT draft the rewrite).
- Bullet 1 says "no impact on me / I can't tell" AND nothing else signals user-facing change → **delete**.

Spawn all subagents for a single audit in **one parallel batch** (one message with multiple Agent tool calls). Do not serialize — wall-clock matters and the subagents are independent.

## Judgment philosophy

- **Err on "drop, don't keep"** for entries that don't clearly belong. If someone wants to add a CHANGELOG entry back later, they can; if a non-entry ships, it adds noise to release notes that get scraped from this file.
- **Err on "ask, don't auto-edit"** for non-trivial rewrites. The mechanical filters delete confidently; for rewrites of borderline entries, route through the subagent step.
- **Never invent facts.** If an entry's claim seems wrong (e.g., the GH ref doesn't match the topic), flag it in the final report rather than silently rewriting around it.
- **Keep what is good.** If an entry is already clear, user-facing, and well-scoped, mark it kept and move on. The pass is not a justification for redrafting prose that already works.
