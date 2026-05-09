# Impact Sorting Rubric

Loaded by Phase 4. Use these criteria to score each CHANGELOG entry's user-impact when sorting within a subsection. Sort high → mid → low; within a tier, prefer topic clustering as a soft tie-breaker.

## High impact (top of subsection)

- Brand-new public API that unlocks a workflow not previously possible (a new function family, type, class, decorator).
- A new scalar type, public protocol, or platform / language interop boundary (`wp.bfloat16`, external texture interop, C++ replay of Python-captured graphs).
- A breaking change (entry contains `**Breaking:**`).
- A removal (entry is in `### Removed`).
- A change to a default that affects every user (e.g., default optimization level, default codegen path, default alignment behavior).
- Headline experimental features flagged with `**Experimental**`. The marker softens the stability contract but does not soften the *importance* — a major experimental feature still belongs near the top.
- Platform / Python / dependency support changes (Python 3.9 dropped, new CUDA Toolkit minimum). These are high-impact because they can keep a user from upgrading at all.

## Mid impact

- New parameter or option on an existing API (`fill_mode` on Cholesky, `aligned` on `tile_load`, `module_options` dict on `@wp.kernel`).
- Bug fix for behavior users would hit in normal use.
- Performance improvement on a hot path with a quantitative claim (e.g., "~4x faster shared memory writes").
- New documentation page, example, or user guide section. (A *typo fix* is not mid-impact and likely does not belong in the CHANGELOG at all — see `language-conventions.md`.)
- New public utility / context manager / tracking facility that is optional (off by default) but useful when needed.

## Low impact (bottom of subsection)

- Bug fix for a corner case unlikely to be triggered in normal use.
- Performance tweak with no user-visible threshold change.
- Internal helper newly exposed for advanced users.
- Minor docstring / log-message improvements (most of these belong off the CHANGELOG; if they made it in, push to the bottom).

## Tie-breakers within a tier

Apply in order:

1. **Topic clustering.** If an entry has neighbors above or below at the same impact tier on its topic (e.g., several `wp.tile_*` adds), prefer the placement that keeps the cluster intact.
2. **Specificity.** Entries that name a single symbol read more crisply than topic-style entries. Within a tier, prefer specific over topic-style.

Topic clustering is a soft rule. **Never demote a higher-impact entry to enforce grouping.** If the top of a subsection is a tile feature, the second slot is whatever is second-most-impactful, even if it is a fem feature. The cluster picks up wherever the next tile entry naturally falls.

## Worked example

Before (chronological merge order):

```
### Added
- Add `wp.tile_query_valid()` for tile BVH/mesh AABB query loops [GH-1335].
- Add `wp.bfloat16` scalar type with NumPy + DLPack + Torch + JAX interop [GH-1332].
- Add `aligned` parameter to `tile_load()` and `tile_store()` [GH-1236].
- Add `wp.tile_dot()` for tile dot products [GH-1364].
- Add `wp.handle` scalar for graph-capture mesh remapping [GH-1349].
- Add `module_options` dict parameter to `@wp.kernel` [GH-1250].
```

After:

```
### Added
- Add `wp.bfloat16` scalar type with NumPy + DLPack + Torch + JAX interop [GH-1332].   # high — new scalar type
- Add `wp.handle` scalar for graph-capture mesh remapping [GH-1349].                   # high — new scalar type, clusters with bfloat16
- Add `wp.tile_dot()` for tile dot products [GH-1364].                                 # high — new public API, different topic
- Add `wp.tile_query_valid()` for tile BVH/mesh AABB query loops [GH-1335].            # mid — tile cluster
- Add `aligned` parameter to `tile_load()` and `tile_store()` [GH-1236].               # mid — tile cluster (extends existing tile API)
- Add `module_options` dict parameter to `@wp.kernel` [GH-1250].                       # mid — different topic, lower in tier
```

The two new scalar types cluster at the very top because both are high-impact and on the same topic. `wp.tile_dot` is high-impact but a different topic; it stays third (still high), not demoted to keep the scalars adjacent. The two tile entries cluster in the mid tier because they share a topic; `module_options` ends the subsection.
