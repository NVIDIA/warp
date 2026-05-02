# Report Rendering Rules

Loaded during Phase 6b. These are hard constraints on the generated report — every rule applies to every line of output Claude writes.

## URL shapes

- Full commit URLs: `https://github.com/NVIDIA/warp/commit/<full-sha>`.
- Full issue URLs: `https://github.com/NVIDIA/warp/issues/<num>`.

## Layout

- Preserve `diff` fence blocks for signature diffs.
- **Anomaly banner** appears ONLY if any commit has `main_match_state: "missing"` (not `"ambiguous"`).
- **Table of contents** sits immediately after the Release Highlights section (front matter reads: counts → bake → highlights → TOC → body). Link every top-level `##` section and every per-symbol / per-topic `###` heading under them.
- **Behavioral & Support Changes** section: group by topic with short descriptive titles (e.g., "Anisotropic voxel spacing", "CPU compile performance", "Build requirements"). Claude synthesizes the titles from the entry content.

## Output-style hard constraints

1. **No em dashes (`—`) anywhere in the report output.** Use colons, parentheses, or rewrite the sentence. This includes headings (e.g., use `` `wp.tile_fft`: new parameter `` not `` `wp.tile_fft` — new parameter ``), bullet points, table cells, prose. Check every line before writing.

2. **No internal skill terminology in the output.** The reader does not know what "Phase 4f", "Phase 5b", "tier-1 heuristic", "the language-review pass", or similar skill-internal names refer to. If a section needs explanation of how flags were produced, write it in plain user terms (e.g., "Flagged because commits tagged with this GH ref don't touch any kernel code" instead of "Tier-1 topic mismatch from Phase 5a").

3. **No mention of previously-shipped patch-release fixes.** The commit-list tool already scopes to `<base>..<head>`, so patch-release content is excluded automatically; do not manufacture a comparison to it (e.g., do not write "5 fixes, plus 11 already shipped in 1.12.1 and not re-listed here").

4. **No "end of report" or similar terminal markers.** The last section is the last section. No "— end —", no "Thanks for reading", no concluding paragraph, no closing quote.

5. **Every GH ref is a markdown hyperlink.** Anywhere `GH-NNNN` appears in the generated report — highlight bullets, table cells, section headings, prose paragraphs, CHANGELOG blockquotes (leave those alone if already linked upstream) — it must be a markdown link to `https://github.com/NVIDIA/warp/issues/NNNN`. Plain-text `GH-NNNN` tokens, paren-grouped lists like `(GH-1287, GH-1298, ...)`, and shorthand like `(multiple GHs)` / `(see CHANGELOG)` are NOT acceptable, even when many refs bunch into one bullet or cell. When six GH refs share a row, render six links, not a plain-text list.

6. **Signature and docstring render as a single fenced code block**, shaped like Python source so users see them in one glance. The exact shape depends on the kind of symbol.

   **Functions and methods:**

   ```python
   get_device_allocator(device: DeviceLike) -> Allocator
   """Return the current allocator for the given device.

   Longer description if present in the real docstring.
   """
   ```

   **Classes with only a constructor** (simple data holders or context managers):

   ```python
   class ScopedMemoryTracker:
       """Context manager tracking memory allocations within a scope.

       Longer description if present.
       """

       def __init__(self, label: str, devices: Optional[Sequence[DeviceLike]] = None)
   ```

   **Classes with additional public methods** (interfaces, allocators, trackers with user-facing API beyond init): list EVERY public (non-dunder, non-leading-underscore) method with its signature, plus the class docstring and each method's docstring if present. Do not just show `__init__`.

   ```python
   class Allocator:
       """Base class for custom GPU memory allocators.

       Subclass and implement `alloc` and `free` to plug into Warp's memory
       system via `wp.set_cuda_allocator`.
       """

       def __init__(self, device: DeviceLike)

       def alloc(self, size: int) -> int
       """Allocate `size` bytes; return a raw device pointer."""

       def free(self, ptr: int, size: int) -> None
       """Release a previously-allocated block."""

       def empty_cache(self) -> None
       """Release cached but unused blocks back to the driver."""
   ```

   **Enum / IntEnum / IntFlag classes:** list every member with its integer value and its attribute docstring or comment, plus the class docstring. Do not show a constructor for enums.

   ```python
   class TextureResourceFlags(IntFlag):
       """Flags controlling how a texture resource is bound and accessed."""

       NONE = 0
       """No special flags (default)."""

       COLOR_ATTACHMENT = 1 << 0
       """Texture may be used as a color render target."""

       DEPTH_ATTACHMENT = 1 << 1
       """Texture may be used as a depth render target."""

       STORAGE = 1 << 2
       """Texture may be bound for generic read/write in kernels."""
   ```

   Extract member docstrings from the source using `ast` attribute-docstring form (`"""..."""` immediately following the assignment). If the member uses a `#:` comment or a trailing `#` comment, preserve that instead. If there is no per-member doc, show the member without one.

   **Kernel built-ins** (synthesized Python-style, NOT the `add_builtin()` call):

   ```python
   tile_scatter_add(dest: tile, indices: array(dtype=int), values: tile, *, atomic: bool = True) -> None
   """Per-thread cooperative atomic adds into a shared-memory tile.

   Longer description pulled from the doc= arg of add_builtin(), verbatim.
   """
   ```

   Do NOT separate "Signature" and "Docstring" into two headed subsections. Do NOT blockquote the docstring line by line; it lives inside the code block as Python source.

7. **API summary tables** include a Description column AND a short-form signature in the Symbol cell. The Symbol cell shows the call shape WITHOUT type annotations so readers can skim the args at a glance. Defaults ARE included. The column order for New API tables is: `Symbol | Description | GH | Bake`. Examples of Symbol cells:

   - Function: `wp.print_memory_report(file=None, sort="size", max_items=10)`
   - Function with kwonly: `wp.tile_scatter_add(dest, indices, values, *, atomic=True)`
   - Class with constructor: `wp.ScopedMemoryTracker(label, devices=None)`
   - Enum/flag (no call form): `wp.TextureResourceFlags`
   - Scalar type: `wp.bfloat16`
   - Decorator: `@wp.kernel`

   Description is a short (≤ 10 word) phrase summarizing what the symbol does, pulled from the first sentence of its docstring or the CHANGELOG entry.

8. **New API tables are grouped by Kind.** Inside each scope (Python / Kernel), render one table per Kind (for example: "Functions", "Classes / context managers", "Scalar types", "Enums / flags", "Decorators"). If a scope only has one kind, one table is fine.

9. **Changes-to-Existing-API table columns**: `API | Kind | Breaking | Description | GH | Commits | Bake`. The API cell uses the same short-form call-shape convention as rule 7 (parameter names + defaults, no annotations). Description is a short phrase. Kind values include: `signature change`, `new parameter`, `capability extension`, `removed`, `deprecated`, `semantic change`. For any entry tagged experimental in Phase 4g, the Breaking cell reads `Experimental` rather than `Yes`/`No`, and the Description should include the release where the symbol was introduced as experimental (e.g., "Experimental since 1.12.0: ..."). For any Removed entry, the Description also includes the deprecation-window fact from Phase 4d (e.g., "Deprecated in 1.11.0; removed here.").

10. **Audit appendix rendering is conditional.** If only one of the two audit sections (CHANGELOG-orphan entries / language-review flags) has any content, do not render an "Audit Appendix" umbrella heading. Just render the single non-empty section with its own top-level heading (e.g., `## CHANGELOG Review Notes`). If neither has content, render neither. Only use the "Audit Appendix" umbrella when both are non-empty.

11. **No Phase names anywhere.** If the report needs to explain a flag, write it in user-facing terms. Never write "Phase 4e", "the Phase 5 pass", etc.
