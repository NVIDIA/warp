# Design Documents

This directory contains developer design documents for Warp features.
These are internal reference material explaining the motivation, requirements,
and design choices behind complex features. They are **not** part of the
Sphinx documentation build in `docs/`.

## When to write one

Use your judgment. Good candidates:

- New user-facing APIs or language features
- Architectural changes that affect multiple modules
- Features with non-obvious design trade-offs
- Any change where the "why" isn't obvious from the code alone

Design docs are not required for bug fixes, small tweaks, or mechanical refactors.

Reviewers may request a design doc during PR/MR review as a condition for
approval. This is a normal part of the review process, not an escalation.

## Audience

All contributors, including external. Documents should be self-contained enough
that someone unfamiliar with Warp internals can understand the motivation and
high-level design. Implementation details can reference internal modules but
should briefly explain what they do.

## How to create one

1. Copy `TEMPLATE.md` to a new file named after the feature: `kebab-case-feature-name.md`
2. Fill in the sections that apply. Only **Motivation** and **Design** are required;
   skip or add sections as appropriate.
3. Include the design doc in the same PR/MR as the implementation.

## AI-authored documents

These documents are expected to be produced by AI coding agents (e.g., Claude
Code), not written by hand. Developers should direct their AI agent to generate
the design doc as part of the feature workflow -- the agent has full context of
the codebase and the changes being made, which makes it well-suited to capture
design rationale accurately.

If a design doc becomes outdated (e.g., after a refactor or follow-up changes),
use an AI agent to update it rather than letting it rot. Point the agent at the
doc and the relevant code changes and ask it to reconcile.

## Conventions

- **Flat structure** -- No subdirectories. One file per feature.
- **Naming** -- `kebab-case-feature-name.md`. No date prefixes, no `-srd`/`-sdd` suffixes.
- **Status field** -- Keep the `Status` line in the header block up to date.
- **Issue links** -- Always link to the relevant GitHub issue if one exists.
