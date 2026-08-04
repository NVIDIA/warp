#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a Warp evaluation report directory.

Checks the properties that catch real defects, not table bookkeeping:

  * structure     — visible schema declaration, required sections, sibling directories
  * mapping       — every census ID resolves to a section, and vice versa
  * one number    — a bottleneck's baseline figures are identical in the census
                    row and in its own table (two numbers for one fact is a bug)
  * ratios        — only the readable vocabulary; bare-multiplier forms rejected
  * placeholders  — no unfilled `<...>`, no empty cells, no invented tokens
  * artifacts     — every linked patch, script and raw result exists
  * integrity     — the discarded-measurements line is present

Usage:
    uv run python scripts/validate_report_schema.py <report-directory>
Arguments:
    report-directory containing warp-evaluation-report.md and its artifacts.
Output:
    One success line on stdout; validation diagnostics on stderr.
Exit codes:
    0 for a conforming report, 1 for handled validation failures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA_DECLARATION = "**Report schema:** `warp-evaluation/v1`"
REPORT_FILENAME = "warp-evaluation-report.md"
REQUIRED_DIRECTORIES = ("solutions", "benchmarks", "results")

REQUIRED_SECTIONS = (
    "## Where the time goes",
    "## Caveats that bound these numbers",
    "## Environment and reproduction",
)

MISSING_TOKENS = {
    "not measured",
    "not available",
    "no representative data",
    "unknown",
    "n/a",
}

MULTIPLICATION_SIGN = "\N{MULTIPLICATION SIGN}"

# `2,100x faster` / `1.2x slower` / `17x less` / `2.0x more` / `same` / em-dash
RATIO_RE = re.compile(
    rf"^(?:—|-|same|[\d,]+(?:\.\d+)?{MULTIPLICATION_SIGN}\s+(?:faster|slower|less|more))$",
    re.I,
)
# the form the template bans: a bare multiplier such as `0.0012x` or `1.00x`
BARE_MULTIPLIER_RE = re.compile(rf"^[\d,]+(?:\.\d+)?{MULTIPLICATION_SIGN}$")

PLACEHOLDER_RE = re.compile(r"<[a-z][^>]{2,}>")
ID_RE = re.compile(r"^B\d+$")
EVALUATION_STATE_PREFIX = "**Evaluation state:**"
EVALUATION_STATES = frozenset({"complete", "aborted", "incomplete"})


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|") and "---" not in line


def strip_md(text: str) -> str:
    """Drop emphasis and code ticks so cell values compare cleanly."""
    return text.replace("**", "").replace("`", "").replace("*", "").strip()


def parse_evaluation_states(text: str) -> list[tuple[str, str]]:
    """Return valid evaluation-state declarations without regex backtracking."""
    matches = []
    prefix = EVALUATION_STATE_PREFIX.casefold()
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate.casefold().startswith(prefix):
            continue
        declaration = candidate[len(EVALUATION_STATE_PREFIX) :].strip()
        state_text, separator, detail = declaration.partition("—")
        state = state_text.strip().casefold()
        if state in EVALUATION_STATES:
            matches.append((state, detail.strip() if separator else ""))
    return matches


def section_lines(lines: list[str], heading_re: str):
    """Yield the lines belonging to one `## ` section."""
    inside = False
    for line in lines:
        if re.match(heading_re, line):
            inside = True
            continue
        if inside and line.startswith("## "):
            return
        if inside:
            yield line


def _has_measured_record(results_dir: Path) -> bool:
    """True if results/ holds at least one record emitted by measure.py."""
    for path in sorted(results_dir.rglob("*")):
        if path.suffix not in (".json", ".jsonl") or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if path.suffix == ".json":
            try:
                record = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "env" in record:
                return True
            continue
        for raw_line in content.splitlines():
            raw = raw_line.strip()
            if not raw.startswith("{"):
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "env" in record:
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    root: Path = args.directory
    errors: list[str] = []

    report = root / REPORT_FILENAME
    if not report.is_file():
        print(f"ERROR: missing {REPORT_FILENAME} in {root}", file=sys.stderr)
        return 1
    text = report.read_text(encoding="utf-8")
    lines = text.split("\n")

    # ---------------------------------------------------------------- structure
    if text.count(SCHEMA_DECLARATION) != 1:
        errors.append(f"expected exactly one visible schema declaration: {SCHEMA_DECLARATION}")
    if "<!--" in text or "-->" in text:
        errors.append("HTML comments are not allowed; use visible report text")
    for directory in REQUIRED_DIRECTORIES:
        if not (root / directory).is_dir():
            errors.append(f"missing required directory: {directory}/")
    for section in REQUIRED_SECTIONS:
        if section not in text:
            errors.append(f"missing required section: {section}")
    if "Delete these quoted instruction lines" in text:
        errors.append("template instruction block was not deleted before delivery")
    state_matches = parse_evaluation_states(text)
    evaluation_state = None
    if len(state_matches) != 1:
        errors.append(
            "expected exactly one '**Evaluation state:** complete / aborted — <gate and scope> / "
            "incomplete — <missing evidence and scope>' line"
        )
    else:
        evaluation_state, state_detail = state_matches[0]
        evaluation_state = evaluation_state.lower()
        if evaluation_state == "aborted" and not state_detail.strip():
            errors.append("aborted evaluation state must name the gate and affected scope")
        if evaluation_state == "incomplete" and not state_detail.strip():
            errors.append("incomplete evaluation state must name the missing evidence and affected scope")

    # ------------------------------------------------------------------ mapping
    section_ids: set[str] = set()
    for match in re.finditer(r"^## (B\d+) — ", text, re.M):
        section_id = match.group(1)
        if section_id in section_ids:
            errors.append(f"duplicate section ID {section_id}")
        section_ids.add(section_id)
    census_rows: dict[str, list[str]] = {}
    for line in section_lines(lines, r"^## Where the time goes"):
        if is_table_row(line):
            row = cells(line)
            if row and ID_RE.match(strip_md(row[0])):
                census_id = strip_md(row[0])
                if census_id in census_rows:
                    errors.append(f"duplicate census row {census_id}")
                    continue
                census_rows[census_id] = row

    if not census_rows:
        errors.append("census table has no B<n> rows — nothing maps to a section")
    for cid in census_rows:
        if cid not in section_ids:
            errors.append(f"census row {cid} has no matching '## {cid} — ' section")
    for sid in sorted(section_ids - set(census_rows)):
        errors.append(f"section {sid} has no row in the census table")

    # --------------------------------------------------- one number per fact
    for cid, row in sorted(census_rows.items()):
        if len(row) < 4:
            errors.append(f"census row {cid} has too few columns")
            continue
        census_time, census_mem = strip_md(row[2]), strip_md(row[3])
        baseline = None
        for line in section_lines(lines, rf"^## {cid} — "):
            if is_table_row(line):
                candidate = cells(line)
                if candidate and strip_md(candidate[0]).lower().startswith("baseline"):
                    baseline = candidate
                    break
        if baseline is None:
            errors.append(f"{cid}: no row whose first cell starts with 'Baseline'")
            continue
        if len(baseline) >= 4:
            if strip_md(baseline[1]) != census_time:
                errors.append(
                    f"{cid}: census time {census_time!r} != baseline time "
                    f"{strip_md(baseline[1])!r} — report one measurement, not two"
                )
            if strip_md(baseline[3]) != census_mem:
                errors.append(
                    f"{cid}: census peak mem {census_mem!r} != baseline "
                    f"{strip_md(baseline[3])!r} — report one measurement, not two"
                )

    # ------------------------------------------------------- solution tables
    for cid in sorted(section_ids):
        seen = 0
        baseline_first = False
        warp_row = None
        for line in section_lines(lines, rf"^## {cid} — "):
            if line.startswith("### "):
                break  # regime sub-table has different columns
            if not is_table_row(line):
                continue
            row = cells(line)
            if len(row) < 6 or strip_md(row[0]).lower() in ("solution", ""):
                continue
            seen += 1
            if seen == 1:
                baseline_first = strip_md(row[0]).lower().startswith("baseline")
            if strip_md(row[0]).lower().startswith("warp"):
                warp_row = row
            for idx, label in ((2, "vs baseline (time)"), (4, "vs baseline (mem)")):
                value = strip_md(row[idx])
                if value.lower() in MISSING_TOKENS or value == "":
                    continue
                if BARE_MULTIPLIER_RE.match(value):
                    errors.append(
                        f"{cid}: {label} {value!r} — use "
                        f"'N{MULTIPLICATION_SIGN} faster' / "
                        f"'N{MULTIPLICATION_SIGN} less' / 'same', "
                        "not a bare multiplier"
                    )
                elif not RATIO_RE.match(value):
                    errors.append(f"{cid}: {label} {value!r} is not a valid ratio")
            for idx, value in enumerate(row):
                if value == "":
                    errors.append(f"{cid}: empty cell in column {idx + 1} — use a missing-evidence token")
        if seen and not baseline_first:
            errors.append(f"{cid}: first solution row must be the baseline")
        if evaluation_state == "complete":
            if warp_row is None:
                errors.append(f"{cid}: complete evaluation has no Warp row")
            elif strip_md(warp_row[1]).lower() in MISSING_TOKENS or not strip_md(warp_row[1]):
                errors.append(f"{cid}: complete evaluation has no measured Warp time")

    # ------------------------------------------------------------- artifacts
    for rel in sorted(set(re.findall(r"`(solutions/[^`]+)`", text))):
        if not rel.endswith(".patch"):
            errors.append(f"solution must be a .patch file: {rel}")
        elif not (root / rel).is_file():
            errors.append(f"linked solution does not exist: {rel}")
    for rel in sorted(set(re.findall(r"`(results/[^`]+)`", text))):
        if not (root / rel).is_file():
            errors.append(f"linked raw result does not exist: {rel}")
    for rel in sorted(set(re.findall(r"`(benchmarks/[^`]+)`", text))):
        if not (root / rel).exists():
            errors.append(f"linked benchmark does not exist: {rel}")
    # ----------------------------------------------------- measured provenance
    # A directive to use measure.py is only as strong as its enforcement: prose
    # cannot stop an agent hand-rolling a harness that skips synchronization or
    # reads an inherited ru_maxrss. Every record measure.py emits carries an
    # `env` fingerprint, so require that results/ actually contains some.
    results_dir = root / "results"
    if results_dir.is_dir() and not _has_measured_record(results_dir):
        errors.append(
            "results/ has no record emitted by scripts/measure.py (a JSON object "
            "carrying an 'env' fingerprint) — time and memory must be measured "
            "through it rather than a hand-rolled harness; record any deviation "
            "under 'Measurements discarded, and why'"
        )

    # -------------------------------------------------------------- integrity
    if "Measurements discarded" not in text:
        errors.append(
            "caveats must include a 'Measurements discarded, and why' line (write 'none' if nothing was discarded)"
        )

    # ------------------------------------------------------------ placeholders
    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith(">"):
            continue
        for match in PLACEHOLDER_RE.finditer(line):
            errors.append(f"line {n}: unfilled placeholder {match.group(0)}")

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"OK: {root} conforms to warp-evaluation/v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
