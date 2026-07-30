#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a Warp adoption report directory and its linked artifacts."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCHEMA_MARKER = "<!-- warp-adoption-report-schema: 2 -->"
SCHEMA_LABEL = "**Report schema:** `warp-adoption/v2`"
REPORT_FILENAME = "warp-adoption-report.md"
REQUIRED_DIRECTORIES = ("solutions", "benchmarks", "results")
DEPTH_VALUES = {
    "full",
    "profile-only",
    "correctness-only",
    "inconclusive-minimal",
}
SUMMARY_DERIVATION_VALUES = {
    "single-decision",
    "unanimous-decisions",
    "dominant-regime",
    "mixed-decisions",
}
ENVIRONMENT_ITEMS = [
    "Repository / commit",
    "Warp version (project pin)",
    "Framework versions",
    "GPU / driver / CUDA",
    "OS / Python",
    "Cache state during measurement",
    "**Execution regimes present in the product**",
    "Device memory budget assumed",
    "Device lock held for sweeps",
    "Representative dataset",
]
PRE_REGISTRATION_FIELDS = [
    "Workload provenance",
    "State variable and production range",
    "Candidate tuning knobs",
    "Baseline run-to-run spread",
    "Oracle plan and both-sides scoring",
]
INTEGRATION_DIMENSIONS = [
    "API seam and ownership model",
    "Cache / structure lifetime and invalidation",
    "Streams and synchronization contract",
    "Concurrency and thread/process policy",
    "Graph capture and retention requirements",
    "Fallback path (device, dtype, shape, size)",
    "Optional dependency and lazy import",
    "Kernel cache in deployment (writable FS, prewarming)",
    "Version qualification matrix",
    "Who owns the second implementation and its tests",
    "Dependency friction encountered",
]

SECTIONS = [
    "1. Scope, requested outcomes, and non-goals",
    "2. Executive verdict",
    "3. Assumptions and missing information",
    "4. Environment, target workload, and evidence provenance",
    "5. Baseline profile and strongest alternatives",
    "6. Candidate matrix",
    "7. Correctness and semantic-contract results",
    "8. Reproducible experiment and benchmark method",
    "9. Optimization experiments and runtime results",
    "10. Memory results",
    "11. Integration, lifecycle, portability, and maintenance costs",
    "12. Decisions by candidate, narrow seam, and regime",
    "13. Smallest next steps requiring user review",
    "14. Evidence and raw artifacts",
]

TABLES = {
    "executive-summary": [
        "Candidate",
        "Seam",
        "Regime",
        "Verdict",
        "Recommended action",
        "Decisive evidence",
    ],
    "assumptions-gaps": [
        "#",
        "Assumption or gap",
        "Why it matters",
        "How to resolve",
    ],
    "environment": ["Item", "Value"],
    "entry-point-surface": [
        "Entry point / variant",
        "Regime",
        "Status",
        "Workload",
        "Gate / reason",
        "Evidence",
    ],
    "pre-registration": [
        "Field",
        "Pre-registered value",
        "Status / evidence",
    ],
    "entry-point-census": [
        "Entry point / variant",
        "Regime",
        "Workload",
        "Host time (ms)",
        "Device time (ms)",
        "Peak memory (MiB; domain)",
        "Status / gate",
        "Evidence",
    ],
    "pipeline-composition": [
        "Pipeline",
        "Regime",
        "Workload",
        "Stage",
        "Time (ms)",
        "Share of total",
        "Peak memory (MiB; domain)",
        "Evidence",
    ],
    "candidate-stage-profile": [
        "Stage",
        "Time (ms)",
        "Share of total",
        "Peak memory (MiB; domain)",
        "Evidence",
    ],
    "alternatives": [
        "Candidate",
        "Alternative",
        "Applicable?",
        "Outcome",
        "Evidence",
    ],
    "candidate-matrix": [
        "ID",
        "Candidate",
        "Source location",
        "Share of profile",
        "Objective / gate metric",
        "Proposed seam",
        "Status",
        "Notes",
    ],
    "correctness-results": ["Candidate", "Check", "Result", "Evidence"],
    "optimization-ledger": [
        "ID",
        "Candidate",
        "Optimization / implementation",
        "Workload / regime",
        "Timed boundary",
        "Correctness",
        "Time (ms)",
        "Peak memory (MiB; domain)",
        "Relative to baseline",
        "Decision",
        "Solution type",
        "Solution diff",
        "Benchmark entry point",
        "Raw result",
        "Decisive note",
    ],
    "cold-costs": [
        "Experiment",
        "Implementation",
        "Phase",
        "Cache state",
        "Time (ms)",
        "Charged in regime?",
        "Evidence",
    ],
    "repetition-totals": [
        "Workload / regime",
        "Implementation",
        "1 call (ms)",
        "10 calls (ms)",
        "100 calls (ms)",
        "Measurement status",
        "Evidence",
    ],
    "crossovers": [
        "Candidate",
        "Axis",
        "Crossover bracket",
        "Production distribution sits",
        "Evidence",
    ],
    "memory-results": [
        "Experiment",
        "Candidate",
        "Workload",
        "Implementation",
        "Domain",
        "Peak memory (MiB)",
        "Delta vs baseline (MiB)",
        "How measured",
        "Evidence",
    ],
    "integration-costs": ["Dimension", "Finding"],
    "seam-decisions": [
        "Candidate",
        "Seam",
        "Regime",
        "Verdict",
        "Approved envelope",
        "Dispatch condition",
        "Fallback",
        "Recommended action",
        "Decisive evidence",
    ],
    "evidence-artifacts": ["Claim", "Artifact", "Location"],
}

PLACEHOLDER_RE = re.compile(r"(?<!<)<[A-Za-z][^>\n]{0,160}>")
SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
# The en and em dashes are the placeholder glyphs this check looks for in
# report cells, so the ambiguous-character lint does not apply here.
MISSING_SHORTHANDS = {"-", "–", "—", "tbd", "todo"}  # noqa: RUF001
MISSING_VALUES = {
    "not measured",
    "not available",
    "no representative data",
    "unknown",
    "n/a",
}
CORRECTNESS_VALUES = {"PASS", "FAIL", "PARTIAL", "NOT TESTED", "N/A"}
VERDICT_VALUES = {"GO", "CONDITIONAL GO", "INCONCLUSIVE", "NO-GO"}
SUMMARY_VALUES = VERDICT_VALUES | {"MIXED"}
CANDIDATE_STATUS_VALUES = {"rejected", "hypothesis", "measured", "recommended"}
SOLUTION_TYPE_VALUES = {"baseline", "code", "configuration"}
DECISION_VALUES = {
    "BASELINE",
    "CARRY FORWARD",
    "RECOMMEND",
    "REJECT",
    "INCONCLUSIVE",
    "EXCLUDE INVALID",
}


def table_cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    return [cell.replace(r"\|", "|").strip() for cell in re.split(r"(?<!\\)\|", stripped[1:-1])]


def next_nonempty(lines: list[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].strip():
            return index
    return None


def plain(cell: str) -> str:
    return cell.strip().strip("`*").strip()


def prose_without_code(text: str) -> str:
    prose: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            prose.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(prose)


def is_missing(value: str) -> bool:
    return plain(value).lower() in MISSING_VALUES


def validate_artifact(
    root: Path,
    value: str,
    expected_directory: str,
    label: str,
    suffix: str | None = None,
) -> tuple[Path | None, str | None]:
    artifact = Path(plain(value))
    if artifact.is_absolute() or ".." in artifact.parts:
        return None, f"{label}: path must be relative and stay inside the report directory"
    if not artifact.parts or artifact.parts[0] != expected_directory:
        return None, f"{label}: path must be under {expected_directory}/"
    if suffix is not None and artifact.suffix != suffix:
        return None, f"{label}: expected a {suffix} file"

    full_path = root / artifact
    if not full_path.is_file():
        return None, f"{label}: file does not exist: {artifact}"
    try:
        full_path.resolve().relative_to(root.resolve())
    except ValueError:
        return None, f"{label}: resolved path leaves the report directory"
    return full_path, None


def validate(path: Path, template: bool) -> list[str]:
    errors: list[str] = []
    report_root: Path | None = None

    if template:
        report_path = path
    else:
        if not path.is_dir():
            return [f"expected a report directory, found: {path}"]
        report_root = path
        report_path = path / REPORT_FILENAME
        for directory in REQUIRED_DIRECTORIES:
            artifact_directory = report_root / directory
            if not artifact_directory.is_dir():
                errors.append(f"missing required directory: {directory}/")
        if not (report_root / "benchmarks" / "README.md").is_file():
            errors.append("missing required file: benchmarks/README.md")

    text = report_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    table_rows: dict[str, list[list[str]]] = {}

    if text.count(SCHEMA_MARKER) != 1:
        errors.append(f"expected exactly one schema marker: {SCHEMA_MARKER}")
    if text.count(SCHEMA_LABEL) != 1:
        errors.append(f"expected exactly one schema label: {SCHEMA_LABEL}")
    authorization_lines = [line for line in lines if line.startswith("**Assessment authorization:**")]
    if len(authorization_lines) != 1:
        errors.append("expected exactly one assessment authorization record")

    actual_sections = [line[3:].strip() for line in lines if line.startswith("## ")]
    if actual_sections != SECTIONS:
        errors.append(
            f"numbered sections differ from warp-adoption/v2\n  expected: {SECTIONS}\n  actual:   {actual_sections}"
        )

    for name, expected_header in TABLES.items():
        marker = f"<!-- required-table: {name} -->"
        positions = [index for index, line in enumerate(lines) if line.strip() == marker]
        if len(positions) != 1:
            errors.append(f"expected exactly one required-table marker: {name}")
            continue

        header_index = next_nonempty(lines, positions[0] + 1)
        if header_index is None:
            errors.append(f"{name}: missing table header")
            continue

        actual_header = table_cells(lines[header_index])
        if actual_header != expected_header:
            errors.append(f"{name}: columns differ\n  expected: {expected_header}\n  actual:   {actual_header}")
            continue

        separator_index = next_nonempty(lines, header_index + 1)
        separator = table_cells(lines[separator_index]) if separator_index is not None else None
        if (
            separator is None
            or len(separator) != len(expected_header)
            or any(SEPARATOR_RE.fullmatch(cell) is None for cell in separator)
        ):
            errors.append(f"{name}: invalid Markdown table separator")
            continue

        row_count = 0
        rows: list[list[str]] = []
        row_index = separator_index + 1
        while row_index < len(lines):
            cells = table_cells(lines[row_index])
            if cells is None:
                break
            row_count += 1
            if len(cells) != len(expected_header):
                errors.append(f"{name}:{row_index + 1}: expected {len(expected_header)} cells, found {len(cells)}")
            else:
                rows.append(cells)
                if not template and any(not cell for cell in cells):
                    errors.append(f"{name}:{row_index + 1}: empty cells must use a standard missing-evidence token")
                if not template:
                    for cell in cells:
                        if plain(cell).lower() in MISSING_SHORTHANDS:
                            errors.append(
                                f"{name}:{row_index + 1}: use a standard missing-evidence token instead of {cell!r}"
                            )
            row_index += 1

        if row_count == 0:
            errors.append(f"{name}: required table has no data rows")
        table_rows[name] = rows

    for table_name, required_first_column in [
        ("environment", ENVIRONMENT_ITEMS),
        ("pre-registration", PRE_REGISTRATION_FIELDS),
        ("integration-costs", INTEGRATION_DIMENSIONS),
    ]:
        actual_first_column = [row[0] for row in table_rows.get(table_name, []) if row]
        if actual_first_column != required_first_column:
            errors.append(
                f"{table_name}: required rows differ\n"
                f"  expected: {required_first_column}\n"
                f"  actual:   {actual_first_column}"
            )

    if not template:
        placeholders = PLACEHOLDER_RE.findall(prose_without_code(text))
        if placeholders:
            preview = ", ".join(sorted(set(placeholders))[:5])
            errors.append(f"unfilled template placeholders remain: {preview}")

        candidate_rows = table_rows.get("candidate-matrix", [])
        candidate_ids = [plain(row[0]) for row in candidate_rows]
        if len(candidate_ids) != len(set(candidate_ids)):
            errors.append("candidate-matrix: candidate IDs must be unique")
        for candidate_id in candidate_ids:
            if re.fullmatch(r"C[1-9]\d*", candidate_id) is None:
                errors.append(f"candidate-matrix: invalid candidate ID {candidate_id!r}")
        for row in candidate_rows:
            status = plain(row[6])
            if status not in CANDIDATE_STATUS_VALUES:
                errors.append(
                    f"candidate-matrix: Status must be one of {sorted(CANDIDATE_STATUS_VALUES)}, found {status!r}"
                )

        ledger_rows = table_rows.get("optimization-ledger", [])
        experiment_ids = [plain(row[0]) for row in ledger_rows]
        if len(experiment_ids) != len(set(experiment_ids)):
            errors.append("optimization-ledger: experiment IDs must be unique")
        for row in ledger_rows:
            experiment_id, candidate_id = plain(row[0]), plain(row[1])
            correctness, decision = plain(row[5]), plain(row[9])
            solution_type = plain(row[10])
            if re.fullmatch(r"E[1-9]\d*", experiment_id) is None:
                errors.append(f"optimization-ledger: invalid experiment ID {experiment_id!r}")
            if candidate_id not in candidate_ids:
                errors.append(f"optimization-ledger: unknown candidate ID {candidate_id!r}")
            if correctness not in CORRECTNESS_VALUES:
                errors.append(
                    "optimization-ledger: Correctness must be one of "
                    f"{sorted(CORRECTNESS_VALUES)}, found {correctness!r}"
                )
            if decision not in DECISION_VALUES:
                errors.append(
                    f"optimization-ledger: Decision must be one of {sorted(DECISION_VALUES)}, found {decision!r}"
                )
            if solution_type not in SOLUTION_TYPE_VALUES:
                errors.append(
                    "optimization-ledger: Solution type must be one of "
                    f"{sorted(SOLUTION_TYPE_VALUES)}, found {solution_type!r}"
                )
            if (decision == "BASELINE") != (solution_type == "baseline"):
                errors.append(
                    f"optimization-ledger {experiment_id}: BASELINE decision "
                    "and baseline Solution type must occur together"
                )
            if decision == "RECOMMEND":
                if correctness != "PASS":
                    errors.append("optimization-ledger: RECOMMEND requires correctness PASS")
                for column, label in [(6, "Time"), (7, "Peak memory")]:
                    if plain(row[column]).lower() in MISSING_VALUES:
                        errors.append(f"optimization-ledger: RECOMMEND requires measured {label.lower()}")

            if report_root is not None:
                diff_value = plain(row[11])
                benchmark_value = plain(row[12])
                result_value = plain(row[13])
                timed = not is_missing(row[6])

                if solution_type in {"baseline", "configuration"}:
                    if diff_value.lower() != "n/a":
                        errors.append(
                            f"optimization-ledger {experiment_id}: "
                            f"{solution_type} solution must use n/a for "
                            "Solution diff"
                        )
                elif solution_type == "code":
                    if is_missing(diff_value):
                        errors.append(
                            f"optimization-ledger {experiment_id}: code solution requires an independent diff"
                        )
                    else:
                        diff_path, error = validate_artifact(
                            report_root,
                            diff_value,
                            "solutions",
                            f"optimization-ledger {experiment_id} Solution diff",
                            ".patch",
                        )
                        if error is not None:
                            errors.append(error)
                        elif diff_path is not None:
                            patch_text = diff_path.read_text(encoding="utf-8", errors="replace")
                            if "diff --git " not in patch_text:
                                errors.append(
                                    f"optimization-ledger {experiment_id}: solution patch is not a unified git diff"
                                )

                for value, directory, label in [
                    (benchmark_value, "benchmarks", "Benchmark entry point"),
                    (result_value, "results", "Raw result"),
                ]:
                    if is_missing(value):
                        if timed:
                            errors.append(f"optimization-ledger {experiment_id}: timed row requires {label}")
                        continue
                    _, error = validate_artifact(
                        report_root,
                        value,
                        directory,
                        f"optimization-ledger {experiment_id} {label}",
                    )
                    if error is not None:
                        errors.append(error)

        for table_name, candidate_column in [
            ("executive-summary", 0),
            ("alternatives", 0),
            ("correctness-results", 0),
            ("crossovers", 0),
            ("memory-results", 1),
            ("seam-decisions", 0),
        ]:
            for row in table_rows.get(table_name, []):
                candidate_id = plain(row[candidate_column])
                if candidate_id not in candidate_ids:
                    errors.append(f"{table_name}: unknown candidate ID {candidate_id!r}")

        for table_name, experiment_column in [
            ("cold-costs", 0),
            ("memory-results", 0),
        ]:
            for row in table_rows.get(table_name, []):
                experiment_id = plain(row[experiment_column])
                if experiment_id not in experiment_ids:
                    errors.append(f"{table_name}: unknown experiment ID {experiment_id!r}")

        for table_name, verdict_column in [
            ("executive-summary", 3),
            ("seam-decisions", 3),
        ]:
            for row in table_rows.get(table_name, []):
                verdict = plain(row[verdict_column])
                if verdict not in VERDICT_VALUES:
                    errors.append(f"{table_name}: Verdict must be one of {sorted(VERDICT_VALUES)}, found {verdict!r}")

        summary_lines = [line for line in lines if line.startswith("**Report-level summary:**")]
        if len(summary_lines) != 1:
            errors.append("expected exactly one report-level summary")
        else:
            summary = plain(summary_lines[0].split(":", 1)[1])
            if summary not in SUMMARY_VALUES:
                errors.append(f"report-level summary must be one of {sorted(SUMMARY_VALUES)}, found {summary!r}")

        depth_lines = [line for line in lines if line.startswith("**Assessment depth:**")]
        if len(depth_lines) != 1:
            errors.append("expected exactly one assessment depth")
        else:
            depth = plain(depth_lines[0].split(":", 1)[1])
            if depth not in DEPTH_VALUES:
                errors.append(f"assessment depth must be one of {sorted(DEPTH_VALUES)}, found {depth!r}")

        derivation_lines = [line for line in lines if line.startswith("**Summary derivation:**")]
        if len(derivation_lines) != 1:
            errors.append("expected exactly one summary derivation")
        else:
            derivation = plain(derivation_lines[0].split(":", 1)[1])
            if derivation not in SUMMARY_DERIVATION_VALUES:
                errors.append(
                    f"summary derivation must be one of {sorted(SUMMARY_DERIVATION_VALUES)}, found {derivation!r}"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a report directory against warp-adoption/v2.")
    parser.add_argument(
        "report",
        type=Path,
        help="report directory, or template Markdown file with --template",
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="allow placeholders and empty cells while validating the template",
    )
    args = parser.parse_args()

    try:
        errors = validate(args.report, args.template)
    except OSError as error:
        print(f"{args.report}: {error}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: {args.report} conforms to warp-adoption/v2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
