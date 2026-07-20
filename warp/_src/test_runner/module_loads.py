# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact module-compilation diagnostics for Warp's test runner."""

import collections
import dataclasses
import heapq
import pathlib
import re
from typing import Any

SLOWEST_LIMIT = 20
_OUTPUT_GLOB = "worker-*-pid-*.output.log"
_HASH_PATTERN = re.compile(r"[0-9a-fA-F]{7}")
_MODULE_LOAD_PATTERN = re.compile(
    r"^Module (?P<identity>.+) load on device '(?P<device>[^'\r\n]+)'"
    r"(?: \([^\r\n)]*\))? took "
    r"(?P<elapsed>(?:0|[1-9]\d*)(?:\.\d+)?) ms\s+"
    r"\((?P<status>compiled|cached|error)\)$"
)


@dataclasses.dataclass(frozen=True)
class ModuleLoadRecord:
    module: str
    module_hash: str | None
    device: str
    elapsed_ms: float
    status: str


@dataclasses.dataclass
class _HashBuildStats:
    attempts: int = 0
    compiled: int = 0
    errors: int = 0
    attempt_ms: float = 0.0
    compiled_ms: float = 0.0


@dataclasses.dataclass(frozen=True)
class HashChurn:
    module: str
    device: str
    hashes: tuple[str, ...]
    attempts: int
    compiled_count: int
    error_count: int
    aggregate_ms: float


@dataclasses.dataclass(frozen=True)
class RepeatedCompilation:
    module: str
    module_hash: str
    device: str
    compilations: int
    aggregate_ms: float


@dataclasses.dataclass(frozen=True)
class ModuleLoadSummary:
    files_inspected: int
    parsed_records: int
    complete: bool
    no_shared_cache: bool
    status_counts: dict[str, int]
    hash_churn: tuple[HashChurn, ...]
    repeated_compilations: tuple[RepeatedCompilation, ...]
    slowest_compiled: tuple[ModuleLoadRecord, ...]
    read_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_inspected": self.files_inspected,
            "parsed_records": self.parsed_records,
            "complete": self.complete,
            "no_shared_cache": self.no_shared_cache,
            "status_counts": dict(self.status_counts),
            "hash_churn": [dataclasses.asdict(record) for record in self.hash_churn],
            "repeated_compilations": [dataclasses.asdict(record) for record in self.repeated_compilations],
            "slowest_compiled": [dataclasses.asdict(record) for record in self.slowest_compiled],
        }


def parse_module_load_line(line: str) -> ModuleLoadRecord | None:
    """Parse one complete Warp module-load timer line."""
    match = _MODULE_LOAD_PATTERN.fullmatch(line.rstrip("\r\n"))
    if match is None:
        return None

    identity = match.group("identity")
    module, separator, candidate_hash = identity.rpartition(" ")
    if separator and _HASH_PATTERN.fullmatch(candidate_hash):
        module_hash = candidate_hash.lower()
    else:
        module = identity
        module_hash = None

    return ModuleLoadRecord(
        module=module,
        module_hash=module_hash,
        device=match.group("device"),
        elapsed_ms=float(match.group("elapsed")),
        status=match.group("status"),
    )


def collect_module_load_summary(
    run_dir: pathlib.Path,
    *,
    no_shared_cache: bool,
    slowest_limit: int = SLOWEST_LIMIT,
) -> ModuleLoadSummary | None:
    """Collect compact module-load diagnostics from captured worker logs."""
    paths = tuple(sorted(pathlib.Path(run_dir).glob(_OUTPUT_GLOB)))
    if not paths:
        return None

    status_counts = collections.Counter({"compiled": 0, "cached": 0, "error": 0})
    build_stats = collections.defaultdict(dict)  # (module, device) -> {module_hash: _HashBuildStats}
    compiled_records = []
    read_errors = []
    parsed_records = 0

    for path in paths:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    record = parse_module_load_line(line)
                    if record is None:
                        continue
                    parsed_records += 1
                    status_counts[record.status] += 1
                    if record.status == "compiled":
                        compiled_records.append(record)
                    if record.module_hash is None or record.status not in {"compiled", "error"}:
                        continue
                    stats = build_stats[(record.module, record.device)].setdefault(
                        record.module_hash, _HashBuildStats()
                    )
                    stats.attempts += 1
                    stats.compiled += record.status == "compiled"
                    stats.errors += record.status == "error"
                    stats.attempt_ms += record.elapsed_ms
                    if record.status == "compiled":
                        stats.compiled_ms += record.elapsed_ms
        except OSError as error:
            read_errors.append(f"{path}: {error}")

    hash_churn = []
    repeated_compilations = []
    for (module, device), hash_stats in build_stats.items():
        if len(hash_stats) > 1:
            hash_churn.append(
                HashChurn(
                    module=module,
                    device=device,
                    hashes=tuple(sorted(hash_stats)),
                    attempts=sum(stats.attempts for stats in hash_stats.values()),
                    compiled_count=sum(stats.compiled for stats in hash_stats.values()),
                    error_count=sum(stats.errors for stats in hash_stats.values()),
                    aggregate_ms=round(sum(stats.attempt_ms for stats in hash_stats.values()), 2),
                )
            )
        for module_hash, stats in hash_stats.items():
            if stats.compiled > 1:
                repeated_compilations.append(
                    RepeatedCompilation(
                        module=module,
                        module_hash=module_hash,
                        device=device,
                        compilations=stats.compiled,
                        aggregate_ms=round(stats.compiled_ms, 2),
                    )
                )
    hash_churn.sort(key=lambda record: (-record.aggregate_ms, record.module, record.device))
    repeated_compilations.sort(
        key=lambda record: (-record.aggregate_ms, record.module, record.device, record.module_hash)
    )
    slowest_compiled = heapq.nsmallest(
        max(slowest_limit, 0),
        compiled_records,
        key=lambda record: (-record.elapsed_ms, record.module, record.device, record.module_hash or ""),
    )

    return ModuleLoadSummary(
        files_inspected=len(paths),
        parsed_records=parsed_records,
        complete=not read_errors,
        no_shared_cache=no_shared_cache,
        status_counts={status: status_counts[status] for status in ("compiled", "cached", "error")},
        hash_churn=tuple(hash_churn),
        repeated_compilations=tuple(repeated_compilations),
        slowest_compiled=tuple(slowest_compiled),
        read_errors=tuple(read_errors),
    )


def format_module_load_summary(
    summary: ModuleLoadSummary,
    *,
    redundancy_limit: int = SLOWEST_LIMIT,
) -> str:
    """Format potential redundancy and the slowest compiled module loads."""
    potential = [("hash_churn", record) for record in summary.hash_churn] + [
        ("repeated_compilation", record) for record in summary.repeated_compilations
    ]
    potential.sort(
        key=lambda item: (
            -item[1].aggregate_ms,
            item[1].module,
            item[1].device,
            getattr(item[1], "module_hash", ""),
            item[0],
        )
    )

    lines = ["Potential redundant module builds:"]
    if summary.no_shared_cache and summary.repeated_compilations:
        lines.append("  Note: --no-shared-cache was enabled; repeated hashes may be expected.")
    if not potential:
        lines.append("  none")
    else:
        for rank, (kind, record) in enumerate(potential[:redundancy_limit], start=1):
            if kind == "hash_churn":
                error_label = "error" if record.error_count == 1 else "errors"
                lines.extend(
                    (
                        f"  {rank}. {record.module} on {record.device}: "
                        f"{len(record.hashes)} hashes across {record.attempts} build attempts",
                        f"     {record.compiled_count} compiled, {record.error_count} "
                        f"{error_label}, {record.aggregate_ms * 0.001:.3f}s aggregate",
                    )
                )
            else:
                lines.extend(
                    (
                        f"  {rank}. {record.module} {record.module_hash} on {record.device}:",
                        f"     compiled {record.compilations} times, {record.aggregate_ms * 0.001:.3f}s aggregate",
                    )
                )
        omitted = len(potential) - redundancy_limit
        if omitted > 0:
            lines.append(f"  ... {omitted} more in module-load-summary.json")

    lines.extend(("", f"Slowest {SLOWEST_LIMIT} compiled module loads:"))
    if not summary.slowest_compiled:
        lines.append("  none")
    else:
        for rank, record in enumerate(summary.slowest_compiled, start=1):
            identity = record.module
            if record.module_hash is not None:
                identity += f" {record.module_hash}"
            lines.append(f"  {rank}. {identity} on {record.device}: {record.elapsed_ms * 0.001:.3f}s")
    return "\n".join(lines)
