# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import warp  # ensure all API functions are loaded  # noqa: F401
from warp._src.context import export_stubs

logger = logging.getLogger(__name__)

_DOCS_SOURCES_PREPARED_ENV = "WARP_DOCS_SOURCES_PREPARED"
_EXCLUDED_DOCS_DIRECTORIES = frozenset(("_build", "_src", "_templates", "superpowers"))
_DOCTEST_DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+(?:doctest|testcode)::", re.MULTILINE)
_DOCTEST_PROMPT_RE = re.compile(r"^\s*>>>\s", re.MULTILINE)


def positive_int(value: str) -> int:
    """Parse a positive integer command-line value."""
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed_value


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Warp Sphinx Documentation Builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--html",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build HTML documentation",
    )
    parser.add_argument(
        "--doctest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run doctest tests of code blocks",
    )
    parser.add_argument(
        "--doctest-jobs",
        type=positive_int,
        default=1,
        help="Number of concurrent Sphinx doctest processes",
    )
    parser.add_argument(
        "--warnings-as-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Treat Sphinx warnings as errors (passes -W). Off by default so local "
            "builds stay lenient (e.g. unreachable intersphinx inventories when "
            "offline do not abort the build). CI/CD opts in to enforce strictness."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser


def format_file_with_ruff(file_path):
    """Format a file with Ruff using pre-commit for version consistency."""
    try:
        import pre_commit.main  # noqa: PLC0415

        result = pre_commit.main.main(["run", "ruff-format", "--files", file_path])
        logger.debug(f"pre-commit returned exit code {result} (first run)")

        if result == 0:
            # Success - file was already formatted or no changes needed
            logger.info(f"File {file_path} is already formatted")
        elif result == 1:
            # Exit code 1 typically means files were modified
            # Run again to verify the file is now properly formatted
            logger.info("Running pre-commit again to verify formatting (a 'Passed' message below is expected)")
            result = pre_commit.main.main(["run", "ruff-format", "--files", file_path])
            logger.debug(f"pre-commit returned exit code {result} (second run)")

            if result == 0:
                # Success - file is now properly formatted
                logger.info(f"Formatted {file_path}")
            else:
                # Still failing after formatting - this is a real error
                raise RuntimeError(
                    f"pre-commit formatting failed for {file_path}. "
                    f"File was modified but still has issues (exit code {result})"
                )
        else:
            raise RuntimeError(f"pre-commit formatting failed for {file_path} with exit code {result}")
    except ImportError as err:
        raise ImportError(
            "Could not format generated stubs: pre-commit is not available. "
            "Install with 'pip install warp-lang[docs]' or equivalent."
        ) from err


def sphinx_args(
    source_dir: Path,
    output_dir: Path,
    builder: str,
    warnings_as_errors: bool = False,
    config_overrides: tuple[str, ...] = (),
) -> list[str]:
    """Build a Sphinx argument list."""
    args = ["-j", "auto", "-b", builder]
    if warnings_as_errors:
        args.insert(0, "-W")
    for config_override in config_overrides:
        args.extend(("-D", config_override))
    args.extend((os.fspath(source_dir), os.fspath(output_dir)))
    return args


def build_sphinx_docs(
    source_dir: Path,
    output_dir: Path,
    builder: str = "html",
    warnings_as_errors: bool = False,
) -> None:
    """Build Sphinx documentation programmatically."""
    logger.info(f"Building {builder} documentation: {source_dir} -> {output_dir}")
    try:
        from sphinx.cmd.build import build_main  # noqa: PLC0415

        if output_dir.exists():
            logger.debug(f"Cleaning previous output directory: {output_dir}")
            shutil.rmtree(output_dir)

        args = sphinx_args(source_dir, output_dir, builder, warnings_as_errors)
        logger.debug(f"Running sphinx-build {' '.join(args)}")
        result = build_main(args)
        if result != 0:
            raise RuntimeError(f"Sphinx build failed with exit code {result}")

        logger.info(f"Successfully built {builder} documentation")

    except ImportError as err:
        raise ImportError(
            "Could not build docs: Sphinx is not available. Install with 'pip install warp-lang[docs]' or equivalent."
        ) from err


def discover_documentation_sources(source_dir: Path) -> tuple[Path, ...]:
    """Return all Sphinx source files that are eligible for doctesting."""
    sources = []
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix not in (".md", ".rst"):
            continue
        relative_path = path.relative_to(source_dir)
        if any(part in _EXCLUDED_DOCS_DIRECTORIES for part in relative_path.parts):
            continue
        sources.append(path)
    return tuple(sorted(sources))


def estimate_doctest_weight(path: Path) -> int:
    """Estimate a source file's doctest cost for deterministic sharding."""
    source = path.read_text(encoding="utf-8")
    weight = len(_DOCTEST_DIRECTIVE_RE.findall(source)) + len(_DOCTEST_PROMPT_RE.findall(source))

    # Autosummary pages contain directives rather than their extracted docstrings,
    # so their doctests are not visible until Sphinx reads the document.
    if "_generated" in path.parts:
        weight += 1

    return max(1, weight)


def partition_doctest_sources(sources: tuple[Path, ...], job_count: int) -> tuple[tuple[Path, ...], ...]:
    """Partition Sphinx sources into deterministic, approximately balanced shards."""
    if job_count < 1:
        raise ValueError("job_count must be at least 1")
    if job_count > len(sources):
        raise ValueError("job_count cannot exceed the number of documentation sources")

    weighted_sources = sorted(
        ((estimate_doctest_weight(source), source) for source in sources),
        key=lambda item: (-item[0], os.fspath(item[1])),
    )
    shards: list[list[Path]] = [[] for _ in range(job_count)]
    shard_weights = [0] * job_count

    for weight, source in weighted_sources:
        shard_index = min(range(job_count), key=lambda index: (shard_weights[index], index))
        shards[shard_index].append(source)
        shard_weights[shard_index] += weight

    for index, (shard, weight) in enumerate(zip(shards, shard_weights, strict=True), start=1):
        shard.sort()
        logger.info(
            "Planned doctest shard %d/%d with %d sources and estimated weight %d",
            index,
            job_count,
            len(shard),
            weight,
        )

    return tuple(tuple(shard) for shard in shards)


def combine_doctest_output(output_dir: Path, job_count: int) -> None:
    """Combine per-shard Sphinx summaries into the conventional output file."""
    combined_output = output_dir / "output.txt"
    with combined_output.open("w", encoding="utf-8") as output_file:
        for shard_index in range(job_count):
            shard_output = output_dir / f"shard-{shard_index}" / "output.txt"
            output_file.write(f"Doctest shard {shard_index + 1}/{job_count}\n")
            output_file.write("=" * 24 + "\n")
            if shard_output.exists():
                output_file.write(shard_output.read_text(encoding="utf-8"))
            else:
                output_file.write("No Sphinx doctest output was produced.\n")
            output_file.write("\n")


def run_parallel_doctests(
    source_dir: Path,
    output_dir: Path,
    job_count: int,
    warnings_as_errors: bool,
    sources_prepared: bool,
) -> None:
    """Run filename-sharded Sphinx doctests in concurrent subprocesses."""
    if output_dir.exists():
        logger.debug(f"Cleaning previous output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    if not sources_prepared:
        preparation_output = output_dir / "prepare"
        build_sphinx_docs(source_dir, preparation_output, "dummy")
        shutil.rmtree(preparation_output)

    sources = discover_documentation_sources(source_dir)
    if not sources:
        raise RuntimeError(f"No Sphinx sources found under {source_dir}")
    shards = partition_doctest_sources(sources, job_count)
    assigned_sources = [source for shard in shards for source in shard]
    if len(assigned_sources) != len(sources) or set(assigned_sources) != set(sources):
        raise RuntimeError("Doctest sharding must assign every documentation source exactly once")

    processes: list[tuple[int, subprocess.Popen]] = []
    for shard_index, shard in enumerate(shards):
        shard_output = output_dir / f"shard-{shard_index}"
        cache_path = output_dir / "warp-cache" / f"shard-{shard_index}"
        manifest_path = output_dir / f"shard-{shard_index}.txt"
        cache_path.mkdir(parents=True)
        manifest_path.write_text(
            "\n".join(path.relative_to(source_dir).with_suffix("").as_posix() for path in shard) + "\n",
            encoding="utf-8",
        )

        env = os.environ.copy()
        env[_DOCS_SOURCES_PREPARED_ENV] = "1"
        env["WARP_CACHE_PATH"] = os.fspath(cache_path)
        command = [
            sys.executable,
            "-m",
            "sphinx",
            *sphinx_args(
                source_dir,
                shard_output,
                "doctest-shard",
                warnings_as_errors,
                (f"warp_doctest_shard_manifest={manifest_path}",),
            ),
        ]
        logger.info("Starting doctest shard %d/%d", shard_index + 1, job_count)
        processes.append((shard_index, subprocess.Popen(command, env=env)))

    failed_shards = []
    for shard_index, process in processes:
        result = process.wait()
        if result == 0:
            logger.info("Doctest shard %d/%d completed successfully", shard_index + 1, job_count)
        else:
            failed_shards.append((shard_index, result))
            logger.error("Doctest shard %d/%d failed with exit code %d", shard_index + 1, job_count, result)

    combine_doctest_output(output_dir, job_count)
    if failed_shards:
        failed_summary = ", ".join(f"{index + 1} (exit code {result})" for index, result in failed_shards)
        raise RuntimeError(f"Sphinx doctest shard(s) failed: {failed_summary}")


def main(argv: list[str] | None = None) -> None:
    """Build the requested Warp documentation outputs."""
    parser = create_parser()
    args = parser.parse_args(argv)
    if not args.html and not args.doctest:
        parser.error("At least one of --html or --doctest must be enabled")
    if not args.doctest and args.doctest_jobs != 1:
        parser.error("--doctest-jobs requires --doctest")

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    base_path = Path(__file__).resolve().parent
    source_dir = base_path / "docs"

    logger.info("Starting Warp documentation build")
    logger.info("Generating API stubs for autocomplete")
    stub_path = base_path / "warp" / "__init__.pyi"
    with stub_path.open("w", encoding="utf-8") as stub_file:
        export_stubs(stub_file)

    logger.info("Formatting __init__.pyi (a 'Failed' message in the output below is expected)")
    format_file_with_ruff(os.fspath(stub_path))

    sources_prepared = False
    if args.html:
        html_output_dir = source_dir / "_build" / "html"
        build_sphinx_docs(source_dir, html_output_dir, "html", warnings_as_errors=args.warnings_as_errors)
        sources_prepared = True

    if args.doctest:
        logger.info("Running doctest...")
        doctest_output_dir = source_dir / "_build" / "doctest"
        if args.doctest_jobs == 1:
            build_sphinx_docs(source_dir, doctest_output_dir, "doctest", warnings_as_errors=args.warnings_as_errors)
        else:
            run_parallel_doctests(
                source_dir,
                doctest_output_dir,
                args.doctest_jobs,
                args.warnings_as_errors,
                sources_prepared,
            )

    logger.info("Documentation build completed successfully")


if __name__ == "__main__":
    main()
