# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import warp  # ensure all API functions are loaded  # noqa: F401
from warp._src.generated_files import generate_stubs_file

logger = logging.getLogger(__name__)

# Environment flag telling doctest workers that generated sources are ready.
_DOCS_SOURCES_PREPARED_ENV = "WARP_DOCS_SOURCES_PREPARED"

# Directories that do not contain standalone Sphinx documents to shard.
_EXCLUDED_DOCS_DIRECTORIES = frozenset(("_build", "_src", "_templates", "superpowers"))


@dataclass(eq=False)
class DoctestProcess:
    """State for one running doctest shard."""

    shard_index: int
    process: subprocess.Popen
    manifest_path: Path
    output_path: Path
    log_path: Path
    log_file: TextIO


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
        default=4,
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
    jobs: str | int = "auto",
) -> list[str]:
    """Build a Sphinx argument list."""
    args = ["-j", str(jobs), "-b", builder]
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


def partition_doctest_sources(sources: tuple[Path, ...], job_count: int) -> tuple[tuple[Path, ...], ...]:
    """Distribute sorted Sphinx sources evenly across doctest shards."""
    if job_count < 1:
        raise ValueError("job_count must be at least 1")
    if job_count > len(sources):
        raise ValueError("job_count cannot exceed the number of documentation sources")

    shards = tuple(sources[index::job_count] for index in range(job_count))
    for index, shard in enumerate(shards, start=1):
        logger.info("Planned doctest shard %d/%d with %d sources", index, job_count, len(shard))
    return shards


def stop_doctest_processes(processes: list[DoctestProcess]) -> None:
    """Stop running doctest shards and close their log files."""
    for shard in processes:
        if shard.process.poll() is None:
            try:
                shard.process.terminate()
            except ProcessLookupError:
                pass

    for shard in processes:
        try:
            if shard.process.poll() is None:
                try:
                    shard.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    shard.process.kill()
                    shard.process.wait()
        finally:
            shard.log_file.close()


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
        build_sphinx_docs(
            source_dir,
            preparation_output,
            "dummy",
            warnings_as_errors=warnings_as_errors,
        )
        shutil.rmtree(preparation_output)

    sources = discover_documentation_sources(source_dir)
    if not sources:
        raise RuntimeError(f"No Sphinx sources found under {source_dir}")
    shards = partition_doctest_sources(sources, job_count)

    processes: list[DoctestProcess] = []
    try:
        for shard_index, shard in enumerate(shards):
            shard_output = output_dir / f"shard-{shard_index}"
            cache_path = output_dir / "warp-cache" / f"shard-{shard_index}"
            manifest_path = output_dir / f"shard-{shard_index}.txt"
            shard_output.mkdir(parents=True)
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
                "-q",
                *sphinx_args(
                    source_dir,
                    shard_output,
                    "doctest-shard",
                    warnings_as_errors,
                    (f"warp_doctest_shard_manifest={manifest_path}",),
                    jobs=1,
                ),
            ]
            logger.info("Starting doctest shard %d/%d", shard_index + 1, job_count)
            log_path = shard_output / "log.txt"
            log_file = log_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            except Exception:
                log_file.close()
                raise
            processes.append(
                DoctestProcess(
                    shard_index=shard_index,
                    process=process,
                    manifest_path=manifest_path,
                    output_path=shard_output / "output.txt",
                    log_path=log_path,
                    log_file=log_file,
                )
            )

        failed_shards = []
        for shard in processes:
            result = shard.process.wait()
            shard.log_file.close()
            if result == 0:
                logger.info("Doctest shard %d/%d completed successfully", shard.shard_index + 1, job_count)
                continue

            failed_shards.append((shard.shard_index, result))
            logger.error("Doctest shard %d/%d failed with exit code %d", shard.shard_index + 1, job_count, result)
            details_path = shard.output_path if shard.output_path.exists() else shard.log_path
            details = details_path.read_text(encoding="utf-8", errors="replace").rstrip()
            if details_path == shard.log_path:
                log_lines = details.splitlines()
                if len(log_lines) > 80:
                    omitted_count = len(log_lines) - 80
                    details = f"... {omitted_count} earlier log lines omitted ...\n" + "\n".join(log_lines[-80:])
            logger.error(
                "Doctest shard %d/%d details from %s (full log: %s; manifest: %s):\n%s",
                shard.shard_index + 1,
                job_count,
                details_path,
                shard.log_path,
                shard.manifest_path,
                details,
            )

        if failed_shards:
            failed_summary = ", ".join(f"{index + 1} (exit code {result})" for index, result in failed_shards)
            raise RuntimeError(f"Sphinx doctest shard(s) failed: {failed_summary}")
    finally:
        stop_doctest_processes(processes)


def main(argv: list[str] | None = None) -> None:
    """Build the requested Warp documentation outputs."""
    parser = create_parser()
    args = parser.parse_args(argv)
    if not args.html and not args.doctest:
        parser.error("At least one of --html or --doctest must be enabled")

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
    if generate_stubs_file(os.fspath(base_path)):
        logger.info(f"Generated {stub_path}")
        logger.info("Formatting __init__.pyi (a 'Failed' message in the output below is expected)")
        format_file_with_ruff(os.fspath(stub_path))
    else:
        logger.info(f"{stub_path} is up to date")

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
