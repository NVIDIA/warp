# Licensed under the MIT License
# https://github.com/craigahobbs/unittest-parallel/blob/main/LICENSE

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel process-pool supervision for Warp's internal test runner."""

import concurrent.futures
import multiprocessing
import multiprocessing.connection
import signal
import time
from concurrent.futures.process import BrokenProcessPool

from warp._src.test_runner.common import (
    PARALLEL_RUN_TIMEOUT,
    ParallelRunResult,
    PoolFailure,
    ProcessExit,
    ProcessExitProvenance,
    warn,
)
from warp._src.test_runner.events import replay_worker_event_journals
from warp._src.test_runner.postmortem import (
    build_crash_snapshot,
    classify_suites,
    format_pool_failure,
    print_pool_failure_evidence,
    write_pool_failure_reports,
)
from warp._src.test_runner.worker import ParallelTestManager, initialize_test_process

_CLEANUP_EXIT_SIGNALS = frozenset({"SIGTERM", "SIGKILL"})


def _register_executor_processes(executor, processes):
    """Record executor-owned worker processes so exits stay observable after shutdown."""
    try:
        current = dict(executor._processes) if executor._processes else {}
    except (AttributeError, RuntimeError, TypeError):
        return
    for pid, process in current.items():
        processes.setdefault(pid, process)


def _process_is_alive(process):
    # ``Process.is_alive()`` calls ``waitpid()``/``poll()`` under the hood, which only
    # resolves once *this* thread performs the reap. Immediately after a pool failure
    # surfaces, the executor's own manager thread may not have reaped a just-exited
    # worker yet, so ``is_alive()`` can still (wrongly) report True for a worker that
    # has already died. The process sentinel reflects kernel-level exit status directly
    # (no reap required), so prefer it and fall back to ``is_alive()`` when unavailable.
    try:
        sentinel = process.sentinel
    except (AssertionError, OSError, ValueError, AttributeError):
        sentinel = None
    if sentinel is not None:
        try:
            return sentinel not in multiprocessing.connection.wait([sentinel], timeout=0)
        except (OSError, ValueError):
            pass
    try:
        return bool(process.is_alive())
    except (AssertionError, OSError, ValueError):
        return False


def _signal_name(exit_code):
    if exit_code is None or exit_code >= 0:
        return None
    try:
        return signal.Signals(-exit_code).name
    except ValueError:
        return f"SIGNAL_{-exit_code}"


def _derive_provenance(exit_code, alive_at_failure, terminated_while_working=False):
    """Attribute one worker exit from its liveness, journaled phase, and exit code.

    Workers alive when the failure surfaced can only have been killed by our own
    cleanup. That liveness snapshot is not reliable on its own: CPython's
    executor manager thread publishes ``BrokenProcessPool`` to the pending
    futures before it terminates the surviving workers, so a worker we are about
    to classify may already be reaped by the time we look. The worker's own last
    journaled phase does not depend on which thread wins that race, so a
    ``SIGTERM`` delivered to a worker that had not yet reached shutdown is
    attributed to the pool teardown either way.

    ``SIGKILL`` stays ambiguous: our cleanup sends it, but so does the OOM
    killer, and conflating the two would hide the more serious diagnosis.
    """
    if alive_at_failure:
        return ProcessExitProvenance.PARENT_TERMINATED
    if exit_code is None or exit_code == 0:
        return ProcessExitProvenance.UNRESOLVED
    signal_name = _signal_name(exit_code)
    if signal_name == "SIGTERM" and terminated_while_working:
        return ProcessExitProvenance.PARENT_TERMINATED
    if signal_name in _CLEANUP_EXIT_SIGNALS:
        return ProcessExitProvenance.UNRESOLVED
    return ProcessExitProvenance.INDEPENDENTLY_ABNORMAL


def _snapshot_alive_pids(processes):
    return frozenset(pid for pid, process in processes.items() if _process_is_alive(process))


def _snapshot_working_pids(tracker):
    """Collect PIDs whose last journaled phase shows work still in progress.

    A worker that finished on its own terms reports ``shutdown``; anything else
    means the pool still owned it when it died.
    """
    if tracker is None:
        return frozenset()
    try:
        snapshots = tracker.snapshots(now_ns=time.monotonic_ns())
    except Exception:
        # Attribution is best-effort; falling back to liveness alone is better
        # than letting a tracker failure mask the pool failure being reported.
        return frozenset()
    return frozenset(snapshot.pid for snapshot in snapshots if snapshot.phase != "shutdown")


def _process_exit_records(processes, alive_at_failure=frozenset(), working_at_failure=frozenset()):
    records = []
    for pid, process in processes.items():
        exit_code = getattr(process, "exitcode", None)
        records.append(
            ProcessExit(
                pid=pid,
                exit_code=exit_code,
                signal_name=_signal_name(exit_code),
                provenance=_derive_provenance(
                    exit_code,
                    pid in alive_at_failure,
                    pid in working_at_failure,
                ).value,
            )
        )
    return tuple(sorted(records, key=lambda record: record.pid))


def kill_process_pool(executor, processes=None, alive_at_failure=frozenset(), working_at_failure=frozenset()):
    """Cancel pending work, kill surviving workers, and report their exits."""
    if processes is None:
        processes = {}
    _register_executor_processes(executor, processes)
    manager_thread = getattr(executor, "_executor_manager_thread", None)

    kill_workers = getattr(executor, "kill_workers", None)
    if callable(kill_workers):
        kill_workers()
    else:
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes.values():
            if not _process_is_alive(process):
                continue
            try:
                process.kill()
            except ProcessLookupError:
                continue

    for process in processes.values():
        try:
            process.join()
        except (AssertionError, ValueError):
            pass
    if manager_thread is not None:
        manager_thread.join()

    return _process_exit_records(processes, alive_at_failure, working_at_failure)


def _store_future_result(future, index, results_by_index, future_states, failfast):
    if future.cancelled():
        future_states[index] = "cancelled"
        return None
    result = future.result()
    results_by_index[index] = result
    if failfast.is_set() and result[0] == 0:
        future_states[index] = "skipped_by_failfast"
    else:
        future_states[index] = "confirmed"
    return result


def _salvage_done_futures(future_to_index, results_by_index, future_states, failfast):
    for future, index in future_to_index.items():
        if index in results_by_index or not future.done():
            continue
        try:
            _store_future_result(future, index, results_by_index, future_states, failfast)
        except concurrent.futures.CancelledError:
            future_states[index] = "cancelled"
        except Exception:
            future_states[index] = "unresolved"


def run_parallel_suites(
    test_suites,
    process_count,
    manager,
    args,
    temp_dir,
    event_queue,
    tracker,
    monitor,
    run_dir,
    run_start_monotonic_ns,
):
    """Run shared-pool suites and retain every result confirmed by its future."""
    shared_index = manager.Value("i", -1)
    test_manager = ParallelTestManager(manager, args, temp_dir)
    executor_options = {}
    if getattr(args, "isolate_test_processes", False):
        executor_options["max_tasks_per_child"] = 1

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=process_count,
        mp_context=multiprocessing.get_context(method="spawn"),
        initializer=initialize_test_process,
        initargs=(
            manager.Lock(),
            shared_index,
            args,
            temp_dir,
            event_queue,
            run_dir,
            run_start_monotonic_ns,
        ),
        **executor_options,
    )
    results_by_index = {}
    future_states = dict.fromkeys(range(len(test_suites)), "pending")
    parallel_failure = None
    processes = {}
    process_exits = ()
    future_to_index = {}

    try:
        for index, suite in enumerate(test_suites):
            future = executor.submit(test_manager.run_tests, index, suite)
            future_to_index[future] = index
            _register_executor_processes(executor, processes)
        try:
            for future in concurrent.futures.as_completed(
                future_to_index,
                timeout=PARALLEL_RUN_TIMEOUT,
            ):
                index = future_to_index[future]
                try:
                    result = _store_future_result(
                        future,
                        index,
                        results_by_index,
                        future_states,
                        test_manager.failfast,
                    )
                except concurrent.futures.CancelledError:
                    future_states[index] = "cancelled"
                    continue
                except BrokenProcessPool as error:
                    parallel_failure = error
                    break
                except Exception as error:
                    parallel_failure = error
                    break

                if result is not None and test_manager.failfast.is_set():
                    for pending_future, pending_index in future_to_index.items():
                        if pending_future.done():
                            continue
                        if pending_future.cancel():
                            future_states[pending_index] = "cancelled"
        except (concurrent.futures.TimeoutError, BrokenProcessPool) as error:
            parallel_failure = error

        if parallel_failure is not None:
            _salvage_done_futures(
                future_to_index,
                results_by_index,
                future_states,
                test_manager.failfast,
            )

            for future, index in future_to_index.items():
                if future.done():
                    continue
                if future.cancel():
                    future_states[index] = "cancelled"
                else:
                    future_states[index] = "unresolved"
    except BaseException as error:
        parallel_failure = error
        if not isinstance(error, Exception):
            raise
    finally:
        _register_executor_processes(executor, processes)
        if parallel_failure is None:
            executor.shutdown(wait=True)
            process_exits = _process_exit_records(processes)
        else:
            alive_at_failure = _snapshot_alive_pids(processes)
            working_at_failure = _snapshot_working_pids(tracker)
            process_exits = kill_process_pool(executor, processes, alive_at_failure, working_at_failure)

    if parallel_failure is not None:
        _salvage_done_futures(
            future_to_index,
            results_by_index,
            future_states,
            test_manager.failfast,
        )
        if run_dir is not None:
            for error in replay_worker_event_journals(run_dir, tracker):
                warn(f"Failed to replay durable worker diagnostics: {error}")

    diagnostics_degraded = parallel_failure is not None
    if parallel_failure is None:
        try:
            monitor.stop_and_drain()
        except Exception as error:
            diagnostics_degraded = True
            warn(f"Failed to drain worker diagnostics: {error}")
    snapshots = tracker.snapshots(now_ns=time.monotonic_ns() - run_start_monotonic_ns)
    classifications = classify_suites(
        test_suites,
        results_by_index,
        future_states,
        snapshots,
        unit_type=args.level,
    )
    ordered_results = {index: results_by_index[index] for index in sorted(results_by_index)}
    pool_failure = None
    if parallel_failure is not None:
        snapshot = build_crash_snapshot(
            parallel_failure,
            classifications,
            snapshots,
            process_exits,
            run_dir,
        )
        formatted_summary = format_pool_failure(snapshot)
        pool_failure = PoolFailure(
            exception_type=type(parallel_failure).__name__,
            reason=str(parallel_failure) or type(parallel_failure).__name__,
            snapshot=snapshot,
            formatted_summary=formatted_summary,
        )
        if run_dir is not None:
            try:
                write_pool_failure_reports(run_dir, snapshot)
            except Exception as error:
                warn(f"Failed to write pool-failure diagnostics: {error}")
        try:
            print_pool_failure_evidence(snapshot)
        except Exception as error:
            warn(f"Failed to print pool-failure diagnostics: {error}")

    return ParallelRunResult(
        results_by_index=ordered_results,
        pool_failure=pool_failure,
        suite_classifications=classifications,
        diagnostics_degraded=diagnostics_degraded,
    )
