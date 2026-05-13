# Unified Logging

**Issue**: [GH-1315](https://github.com/NVIDIA/warp/issues/1315) (host-side logging infrastructure)

## Motivation

Warp emits host-side diagnostics during initialization, module loading, kernel
code generation, compilation, array operations, and interop workflows. Before
this change those messages were routed through a mix of direct `print()` calls,
`warnings.warn()` calls, `sys.stdout` / `sys.stderr` writes, and small local
helpers.

This branch introduces one host-side logging path controlled by
`wp.config.log_level` and a small pluggable `wp.Logger` protocol. Applications
and frameworks can now redirect Warp's host diagnostics without Warp carrying
application-specific integrations.

Kernel-side user logging, such as a future `wp.log_*()` device builtin or device
ring buffer, is outside the scope of this branch.

## Requirements

| ID  | Requirement                                                              | Priority | Notes                                      |
| --- | ------------------------------------------------------------------------ | -------- | ------------------------------------------ |
| R1  | Define named log-level constants in one place                            | Must     | Re-exported as `wp.LOG_*`                  |
| R2  | Add one global threshold for Warp host diagnostics                       | Must     | `wp.config.log_level`                      |
| R3  | Route host warnings through Python's `warnings` machinery by default     | Must     | Honors `-W`, `simplefilter()`, etc.        |
| R4  | Provide a pluggable host logger for framework integration                | Must     | `wp.Logger`, `wp.set_logger()`             |
| R5  | Preserve legacy `verbose` and `quiet` behavior during deprecation        | Must     | Existing callers should keep working       |
| R6  | Avoid bundling application-specific logger adapters in Warp              | Must     | App-side adapters live outside the library |

**Non-goals:**

- Public host-side `wp.log_*()` functions. Warp's host emitters are internal
  helpers in `warp._src.logger`.
- Kernel-side logging. `wp.printf()` remains the existing kernel-side debugging
  primitive; device-side `wp.log_*()` builtins are future work.
- Structured or machine-readable log records. This branch focuses on routing
  and thresholding human-readable diagnostics.
- Built-in Omniverse Kit, `carb`, or other application-specific adapters.
  Applications install those adapters via `wp.set_logger()`.

## Design

### Log-Level Constants

Four levels are defined in `warp/_src/logger.py` and re-exported from
`warp/__init__.py`. Numeric values match Python's `logging` module:

| Constant         | Value | Purpose                              |
| ---------------- | ----- | ------------------------------------ |
| `wp.LOG_DEBUG`   | 10    | Verbose compilation/debug output     |
| `wp.LOG_INFO`    | 20    | Informational messages               |
| `wp.LOG_WARNING` | 30    | Warnings and deprecation notices     |
| `wp.LOG_ERROR`   | 40    | Errors                               |

`wp.config.log_level` defaults to `warp.LOG_INFO` (20). Documentation renders
the default symbolically as `warp.LOG_INFO (20)` so readers do not have to infer
the meaning of the raw integer.

### Public API Surface

The public host-side surface is:

- `wp.LOG_DEBUG`, `wp.LOG_INFO`, `wp.LOG_WARNING`, `wp.LOG_ERROR` for assigning
  or comparing `wp.config.log_level`.
- `wp.Logger`, a runtime-checkable `Protocol` with `debug()`, `info()`,
  `warning()`, and `error()` methods.
- `wp.set_logger()` and `wp.get_logger()` to install or retrieve the active
  logger. Passing `None` to `wp.set_logger()` restores Warp's built-in default.
- `wp.ScopedLogger(logger)` to temporarily install a logger and restore the
  previous logger on exit. Passing `None` temporarily restores the built-in
  default.
- `wp.ScopedLogLevel(log_level)` to temporarily override `wp.config.log_level`
  and restore the previous level on exit.

The built-in default logger remains an internal implementation detail
(`warp._src.logger.LoggerBasic`). Users and frameworks should depend on the
`wp.Logger` protocol, not subclass the default implementation.

### Internal Emitters

Warp's own diagnostics flow through internal helpers in `warp/_src/logger.py`:

```python
def log_debug(message: str) -> None: ...
def log_info(message: str) -> None: ...
def log_warning(message: str, category=None, stacklevel=1, once=False) -> None: ...
def log_error(message: str) -> None: ...
```

These helpers are not re-exported as public API. Internal callers import them
directly from `warp._src.logger`.

`log_debug()` emits when either legacy `wp.config.verbose` is true or
`wp.config.log_level <= wp.LOG_DEBUG`. Keeping both checks preserves behavior
for code that still toggles `verbose` after `wp.init()`.

`log_info()` emits when `wp.config.log_level <= wp.LOG_INFO`.

`log_warning()` emits when `wp.config.log_level <= wp.LOG_WARNING`. It accepts:

- `category`, passed to the logger's `warning()` method for Python warning
  integration.
- `stacklevel`, expressed relative to the `log_warning()` call site. The helper
  adjusts it before dispatch, so custom logger adapters can pass the received
  value directly to `warnings.warn(..., stacklevel=stacklevel)`.
- `once`, which deduplicates non-deprecation warnings by `(category, message)`.

`DeprecationWarning` messages are deduplicated unconditionally to preserve the
old `warp._src.utils.warn()` behavior.

`log_error()` always emits regardless of `wp.config.log_level`.

### Default Logger

The built-in logger routes output as follows:

| Method      | Destination / behavior                                      |
| ----------- | ------------------------------------------------------------ |
| `debug()`   | Writes message text to `sys.stdout`                          |
| `info()`    | Writes message text to `sys.stdout`                          |
| `warning()` | Calls `warnings.warn()` with temporary Warp warning formatting |
| `error()`   | Writes `Warp Error: <message>` to `sys.stderr`                |

Warnings intentionally pass through Python's `warnings` machinery so existing
filters, `-W` flags, and `warnings.simplefilter()` keep working. The default
formatter writes warnings as `Warp <CategoryName>: <message>`. If
`wp.config.verbose_warnings` is true, it also appends the source filename, line
number, and source line when available.

### Custom Loggers

Custom loggers are duck-typed against `wp.Logger`. They must provide all four
methods and `warning()` must accept `category` and `stacklevel` keyword
arguments:

```python
class MyLogger:
    def debug(self, message): ...
    def info(self, message): ...
    def warning(self, message, category=None, stacklevel=1): ...
    def error(self, message): ...

wp.set_logger(MyLogger())
```

Adapters that want Python warning filters should call:

```python
warnings.warn(message, category, stacklevel=stacklevel)
```

Adapters that forward to another logging system may ignore `category` and
`stacklevel`, but then warning filtering and warning-source attribution are the
adapter's responsibility.

Warp does not ship a `LoggerKit` or other application-specific adapter. For
example, an Omniverse Kit integration should live in the application layer and
install itself with `wp.set_logger()`.

### Deprecated Config Flags

`wp.config.verbose` and `wp.config.quiet` remain available during the
deprecation window:

- External reads or writes of either flag emit a one-time `DeprecationWarning`
  routed through the active Warp logger.
- Internal Warp reads do not emit those access warnings.
- `Runtime.__init__` still maps `quiet=True` to at least `wp.LOG_WARNING` and
  `verbose=True` to `wp.LOG_DEBUG`, unless the diagnostics path is temporarily
  suppressing the init banner.
- Call sites that historically honored runtime changes to `verbose` continue to
  check `wp.config.verbose or wp.config.log_level <= wp.LOG_DEBUG`.

The deprecated setter does not directly mutate `log_level`; compatibility is
handled at initialization and at the legacy verbose-sensitive call sites. This
avoids surprising side effects while keeping existing code functional.

`wp.config.verbose_warnings` is not deprecated. It only controls warning
formatting and has no `log_level` replacement.

### Migrated Call Sites

This branch migrates Warp's host-side diagnostics onto the new logger path,
including:

- Initialization banners and diagnostics-related output.
- Module load, code generation, compilation, cache, and launch diagnostics.
- Warnings and deprecation notices previously routed through
  `warp._src.utils.warn()` or direct `warnings.warn()` calls.
- Error output in renderer and JAX interop paths.
- `wp.config.print_launches` output, which now goes through the active logger.

API deprecation warnings converted to `log_warning()` pass an explicit
`stacklevel` so default Python filters attribute them to user call sites instead
of internal Warp files.

## Testing Strategy

- **Logger protocol and routing:** Verify `set_logger()` / `get_logger()`,
  `ScopedLogger`, duck-typed custom loggers, validation failures, default logger
  stdout/stderr routing, and restoration semantics.
- **Level gating:** Verify `LOG_DEBUG`, `LOG_INFO`, `LOG_WARNING`, and
  `LOG_ERROR` thresholds, including the legacy `verbose` compatibility path.
- **Warnings:** Verify Python warning filters still work with the default
  logger, custom warning adapters receive a `warnings.warn()`-ready stack level,
  duplicate deprecation warnings are suppressed, and deprecation warnings remain
  visible under default filters when attributed to user call sites.
- **Deprecated config flags:** Verify external `verbose` / `quiet` access emits
  one-time warnings through the active logger, respects `log_level`, and does
  not warn for internal Warp callers.
- **Integration:** Verify print-launch output, diagnostics banner suppression,
  CUDA compiler verbosity, and JAX FFI debug gates use the new logging behavior
  without dropping legacy `verbose` support.
