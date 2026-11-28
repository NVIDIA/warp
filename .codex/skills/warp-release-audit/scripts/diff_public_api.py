#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit deterministic public runtime/stub API diffs for release audits.

This helper is intentionally static and stdlib-only. It reads source files from
git refs, parses them with ``ast``, and emits JSON facts for the warp-release-audit
skill to interpret. It does not import Warp or build native libraries.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import textwrap
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

_GIT_SHOW_CACHE: dict[tuple[str, str, str], str | None] = {}
_PARSE_CACHE: dict[tuple[str, str, str, str], ModuleSurface] = {}


@dataclass(frozen=True)
class Param:
    name: str
    kind: str
    has_default: bool


@dataclass(frozen=True)
class ApiEntry:
    symbol: str
    surface: str
    entry_kind: str
    signature: str | None = None
    params: tuple[Param, ...] = ()
    source: str | None = None
    definition: str | None = None
    documented: bool = False

    @property
    def signature_key(self) -> str:
        if self.signature is None:
            return self.entry_kind
        return self.signature


@dataclass
class Change:
    kind: str
    symbol: str
    surface: str
    reasons: list[str]
    breaking: bool = True
    base: str | None = None
    head: str | None = None
    source: str | None = None


@dataclass
class ModuleSurface:
    entries: dict[str, list[ApiEntry]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    star_exports: tuple[str, ...] = ()

    def add(self, entry: ApiEntry) -> None:
        self.entries.setdefault(entry.symbol, []).append(entry)


def git(repo: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        sys.exit("diff_public_api: 'git' executable not found on PATH")

    if proc.returncode != 0:
        sys.exit(
            f"diff_public_api: git {' '.join(args)} failed in {repo} (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def git_show(repo: Path, ref: str, path: str) -> str | None:
    key = (str(repo), ref, path)
    if key in _GIT_SHOW_CACHE:
        return _GIT_SHOW_CACHE[key]

    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        _GIT_SHOW_CACHE[key] = None
        return None
    _GIT_SHOW_CACHE[key] = proc.stdout
    return proc.stdout


def resolve_sha(repo: Path, ref: str) -> str:
    return git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()


def module_path(repo: Path, ref: str, module: str, suffix: str) -> tuple[str, bool] | None:
    base = module.replace(".", "/")
    package_path = f"{base}/__init__{suffix}"
    module_file = f"{base}{suffix}"
    if git_show(repo, ref, package_path) is not None:
        return package_path, True
    if git_show(repo, ref, module_file) is not None:
        return module_file, False
    return None


def is_public(name: str) -> bool:
    return not name.startswith("_")


def api_reference(repo: Path, ref: str) -> tuple[set[str], dict[str, str]]:
    """Read the public modules and symbols listed in Warp's API reference."""
    modules = {"warp"}
    symbols: dict[str, str] = {}
    paths = git(repo, "ls-tree", "-r", "--name-only", ref, "--", "docs/api_reference")
    for path in paths.splitlines():
        if not path.endswith(".rst"):
            continue
        source = git_show(repo, ref, path)
        if source is None:
            continue
        module = "warp"
        in_autosummary = False
        for line in source.splitlines():
            if line and not line[0].isspace():
                in_autosummary = line.rstrip() == ".. autosummary::"
            namespace = re.fullmatch(r"\.\. (?:currentmodule|automodule)::\s+(warp(?:\.\w+)*)\s*", line)
            if namespace:
                module = namespace[1]
                if all(is_public(part) for part in module.split(".")):
                    modules.add(module)
            if in_autosummary:
                name = re.fullmatch(r"\s+~?([A-Za-z_]\w*(?:\.\w+)*)\s*", line)
            else:
                name = re.fullmatch(r"\.\. auto(?:class|function|method|attribute|data)::\s+~?([\w.]+)\s*", line)
            if name:
                symbol = name[1] if name[1].startswith("warp.") else f"{module}.{name[1]}"
                if all(is_public(part) for part in symbol.split(".")):
                    symbols.setdefault(symbol, path)
    # A fully qualified autodoc entry can document a module without a separate page.
    for symbol in symbols:
        namespace = symbol.rpartition(".")[0]
        while namespace not in modules:
            if module_path(repo, ref, namespace, ".py") is not None:
                modules.add(namespace)
                break
            namespace = namespace.rpartition(".")[0]
    return modules, symbols


def overload_decorator_names(tree: ast.Module) -> set[str]:
    """Return names bound to ``typing.overload`` in ``tree``."""

    names = {"overload"}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module not in {"typing", "typing_extensions"}:
            continue
        for alias in node.names:
            if alias.name == "overload":
                names.add(alias.asname or alias.name)
    return names


def is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef, decorator_names: set[str]) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id in decorator_names:
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "overload":
            return True
    return False


def unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    return ast.unparse(node)


def param_text(arg: ast.arg, default: ast.expr | None, prefix: str = "") -> str:
    text = f"{prefix}{arg.arg}"
    if arg.annotation is not None:
        text += f": {unparse(arg.annotation)}"
    if default is not None:
        text += f"={unparse(default)}"
    return text


def params_from_args(args: ast.arguments) -> tuple[list[Param], list[str]]:
    params: list[Param] = []
    pieces: list[str] = []
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)

    for index, (arg, default) in enumerate(zip(positional, defaults, strict=True)):
        kind = "posonly" if index < len(args.posonlyargs) else "poskw"
        params.append(Param(arg.arg, kind, default is not None))
        pieces.append(param_text(arg, default))

    if args.vararg is not None:
        params.append(Param(args.vararg.arg, "vararg", False))
        pieces.append(param_text(args.vararg, None, "*"))
    elif args.kwonlyargs:
        pieces.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        params.append(Param(arg.arg, "kwonly", default is not None))
        pieces.append(param_text(arg, default))

    if args.kwarg is not None:
        params.append(Param(args.kwarg.arg, "varkw", False))
        pieces.append(param_text(args.kwarg, None, "**"))

    return params, pieces


def callable_entry(
    public_symbol: str,
    surface: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    entry_kind: str,
    source: str,
) -> ApiEntry:
    params, pieces = params_from_args(node.args)
    signature = f"{public_symbol}({', '.join(pieces)})"
    if node.returns is not None:
        signature += f" -> {unparse(node.returns)}"
    return ApiEntry(
        symbol=public_symbol,
        surface=surface,
        entry_kind=entry_kind,
        signature=signature,
        params=tuple(params),
        source=source,
        documented=bool(ast.get_docstring(node)),
    )


def assignment_targets(node: ast.stmt) -> Sequence[ast.expr]:
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, ast.AnnAssign):
        return (node.target,)
    return ()


def unpack_targets(targets: Sequence[ast.expr]) -> Iterator[ast.expr]:
    for target in targets:
        if isinstance(target, (ast.Tuple, ast.List)):
            yield from unpack_targets(target.elts)
        elif isinstance(target, ast.Starred):
            yield from unpack_targets((target.value,))
        else:
            yield target


def statement_blocks(node: ast.AST) -> Iterator[list[ast.stmt]]:
    """Yield ordered statement lists, stopping at nested definitions."""
    for _, children in ast.iter_fields(node):
        if not isinstance(children, list) or not children:
            continue
        if isinstance(children[0], ast.stmt):
            yield children
        for child in children:
            if isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case)) and not isinstance(
                child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                yield from statement_blocks(child)


def attributes(
    nodes: Sequence[ast.stmt], prefix: str, surface: str, source: str, instance_docs: set[str] | None = None
) -> list[ApiEntry]:
    entries = []
    for index, node in enumerate(nodes):
        targets = assignment_targets(node)
        if not targets:
            continue
        following = nodes[index + 1] if index + 1 < len(nodes) else None
        documented = (
            isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        )
        for target in unpack_targets(targets):
            if instance_docs is None and isinstance(target, ast.Name):
                name = target.id
                has_doc = documented
            elif (
                instance_docs is not None
                and isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                name = target.attr
                has_doc = documented or name in instance_docs
            else:
                continue
            if is_public(name):
                entries.append(ApiEntry(f"{prefix}.{name}", surface, "attribute", source=source, documented=has_doc))
    return entries


def class_entries(
    public_symbol: str,
    surface: str,
    node: ast.ClassDef,
    source: str,
    overload_names: set[str],
) -> list[ApiEntry]:
    entries: list[ApiEntry] = []
    init_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__":
            init_node = child
            break

    if init_node is not None:
        entries.append(callable_entry(public_symbol, surface, init_node, "class", source))
        doc = ast.get_docstring(node) or ""
        section = re.search(r"^Attributes:[ \t]*\n((?:[ \t]+[^\n]+\n?|\n)*)", doc, re.MULTILINE)
        instance_docs = (
            set(re.findall(r"^(\w+)(?: \([^\n]+\))?:", textwrap.dedent(section[1]), re.MULTILINE)) if section else set()
        )
        for block in statement_blocks(init_node):
            entries.extend(attributes(block, public_symbol, surface, source, instance_docs))
    else:
        entries.append(ApiEntry(public_symbol, surface, "class", source=source))

    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public(child.name):
            kind = "overload" if is_overload(child, overload_names) else "method"
            entries.append(callable_entry(f"{public_symbol}.{child.name}", surface, child, kind, source))
    entries.extend(attributes(node.body, public_symbol, surface, source))
    return entries


def resolve_import_module(current_module: str, is_package: bool, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module

    parts = current_module.split(".") if is_package else current_module.split(".")[:-1]
    if node.level > 1:
        parts = parts[: -(node.level - 1)]
    if node.module:
        parts.extend(node.module.split("."))
    return ".".join(part for part in parts if part)


def static_star_exports(tree: ast.Module) -> tuple[str, ...]:
    """Read a literal module-level __all__ without importing the module."""
    for node in reversed(tree.body):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in assignment_targets(node)
        ):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple)) and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts
        ):
            return tuple(item.value for item in value.elts)
        break
    return ()


def entries_from_source_module(
    parsed: ModuleSurface | None,
    source_module: str,
    imported_name: str,
    public_symbol: str,
    surface: str,
) -> list[ApiEntry]:
    if parsed is None:
        return [ApiEntry(public_symbol, surface, "alias", source=source_module)]

    source_symbol = f"{source_module}.{imported_name}"
    source_entries = parsed.entries.get(source_symbol, [])
    if not source_entries:
        return [ApiEntry(public_symbol, surface, "alias", source=source_module)]
    if any(entry.entry_kind == "class" for entry in source_entries):
        prefix = f"{source_symbol}."
        source_entries = [*source_entries]
        for symbol, members in parsed.entries.items():
            if symbol.startswith(prefix):
                source_entries.extend(members)

    rewritten: list[ApiEntry] = []
    for entry in source_entries:
        suffix = entry.symbol[len(source_symbol) :]
        new_symbol = public_symbol + suffix
        signature = entry.signature
        if signature is not None:
            signature = new_symbol + signature[len(entry.symbol) :]
        rewritten.append(
            replace(
                entry,
                symbol=new_symbol,
                signature=signature,
                definition=entry.definition or entry.symbol,
            )
        )
    return rewritten


def parse_module(
    repo: Path,
    ref: str,
    module: str,
    surface: str,
    seen: set[tuple[str, str]] | None = None,
) -> ModuleSurface | None:
    cache_key = (str(repo), ref, module, surface)
    if cache_key in _PARSE_CACHE:
        return _PARSE_CACHE[cache_key]

    seen = seen or set()
    marker = (ref, module)
    if marker in seen:
        return None
    seen.add(marker)

    suffix = ".pyi" if surface == "stub" else ".py"
    resolved = module_path(repo, ref, module, suffix)
    if resolved is None:
        return None
    path, is_package = resolved
    text = git_show(repo, ref, path)
    if text is None:
        return None

    surface_data = ModuleSurface()
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        surface_data.warnings.append(f"{path}: cannot parse: {exc}")
        _PARSE_CACHE[cache_key] = surface_data
        return surface_data

    surface_data.star_exports = static_star_exports(tree)
    overload_names = overload_decorator_names(tree)

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            source_module = resolve_import_module(module, is_package, node)
            if source_module is None:
                continue
            root_module = module.split(".", 1)[0]
            if source_module != root_module and not source_module.startswith(f"{root_module}."):
                continue
            wildcard = node.names[0].name == "*"
            names = [
                (alias.name, alias.asname or alias.name)
                for alias in node.names
                if alias.name != "*" and is_public(alias.asname or alias.name)
            ]
            if not wildcard and not names:
                continue
            # Keep the cycle guard local to this import chain so it does not
            # affect independent re-exports from the same source module.
            imported = parse_module(repo, ref, source_module, surface, seen.copy())
            if wildcard and imported is not None:
                names = [(name, name) for name in imported.star_exports if is_public(name)]
            for imported_name, public_name in names:
                public_symbol = f"{module}.{public_name}"
                for entry in entries_from_source_module(
                    imported,
                    source_module,
                    imported_name,
                    public_symbol,
                    surface,
                ):
                    surface_data.add(entry)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public(node.name):
            kind = "overload" if is_overload(node, overload_names) else "function"
            surface_data.add(callable_entry(f"{module}.{node.name}", surface, node, kind, path))
        elif isinstance(node, ast.ClassDef) and is_public(node.name):
            for entry in class_entries(f"{module}.{node.name}", surface, node, path, overload_names):
                surface_data.add(entry)
    for entry in attributes(tree.body, module, surface, path):
        surface_data.add(entry)

    _PARSE_CACHE[cache_key] = surface_data
    return surface_data


def comparable_params(params: tuple[Param, ...]) -> list[Param]:
    return [param for param in params if param.name not in {"self", "cls"}]


def signature_breaking_reasons(base: ApiEntry, head: ApiEntry) -> list[str]:
    base_params = comparable_params(base.params)
    head_params = comparable_params(head.params)
    reasons: list[str] = []

    base_by_name = {param.name: param for param in base_params}
    head_by_name = {param.name: param for param in head_params}

    for name, old in base_by_name.items():
        new = head_by_name.get(name)
        if new is None:
            reasons.append("removed_parameter")
            continue
        if old.kind in {"posonly", "poskw"} and new.kind == "kwonly":
            reasons.append("positional_to_keyword_only")

    base_pos_names = [param.name for param in base_params if param.kind in {"posonly", "poskw"}]
    head_pos_names = [param.name for param in head_params if param.kind in {"posonly", "poskw"}]
    base_pos_set = set(base_pos_names)
    for index, name in enumerate(head_pos_names):
        if name in base_pos_set:
            continue
        if any(later_name in base_pos_set for later_name in head_pos_names[index + 1 :]):
            reasons.append("inserted_positional_parameter")
            break

    common_head_order = [name for name in head_pos_names if name in base_pos_set]
    common_base_order = [name for name in base_pos_names if name in set(head_pos_names)]
    if common_head_order != common_base_order:
        reasons.append("reordered_parameter")

    for param in head_params:
        if param.name in base_by_name:
            continue
        if param.kind in {"posonly", "poskw", "kwonly"} and not param.has_default:
            reasons.append("new_required_parameter")

    return sorted(set(reasons))


def runtime_implementations(surface: ModuleSurface, symbol: str) -> list[ApiEntry]:
    return [
        entry
        for entry in surface.entries[symbol]
        if entry.signature is not None and entry.entry_kind in {"function", "method", "class"}
    ]


def choose_signature_entry(
    callable_entries: list[ApiEntry], surface: ModuleSurface, symbol: str, ref: str
) -> ApiEntry | None:
    if len(callable_entries) == 1:
        return callable_entries[0]
    if len(callable_entries) > 1:
        sources = ", ".join(sorted({entry.source or "<unknown>" for entry in callable_entries}))
        surface.warnings.append(
            f"{symbol}: multiple non-overload callable entries at {ref} from {sources}; signature comparison skipped"
        )
    return None


def diff_signatures(base: ModuleSurface, head: ModuleSurface, base_ref: str, head_ref: str) -> list[Change]:
    changes: list[Change] = []
    for symbol in sorted(set(base.entries) & set(head.entries)):
        base_implementations = runtime_implementations(base, symbol)
        head_implementations = runtime_implementations(head, symbol)
        if [(entry.signature, entry.params) for entry in base_implementations] == [
            (entry.signature, entry.params) for entry in head_implementations
        ]:
            continue
        base_entry = choose_signature_entry(base_implementations, base, symbol, base_ref)
        head_entry = choose_signature_entry(head_implementations, head, symbol, head_ref)
        if base_entry is None or head_entry is None:
            continue
        reasons = signature_breaking_reasons(base_entry, head_entry)
        if not reasons:
            continue
        changes.append(
            Change(
                kind="signature_change",
                symbol=symbol,
                surface=base_entry.surface,
                reasons=reasons,
                breaking=True,
                base=base_entry.signature,
                head=head_entry.signature,
                source=head_entry.source,
            )
        )
    return changes


def diff_stub_removals(base: ModuleSurface, head: ModuleSurface) -> list[Change]:
    changes: list[Change] = []
    for symbol in sorted(set(base.entries) - set(head.entries)):
        base_entry = base.entries[symbol][0]
        reason = "removed_stub_attribute" if base_entry.entry_kind == "attribute" else "removed_stub_symbol"
        changes.append(
            Change(
                kind="public_stub_removal",
                symbol=symbol,
                surface="stub",
                reasons=[reason],
                breaking=True,
                base=base_entry.signature,
                source=base_entry.source,
            )
        )

    for symbol in sorted(set(base.entries) & set(head.entries)):
        base_overloads = {entry.signature_key for entry in base.entries[symbol] if entry.entry_kind == "overload"}
        head_overloads = {entry.signature_key for entry in head.entries[symbol] if entry.entry_kind == "overload"}
        for signature in sorted(base_overloads - head_overloads):
            changes.append(
                Change(
                    kind="public_stub_removal",
                    symbol=symbol,
                    surface="stub",
                    reasons=["removed_stub_overload"],
                    breaking=True,
                    base=signature,
                )
            )
    return changes


def dedupe_changes(changes: list[Change]) -> list[Change]:
    seen: set[tuple[str, str, str, tuple[str, ...], str | None, str | None]] = set()
    out: list[Change] = []
    for change in sorted(
        changes,
        key=lambda item: (item.kind, item.symbol, item.surface, item.reasons, item.base or "", item.head or ""),
    ):
        key = (
            change.kind,
            change.symbol,
            change.surface,
            tuple(change.reasons),
            change.base,
            change.head,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(change)
    return out


def documented_additions(base: ModuleSurface, head: ModuleSurface, documentation: dict[str, str]) -> list[dict]:
    additions = []
    for symbol in sorted(head.entries.keys() - base.entries.keys()):
        entries = head.entries[symbol]
        entry = entries[0]
        doc_path = documentation.get(symbol)
        owner = symbol.rpartition(".")[0]
        if doc_path is None and any(item.documented for item in entries):
            if any(item.entry_kind == "class" for item in head.entries.get(owner, [])):
                doc_path = documentation.get(owner)
        if doc_path is not None:
            additions.append(
                {
                    "symbol": symbol,
                    "kind": entry.entry_kind,
                    "source": entry.source,
                    "definition": entry.definition or symbol,
                    "signatures": sorted({item.signature for item in entries if item.signature is not None}),
                    "documentation": doc_path,
                }
            )
    return additions


def build_report(repo: Path, base: str, head: str, modules: list[str]) -> dict:
    base_modules, _ = api_reference(repo, base)
    head_modules, documentation = api_reference(repo, head)
    modules = sorted({"warp", *modules} | base_modules | head_modules)
    additions: list[dict] = []
    changes: list[Change] = []
    warnings: list[dict[str, str]] = []

    for module in modules:
        base_runtime = parse_module(repo, base, module, "runtime") or ModuleSurface()
        head_runtime = parse_module(repo, head, module, "runtime") or ModuleSurface()
        base_stub = parse_module(repo, base, module, "stub") or ModuleSurface()
        head_stub = parse_module(repo, head, module, "stub") or ModuleSurface()

        additions.extend(documented_additions(base_runtime, head_runtime, documentation))
        changes.extend(diff_signatures(base_runtime, head_runtime, base, head))
        changes.extend(diff_stub_removals(base_stub, head_stub))

        for surface, data in (
            ("runtime", base_runtime),
            ("runtime", head_runtime),
            ("stub", base_stub),
            ("stub", head_stub),
        ):
            for message in data.warnings:
                warnings.append({"module": module, "surface": surface, "message": message})

    return {
        "base": {"ref": base, "sha": resolve_sha(repo, base)},
        "head": {"ref": head, "sha": resolve_sha(repo, head)},
        "modules": modules,
        "additions": additions,
        "changes": [asdict(change) for change in dedupe_changes(changes)],
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Git repository root")
    parser.add_argument("--base", required=True, help="Base git ref")
    parser.add_argument("--head", required=True, help="Head git ref")
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        help="Additional public module to scan alongside the modules in the API reference. Repeat as needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = args.modules or ["warp"]
    report = build_report(args.repo.resolve(), args.base, args.head, modules)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
