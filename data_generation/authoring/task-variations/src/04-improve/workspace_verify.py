"""Standalone verifier copied into each reviewed-generator workspace."""

from __future__ import annotations

import ast
import argparse
import json
import secrets
import sys
from pathlib import Path

try:
    from .host_checks import run_checks
except ImportError:  # standalone copy in a puzzle workspace
    from host_checks import run_checks


REQUIRED_FUNCTIONS = (
    "generate_puzzle",
    "validate_generated_puzzle",
    "self_test",
)
FORBIDDEN_MODULES = {
    "asyncio",
    "ftplib",
    "http",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "random",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "telnetlib",
    "urllib",
}
FORBIDDEN_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}
FORBIDDEN_NUMPY_IO = {
    "fromfile",
    "genfromtxt",
    "load",
    "loadtxt",
    "save",
    "savetxt",
    "tofile",
}


def _positional_names(function: ast.FunctionDef) -> list[str]:
    return [item.arg for item in function.args.posonlyargs + function.args.args]


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _validate_signature(
    function: ast.FunctionDef,
    expected: list[str],
    default_count: int,
) -> list[str]:
    errors: list[str] = []
    arguments = function.args
    if (
        _positional_names(function) != expected
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or arguments.kwonlyargs
    ):
        errors.append(
            "%s must have exactly this positional signature: (%s)."
            % (function.name, ", ".join(expected))
        )
    if len(arguments.defaults) != default_count or any(
        not _is_none(item) for item in arguments.defaults
    ):
        if default_count:
            errors.append(
                "%s must default its final %d arguments to None."
                % (function.name, default_count)
            )
        else:
            errors.append("%s must not define default arguments." % function.name)
    return errors


def _attribute_parts(node: ast.AST) -> list[str]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return list(reversed(parts))


def _placeholder_body(function: ast.FunctionDef) -> bool:
    if len(function.body) != 1:
        return False
    statement = function.body[0]
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
        return statement.value.value is Ellipsis
    if isinstance(statement, ast.Raise) and isinstance(statement.exc, ast.Call):
        return isinstance(statement.exc.func, ast.Name) and statement.exc.func.id == "NotImplementedError"
    return False


def _static_check(source: str) -> tuple[ast.Module | None, list[str]]:
    errors: list[str] = []
    try:
        tree = ast.parse(source, filename="generator.py", mode="exec")
        compile(tree, "generator.py", "exec")
    except SyntaxError as exc:
        return None, ["Generator code does not compile: %s" % exc]

    top_level_functions: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name in REQUIRED_FUNCTIONS:
            errors.append("Required function %s must be synchronous." % node.name)
        elif isinstance(node, ast.FunctionDef):
            if node.name in top_level_functions:
                errors.append("Top-level function %s is defined more than once." % node.name)
            top_level_functions[node.name] = node
        elif isinstance(node, ast.If):
            try:
                rendered = ast.unparse(node.test)
            except Exception:
                rendered = ""
            if "__name__" in rendered and "__main__" in rendered:
                errors.append("A command-line __main__ block is not allowed.")

    for name in REQUIRED_FUNCTIONS:
        if name not in top_level_functions:
            errors.append("Missing required top-level function %s." % name)
    if "generate_puzzle" in top_level_functions:
        errors.extend(
            _validate_signature(
                top_level_functions["generate_puzzle"],
                ["seed", "num_train", "num_test"],
                2,
            )
        )
    if "validate_generated_puzzle" in top_level_functions:
        errors.extend(
            _validate_signature(
                top_level_functions["validate_generated_puzzle"], ["puzzle"], 0
            )
        )
    if "self_test" in top_level_functions:
        errors.extend(_validate_signature(top_level_functions["self_test"], [], 0))
    for name in REQUIRED_FUNCTIONS:
        function = top_level_functions.get(name)
        if function is not None and _placeholder_body(function):
            errors.append("Required function %s is only a placeholder." % name)

    numpy_alias_found = False
    default_rng_found = False
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if alias.name == "numpy" and alias.asname == "np":
                    numpy_alias_found = True
                if root in FORBIDDEN_MODULES:
                    errors.append("Importing %s is not allowed." % alias.name)
                elif root != "numpy" and root not in stdlib:
                    errors.append("External package %s is not allowed." % alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in FORBIDDEN_MODULES:
                errors.append("Importing from %s is not allowed." % (node.module or ""))
            elif root not in ("", "__future__", "numpy") and root not in stdlib:
                errors.append("External package %s is not allowed." % (node.module or ""))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                errors.append("Calling %s is not allowed." % node.func.id)
            parts = _attribute_parts(node.func)
            if parts == ["np", "random", "default_rng"]:
                default_rng_found = True
            if (
                parts[:2] == ["np", "random"]
                and parts
                not in (
                    ["np", "random", "default_rng"],
                    ["np", "random", "SeedSequence"],
                )
            ):
                errors.append(
                    "Global NumPy random APIs are not allowed; use a Generator instance."
                )
            if len(parts) >= 2 and parts[0] == "np" and parts[-1] in FORBIDDEN_NUMPY_IO:
                errors.append("NumPy file I/O via np.%s is not allowed." % parts[-1])
            if parts and parts[-1] == "tofile":
                errors.append("Array file I/O via tofile is not allowed.")
    if not numpy_alias_found:
        errors.append("Generator code must contain `import numpy as np`.")
    if not default_rng_found:
        errors.append(
            "Generator code must construct randomness with np.random.default_rng(...)."
        )
    return tree, list(dict.fromkeys(errors))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run independent host checks and generator self-tests")
    parser.add_argument("--seed-base", type=int, help="reproduce the seed suite reported by a previous check")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    generator_path = root / "generator.py"
    source = generator_path.read_text(encoding="utf-8")
    tree, errors = _static_check(source)
    if errors:
        for error in errors:
            print("ERROR: %s" % error, file=sys.stderr)
        return 1
    assert tree is not None
    namespace = {
        "__name__": "workspace_generator",
        "__file__": str(generator_path),
    }
    try:
        exec(compile(tree, "generator.py", "exec"), namespace)
        source_task = json.loads((root / "task.json").read_text(encoding="utf-8"))
        policy = json.loads((root / "acceptance.json").read_text(encoding="utf-8"))
        reference = {}
        reference_path = root / "reference_checks.py"
        if reference_path.is_file():
            # This optional module is supplied by the host and protected like evidence.
            reference = {"__name__": "reference_checks"}
            exec(compile(reference_path.read_text(encoding="utf-8"), str(reference_path), "exec"), reference)
        report = run_checks(namespace, source_task, policy,
                            args.seed_base if args.seed_base is not None else secrets.randbits(63), reference)
        if report["valid"]:
            try:
                namespace["self_test"]()
                report["self_test"] = "passed"
            except BaseException as exc:
                report["self_test"] = "failed"
                report["valid"] = False
                report["errors"].append(f"self_test: {type(exc).__name__}: {exc}")
        scratch = root / "scratch"
        scratch.mkdir(exist_ok=True)
        (scratch / "verification.json").write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")
        for error in report["errors"]:
            print("ERROR: " + error, file=sys.stderr)
        print("Reproduce with: python verify.py --seed-base %d" % report["seed_base"])
        print("Measurements: scratch/verification.json; reference solver: %s; review criteria: %s"
              % (report["reference_solver"], report["review_criteria"]))
        if not report["valid"]:
            return 1
    except BaseException as exc:
        print("ERROR: verification failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    print("generator.py passed static checks, independent host checks, and self_test()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
