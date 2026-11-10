"""Extract and statically validate a model-written puzzle generator."""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


PYTHON_FENCE_RE = re.compile(r"```([^\n`]*)\r?\n(.*?)\r?\n```", re.DOTALL)
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
FORBIDDEN_NUMPY_IO = {"fromfile", "genfromtxt", "load", "loadtxt", "save", "savetxt", "tofile"}
RESPONSE_VALIDATION_MODE = "fence extraction, compilation, and static AST checks"
CODE_VALIDATION_MODE = "compilation and static AST checks"


@dataclass(frozen=True)
class GeneratorValidation:
    valid: bool
    errors: List[str]
    extracted_code: Optional[str]
    code_sha256: Optional[str]
    code_line_count: int
    discovered_functions: List[str]
    imported_modules: List[str]
    validation_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "code_extracted": self.extracted_code is not None,
            "code_sha256": self.code_sha256,
            "code_line_count": self.code_line_count,
            "required_functions": list(REQUIRED_FUNCTIONS),
            "discovered_functions": self.discovered_functions,
            "imported_modules": self.imported_modules,
            "execution_performed": False,
            "validation_mode": self.validation_mode,
        }


def _result(
    errors: List[str],
    code: Optional[str] = None,
    tree: Optional[ast.Module] = None,
    validation_mode: str = RESPONSE_VALIDATION_MODE,
) -> GeneratorValidation:
    functions: List[str] = []
    imports: List[str] = []
    if tree is not None:
        functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest() if code is not None else None
    return GeneratorValidation(
        valid=not errors,
        errors=errors,
        extracted_code=code,
        code_sha256=code_hash,
        code_line_count=len(code.splitlines()) if code is not None else 0,
        discovered_functions=functions,
        imported_modules=sorted(set(imports)),
        validation_mode=validation_mode,
    )


def _positional_names(function: ast.FunctionDef) -> List[str]:
    return [item.arg for item in function.args.posonlyargs + function.args.args]


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _validate_signature(function: ast.FunctionDef, expected: List[str], default_count: int) -> List[str]:
    errors: List[str] = []
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
    if len(arguments.defaults) != default_count or any(not _is_none(item) for item in arguments.defaults):
        if default_count:
            errors.append("%s must default its final %d arguments to None." % (function.name, default_count))
        else:
            errors.append("%s must not define default arguments." % function.name)
    return errors


def _attribute_parts(node: ast.AST) -> List[str]:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return list(reversed(parts))


def _is_main_guard(node: ast.If) -> bool:
    try:
        rendered = ast.unparse(node.test)
    except Exception:  # pragma: no cover - ast.unparse exists on supported Python.
        return False
    return "__name__" in rendered and "__main__" in rendered


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


def validate_generator_response(response: str) -> GeneratorValidation:
    errors: List[str] = []
    if not isinstance(response, str) or not response.strip():
        return _result(["Response is empty."])

    matches = list(PYTHON_FENCE_RE.finditer(response))
    if response.count("```") != 2 or len(matches) != 1:
        return _result(["Response must contain exactly one fenced Python code block."])
    match = matches[0]
    language = match.group(1).strip().lower()
    if language not in ("python", "py"):
        errors.append("The single code fence must be labeled python.")
    if not response[: match.start()].strip():
        errors.append("The required explanation must appear before the code block.")
    if response[match.end() :].strip():
        errors.append("No content may appear after the Python code block.")

    code = match.group(2)
    return _validate_generator_code(code, errors, RESPONSE_VALIDATION_MODE)


def validate_generator_code(code: str) -> GeneratorValidation:
    """Validate generator source read from a workspace file."""
    if not isinstance(code, str) or not code.strip():
        return _result(
            ["Generator code is empty."],
            code if isinstance(code, str) else None,
            validation_mode=CODE_VALIDATION_MODE,
        )
    return _validate_generator_code(code, [], CODE_VALIDATION_MODE)


def _validate_generator_code(
    code: str,
    errors: List[str],
    validation_mode: str,
) -> GeneratorValidation:
    if not code.strip():
        return _result(
            errors + ["The Python code block is empty."],
            code,
            validation_mode=validation_mode,
        )
    try:
        tree = ast.parse(code, filename="generator.py", mode="exec")
        compile(tree, "generator.py", "exec")
    except SyntaxError as exc:
        return _result(
            errors + ["Generator code does not compile: %s" % exc],
            code,
            validation_mode=validation_mode,
        )

    top_level_functions: Dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name in REQUIRED_FUNCTIONS:
            errors.append("Required function %s must be synchronous." % node.name)
        elif isinstance(node, ast.FunctionDef):
            if node.name in top_level_functions:
                errors.append("Top-level function %s is defined more than once." % node.name)
            top_level_functions[node.name] = node
        elif isinstance(node, ast.If) and _is_main_guard(node):
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
                and parts not in (["np", "random", "default_rng"], ["np", "random", "SeedSequence"])
            ):
                errors.append("Global NumPy random APIs are not allowed; use a Generator instance.")
            if len(parts) >= 2 and parts[0] == "np" and parts[-1] in FORBIDDEN_NUMPY_IO:
                errors.append("NumPy file I/O via np.%s is not allowed." % parts[-1])
            if parts and parts[-1] == "tofile":
                errors.append("Array file I/O via tofile is not allowed.")
    if not numpy_alias_found:
        errors.append("Generator code must contain `import numpy as np`.")
    if not default_rng_found:
        errors.append("Generator code must construct randomness with np.random.default_rng(...).")

    return _result(
        list(dict.fromkeys(errors)),
        code,
        tree,
        validation_mode=validation_mode,
    )
