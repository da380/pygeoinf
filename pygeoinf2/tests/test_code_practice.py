"""
The project's code-practice standard, enforced.

Docstrings, type hints and keyword-only optional arguments are house rules for
this package. A rule that is not checked drifts, so it is checked here rather
than recorded in a document nobody re-reads.

Scope is the package itself, not the tests: test functions document themselves
by their names, and pytest fixtures take positional arguments by design.
"""

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
SKIP_DIRS = {"tests", "__pycache__"}


def source_files() -> list[pathlib.Path]:
    """Every module in the package, excluding the test suite."""
    return sorted(
        path
        for path in PACKAGE.rglob("*.py")
        if not SKIP_DIRS & set(path.relative_to(PACKAGE).parts)
    )


def functions_and_classes(path: pathlib.Path):
    """Yield ``(node, class_name_or_None)`` for every top-level definition."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, None
        elif isinstance(node, ast.ClassDef):
            yield node, None
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield sub, node.name


def label(path: pathlib.Path, node, owner: str | None) -> str:
    name = f"{owner}.{node.name}" if owner else node.name
    return f"{path.relative_to(PACKAGE.parent)}:{node.lineno} {name}"


def is_public(name: str) -> bool:
    return not name.startswith("_")


def is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
class TestCodePractice:
    def test_public_definitions_have_docstrings(self, path):
        offenders = [
            label(path, node, owner)
            for node, owner in functions_and_classes(path)
            if is_public(node.name) and not ast.get_docstring(node)
        ]
        assert not offenders, "missing docstrings:\n  " + "\n  ".join(offenders)

    def test_optional_arguments_are_keyword_only(self, path):
        """An optional positional argument is a compatibility hazard.

        Once callers pass it positionally, its position is part of the API and
        no argument can be inserted before it.
        """
        offenders = []
        for node, owner in functions_and_classes(path):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if is_dunder(node.name):
                continue
            args = node.args
            if not args.defaults:
                continue
            positional = list(args.posonlyargs) + list(args.args)
            named = [a.arg for a in positional[len(positional) - len(args.defaults) :]]
            offenders.append(f"{label(path, node, owner)} -> {named}")
        assert not offenders, "positional optional arguments:\n  " + "\n  ".join(
            offenders
        )

    def test_signatures_are_annotated(self, path):
        offenders = []
        for node, owner in functions_and_classes(path):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            where = label(path, node, owner)
            args = node.args
            for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
                if arg.arg in ("self", "cls"):
                    continue
                if arg.annotation is None:
                    offenders.append(f"{where} (parameter {arg.arg})")
            if node.returns is None and node.name != "__init__":
                offenders.append(f"{where} (return)")
        assert not offenders, "missing annotations:\n  " + "\n  ".join(offenders)
