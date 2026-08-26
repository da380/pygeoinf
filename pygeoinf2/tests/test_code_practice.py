"""
The project's code-practice standard, enforced.

Docstrings, type hints and keyword-only optional arguments are house rules for
this package. A rule that is not checked drifts, so it is checked here rather
than recorded in a document nobody re-reads.

Scope is the package itself. The tests are excluded because test functions
document themselves by their names and pytest fixtures take positional
arguments by design. The examples are excluded because they are teaching
material: annotating a three-line helper written to illustrate one idea makes
the illustration worse, and the examples have their own test, which is that
they run.
"""

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
SKIP_DIRS = {"tests", "examples", "__pycache__"}


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


CORE_VECTOR_API = frozenset(
    {
        "add",
        "axpy",
        "copy",
        "dim",
        "inner_product",
        "mean",
        "negative",
        "norm",
        "random",
        "scale",
        "scale_inplace",
        "squared_norm",
        "subtract",
        "white_noise",
        "zero",
    }
)


def concrete_spaces() -> list[type]:
    """Every concrete HilbertSpace subclass the package defines."""
    import pygeoinf2
    import pygeoinf2.spaces
    from pygeoinf2.algebra.spaces import HilbertSpace

    seen: dict[str, type] = {}
    for module in (pygeoinf2, pygeoinf2.spaces):
        for name in dir(module):
            value = getattr(module, name)
            if isinstance(value, type) and issubclass(value, HilbertSpace):
                seen[value.__qualname__] = value
    return sorted(seen.values(), key=lambda c: c.__qualname__)


@pytest.mark.parametrize("space_class", concrete_spaces(), ids=lambda c: c.__name__)
def test_space_attributes_do_not_shadow_the_vector_api(space_class):
    """A space's own attributes share a namespace with the whole vector API.

    A ``PeriodicBox`` given a ``scale`` property for its Sobolev length shadows
    ``HilbertSpace.scale(a, x)``, and every vector operation that routes
    through it fails with ``'float' object is not callable`` somewhere far from
    the cause. Cheap to check, and not obvious until it happens.
    """
    from pygeoinf2.algebra.spaces import HilbertSpace

    offenders = []
    for name in CORE_VECTOR_API:
        base = getattr(HilbertSpace, name, None)
        override = getattr(space_class, name, None)
        if base is None or override is None or override is base:
            continue
        # A method may be overridden by a method; only a value shadowing a
        # callable, or the reverse, is a mistake.
        if callable(base) and isinstance(override, property):
            offenders.append(f"{name}: method shadowed by a property")
        elif isinstance(base, property) and callable(override):
            offenders.append(f"{name}: property shadowed by a method")
    assert not offenders, f"{space_class.__name__} shadows the vector API:\n  " + (
        "\n  ".join(offenders)
    )
