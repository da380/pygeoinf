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


def source_line(path, number: int) -> str:
    """One line of a file, for reading an inline escape comment."""
    return path.read_text().split("\n")[number - 1]


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
            # __init__ is *not* exempt here, unlike the other rules: a
            # constructor is exactly where a positional argument's position
            # gets locked into every call site in the wild.
            if is_dunder(node.name) and node.name != "__init__":
                continue
            args = node.args
            if not args.defaults:
                continue
            # An explicit escape, for the case the rule genuinely cannot
            # accommodate: cooperative multiple inheritance, where a superclass
            # calls __init__ positionally and a keyword-only argument would
            # break the chain. It must be written on the def line with a
            # reason, so it is a decision rather than an oversight.
            if "noqa: positional" in source_line(path, node.lineno):
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
    import pygeoinf2.symmetric_space
    from pygeoinf2.algebra.spaces import HilbertSpace

    seen: dict[str, type] = {}
    for module in (pygeoinf2, pygeoinf2.symmetric_space):
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


class TestTheCatalogueMatchesTheCode:
    """The review found the catalogue "materially overstating what was
    ported", and the same class of error twice more since. A document nobody
    can trust is worse than no document, so the claims are checked here rather
    than re-read."""

    @staticmethod
    def catalogue() -> str:
        import pathlib

        return (
            pathlib.Path(__file__).resolve().parent.parent / "V1_CATALOGUE.md"
        ).read_text()

    def test_no_row_contradicts_its_own_status(self):
        """A row marked Ported whose text says "not ported" is the exact
        defect the review named, and there were twenty of them."""
        import re

        rows = re.findall(
            r"^\| ([^|]+) \| (Ported[^|]*) \| ([^|]*) \|", self.catalogue(), re.M
        )
        contradictory = [
            v1.strip()
            for v1, _, v2 in rows
            if re.search(r"\bnot ported\b|\bno v2 home\b", v2, re.I)
        ]
        assert not contradictory, f"marked Ported but say otherwise: {contradictory}"

    def test_every_named_v2_symbol_exists(self):
        """A row naming a v2 symbol that does not exist is a promise the code
        does not keep. Keyword arguments and prose are backticked too, so this
        checks only names that look like symbols and are not obviously words."""
        import importlib
        import inspect
        import pkgutil
        import re

        import pygeoinf2

        known: set[str] = set()
        for module in pkgutil.walk_packages(pygeoinf2.__path__, "pygeoinf2."):
            if ".tests" in module.name or ".examples" in module.name:
                continue
            try:
                loaded = importlib.import_module(module.name)
            except Exception:  # pragma: no cover - optional dependencies
                continue
            for name in dir(loaded):
                if name.startswith("__"):
                    continue
                known.add(name)
                attribute = getattr(loaded, name, None)
                if inspect.isclass(attribute):
                    known.update(
                        item for item in dir(attribute) if not item.startswith("__")
                    )

        # Only paths rooted in one of v2's own subpackages. A dotted name is
        # how a symbol is written when it is meant as one -- but plenty of
        # them belong to somebody else (``scipy.stats``, ``SHCoeffs.spectrum``)
        # or are not symbols at all (``py.typed``), and this is a check on
        # v2's promises rather than on anyone else's.
        roots = {module.name for module in pkgutil.iter_modules(pygeoinf2.__path__)}
        # Submodule names count as things that exist: the catalogue refers to
        # `backends.mfem` and the like.
        for module in pkgutil.walk_packages(pygeoinf2.__path__, "pygeoinf2."):
            known.add(module.name.rsplit(".", 1)[-1])
        dotted = re.findall(
            r"`(?:pygeoinf2\.)?((?:\w+\.)+\w+)`", self.catalogue()
        )
        missing = sorted(
            {
                path.split(".")[-1]
                for path in dotted
                # A trailing "py" is a file name -- `plotting.py` names the
                # module, not a symbol called py.
                if path.split(".")[0] in roots
                and path.split(".")[-1] not in known
                and path.split(".")[-1] != "py"
            }
        )
        assert not missing, f"named in the catalogue but not in the code: {missing}"

    # Names the catalogue claimed as Ported that do not exist anywhere in v2.
    # The dotted-path check above cannot see them, because the rows write them
    # bare: `deflated_pointwise_variance`, not `gaussian.deflated_...`. Each
    # was verified absent by grep over the package, and each row now says so.
    ABSENT_FROM_V2 = (
        "deflated_pointwise_variance",
        "deflated_pointwise_std",
        "LevelSet",
        "SublevelSet",
        "Cut",
        "Bundle",
    )

    @pytest.mark.parametrize("name", ABSENT_FROM_V2)
    def test_a_row_naming_something_absent_does_not_claim_it_is_ported(self, name):
        """The status of the row whose *v1* column names it must not begin
        with "Ported": that is the exact defect the review named, and these
        six survived the first pass because the names are written bare."""
        import importlib
        import pkgutil
        import re

        import pygeoinf2

        for module in pkgutil.walk_packages(pygeoinf2.__path__, "pygeoinf2."):
            if ".tests" in module.name or ".examples" in module.name:
                continue
            try:
                loaded = importlib.import_module(module.name)
            except Exception:  # pragma: no cover - optional dependencies
                continue
            assert not hasattr(loaded, name), (
                f"{name} exists in {module.name}: the catalogue row can be "
                f"restored to Ported, and this list shortened"
            )

        rows = re.findall(r"^\| ([^|]+) \| ([^|]+) \|", self.catalogue(), re.M)
        claiming = [
            v1.strip()
            for v1, status in rows
            if f"`{name}`" in v1 and status.strip().startswith("Ported")
        ]
        assert not claiming, f"{name} does not exist but its row claims Ported"


# Parameters whose meaning is fixed across the whole library and documented in
# one place, so repeating them in every signature is noise rather than
# information: `rng` is always the generator, `n_jobs` and `backend` are always
# joblib's, and DESIGN.md 7 and pygeoinf2/parallel.py say so once.
CONVENTIONAL = {"rng", "n_jobs", "backend"}



def relative(path) -> str:
    """The path within the package, since several files share a bare name.

    Three modules are called ``base.py`` and two ``convex.py``; keying the
    debt list on the bare name silently merged them.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    return str(pathlib.Path(path).resolve().relative_to(root))


def documentation_gaps(path) -> list[str]:
    """Public functions whose docstring omits part of their contract.

    Two things, and deliberately not a blanket "document every parameter":

    * **Raises**, wherever the body raises. What a function refuses is part of
      its contract and is the one thing a reader cannot guess.
    * **Args**, for parameters carrying a *choice* -- anything optional or
      keyword-only. Those are the ones a caller has to understand. A required
      ``x`` on ``copy(x)``, whose summary already says what it copies, is not.

    The blanket rule was tried first and would have added "Args: x: the
    vector." to some fifty vector-algebra methods, diluting docstrings that
    currently explain *why*. This targets what a reader cannot infer.
    """
    gaps = []
    tree = ast.parse(path.read_text())
    # Only what a caller can reach: module-level functions and methods. A
    # closure defined inside one is an implementation detail, and requiring
    # its arguments to be documented documents nothing.
    reachable = []
    for parent in [tree] + [
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
    ]:
        reachable.extend(
            n
            for n in parent.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    for node in reachable:
        if node.name.startswith("_"):
            continue
        doc = ast.get_docstring(node) or ""
        if (
            any(isinstance(n, ast.Raise) and n.exc is not None for n in ast.walk(node))
            and "Raises:" not in doc
        ):
            gaps.append(f"{path.name}:{node.lineno} {node.name} (no Raises)")

        args = node.args
        choices = [a.arg for a in args.kwonlyargs]
        positional = list(args.posonlyargs) + list(args.args)
        if args.defaults:
            choices += [a.arg for a in positional[len(positional) - len(args.defaults) :]]
        choices = [
            c for c in choices if c not in ("self", "cls") and c not in CONVENTIONAL
        ]
        undocumented = (
            [c for c in choices if f"{c}:" not in doc] if "Args:" in doc else choices
        )
        if undocumented:
            gaps.append(f"{path.name}:{node.lineno} {node.name} -> {undocumented}")
    return gaps


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
class TestDocstringsCarryTheContract:
    """What a function refuses, and what its options mean.

    K Should-12. This was enforced for a while against a shrinking list of
    files with a recorded number of gaps each, so the standard could bite
    before the backlog was cleared. The backlog is cleared -- 278 gaps across
    37 files, taken to zero -- so the list is gone and the rule is simply the
    rule.
    """

    def test_the_contract_is_documented(self, path):
        gaps = documentation_gaps(path)
        assert not gaps, (
            f"{path.name} has {len(gaps)} documentation gaps:\n  "
            + "\n  ".join(gaps)
        )


class TestTheNamespaceHasNoHoles:
    """A name that exists but cannot be imported from the package is a hole.

    The re-review found five: ``MassWeightedSpace``, ``HilbertModule`` and
    ``require_module`` reachable only as ``pygeoinf2.algebra.spaces.X``,
    ``resolve_solver`` only as ``pygeoinf2.numerics.solvers.resolve_solver``,
    and the MFEM backend's ``matern_measure`` — which one of the examples
    imports — missing from its ``__all__``. D-5 says the top-level namespace
    is the interface; a class documented as the reason
    ``from_formal_adjoint`` exists has to be in it.
    """

    @staticmethod
    def modules_with_exports():
        """Every importable module in the package that declares ``__all__``."""
        import importlib
        import pkgutil

        import pygeoinf2

        found = []
        for info in pkgutil.walk_packages(pygeoinf2.__path__, prefix="pygeoinf2."):
            if any(part in SKIP_DIRS for part in info.name.split(".")):
                continue
            try:
                module = importlib.import_module(info.name)
            except ImportError:  # an optional backend that is not installed
                continue
            if hasattr(module, "__all__"):
                found.append(module)
        return found

    def test_every_exported_name_resolves(self):
        offenders = [
            f"{module.__name__}.{name}"
            for module in self.modules_with_exports()
            for name in module.__all__
            if not hasattr(module, name)
        ]
        assert not offenders, "names in __all__ that do not exist:\n  " + "\n  ".join(
            offenders
        )

    @pytest.mark.parametrize(
        "package, names",
        [
            (
                "pygeoinf2",
                ["MassWeightedSpace", "HilbertModule", "require_module"],
            ),
            (
                "pygeoinf2.algebra",
                ["MassWeightedSpace", "HilbertModule", "require_module"],
            ),
            (
                "pygeoinf2.numerics",
                ["resolve_solver", "weighted_chi2_cdf", "weighted_chi2_quantile"],
            ),
            (
                "pygeoinf2.backends.mfem",
                ["white_noise_load", "matern_measure"],
            ),
        ],
    )
    def test_the_named_holes_are_closed(self, package, names):
        import importlib

        module = importlib.import_module(package)
        for name in names:
            assert hasattr(module, name), f"{package}.{name} is not reachable"
            assert name in module.__all__, f"{package}.{name} is not in __all__"
