"""Simple dependency graph generator for classes in this repository.

Usage:
    python scripts/dependency_graph.py <module> <ClassName>
    [-o out.dot] [--depth N]

Example:
    python scripts/dependency_graph.py \
        pygeoinf.hilbert_space HilbertSpace -o hilbert.dot

The script performs lightweight static analysis (AST) of the module containing
the target class and extracts:
 - base classes
 - names used in annotations on methods/attributes
 - names referenced in the class body (Name nodes)
It maps referenced names to modules using the module's import table when
possible and emits a Graphviz DOT file showing edges from the target class to
the modules / symbols it depends on.

This is intentionally small and conservative (no runtime imports).
"""

from __future__ import annotations

import ast
import argparse
import importlib.util
# no additional stdlib imports required
from typing import Dict, Set, Tuple


def find_module_file(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"Cannot find module {module_name}")
    return spec.origin


def parse_module_ast(path: str) -> ast.Module:
    with open(path, "r", encoding="utf8") as fh:
        src = fh.read()
    return ast.parse(src, filename=path)


def get_import_map(module_node: ast.Module) -> Dict[str, str]:
    """Return a map alias -> fully qualified module or name.

    Examples:
        import numpy as np    -> {'np': 'numpy'}
        from foo import Bar    -> {'Bar': 'foo.Bar'}
        from foo import Bar as B -> {'B': 'foo.Bar'}
    """
    imap: Dict[str, str] = {}
    for node in module_node.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                imap[name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                asname = alias.asname or alias.name
                if module:
                    imap[asname] = module + "." + alias.name
                else:
                    imap[asname] = alias.name
    return imap


def get_name_from_node(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    try:
        # Fallback to ast.unparse for complex nodes (Py3.9+)
        return ast.unparse(node)  # type: ignore
    except Exception:
        return str(type(node))


def collect_references(class_node: ast.ClassDef) -> Set[str]:
    refs: Set[str] = set()

    # Add base class names
    for base in class_node.bases:
        refs.add(get_name_from_node(base))

    # Walk the class body for annotations and Name nodes
    for node in ast.walk(class_node):
        if isinstance(node, ast.AnnAssign):
            # annotation could be Name/Subscript/etc
            if node.annotation is not None:
                refs.update(_names_in_annotation(node.annotation))
        elif isinstance(node, ast.FunctionDef):
            # arguments annotations
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.annotation is not None:
                    refs.update(_names_in_annotation(arg.annotation))
            if node.returns is not None:
                refs.update(_names_in_annotation(node.returns))
            # decorators
            for d in node.decorator_list:
                refs.add(get_name_from_node(d))
        elif isinstance(node, ast.Call):
            # function/class call - include the callee name
            refs.add(get_name_from_node(node.func))
        elif isinstance(node, ast.Attribute):
            refs.add(get_name_from_node(node))
        elif isinstance(node, ast.Name):
            refs.add(node.id)

    # Filter out builtin names
    builtin_names = (
        set(dir(__builtins__))
        if isinstance(__builtins__, dict)
        else set(dir(__builtins__))
    )
    refs = {r for r in refs if r and r not in builtin_names}
    return refs


def _names_in_annotation(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(get_name_from_node(n))
    return names


def build_dependency_edges(module_name: str, class_name: str, depth: int = 0):
    """Return list of edges (from_node, to_node) for the class dependencies.

    from_node will be represented as 'module.ClassName'. to_node will be either
    a module name (if resolved via imports) or 'module.Name'.
    """
    path = find_module_file(module_name)
    mod_ast = parse_module_ast(path)
    imap = get_import_map(mod_ast)

    # find class
    target: ast.ClassDef | None = None
    for node in mod_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            target = node
            break
    if target is None:
        raise ValueError(
            "Class %s not found in module %s" % (class_name, module_name)
        )

    refs = collect_references(target)

    edges = []
    from_node = f"{module_name}.{class_name}"
    for r in sorted(refs):
        # Skip references to the class itself
        if r.split(".")[-1] == class_name:
            continue
        # Try resolve via imports map
        if r in imap:
            to = imap[r]
        else:
            # If r contains a dot and starts with an imported alias,
            # try splitting
            if "." in r:
                root = r.split(".")[0]
                if root in imap:
                    to = imap[root] + "." + ".".join(r.split(".")[1:])
                else:
                    to = f"{module_name}.{r}"
            else:
                # default to same module
                to = f"{module_name}.{r}"
        edges.append((from_node, to))

    # depth > 0 not implemented fully (placeholder for future recursion)
    return edges


def emit_dot(edges: "list[Tuple[str, str]]") -> str:
    lines = ["digraph G {", "  rankdir=LR;"]
    nodes = set()
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)
        lines.append(f'  "{a}" -> "{b}";')
    for n in sorted(nodes):
        lines.append(f'  "{n}";')
    lines.append("}")
    return "\n".join(lines)


def emit_dot_collapsed(edges: "list[Tuple[str, str]]") -> str:
    """Emit a simplified dot graph collapsing nodes to their module root.

    Example: 'pygeoinf.hilbert_space.HilbertSpace' -> 'pygeoinf.hilbert_space'
    """
    def root(s: str) -> str:
        # remove trailing symbol name if present
        parts = s.split('.')
        if len(parts) <= 2:
            return '.'.join(parts[:-0]) if parts else s
        return '.'.join(parts[:2])

    simple_edges = set()
    for a, b in edges:
        simple_edges.add((root(a), root(b)))

    lines = ["digraph G {", "  rankdir=LR;"]
    nodes = set()
    for a, b in sorted(simple_edges):
        nodes.add(a)
        nodes.add(b)
        lines.append(f'  "{a}" -> "{b}";')
    for n in sorted(nodes):
        lines.append(f'  "{n}";')
    lines.append("}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "module",
        help=(
            "Module containing the class, e.g. 'pygeoinf.hilbert_space'"
        ),
    )
    parser.add_argument(
        "class_name",
        help=("Class name inside the module, e.g. 'HilbertSpace'"),
    )
    parser.add_argument(
        "-o",
        "--out",
        help=("Output DOT file (default: stdout)")
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help=("Recursion depth for dependencies (0: only direct)"),
    )
    args = parser.parse_args(argv)

    edges = build_dependency_edges(
        args.module, args.class_name, depth=args.depth
    )
    dot = emit_dot(edges)

    if args.out:
        with open(args.out, "w", encoding="utf8") as fh:
            fh.write(dot)
        print(f"Wrote DOT file to {args.out}")
    else:
        print(dot)


if __name__ == "__main__":
    main()
