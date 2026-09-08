"""Source-backed macro definitions and hygienic AST names."""

from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
from dataclasses import dataclass

from . import language
from .ir import CompileError, Expr


class ValueNode(ast.expr):
    """A parsed macro value embedded in an AST without evaluating Python code."""

    _fields = ()

    def __init__(self, value, source):
        self.value = value
        ast.copy_location(self, source)


@dataclass(frozen=True)
class RegionValue:
    buffer: str
    origin: tuple[Expr, ...]
    extents: tuple[int, ...]


@dataclass(frozen=True)
class ReferenceValue:
    target: ast.expr


@dataclass(frozen=True)
class Definition:
    node: ast.FunctionDef
    filename: str
    first_line: int
    environment: dict


def definition(macro):
    function = macro.function
    try:
        lines, first_line = inspect.getsourcelines(function)
        node = next(
            n for n in ast.parse(textwrap.dedent("".join(lines))).body if isinstance(n, ast.FunctionDef)
        )
    except (OSError, TypeError, StopIteration) as exc:
        raise CompileError("Macro source must be available in a Python file") from exc
    closure = inspect.getclosurevars(function)
    environment = {
        **vars(builtins),
        **function.__globals__,
        **dict(macro.annotation_locals),
        **closure.nonlocals,
    }
    return Definition(node, inspect.getsourcefile(function) or "<macro>", first_line, environment)


def is_reference(annotation, environment):
    if annotation is language.Ref:
        return True
    if not isinstance(annotation, str):
        return False
    try:
        node = ast.parse(annotation, mode="eval").body
    except SyntaxError:
        return False
    if isinstance(node, ast.Name):
        return environment.get(node.id) is language.Ref
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return environment.get(node.value.id) is language and node.attr == "Ref"
    return False


def rename(definition, fresh):
    node = definition.node
    locals_ = {
        part.id for part in ast.walk(node) if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)
    }
    parameters = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    parameters += [arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None]
    locals_.update(arg.arg for arg in parameters)
    names = locals_ | {part.id for part in ast.walk(node) if isinstance(part, ast.Name)}
    mapping = {name: fresh(name) for name in sorted(names)}
    constants = {
        mapping[name]: definition.environment[name]
        for name in names - locals_
        if name in definition.environment
    }

    class Renamer(ast.NodeTransformer):
        def visit_Name(self, current):
            return ast.copy_location(ast.Name(id=mapping[current.id], ctx=current.ctx), current)

    body = tuple(Renamer().visit(statement) for statement in node.body)
    return body, mapping, constants
