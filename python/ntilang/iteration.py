"""Grid construction and row-major coordinate helpers for the Python frontend."""

from __future__ import annotations

import ast

from . import language
from .ir import Expr
from .macros import ValueNode


def _integer(parser, value, node, label):
    if type(value) is int:
        return value
    if isinstance(value, Expr) and value.op == "const" and type(value.value) is int:
        return value.value
    expression = parser.scalar_value(value, node)
    if not str(parser.value_dtype(expression, node)).startswith(("int", "uint")):
        parser.fail(node, f"{label} requires integer values")
    return expression


def grid(parser, node, *, parallel, nested):
    """Build ordinary serial loops from values bound once in source order.

    As in the pinned TIR Grid builder, each runtime bound is evaluated on entry
    to its nested loop. All construction-time macro expansions happen before the
    loop nest, and all source induction names are rebound only after binding the
    complete argument list.
    """
    parser.keywords(node.iter, set())
    if node.orelse:
        parser.fail(node, "Loop else clauses are not supported")
    targets = node.target.elts if isinstance(node.target, (ast.Tuple, ast.List)) else [node.target]
    extents = []
    for arg in node.iter.args:
        source = arg.value if isinstance(arg, ast.Starred) else arg
        value = parser.macro_value(source)
        if isinstance(arg, ast.Starred):
            if type(value) not in (tuple, list):
                parser.fail(arg, "Starred grid arguments require a tuple or list")
            items = value
        else:
            items = (value,)
        for item in items:
            extent = _integer(parser, item, source, "T.grid extent")
            if type(extent) is int and extent < 0:
                parser.fail(source, "T.grid extents must be nonnegative static integers or runtime integers")
            extents.append(ValueNode(extent, source))
    if not extents or len(targets) != len(extents) or any(not isinstance(t, ast.Name) for t in targets):
        parser.fail(node, "T.grid requires one named variable per extent")
    if len({target.id for target in targets}) != len(targets):
        parser.fail(node, "T.grid variables must have unique names")
    serial_name = parser.fresh("grid_serial")
    parser.constants[serial_name] = language.serial
    body = node.body
    for target, extent in reversed(tuple(zip(targets, extents))):
        call = ast.Call(func=ast.Name(id=serial_name, ctx=ast.Load()), args=[extent], keywords=[])
        loop = ast.For(target=target, iter=call, body=body, orelse=[])
        ast.copy_location(loop, node)
        ast.fix_missing_locations(loop)
        body = [loop]
    return parser.statement(body[0], parallel=parallel, nested=nested)


def coordinates(parser, call):
    arguments = parser.bind_call(call, ["index", "shape"], {})
    values = {key: parser.macro_value(node) for key, node in arguments.items()}
    index = _integer(parser, values["index"], arguments["index"], "Linear index")
    shape = values["shape"]
    if type(shape) not in (tuple, list):
        parser.fail(call, "Coordinate shape requires a tuple or list")
    extents = tuple(_integer(parser, value, arguments["shape"], "Coordinate extent") for value in shape)
    if any(type(value) is int and value <= 0 for value in extents):
        parser.fail(call, "Coordinate extents must be positive")
    if type(index) is int and all(type(value) is int for value in extents):
        return tuple(language.index_to_coordinates(index, extents))
    result = []
    index = Expr("const", value=index) if type(index) is int else index
    for extent in reversed(extents):
        extent = Expr("const", value=extent) if type(extent) is int else extent
        result.append(Expr("%", (index, extent)))
        index = Expr("//", (index, extent))
    return tuple(reversed(result))
