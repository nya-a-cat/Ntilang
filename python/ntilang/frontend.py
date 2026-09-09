"""Restricted Python AST frontend; unsupported syntax is a compilation error."""

from __future__ import annotations

import ast
import builtins
import inspect
import operator
import textwrap
from contextlib import contextmanager
from math import prod

from . import language, macros, metadata, scan
from .ir import (
    DTYPES,
    Buffer,
    CompileError,
    Expr,
    Kernel,
    Partition,
    Region,
    ScalarParameter,
    SourceLocation,
    Statement,
    TensorType,
    loop_controls,
)
from .scalar import (
    BINARY_MATH_OPS,
    BIT_COUNT_OPS,
    FAST_MATH_OPS,
    IEEE_MATH_OPS,
    TRANSCENDENTAL_OPS,
    UNARY_MATH_OPS,
    constant_integer,
    expression_dtype,
)
from .validation import affine, resolved_dtype

NO_CONSTRUCTION_CALL = object()

BINOPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
}
BITWISE_CALLS = {
    "bitwise_and": "&",
    "bitwise_or": "|",
    "bitwise_xor": "^",
    "bitwise_not": "invert",
    "shift_left": "<<",
    "shift_right": ">>",
}
DIVISION_CALLS = {
    "floordiv": "//",
    "floormod": "%",
    "truncdiv": "truncdiv",
    "truncmod": "truncmod",
    "ceildiv": "ceildiv",
    "cdiv": "ceildiv",
}
COMPARISONS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
STATIC_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}


def python_literal(value):
    if type(value) in (int, float, bool, str, type(None)):
        return True
    if type(value) is tuple:
        return all(python_literal(item) for item in value)
    if type(value) is dict:
        return all(python_literal(key) and python_literal(item) for key, item in value.items())
    return False


class Parser:
    def __init__(self, program: language.PrimFunc):
        self.function = program.function
        try:
            lines, self.first_line = inspect.getsourcelines(self.function)
            self.source = textwrap.dedent("".join(lines))
        except (OSError, TypeError) as exc:
            raise CompileError("Kernel source must be available in a Python file") from exc
        self.filename = inspect.getsourcefile(self.function) or "<kernel>"
        self.node = next(n for n in ast.parse(self.source).body if isinstance(n, ast.FunctionDef))
        closure = inspect.getclosurevars(self.function)
        self.constants = {**self.function.__globals__, **dict(program.annotation_locals), **closure.nonlocals}
        self.buffers: dict[str, Buffer] = {}
        self.variables: set[str] = set()
        self.initialized: set[str] = set()
        self.allocated: list[Buffer] = []
        self.threads = 0
        self.parallel_context = None
        self.mutable = {}
        self.bindings = {}
        self.parameter_types = {}
        self.before_launch = False
        self.loops = []
        self.values = {}
        self.pending = []
        self.parse_context = (False, False)
        self.macro_stack = []
        self.boolean_macro_context = 0
        self.frames = [object()]
        self.value_frames = {}
        self.local_names = {
            n.id for n in ast.walk(self.node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        } | {arg.arg for arg in self.node.args.args}
        self.name_counter = 0
        self.source_names = {}
        self.used_names = set(self.constants) | {
            n.id if isinstance(n, ast.Name) else n.arg
            for n in ast.walk(self.node)
            if isinstance(n, (ast.Name, ast.arg))
        }

    def location(self, node):
        return SourceLocation(self.filename, self.first_line + node.lineno - 1, node.col_offset)

    def fail(self, node, message):
        raise CompileError(message, self.location(node))

    @contextmanager
    def lexical_scope(self):
        self.frames.append(object())
        try:
            yield
        finally:
            self.frames.pop()

    def check_name(self, node):
        frame = self.value_frames.get(node.id)
        if frame is not None and frame not in self.frames:
            self.fail(node, f"Variable {node.id} is used outside its defining region")

    def python_value(self, node):
        """Evaluate whitelisted construction-time expressions without running user code."""
        if isinstance(node, macros.ValueNode):
            value = node.value
        elif isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, ast.Name):
            self.check_name(node)
            if node.id in self.values:
                value = self.values[node.id]
            elif node.id in self.variables or node.id in self.buffers or node.id in self.local_names:
                self.fail(node, "Expected a Python construction-time value")
            else:
                value = self.constants.get(node.id, getattr(builtins, node.id, None))
                if value is None and node.id not in self.constants:
                    self.fail(node, "Expected a Python construction-time value")
        elif isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.python_value(item) for item in node.elts)
        elif isinstance(node, ast.Dict) and all(key is not None for key in node.keys):
            values = {}
            for key, item in zip(node.keys, node.values):
                key = self.python_value(key)
                if not python_literal(key) or type(key) is dict:
                    self.fail(node, "Construction-time dictionary keys require built-in immutable values")
                values[key] = self.python_value(item)
            return values
        elif isinstance(node, ast.Subscript):
            value, key = self.python_value(node.value), self.python_value(node.slice)
            if type(value) not in (tuple, dict, str):
                self.fail(node, "Construction-time indexing requires a tuple, dictionary, or string")
            if not python_literal(key) or type(key) is dict:
                self.fail(node, "Construction-time indices require built-in immutable values")
            try:
                return self.python_value(macros.ValueNode(value[key], node))
            except (IndexError, KeyError, TypeError) as exc:
                self.fail(node, f"Invalid construction-time index: {exc}")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            owner = self.python_value(node.value)
            if owner is language:
                value = inspect.getattr_static(owner, node.attr, None)
                if value is None:
                    self.fail(node, "Unknown language attribute")
            else:
                self.fail(node, "Construction-time attributes require a language namespace")
        elif isinstance(node, ast.BinOp) and type(node.op) in {*STATIC_OPS, ast.Div}:
            left, right = self.python_value(node.left), self.python_value(node.right)
            if type(left) not in (int, float, bool, str, tuple) or type(right) not in (
                int,
                float,
                bool,
                str,
                tuple,
            ):
                self.fail(node, "Construction-time arithmetic requires built-in scalar or tuple values")
            try:
                return (operator.truediv if isinstance(node.op, ast.Div) else STATIC_OPS[type(node.op)])(
                    left, right
                )
            except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
                self.fail(node, f"Invalid construction-time arithmetic: {exc}")
        elif isinstance(node, ast.UnaryOp):
            value = self.python_value(node.operand)
            operators = {
                ast.UAdd: operator.pos,
                ast.USub: operator.neg,
                ast.Invert: operator.invert,
                ast.Not: operator.not_,
            }
            if type(node.op) not in operators or type(value) not in (int, float, bool):
                self.fail(node, "Unsupported construction-time unary expression")
            return operators[type(node.op)](value)
        elif isinstance(node, ast.Compare):
            operators = {
                ast.Lt: operator.lt,
                ast.LtE: operator.le,
                ast.Gt: operator.gt,
                ast.GtE: operator.ge,
                ast.Eq: operator.eq,
                ast.NotEq: operator.ne,
                ast.Is: operator.is_,
                ast.IsNot: operator.is_not,
                ast.In: lambda x, y: x in y,
                ast.NotIn: lambda x, y: x not in y,
            }
            left = self.python_value(node.left)
            for op, child in zip(node.ops, node.comparators):
                right = self.python_value(child)
                if (
                    type(op) not in operators
                    or type(left) not in (int, float, bool, str, tuple, type(None))
                    or type(right) not in (int, float, bool, str, tuple, dict, type(None))
                    or not python_literal(left)
                    or not python_literal(right)
                ):
                    self.fail(node, "Unsupported construction-time comparison")
                if not operators[type(op)](left, right):
                    return False
                left = right
            return True
        elif isinstance(node, ast.BoolOp):
            value = self.python_value(node.values[0])
            for child in node.values[1:]:
                if type(value) not in (int, float, bool, str, tuple, dict, type(None)):
                    self.fail(node, "Unsupported construction-time truth value")
                if bool(value) == isinstance(node.op, ast.Or):
                    return value
                value = self.python_value(child)
            return value
        elif isinstance(node, ast.IfExp):
            condition = self.python_value(node.test)
            if type(condition) not in (int, float, bool, str, tuple, dict, type(None)):
                self.fail(node, "Unsupported construction-time truth value")
            return self.python_value(node.body if condition else node.orelse)
        else:
            self.fail(node, "Expected a Python construction-time value")
        if isinstance(value, (Expr, macros.ReferenceValue, macros.RegionValue, Buffer)):
            self.fail(node, "Expected a Python construction-time value")
        if type(value) is language.DType:
            return str(value)
        return value

    def value_dtype(self, value, node):
        if isinstance(value, macros.ReferenceValue):
            target = value.target
            if isinstance(target, ast.Subscript):
                return language.DType(self.buffers[self.buffer_name(target.value)].type.dtype)
            value = self.scalar_value(value, node)
        if not isinstance(value, Expr):
            self.fail(node, "Scalar dtype metadata requires an IR scalar expression")
        definitions = {**self.bindings, **self.parameter_types}
        definitions.update({name: Expr("mutable", value=dtype) for name, (dtype, _) in self.mutable.items()})
        induction = {name: None for name in self.variables if name not in definitions}
        return language.DType(resolved_dtype(value, induction, definitions, self.buffers))

    def attribute_value(self, node):
        owner = self.macro_value(node.value)
        try:
            if owner is language:
                value = inspect.getattr_static(owner, node.attr, NO_CONSTRUCTION_CALL)
                if value is NO_CONSTRUCTION_CALL:
                    self.fail(node, f"Unknown language attribute {node.attr!r}")
                return value
            if isinstance(owner, Buffer):
                return metadata.buffer_attribute(owner, node.attr)
            if type(owner) is language.DType:
                return metadata.dtype_attribute(owner, node.attr)
            if isinstance(owner, (Expr, macros.ReferenceValue)) and node.attr == "dtype":
                return self.value_dtype(owner, node)
            if (
                isinstance(owner, Expr)
                and owner.op == "const"
                and type(owner.value) is int
                and node.attr == "value"
            ):
                return owner.value
        except (TypeError, ValueError) as exc:
            self.fail(node, str(exc))
        self.fail(node, f"Unsupported construction-time attribute {node.attr!r}")

    def construction_call(self, node):
        target = self.macro_value(node.func)
        if target is language.print or target is language.device_assert:
            name = "print" if target is language.print else "device_assert"
            self.pending.append(self.debug_statement(node, name, node, self.parse_context[0]))
            return None
        if type(target) is language.DType:
            if len(node.args) != 1 or node.keywords:
                self.fail(node, "Scalar dtype conversion requires one positional argument")
            return Expr("cast", (self.expr(node.args[0]),), str(target))
        if isinstance(target, metadata.ScopeQuery):
            self.bind_call(node, [], {})
            return target.value
        constructors = (
            language.dtype,
            language.get_tvm_dtype,
            builtins.len,
            builtins.tuple,
            builtins.int,
            builtins.str,
        )
        if not any(target is item for item in constructors):
            return NO_CONSTRUCTION_CALL
        if len(node.args) != 1 or node.keywords:
            self.fail(node, "Construction-time conversion requires one positional argument")
        value = self.macro_value(node.args[0])
        try:
            if target is language.dtype or target is language.get_tvm_dtype:
                return language.get_tvm_dtype(value)
            if target is builtins.len or target is builtins.tuple:
                if type(value) not in (tuple, dict, str):
                    self.fail(node, "Construction-time len/tuple requires a built-in container")
                return target(value)
            if target is builtins.int and isinstance(value, Expr):
                known = constant_integer(value, {})
                if known is None:
                    self.fail(node, "int() requires a constant integer IR value")
                return known
            if type(value) in (int, float, bool, str, language.DType):
                return target(value)
        except (TypeError, ValueError, OverflowError) as exc:
            self.fail(node, str(exc))
        self.fail(node, "Unsupported construction-time conversion operand")

    def fresh(self, label):
        while True:
            self.name_counter += 1
            name = f"_macro_{self.name_counter}_{label}"
            if name not in self.used_names:
                self.used_names.add(name)
                self.source_names[name] = self.source_names.get(label, label)
                return name

    def macro_object(self, node):
        if not isinstance(node, ast.Call):
            return None
        fn = node.func
        value = None
        if isinstance(fn, ast.Name) and (fn.id in self.values or fn.id not in self.variables):
            self.check_name(fn)
            value = self.values.get(fn.id, self.constants.get(fn.id))
        elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            self.check_name(fn.value)
            owner = self.values.get(fn.value.id, self.constants.get(fn.value.id))
            if owner is not None:
                value = inspect.getattr_static(owner, fn.attr, None)
        return value if isinstance(value, language.Macro) else None

    def macro_value(self, node):
        if isinstance(node, macros.ValueNode):
            return node.value
        if self.macro_object(node) is not None:
            return self.expand_macro(node)
        if isinstance(node, ast.Name):
            self.check_name(node)
            if node.id in self.values:
                return self.values[node.id]
            if node.id in self.buffers:
                return self.buffers[node.id]
            if node.id in self.mutable:
                return macros.ReferenceValue(node)
            if node.id in self.variables:
                return Expr("var", value=node.id)
            if (
                node.id not in self.variables
                and node.id not in self.local_names
                and node.id in self.constants
            ):
                return self.constants[node.id]
            if node.id not in self.local_names and hasattr(builtins, node.id):
                return getattr(builtins, node.id)
        if isinstance(node, ast.Attribute):
            return self.attribute_value(node)
        if isinstance(node, ast.Call):
            value = self.construction_call(node)
            if value is not NO_CONSTRUCTION_CALL:
                return value
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.macro_value(value) for value in node.elts)
        if isinstance(node, ast.Dict) and all(key is not None for key in node.keys):
            return {self.static(key): self.macro_value(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.Subscript):
            value = self.macro_value(node.value)
            if type(value) in (tuple, dict, str):
                try:
                    return value[self.static(node.slice)]
                except (IndexError, KeyError, TypeError) as exc:
                    self.fail(node, f"Invalid macro argument access: {exc}")
            else:
                node = ast.copy_location(
                    ast.Subscript(value=macros.ValueNode(value, node.value), slice=node.slice, ctx=node.ctx),
                    node,
                )
                parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                if any(isinstance(part, ast.Slice) for part in parts):
                    name, origin, extents = self.region_spec(node)
                    return macros.RegionValue(name, origin, extents)
                return macros.ReferenceValue(self.element_node(node))
        if isinstance(node, ast.BinOp):
            values = (self.macro_value(node.left), self.macro_value(node.right))
            rewritten = ast.copy_location(
                ast.BinOp(
                    left=macros.ValueNode(values[0], node.left),
                    op=node.op,
                    right=macros.ValueNode(values[1], node.right),
                ),
                node,
            )
            if all(not isinstance(value, (Expr, macros.ReferenceValue)) for value in values):
                return self.python_value(rewritten)
            if type(node.op) not in BINOPS:
                self.fail(node, "Unsupported runtime binary operation")
            return Expr(BINOPS[type(node.op)], tuple(self.scalar_value(value, node) for value in values))
        if isinstance(node, ast.UnaryOp):
            value = self.macro_value(node.operand)
            if not isinstance(value, (Expr, macros.ReferenceValue)):
                return self.python_value(
                    ast.copy_location(
                        ast.UnaryOp(op=node.op, operand=macros.ValueNode(value, node.operand)), node
                    )
                )
            operators = {ast.USub: "neg", ast.UAdd: "pos", ast.Not: "not", ast.Invert: "invert"}
            if type(node.op) not in operators:
                self.fail(node, "Unsupported runtime unary operation")
            return Expr(operators[type(node.op)], (self.scalar_value(value, node),))
        if isinstance(node, ast.Compare):
            if len(node.ops) > 1:
                # DSLMutator.visit_Compare repeats each middle expression in
                # adjacent comparisons and combines them with eager boolop.
                operands = (node.left, *node.comparators)
                comparisons = [
                    ast.copy_location(ast.Compare(left=left, ops=[op], comparators=[right]), node)
                    for left, op, right in zip(operands, node.ops, operands[1:])
                ]
                return self.macro_value(ast.copy_location(ast.BoolOp(op=ast.And(), values=comparisons), node))
            left, right = self.macro_value(node.left), self.macro_value(node.comparators[0])
            if not isinstance(left, (Expr, macros.ReferenceValue)) and not isinstance(
                right, (Expr, macros.ReferenceValue)
            ):
                return self.python_value(
                    ast.copy_location(
                        ast.Compare(
                            left=macros.ValueNode(left, node.left),
                            ops=node.ops,
                            comparators=[macros.ValueNode(right, node.comparators[0])],
                        ),
                        node,
                    )
                )
            if type(node.ops[0]) not in COMPARISONS:
                self.fail(node, "Unsupported runtime comparison")
            return Expr(
                COMPARISONS[type(node.ops[0])],
                (self.scalar_value(left, node), self.scalar_value(right, node)),
            )
        if isinstance(node, ast.BoolOp):
            value = self.macro_value(node.values[0])
            if len(node.values) == 1:
                return value
            rest = ast.copy_location(ast.BoolOp(op=node.op, values=node.values[1:]), node)
            if type(value) in (int, float, bool, str, tuple, dict, type(None)):
                return value if bool(value) == isinstance(node.op, ast.Or) else self.macro_value(rest)
            self.boolean_macro_context += 1
            try:
                right = self.macro_value(rest)
            finally:
                self.boolean_macro_context -= 1
            return Expr(
                "and" if isinstance(node.op, ast.And) else "or",
                (self.scalar_value(value, node), self.scalar_value(right, node)),
            )
        if isinstance(node, ast.IfExp):
            value = self.macro_value(node.test)
            if type(value) in (int, float, bool, str, tuple, dict, type(None)):
                return self.macro_value(node.body if value else node.orelse)
            self.boolean_macro_context += 1
            try:
                left, right = self.macro_value(node.body), self.macro_value(node.orelse)
            finally:
                self.boolean_macro_context -= 1
            return Expr("if_then_else", tuple(self.scalar_value(item, node) for item in (value, left, right)))
        try:
            return self.python_value(node)
        except CompileError:
            return self.expr(node)

    def scalar_value(self, value, node):
        if isinstance(value, Expr):
            return value
        if isinstance(value, macros.ReferenceValue):
            if isinstance(value.target, ast.Name) and value.target.id in self.mutable:
                return Expr("var", value=value.target.id)
            return self.expr(value.target)
        if type(value) in (int, float, bool):
            return Expr("const", value=value)
        self.fail(node, "A scalar expression requires a numeric or Boolean macro result")

    def element_node(self, node):
        name = self.buffer_name(node.value)
        indices = self.indices(node.slice)
        if len(indices) != len(self.buffers[name].type.shape):
            self.fail(node, "Buffer element rank mismatch")
        result = ast.Subscript(
            value=macros.ValueNode(self.buffers[name], node.value),
            slice=ast.copy_location(
                ast.Tuple(elts=[macros.ValueNode(value, node) for value in indices], ctx=ast.Load()), node
            ),
            ctx=node.ctx,
        )
        return ast.copy_location(result, node)

    def bind_macro_scalar(self, name, value, node):
        if self.before_launch and not self.macro_stack:
            self.check_prelude_expr(value, node)
            return value
        self.pending.append(Statement("let", (name, value), self.location(node)))
        self.variables.add(name)
        self.bindings[name] = value
        return Expr("var", value=name)

    def capture_reference(self, value, node):
        def capture(expr):
            if expr.op == "const":
                return expr
            return self.bind_macro_scalar(self.fresh("index"), expr, node)

        if isinstance(value, macros.RegionValue):
            return macros.RegionValue(value.buffer, tuple(capture(x) for x in value.origin), value.extents)
        if not isinstance(value, macros.ReferenceValue):
            self.fail(node, "T.Ref arguments require a mutable scalar, buffer element, or sliced region")
        target = value.target
        if isinstance(target, ast.Name) and target.id in self.mutable:
            return value
        if isinstance(target, ast.Subscript):
            target = self.element_node(target)
            target.slice.elts = [macros.ValueNode(capture(self.expr(x)), x) for x in target.slice.elts]
            return macros.ReferenceValue(target)
        self.fail(node, "T.Ref requires an assignable argument")

    def expand_macro(self, call):
        macro = self.macro_object(call)
        if self.boolean_macro_context:
            self.fail(call, "The upstream eager parser rejects macros inside runtime Boolean branches")
        if len(self.macro_stack) >= 128:
            self.fail(call, "Macro expansion exceeds the depth limit of 128")
        arguments, keywords = [], {}
        for arg in call.args:
            value = self.macro_value(arg.value if isinstance(arg, ast.Starred) else arg)
            if isinstance(arg, ast.Starred):
                if not isinstance(value, tuple):
                    self.fail(arg, "Starred macro arguments require a tuple or list")
                arguments.extend(value)
            else:
                arguments.append(value)
        for keyword in call.keywords:
            value = self.macro_value(keyword.value)
            items = value if keyword.arg is None else {keyword.arg: value}
            if not isinstance(items, dict) or any(not isinstance(key, str) for key in items):
                self.fail(keyword.value, "Expanded macro keyword arguments require a string-keyed dictionary")
            if keywords.keys() & items.keys():
                self.fail(keyword.value, "Duplicate macro keyword argument")
            keywords.update(items)
        signature = inspect.signature(macro.function)
        try:
            bound = signature.bind(*arguments, **keywords)
        except TypeError as exc:
            self.fail(call, f"Macro {macro.__name__}: {exc}")
        bound.apply_defaults()
        source = macros.definition(macro)
        body, names, constants = macros.rename(source, self.fresh)
        saved = (
            self.filename,
            self.first_line,
            self.values,
            self.constants,
            self.pending,
            self.value_frames,
            self.local_names,
        )
        self.filename, self.first_line = source.filename, source.first_line
        self.values, self.constants = self.values.copy(), {**self.constants, **constants}
        self.value_frames = self.value_frames.copy()
        self.local_names = (
            self.local_names
            | {
                part.id
                for part in ast.walk(source.node)
                if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)
            }
            | {names[name] for name in bound.arguments}
        )
        self.pending = []
        self.frames.append(object())
        call_location = SourceLocation(saved[0], saved[1] + call.lineno - 1, call.col_offset)
        self.macro_stack.append((macro, len(self.loops), call_location))
        try:
            for name, value in bound.arguments.items():
                formal = names[name]
                annotation = signature.parameters[name].annotation
                if macros.is_reference(annotation, source.environment):
                    self.values[formal] = self.capture_reference(value, source.node)
                elif isinstance(value, (Expr, macros.ReferenceValue)):
                    self.values[formal] = self.bind_macro_scalar(
                        formal, self.scalar_value(value, source.node), source.node
                    )
                else:
                    self.values[formal] = value
                self.value_frames[formal] = self.frames[-1]
            expansion = self.pending
            self.pending = []
            queued, returned = list(body), None
            parallel, nested = self.parse_context
            if (
                queued
                and isinstance(queued[0], ast.Expr)
                and isinstance(queued[0].value, ast.Constant)
                and isinstance(queued[0].value.value, str)
            ):
                queued.pop(0)
            while queued:
                statement = queued.pop(0)
                if isinstance(statement, ast.Return):
                    returned = statement
                    break
                if isinstance(statement, ast.If):
                    try:
                        condition = self.python_value(statement.test)
                    except CompileError:
                        pass
                    else:
                        if type(condition) in (int, float, bool):
                            queued[:0] = statement.body if condition else statement.orelse
                            continue
                if any(isinstance(part, ast.Return) for part in ast.walk(statement)):
                    self.fail(
                        statement, "The upstream eager parser rejects macro returns inside control flow"
                    )
                expansion.extend(self.statements((statement,), parallel=parallel, nested=nested))
            result = (
                self.macro_value(returned.value)
                if returned is not None and returned.value is not None
                else None
            )

            def unwrap(value):
                if isinstance(value, tuple):
                    return tuple(unwrap(item) for item in value)
                return (
                    self.scalar_value(value, returned) if isinstance(value, macros.ReferenceValue) else value
                )

            result = unwrap(result)
            expansion.extend(self.pending)
        finally:
            self.macro_stack.pop()
            self.frames.pop()
            (
                self.filename,
                self.first_line,
                self.values,
                self.constants,
                self.pending,
                self.value_frames,
                self.local_names,
            ) = saved
        self.pending.extend(expansion)
        return result

    def call_name(self, node):
        if not isinstance(node, ast.Call):
            self.fail(node, "Expected a language operation")
        fn = node.func
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            self.check_name(fn.value)
            if self.values.get(fn.value.id, self.constants.get(fn.value.id)) is language:
                return fn.attr
        if isinstance(fn, ast.Name):
            self.check_name(fn)
            value = self.values.get(fn.id, self.constants.get(fn.id, getattr(builtins, fn.id, None)))
            if isinstance(value, language.DType):
                return str(value)
            if value is builtins.range:
                return "serial"
            if callable(value) and value in language._MARKER_NAMES:
                return language._MARKER_NAMES[value]
            if value is language.ceildiv:
                return "ceildiv"
            if value is language.align_up:
                return "align_up"
            if value is language.Tensor:
                return "Tensor"
        self.fail(node, "Only ntilang.language operations are supported in kernels")

    def static(self, node):
        if isinstance(node, macros.ValueNode):
            value = node.value
            if isinstance(value, language.DType):
                return str(value)
            if isinstance(value, Expr) and value.op == "const":
                return value.value
            if isinstance(value, Expr):
                known = constant_integer(value, self.bindings)
                if known is not None:
                    return known
            if type(value) is tuple:
                return tuple(self.static(macros.ValueNode(item, node)) for item in value)
            if type(value) in (int, float, str, bool, tuple, dict) or value is None:
                return value
            self.fail(node, "Expected a static macro value")
        if isinstance(node, ast.Name):
            self.check_name(node)
        if isinstance(node, ast.Name) and node.id in self.values:
            return self.static(macros.ValueNode(self.values[node.id], node))
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            return self.static(macros.ValueNode(self.macro_value(node), node))
        if isinstance(node, ast.Name) and node.id in self.variables:
            known = constant_integer(Expr("var", value=node.id), self.bindings)
            if known is not None:
                return known
            self.fail(node, "A runtime variable cannot be used as a static specialization constant")
        if isinstance(node, ast.Constant) and (
            type(node.value) in (int, float, str, bool) or node.value is None
        ):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.local_names:
            self.fail(node, f"Expected a static specialization constant; local {node.id} is not bound")
        if isinstance(node, ast.Name) and node.id in self.constants:
            value = self.constants[node.id]
            if isinstance(value, language.DType):
                return str(value)
            if value is None or type(value) in (int, float, str, bool, tuple, dict):
                return value
        if isinstance(node, ast.Dict) and all(key is not None for key in node.keys):
            return {self.static(key): self.static(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.static(x) for x in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
            value = self.static(node.operand)
            if isinstance(node.op, ast.Invert):
                if type(value) not in (int, bool):
                    self.fail(node, "Static bitwise inversion requires an integer")
                return ~value
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in STATIC_OPS:
            try:
                return STATIC_OPS[type(node.op)](self.static(node.left), self.static(node.right))
            except (TypeError, ValueError, ZeroDivisionError) as exc:
                self.fail(node, f"Invalid static expression: {exc}")
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in COMPARISONS:
            operations = {
                ast.Lt: operator.lt,
                ast.LtE: operator.le,
                ast.Gt: operator.gt,
                ast.GtE: operator.ge,
                ast.Eq: operator.eq,
                ast.NotEq: operator.ne,
            }
            return operations[type(node.ops[0])](self.static(node.left), self.static(node.comparators[0]))
        name = self.optional_call_name(node)
        if name in ("ceildiv", "cdiv", "align_up"):
            parameters = ["x", "y"] if name == "align_up" else ["lhs", "rhs", "span"]
            args = self.bind_call(node, parameters, {} if name == "align_up" else {"span": None})
            try:
                operation = language.align_up if name == "align_up" else language.ceildiv
                return operation(*(self.static(args[key]) for key in parameters))
            except (TypeError, ValueError) as exc:
                self.fail(node, str(exc))
        if isinstance(node, ast.Call):
            return self.static(macros.ValueNode(self.macro_value(node), node))
        self.fail(node, "Expected a static specialization constant")

    def optional_call_name(self, node):
        try:
            return self.call_name(node)
        except CompileError:
            return None

    def keywords(self, node, allowed):
        result = {}
        for kw in node.keywords:
            if kw.arg not in allowed or kw.arg in result:
                self.fail(node, f"Unsupported or duplicate keyword {kw.arg!r}")
            result[kw.arg] = self.static(kw.value)
        return result

    def bind_call(self, node, parameters, defaults):
        if len(node.args) > len(parameters):
            self.fail(node, "Too many positional arguments")
        arguments = dict(zip(parameters, node.args))
        for keyword in node.keywords:
            if keyword.arg not in parameters or keyword.arg in arguments:
                self.fail(node, f"Unsupported or duplicate argument {keyword.arg!r}")
            arguments[keyword.arg] = keyword.value
        for name in parameters:
            if name not in arguments:
                if name not in defaults:
                    self.fail(node, f"Missing required argument {name!r}")
                arguments[name] = ast.Constant(value=defaults[name])
        return arguments

    def loop_signature(self, call, name):
        """Bind serial/unroll bounds, a static step, and scheduling hints."""
        unrolled = name in ("unroll", "Unroll")
        parameters = ["start", "stop", "step", "annotations"]
        defaults = {"stop": None, "step": None, "annotations": None}
        if unrolled:
            parameters += ["explicit", "unroll_factor"]
            defaults.update(explicit=False, unroll_factor=None)
        if len(call.args) > 3:
            self.fail(call, "Loop scheduling arguments must be keyword-only")
        arguments = self.bind_call(call, parameters, defaults)
        values = {}
        for key, value in arguments.items():
            if key in ("start", "stop"):
                if any(isinstance(node, ast.Name) and node.id in self.variables for node in ast.walk(value)):
                    values[key] = self.expr(value)
                    continue
                try:
                    values[key] = self.static(value)
                except CompileError:
                    values[key] = self.expr(value)
            else:
                values[key] = self.static(value)
        start, stop, step = (values[key] for key in ("start", "stop", "step"))
        if stop is None:
            start, stop = 0, start
        step = 1 if step is None else step
        annotations = values["annotations"]
        if annotations is None:
            annotations = {}
        if type(annotations) is not dict:
            self.fail(call, "Loop annotations must be a static dictionary")
        annotations = annotations.copy()
        allowed = {"pragma_unroll_explicit", "pragma_unroll_factor"} if unrolled else set()
        if annotations.keys() - allowed:
            self.fail(
                call,
                "Unsupported loop annotations: " + ", ".join(sorted(map(str, annotations.keys() - allowed))),
            )
        if unrolled:
            explicit = values["explicit"] or annotations.get("pragma_unroll_explicit", False)
            factor = values["unroll_factor"]
            if factor is None:
                factor = annotations.get("pragma_unroll_factor")
            if type(values["explicit"]) is not bool or type(explicit) is not bool:
                self.fail(call, "Unroll explicit must be Boolean")
            if factor is not None and (type(factor) is not int or not 0 <= factor <= 2**31 - 1):
                self.fail(call, "Unroll factor must be a nonnegative signed 32-bit integer")
            if explicit and factor is not None:
                self.fail(call, "Unroll explicit and unroll_factor are mutually exclusive")
            annotations = {"pragma_unroll_explicit": explicit}
            if factor is not None:
                annotations["pragma_unroll_factor"] = factor
        return (start, stop, step), tuple(sorted(annotations.items()))

    def scalar_allocation(self, call, target):
        keywords = {}
        for keyword in call.keywords:
            if keyword.arg not in ("dtype", "scope", "init") or keyword.arg in keywords:
                self.fail(call, f"Unsupported or duplicate alloc_var argument {keyword.arg!r}")
            keywords[keyword.arg] = keyword.value
        arguments = list(call.args)
        if arguments:
            if "dtype" in keywords:
                self.fail(call, "Duplicate alloc_var dtype")
            dtype_node = arguments.pop(0)
        elif "dtype" in keywords:
            dtype_node = keywords.pop("dtype")
        else:
            self.fail(call, "alloc_var requires dtype")
        dtype = self.static(dtype_node)
        if not isinstance(dtype, str) or dtype not in DTYPES:
            self.fail(call, "alloc_var requires a supported scalar dtype")
        scope = self.static(keywords.get("scope", ast.Constant(value="local.var")))
        initializer = keywords.get("init", ast.Constant(value=None))

        def optional_static(value):
            try:
                return self.static(value)
            except CompileError:
                return object()

        if len(arguments) == 1:
            argument = optional_static(arguments[0])
            if isinstance(argument, str) and optional_static(initializer) is None and scope == "local.var":
                scope = argument
            else:
                if optional_static(initializer) is not None:
                    self.fail(call, "Initializer specified multiple times in alloc_var")
                initializer = arguments[0]
        elif len(arguments) == 2:
            if optional_static(initializer) is not None:
                self.fail(call, "Initializer specified multiple times in alloc_var")
            initializer, scope_node = arguments
            scope = self.static(scope_node)
        elif len(arguments) > 2:
            self.fail(call, "alloc_var accepts at most three positional arguments")
        if scope != "local.var":
            self.fail(call, "alloc_var currently supports the local.var scope")
        if target.id.startswith("_nt_"):
            self.fail(target, f"Reserved name {target.id}")
        value = Expr("const", value=0) if optional_static(initializer) is None else self.expr(initializer)
        name = self.fresh(target.id)
        self.variables.add(name)
        self.mutable[name] = (dtype, self.parallel_context)
        self.values[target.id] = macros.ReferenceValue(
            ast.copy_location(ast.Name(id=name, ctx=ast.Load()), target)
        )
        self.value_frames[target.id] = self.frames[-1]
        return Statement("declare", (name, dtype, Expr("cast", (value,), dtype)), self.location(call))

    def reduction(self, call, name, loc):
        kinds = {"sum", "abssum", "max", "absmax", "min", "bitand", "bitor", "bitxor"}
        parameters = ["buffer", "out"]
        defaults = {"dim": -1, "clear": True, "batch": 1, "nan_propagate": False, "annotations": None}
        if name == "reduce":
            parameters += ["reduce_type", "dim", "clear", "batch", "nan_propagate", "annotations"]
            del defaults["dim"], defaults["clear"]
        else:
            parameters += ["dim"]
            if name != "reduce_abssum":
                parameters.append("clear")
            parameters.append("batch")
            if name in ("reduce_max", "reduce_min", "reduce_absmax"):
                parameters.append("nan_propagate")
            parameters.append("annotations")
        args = self.bind_call(call, parameters, defaults)
        src, dst = (self.buffer_name(args[key]) for key in ("buffer", "out"))
        source, destination = self.buffers[src], self.buffers[dst]
        kind = self.static(args["reduce_type"]) if name == "reduce" else name.removeprefix("reduce_")
        if kind not in kinds:
            self.fail(call, f"Unsupported reduction kind {kind!r}")
        dim = self.static(args["dim"])
        if type(dim) is not int or not -len(source.type.shape) <= dim < len(source.type.shape):
            self.fail(call, "Reduction dimension is outside the source rank")
        dim %= len(source.type.shape)
        clear = self.static(args["clear"]) if "clear" in args else True
        nan_propagate = self.static(args["nan_propagate"]) if "nan_propagate" in args else False
        if type(clear) is not bool or type(nan_propagate) is not bool:
            self.fail(call, "Reduction clear and nan_propagate must be bool")
        batch = self.static(args["batch"])
        if type(batch) is not int or batch < 1:
            self.fail(call, "Reduction batch must be a positive integer")
        if batch != 1:
            self.fail(call, "Batched AllReduce scheduling is not implemented; reduction batch must be 1")
        annotations = args["annotations"]
        if not (
            isinstance(annotations, ast.Constant)
            and annotations.value is None
            or isinstance(annotations, ast.Dict)
            and not annotations.keys
        ):
            self.fail(call, "Reduction lowering annotations are not implemented")
        if source.space not in ("shared", "fragment") or destination.space not in ("shared", "fragment"):
            self.fail(call, "Reductions require shared or fragment buffers")
        shape = source.type.shape
        removed = shape[:dim] + shape[dim + 1 :]
        kept = shape[:dim] + (1,) + shape[dim + 1 :]
        if destination.type.shape not in (removed, kept):
            self.fail(call, f"Reduction output shape must be {removed} or {kept}")
        if kind.startswith("bit") and not (
            destination.type.dtype.startswith(("int", "uint")) or destination.type.dtype == "bool"
        ):
            self.fail(call, "Bitwise reductions require an integer output dtype")
        if src not in self.initialized or not clear and dst not in self.initialized:
            self.fail(call, "Reduction reads a buffer before initialization")
        self.initialized.add(dst)
        return Statement("reduce", (src, dst, kind, dim, clear, nan_propagate), loc)

    def indices(self, node):
        parts = node.elts if isinstance(node, ast.Tuple) else [node]
        if any(isinstance(p, ast.Slice) for p in parts):
            self.fail(node, "Scalar element access requires indices; use slices in T.copy")

        def canonical(value):
            seen = set()
            original = value
            while value.op == "var" and value.value in self.bindings and value.value not in seen:
                if value.value in self.mutable:
                    return original
                seen.add(value.value)
                candidate = self.bindings[value.value]
                if candidate.op != "var":
                    break
                value = candidate
            return original if value.op == "var" and value.value in self.mutable else value

        return tuple(canonical(self.expr(p)) for p in parts)

    def expr(self, node):
        try:
            value = self.python_value(node)
        except CompileError:
            pass
        else:
            if type(value) in (int, float, bool):
                return Expr("const", value=value)
        if isinstance(node, macros.ValueNode):
            return self.scalar_value(node.value, node)
        if self.macro_object(node) is not None:
            return self.scalar_value(self.expand_macro(node), node)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
            return Expr("const", value=node.value)
        if isinstance(node, ast.Name):
            self.check_name(node)
            if node.id in self.values:
                return self.scalar_value(self.values[node.id], node)
            if node.id in self.variables:
                return Expr("var", value=node.id)
            return Expr("const", value=self.static(node))
        if isinstance(node, ast.Attribute):
            return self.scalar_value(self.macro_value(node), node)
        if isinstance(node, ast.Subscript):
            owner = self.macro_value(node.value)
            if type(owner) in (tuple, dict, str):
                node = ast.copy_location(
                    ast.Subscript(value=macros.ValueNode(owner, node.value), slice=node.slice, ctx=node.ctx),
                    node,
                )
                return self.scalar_value(self.macro_value(node), node)
            name = self.buffer_name(macros.ValueNode(owner, node.value))
            buf = self.buffers[name]
            if self.before_launch:
                self.fail(node, "Buffer reads before T.Kernel require host execution lowering")
            if buf.space != "global" and name not in self.initialized:
                self.fail(node, f"Buffer {name} is read before initialization")
            indices = self.indices(node.slice)
            if len(indices) != len(buf.type.shape):
                self.fail(node, f"Buffer {name} expects {len(buf.type.shape)} indices")
            return Expr("load", indices, name)
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp)):
            return self.scalar_value(self.macro_value(node), node)
        if isinstance(node, ast.Call):
            value = self.construction_call(node)
            if value is not NO_CONSTRUCTION_CALL:
                return self.scalar_value(value, node)
            name = self.call_name(node)
            if name == "likely":
                if len(node.args) > 2:
                    self.fail(node, "T.likely takes at most two positional arguments")
                args = self.bind_call(node, ["cond", "span", "dtype"], {"span": None, "dtype": None})
                values = {
                    key: self.expr(arg) if key == "cond" else self.macro_value(arg)
                    for key, arg in args.items()
                }
                if values["span"] is not None:
                    self.fail(node, "Explicit source span objects require further parser integration")
                return Expr(name, (values["cond"],))
            if name == "reinterpret":
                args = self.bind_call(node, ["dtype", "value", "span"], {"span": None})
                values = {
                    key: self.expr(arg) if key == "value" else self.static(arg) for key, arg in args.items()
                }
                if values["span"] is not None:
                    self.fail(node, "Explicit source span objects require further parser integration")
                return Expr(name, (values["value"],), TensorType((1,), values["dtype"]).dtype)
            if name in BIT_COUNT_OPS:
                if len(node.args) > 1:
                    self.fail(node, f"T.{name} takes one positional argument")
                args = self.bind_call(node, ["x", "dtype"], {"dtype": None})
                # The pinned TileLang wrapper evaluates and discards keyword dtype.
                values = {
                    key: self.expr(arg) if key == "x" else self.macro_value(arg) for key, arg in args.items()
                }
                return Expr(name, (values["x"],))
            if name in IEEE_MATH_OPS:
                parameters = ["x", "y", "z"][: IEEE_MATH_OPS[name][1]]
                defaults = {} if name in ("fma", "fmul", "ieee_frsqrt") else {"rounding_mode": "rn"}
                args = self.bind_call(node, [*parameters, *defaults], defaults)
                mode = self.static(args["rounding_mode"]) if defaults else "rn"
                if not isinstance(mode, str) or mode not in ("rn", "rz", "ru", "rd"):
                    self.fail(node, "IEEE rounding_mode must be rn, rz, ru, or rd")
                return Expr(name, tuple(self.expr(args[key]) for key in parameters), mode)
            if name in BINARY_MATH_OPS:
                parameters = (
                    ["x1", "x2"]
                    if name in ("atan2", "copysign", "hypot", "nextafter", "ldexp")
                    else ["x", "y"]
                )
                defaults = {"span": None} if name == "pow" else {}
                args = self.bind_call(node, [*parameters, *defaults], defaults)
                if defaults and self.static(args["span"]) is not None:
                    self.fail(node, "Explicit source span objects require further parser integration")
                values = tuple(self.expr(args[key]) for key in parameters)
                if name == "pow":
                    exponent = constant_integer(values[1], self.bindings)
                    if exponent is not None and exponent >= 0:
                        if exponent > 2**31 - 1:
                            self.fail(
                                node, "A constant integer power must fit the upstream int template parameter"
                            )
                        return Expr("pow_integer", (values[0],), exponent)
                return Expr(name, values)
            if name in TRANSCENDENTAL_OPS | FAST_MATH_OPS:
                args = self.bind_call(node, ["x"], {})
                return Expr(name, (self.expr(args["x"]),))
            if name in UNARY_MATH_OPS and name != "round_away":
                defaults = {"span": None}
                if name == "round":
                    defaults = {"rounding_mode": "ties-to-even", **defaults}
                args = self.bind_call(node, ["x", *defaults], defaults)
                if self.static(args["span"]) is not None:
                    self.fail(node, "Explicit source span objects require further parser integration")
                if name == "round":
                    mode = self.static(args["rounding_mode"])
                    if mode not in (None, "ties-to-even", "ties-away-from-zero"):
                        self.fail(node, "Round rounding_mode must be ties-to-even or ties-away-from-zero")
                    name = "round_away" if mode == "ties-away-from-zero" else "round"
                return Expr(name, (self.expr(args["x"]),))
            if name in ("Select", "if_then_else"):
                parameters = (
                    ["condition", "true_value", "false_value"] if name == "Select" else ["cond", "t", "f"]
                )
                args = self.bind_call(node, [*parameters, "span"], {"span": None})
                if self.static(args["span"]) is not None:
                    self.fail(node, "Explicit source span objects require further parser integration")
                return Expr(
                    "select" if name == "Select" else "if_then_else",
                    tuple(self.expr(args[key]) for key in parameters),
                )
            if name in DIVISION_CALLS:
                parameters = ["lhs", "rhs"] if name in ("ceildiv", "cdiv") else ["a", "b"]
                args = self.bind_call(node, [*parameters, "span"], {"span": None})
                if self.static(args["span"]) is not None:
                    self.fail(node, "Explicit source span objects require further parser integration")
                return Expr(DIVISION_CALLS[name], tuple(self.expr(args[key]) for key in parameters))
            if name == "align_up":
                args = self.bind_call(node, ["x", "y"], {})
                left, right = (self.expr(args[key]) for key in ("x", "y"))
                return Expr("*", (Expr("ceildiv", (left, right)), right))
            if name in BITWISE_CALLS:
                parameters = ["x"] if name == "bitwise_not" else ["x", "y"]
                args = self.bind_call(node, [*parameters, "span"], {"span": None})
                if self.static(args["span"]) is not None:
                    self.fail(node, "Explicit source span objects require further parser integration")
                return Expr(BITWISE_CALLS[name], tuple(self.expr(args[key]) for key in parameters))
            if name in language.DTYPE_NAMES and len(node.args) == 1 and not node.keywords:
                return Expr("cast", (self.expr(node.args[0]),), language.DTYPE_NAMES[name])
            arity = {
                "maximum": 2,
                "minimum": 2,
                "max": 2,
                "min": 2,
                "cast": 2,
            }
            if name in arity and len(node.args) == arity[name] and not node.keywords:
                if name == "cast":
                    dtype = self.static(node.args[1])
                    TensorType((1,), dtype)
                    return Expr("cast", (self.expr(node.args[0]),), TensorType((1,), dtype).dtype)
                return Expr(name, tuple(self.expr(x) for x in node.args))
        self.fail(node, f"Unsupported expression: {ast.dump(node, include_attributes=False)}")

    def buffer_name(self, node):
        if isinstance(node, ast.Name):
            self.check_name(node)
            if node.id in self.values:
                value = self.values[node.id]
                if isinstance(value, Buffer):
                    return value.name
                self.fail(node, "Expected a declared buffer")
            if node.id in self.buffers:
                return self.buffers[node.id].name
        if isinstance(node, macros.ValueNode):
            value = node.value
        elif (
            self.macro_object(node) is not None
            or isinstance(node, ast.Name)
            and node.id in self.values
            or isinstance(node, ast.Subscript)
        ):
            value = self.macro_value(node)
        else:
            value = None
        if isinstance(value, Buffer):
            return value.name
        self.fail(node, "Expected a declared buffer")

    def region_spec(self, node):
        if (
            isinstance(node, macros.ValueNode)
            or self.macro_object(node) is not None
            or isinstance(node, ast.Name)
            and node.id in self.values
        ):
            value = self.macro_value(node)
            if isinstance(value, macros.RegionValue):
                return value.buffer, value.origin, value.extents
            if isinstance(value, macros.ReferenceValue):
                return self.region_spec(value.target)
            if isinstance(value, Buffer):
                shape = value.type.shape
                return value.name, tuple(Expr("const", value=0) for _ in shape), shape
            self.fail(node, "Expected a buffer or region macro value")
        if isinstance(node, ast.Name):
            name = self.buffer_name(node)
            shape = self.buffers[name].type.shape
            return name, tuple(Expr("const", value=0) for _ in shape), shape
        elif isinstance(node, ast.Subscript):
            name = self.buffer_name(node.value)
        else:
            self.fail(node, "Expected a buffer, a sliced region, or a tile origin")
        shape = self.buffers[name].type.shape
        parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(parts) != len(shape):
            self.fail(node, "Copy regions require one index or slice per buffer dimension")
        if not any(isinstance(part, ast.Slice) for part in parts):
            return name, tuple(self.expr(part) for part in parts), None
        origins, extents = [], []
        for part, size in zip(parts, shape):
            if isinstance(part, ast.Slice):
                if part.step is not None and self.static(part.step) != 1:
                    self.fail(part, "Copy slices require unit stride")
                start = Expr("const", value=0) if part.lower is None else self.expr(part.lower)
                stop = Expr("const", value=size) if part.upper is None else self.expr(part.upper)
                try:
                    extent, coefficients = affine(Expr("-", (stop, start)), {})
                except CompileError:
                    self.fail(part, "Copy slice extents must simplify to static integers")
                if coefficients or extent <= 0:
                    self.fail(part, "Copy slice extents must be positive static integers")
                origins.append(start)
                extents.append(extent)
            else:
                origins.append(self.expr(part))
                extents.append(1)
        return name, tuple(origins), tuple(extents)

    def copy_statement(self, call, node, parallel, nested):
        options = {
            "coalesced_width": None,
            "disable_tma": False,
            "eviction_policy": None,
            "prefer_instruction": None,
            "annotations": None,
            "loop_layout": None,
        }
        if len(call.args) > 2:
            self.fail(call, "Copy options are keyword-only")
        args = self.bind_call(call, ["src", "dst", *options], options)
        settings = {name: self.static(args[name]) for name in options}
        annotations = settings.pop("annotations")
        if annotations is not None and type(annotations) is not dict:
            self.fail(call, "Copy annotations must be a static dictionary")
        settings["parallel_loop_layout"] = settings.pop("loop_layout")
        if annotations:
            if set(annotations) - settings.keys():
                self.fail(call, "Unsupported copy lowering annotation")
            settings.update(annotations)
        if type(settings["disable_tma"]) is not bool:
            self.fail(call, "Copy disable_tma must be bool")
        if settings["coalesced_width"] is not None or settings["parallel_loop_layout"] is not None:
            self.fail(call, "Explicit copy vector widths and layouts need further lowering")
        if settings["eviction_policy"] not in (None, "evict_normal"):
            self.fail(call, "This copy lowering supports the normal cache eviction policy")
        if settings["prefer_instruction"] not in (None, "sync"):
            self.fail(call, "This copy lowering supports prefer_instruction='sync'")
        source = self.region_spec(args["src"])
        destination = self.region_spec(args["dst"])
        if source[2] is None and destination[2] is None:
            assignment = ast.Assign(targets=[args["dst"]], value=args["src"])
            ast.copy_location(assignment, node)
            return self.statement(assignment, parallel=parallel, nested=nested)
        if parallel:
            self.fail(call, "Collective tile operations cannot appear inside T.Parallel")
        if (
            isinstance(args["src"], ast.Name)
            and args["src"].id in self.buffers
            and isinstance(args["dst"], ast.Name)
            and args["dst"].id in self.buffers
            and source[2] != destination[2]
        ):
            self.fail(call, "Whole-buffer copies require equal shapes; use an explicit origin or slice")

        def with_extent(spec, other):
            name, origin, extents = spec
            if extents is None:
                extents = other[2]
                while len(extents) > len(origin) and extents[0] == 1:
                    extents = extents[1:]
                if len(extents) > len(origin):
                    self.fail(call, "Copy origin has insufficient dimensions for the tile")
                extents = (1,) * (len(origin) - len(extents)) + extents
            return name, origin, extents

        source = with_extent(source, destination)
        destination = with_extent(destination, source)
        src_shape = tuple(size for size in source[2] if size != 1) or (1,)
        dst_shape = tuple(size for size in destination[2] if size != 1) or (1,)
        if src_shape != dst_shape:
            self.fail(call, "Copy regions must have equal non-unit extents")
        Partition(src_shape, self.threads)

        def region(spec):
            name, origin, extents = spec
            axes, axis = [], 0
            for extent in extents:
                axes.append(None if extent == 1 else axis)
                axis += extent != 1
            return Region(name, origin, src_shape, tuple(axes))

        src, dst = region(source), region(destination)
        if self.buffers[src.buffer].space != "global" and src.buffer not in self.initialized:
            self.fail(call, f"Buffer {src.buffer} is read before initialization")
        if dst.is_full(self.buffers[dst.buffer].type.shape):
            self.initialized.add(dst.buffer)
        elif self.buffers[dst.buffer].space != "global" and dst.buffer not in self.initialized:
            self.fail(call, "Partial temporary copies require an initialized destination")
        return Statement("copy", (src, dst), self.location(node))

    def parallel_nest(self, node):
        """Collect a rectangular perfect nest into one ownership domain."""
        extents, targets = [], []
        bound_names = self.variables.copy()
        while True:
            self.keywords(node.iter, set())
            if node.orelse:
                self.fail(node, "Loop else clauses are not supported")
            level_targets = (
                node.target.elts if isinstance(node.target, (ast.Tuple, ast.List)) else [node.target]
            )
            if len(level_targets) != len(node.iter.args):
                self.fail(node, "T.Parallel needs one variable per extent")
            for extent in node.iter.args:
                if any(isinstance(part, ast.Name) and part.id in bound_names for part in ast.walk(extent)):
                    self.fail(extent, "Parallel extents require static rectangular domains")
                extents.append(self.static(extent))
            targets.extend(level_targets)
            bound_names.update(target.id for target in level_targets if isinstance(target, ast.Name))
            body = node.body
            if (
                len(body) != 1
                or not isinstance(body[0], ast.For)
                or self.call_name(body[0].iter) != "Parallel"
            ):
                return tuple(extents), targets, body
            node = body[0]

    def statements(self, nodes, *, parallel=False, nested=False):
        body = []
        for node in nodes:
            old_pending, old_context = self.pending, self.parse_context
            self.pending, self.parse_context = [], (parallel, nested)
            try:
                statement = self.statement(node, parallel=parallel, nested=nested)
                prefix = self.pending
            finally:
                self.pending, self.parse_context = old_pending, old_context
            for item in (*prefix, *(statement if isinstance(statement, tuple) else (statement,))):
                body.append(item)
                if item.op in ("break", "continue"):
                    return tuple(body)
        return tuple(body)

    def assignment_target(self, target):
        if isinstance(target, ast.Name) and isinstance(self.values.get(target.id), macros.ReferenceValue):
            self.check_name(target)
            return self.values[target.id].target
        return target

    def bind_name(self, target, value, node):
        name = target.id
        if name.startswith("_nt_"):
            self.fail(node, f"Reserved name {name}")
        if isinstance(value, macros.ReferenceValue):
            value = self.scalar_value(value, node)
        if name in self.mutable and (isinstance(value, Expr) or type(value) in (int, float, bool)):
            dtype, owner = self.mutable[name]
            if owner != self.parallel_context:
                self.fail(node, "Declare mutable scalars inside the parallel loop that updates them")
            return Statement(
                "assign", (name, Expr("cast", (self.scalar_value(value, node),), dtype)), self.location(node)
            )
        if isinstance(value, Expr):
            if self.before_launch and not self.macro_stack:
                self.check_prelude_expr(value, node)
                self.values[name] = value
                self.value_frames[name] = self.frames[-1]
                return Statement("pass", (), self.location(node))
            known = constant_integer(value, {})
            if known is not None and expression_dtype(value, {}, {}) == "int32":
                # Builder.bind converts an int32 IntImm back into a Python int.
                self.values[name] = known
                return Statement("pass", (), self.location(node))
            canonical = self.fresh(name)
            self.variables.add(canonical)
            self.bindings[canonical] = value
            self.values[name] = Expr("var", value=canonical)
            self.value_frames[name] = self.frames[-1]
            return Statement("let", (canonical, value), self.location(node))
        self.values[name] = value
        if not isinstance(value, (int, float, str, tuple)):
            self.value_frames[name] = self.frames[-1]
        return Statement("pass", (), self.location(node))

    def check_prelude_expr(self, value, node):
        if value.op in ("load", "mutable"):
            self.fail(node, "Buffer reads before T.Kernel require host execution lowering")
        if value.op == "var" and value.value not in self.parameter_types:
            self.fail(node, "Only pure scalar parameter expressions can be bound before T.Kernel")
        for child in value.args:
            self.check_prelude_expr(child, node)

    def assign_macro_result(self, target, value, node, *, parallel, nested):
        if isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, tuple) or len(target.elts) != len(value):
                self.fail(node, "Macro tuple unpacking requires matching target and result lengths")

            def capture(item):
                if isinstance(item, tuple):
                    return tuple(capture(child) for child in item)
                if isinstance(item, Expr) and item.op == "const":
                    return item
                if isinstance(item, (Expr, macros.ReferenceValue)):
                    return self.bind_macro_scalar(self.fresh("unpack"), self.scalar_value(item, node), node)
                return item

            values = tuple(capture(item) for item in value)
            result = []
            for left, right in zip(target.elts, values):
                previous_pending, self.pending = self.pending, []
                try:
                    statement = self.assign_macro_result(left, right, node, parallel=parallel, nested=nested)
                    result.extend(self.pending)
                finally:
                    self.pending = previous_pending
                result.extend(statement if isinstance(statement, tuple) else (statement,))
            return tuple(result)
        if isinstance(target, ast.Name):
            if isinstance(value, (Expr, macros.ReferenceValue)) or type(value) in (int, float, bool):
                target = self.assignment_target(target)
            if isinstance(target, ast.Name):
                return self.bind_name(target, value, node)
        assignment = ast.copy_location(
            ast.Assign(targets=[target], value=macros.ValueNode(value, node)), node
        )
        return self.statement(assignment, parallel=parallel, nested=nested)

    def debug_statement(self, call, name, node, parallel):
        if self.before_launch:
            self.fail(node, "Device diagnostics must appear inside T.Kernel")
        defaults = (
            {"obj": None, "msg": "", "warp_group_id": 0, "warp_id": 0}
            if name == "print"
            else {"msg": "", "no_stack_info": False}
        )
        parameters = list(defaults) if name == "print" else ["condition", *defaults]
        arguments = self.bind_call(call, parameters, defaults)
        values = {key: self.macro_value(value) for key, value in arguments.items()}
        if type(values["msg"]) is not str:
            self.fail(node, "Diagnostic messages must be construction-time strings")
        message = values["msg"]
        if name == "device_assert":
            if type(values["no_stack_info"]) is not bool:
                self.fail(node, "no_stack_info must be a construction-time Boolean")
            condition = self.scalar_value(values["condition"], node)
            if not values["no_stack_info"]:
                message += "\n"
                owner = self.macro_stack[-1][0].__name__ if self.macro_stack else self.node.name
                frames = [(owner, self.location(node))]
                for index in range(len(self.macro_stack) - 1, -1, -1):
                    owner = self.macro_stack[index - 1][0].__name__ if index else self.node.name
                    frames.append((owner, self.macro_stack[index][2]))
                message += "".join(f"  at {loc.filename}:{loc.line} in {owner}\n" for owner, loc in frames)
            return Statement(name, (Expr("cast", (condition,), "bool"), message), self.location(node))
        obj = values["obj"]
        if isinstance(obj, macros.ReferenceValue):
            obj = self.scalar_value(obj, node)
        if any(type(values[key]) is not int for key in ("warp_group_id", "warp_id")):
            self.fail(node, "Print warp selectors must be construction-time integers")
        main_lane = values["warp_group_id"] * 128 + values["warp_id"] * 32
        if isinstance(obj, Buffer):
            if obj.space != "global" and obj.name not in self.initialized:
                self.fail(node, f"Buffer {obj.name} is read before initialization")
            if obj.space == "fragment" and parallel:
                self.fail(
                    node, "Printing a full fragment requires uniform collective execution outside T.Parallel"
                )
            if not message and obj.space != "global":
                message = f"buffer<{obj.source_name or obj.name}, {obj.type.dtype}>"
        elif isinstance(obj, Expr):
            if not message:
                text = ast.unparse(arguments["obj"])
                for canonical, original in self.source_names.items():
                    text = text.replace(canonical, original)
                message = f"expr<{text}>"
        elif obj is None:
            if not message:
                self.fail(node, "Message-only T.print requires a nonempty message")
        else:
            self.fail(node, "T.print expects a buffer, scalar IR expression, or None")
        return Statement("print", (obj, message, main_lane), self.location(node))

    def statement(self, node, *, parallel=False, nested=False):
        loc = self.location(node)
        if isinstance(node, ast.Assert):
            condition = self.macro_value(node.test)
            message = self.macro_value(node.msg) if node.msg is not None else "Assertion failed"
            if isinstance(condition, (Expr, macros.ReferenceValue)):
                self.fail(
                    node,
                    "Runtime Python assert requires host error lowering; use T.device_assert inside T.Kernel",
                )
            if not condition:
                raise AssertionError(message)
            return Statement("pass", (), loc)
        if isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name):
                self.fail(node, "Local scalar annotations require a name target")
            dtype = self.static(node.annotation)
            if not isinstance(dtype, str) or dtype not in DTYPES:
                self.fail(node, "Local scalar annotations require a supported scalar dtype")
            if node.value is None:
                if node.target.id not in self.variables and node.target.id not in self.values:
                    self.fail(node, "An annotation alone does not initialize a scalar value")
                self.check_name(node.target)
                return Statement("pass", (), loc)
            # The default upstream eager Builder.bind uses the value dtype;
            # its annotation argument does not insert a numeric conversion.
            assignment = ast.Assign(targets=[node.target], value=node.value)
            ast.copy_location(assignment, node)
            statement = self.statement(assignment, parallel=parallel, nested=nested)
            if isinstance(statement, tuple):
                self.fail(node, "Scalar annotations require one value binding")
            return Statement(statement.op, statement.args, loc, (("scalar_annotation", dtype),))
        if isinstance(node, ast.If):
            value = self.macro_value(node.test)
            if type(value) in (int, float, bool, str, tuple, dict, type(None)):
                return self.statements(node.body if value else node.orelse, parallel=parallel, nested=nested)
            condition = self.scalar_value(value, node.test)
            if self.before_launch:
                self.fail(node, "Runtime control flow before T.Kernel requires host execution lowering")
            before_initialized = self.initialized.copy()
            with self.lexical_scope():
                then_body = self.statements(node.body, parallel=parallel, nested=True)
                then_initialized = self.initialized.copy()
                self.initialized = before_initialized
                else_body = self.statements(node.orelse, parallel=parallel, nested=True)
                self.initialized.intersection_update(then_initialized)
            return Statement("if", (condition, then_body, else_body), loc)
        if isinstance(node, ast.Pass):
            return Statement("pass", (), loc)
        if isinstance(node, (ast.Break, ast.Continue)):
            if self.macro_stack and len(self.loops) <= self.macro_stack[-1][1]:
                self.fail(node, "Macro early exits must target a loop inside that macro")
            if not self.loops or self.loops[-1] == "Parallel":
                self.fail(node, "Early exits require an enclosing serial, unroll, or while loop")
            return Statement("break" if isinstance(node, ast.Break) else "continue", (), loc)
        if isinstance(node, ast.While):
            if self.before_launch:
                self.fail(node, "Loops before T.Kernel require further construction-time lowering")
            if node.orelse:
                self.fail(node, "Loop else clauses are not supported")
            value = self.macro_value(node.test)
            if type(value) in (int, float, bool, str, tuple, dict, type(None)):
                if value:
                    self.fail(node, "A statically true while condition is an infinite loop")
                condition = Expr("const", value=False)
            else:
                condition = self.scalar_value(value, node.test)
            before_initialized = self.initialized.copy()
            self.loops.append("while")
            try:
                with self.lexical_scope():
                    body = self.statements(node.body, parallel=parallel, nested=True)
            finally:
                self.loops.pop()
            self.initialized = before_initialized
            return Statement("while", (condition, body), loc)
        if isinstance(node, ast.AugAssign):
            target = self.assignment_target(node.target)
            if isinstance(target, ast.Subscript):
                target = self.element_node(target)
            if isinstance(target, ast.Name):
                value = self.macro_value(target)
                if isinstance(value, (Buffer, macros.RegionValue)):
                    self.fail(node, "Buffer augmented assignment requires an element target")
            if not isinstance(target, (ast.Name, ast.Subscript)) or type(node.op) not in BINOPS:
                self.fail(node, "Augmented assignment requires a scalar or tensor element")
            assignment = ast.Assign(
                targets=[node.target if isinstance(node.target, ast.Name) else target],
                value=ast.BinOp(left=target, op=node.op, right=node.value),
            )
            ast.copy_location(assignment, node)
            ast.fix_missing_locations(assignment)
            return self.statement(assignment, parallel=parallel, nested=nested)
        if isinstance(node, ast.Assign) and len(node.targets) > 1:
            value = self.macro_value(node.value)
            if isinstance(value, (Expr, macros.ReferenceValue)):
                value = self.bind_macro_scalar(self.fresh("chained"), self.scalar_value(value, node), node)
            targets = ast.copy_location(ast.Tuple(elts=node.targets, ctx=ast.Store()), node)
            return self.assign_macro_result(
                targets, (value,) * len(node.targets), node, parallel=parallel, nested=nested
            )
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Subscript):
                # The eager parser calls assign_slice(buffer, indices, value).
                # Expand each target expression once, before the value's macros.
                target = self.element_node(target)
            if self.macro_object(node.value) is not None:
                return self.assign_macro_result(
                    target, self.expand_macro(node.value), node, parallel=parallel, nested=nested
                )
            if isinstance(target, (ast.Tuple, ast.List)):
                return self.assign_macro_result(
                    target, self.macro_value(node.value), node, parallel=parallel, nested=nested
                )
            if isinstance(node.value, macros.ValueNode) and isinstance(target, ast.Name):
                return self.assign_macro_result(
                    target, node.value.value, node, parallel=parallel, nested=nested
                )
            if isinstance(node.value, ast.Name) and (
                node.value.id in self.buffers
                or isinstance(self.values.get(node.value.id), (Buffer, macros.RegionValue, tuple, dict, str))
            ):
                return self.assign_macro_result(
                    target, self.macro_value(node.value), node, parallel=parallel, nested=nested
                )
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                name = self.optional_call_name(node.value)
                if name == "alloc_var":
                    if self.before_launch:
                        self.fail(node, "Allocations before T.Kernel require host execution lowering")
                    return self.scalar_allocation(node.value, target)
                if name in ("alloc_shared", "alloc_fragment"):
                    if self.before_launch:
                        self.fail(node, "Allocations before T.Kernel require host execution lowering")
                    if nested:
                        self.fail(node, "Allocate buffers directly inside T.Kernel, before loops")
                    if target.id.startswith("_nt_"):
                        self.fail(node, f"Reserved name {target.id}")
                    default_scope = "shared.dyn" if name == "alloc_shared" else "local.fragment"
                    args = self.bind_call(node.value, ["shape", "dtype", "scope"], {"scope": default_scope})
                    shape, dtype, scope = (self.static(args[key]) for key in ("shape", "dtype", "scope"))
                    typ = TensorType((shape,) if type(shape) is int else tuple(shape), dtype)
                    if name == "alloc_shared" and dtype == "bool":
                        scope = "shared"
                    if scope not in ("shared", "shared.dyn", "local.fragment"):
                        self.fail(node, f"Allocation scope {scope!r} requires further memory lowering")
                    buf = Buffer(
                        self.fresh(target.id),
                        typ,
                        "fragment" if scope == "local.fragment" else "shared",
                        source_scope=scope,
                        source_name=self.source_names.get(target.id, target.id),
                    )
                    self.buffers[buf.name] = buf
                    self.allocated.append(buf)
                    self.values[target.id] = buf
                    self.value_frames[target.id] = self.frames[-1]
                    return Statement("alloc", (buf.name,), loc)
            idx = self.indices(target.slice) if isinstance(target, ast.Subscript) else None
            value = self.macro_value(node.value)
            if isinstance(target, ast.Name):
                return self.assign_macro_result(target, value, node, parallel=parallel, nested=nested)
            value = self.scalar_value(value, node)
            if isinstance(target, ast.Subscript):
                name = self.buffer_name(target.value)
                if not parallel:
                    self.fail(node, "Scalar tensor stores require a T.Parallel loop")
                if len(idx) != len(self.buffers[name].type.shape):
                    self.fail(node, "Store rank mismatch")
                if self.buffers[name].space in ("fragment", "shared"):
                    shape, names = self.parallel_context
                    expected = tuple(Expr("var", value=n) for n in names)
                    if self.buffers[name].type.shape != shape or idx != expected:
                        self.fail(
                            node, "Temporary element stores require the matching T.Parallel shape and indices"
                        )
                    self.initialized.add(name)
                return Statement("store", (name, idx, value), loc)
        if isinstance(node, ast.For):
            if self.before_launch:
                self.fail(node, "Loops before T.Kernel require further construction-time lowering")
            name = self.call_name(node.iter)
            if name not in ("Parallel", "serial", "Serial", "Pipelined", "unroll", "Unroll"):
                self.fail(node, f"Unsupported loop {name}")
            if parallel and name == "Parallel":
                self.fail(node, "Parallel nesting currently requires a contiguous rectangular loop nest")
            annotations = ()
            loop_body = node.body
            targets = node.target.elts if isinstance(node.target, (ast.Tuple, ast.List)) else [node.target]
            if name == "Parallel":
                extents, targets, loop_body = self.parallel_nest(node)
            elif name in ("serial", "Serial", "unroll", "Unroll"):
                extents, annotations = self.loop_signature(node.iter, name)
            else:
                kw = self.keywords(node.iter, {"num_stages"} if name == "Pipelined" else set())
                if kw.get("num_stages", 1) not in (0, 1):
                    self.fail(node, "Asynchronous multi-stage pipelines are not implemented; use T.serial")
                extents = tuple(self.static(a) for a in node.iter.args)
            if not targets or any(not isinstance(t, ast.Name) for t in targets):
                self.fail(node, "Loop targets must be names")
            source_names = tuple(t.id for t in targets)
            if len(set(source_names)) != len(source_names) or any(n.startswith("_nt_") for n in source_names):
                self.fail(node, "Loop variables must have unique, non-reserved names")
            names = tuple(self.fresh(label) for label in source_names)
            if name == "Parallel":
                Partition(extents, self.threads)
                if len(names) != len(extents):
                    self.fail(node, "T.Parallel needs one variable per extent")
            else:
                if (
                    len(names) != 1
                    or len(extents) not in (1, 2, 3)
                    or any(type(x) is not int and not isinstance(x, Expr) for x in extents)
                ):
                    self.fail(
                        node, "Serial/unroll loops accept integer stop, (start, stop), or (start, stop, step)"
                    )
                extents = (
                    (0, extents[0], 1)
                    if len(extents) == 1
                    else (*extents, 1)
                    if len(extents) == 2
                    else extents
                )
                if type(extents[2]) is not int:
                    self.fail(node, "Loop step must be a static integer")
                if extents[2] == 0:
                    self.fail(node, "Loop step must be nonzero")
                if any(type(x) is int and (x < -(2**31) or x > 2**31 - 1) for x in extents):
                    self.fail(node, "Loop bounds must fit signed 32-bit integers")
                static_domain = all(type(value) is int for value in extents)
                if static_domain and len(range(*extents)) > 2**31 - 1:
                    self.fail(node, "Loop iteration count exceeds signed 32-bit indexing")
                if not static_domain and dict(annotations).get("pragma_unroll_explicit"):
                    self.fail(node, "Explicit unroll requires static loop bounds")
            if node.orelse:
                self.fail(node, "Loop else clauses are not supported")
            old_initialized = self.initialized.copy()
            old_parallel = self.parallel_context
            self.variables.update(names)
            if name == "Parallel":
                self.parallel_context = (extents, names)
            self.loops.append(name)
            try:
                with self.lexical_scope():
                    for source_name, canonical in zip(source_names, names):
                        self.values[source_name] = Expr("var", value=canonical)
                        self.value_frames[source_name] = self.frames[-1]
                    body = self.statements(loop_body, parallel=parallel or name == "Parallel", nested=True)
            finally:
                self.loops.pop()
            self.parallel_context = old_parallel
            controls = loop_controls(body)
            if name != "Parallel" and (not static_domain or not range(*extents) or controls):
                self.initialized = old_initialized
            kind = (
                "parallel" if name == "Parallel" else "unroll" if name in ("unroll", "Unroll") else "serial"
            )
            if kind == "unroll" and dict(annotations).get("pragma_unroll_explicit") and "break" in controls:
                self.fail(node, "A loop with a targeted break cannot be explicitly expanded")
            return Statement(kind, (names, extents, body), loc, annotations)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if self.macro_object(call) is not None:
                result = self.expand_macro(call)
                if isinstance(result, Expr) or type(result) in (int, float, bool):
                    return Statement("evaluate", (self.scalar_value(result, call),), loc)
                return Statement("pass", (), loc)
            name = self.call_name(call)
            if name in ("print", "device_assert"):
                return self.debug_statement(call, name, node, parallel)
            if name == "likely":
                return Statement("evaluate", (self.expr(call),), loc)
            if name in ("loop_break", "break_loop", "continue_loop"):
                parameters = [] if name == "loop_break" else ["span"]
                arguments = self.bind_call(call, parameters, {} if not parameters else {"span": None})
                if arguments and self.static(arguments["span"]) is not None:
                    self.fail(call, "Explicit source span objects require further parser integration")
                control = ast.Continue() if name == "continue_loop" else ast.Break()
                ast.copy_location(control, node)
                return self.statement(control, parallel=parallel, nested=nested)
            if name == "copy":
                return self.copy_statement(call, node, parallel, nested)
            if parallel:
                self.fail(node, "Collective tile operations cannot appear inside T.Parallel")
            if name in ("cumsum", "cummax", "cumsum_fragment", "cummax_fragment"):
                return scan.parse(self, call, name, loc)
            if name == "reduce" or name.startswith("reduce_"):
                return self.reduction(call, name, loc)
            if name in ("clear", "fill"):
                if len(call.args) != (1 if name == "clear" else 2) or call.keywords:
                    self.fail(node, "T.clear(buffer) or T.fill(buffer, value) expected")
                buf = self.buffer_name(call.args[0])
                if self.buffers[buf].space == "global":
                    self.fail(node, "Fill applies to shared or fragment tiles")
                value = Expr("const", value=0) if name == "clear" else self.expr(call.args[1])
                self.initialized.add(buf)
                return Statement("fill", (buf, value), loc)
            if name == "gemm":
                if len(call.args) != 3:
                    self.fail(node, "T.gemm expects A, B, accumulator")
                kw = self.keywords(call, {"transpose_A", "transpose_B"})
                if any(type(v) is not bool for v in kw.values()):
                    self.fail(node, "Transpose flags must be bool")
                names = tuple(self.buffer_name(x) for x in call.args)
                for buf in names:
                    if buf not in self.initialized:
                        self.fail(node, f"Buffer {buf} is read before initialization")
                return Statement(
                    "gemm", (*names, kw.get("transpose_A", False), kw.get("transpose_B", False)), loc
                )
            self.fail(node, f"Unsupported statement operation T.{name}")
        self.fail(node, f"Unsupported statement {type(node).__name__}")

    def parse(self):
        fn = self.node
        if fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg or fn.args.defaults:
            self.fail(fn, "Kernel parameters must be positional tensors or scalars without defaults")
        parameters = []
        for param in fn.args.args:
            annotation = self.function.__annotations__.get(param.arg)
            if isinstance(annotation, str) and annotation in language.DTYPE_NAMES:
                annotation = language.DTYPE_NAMES[annotation]
            elif not isinstance(annotation, TensorType):
                node = param.annotation
                if isinstance(node, ast.Call) and self.call_name(node) == "Tensor":
                    args = self.bind_call(
                        node,
                        ["shape", "dtype", "data", "scope"],
                        {"dtype": "float32", "data": None, "scope": None},
                    )
                    try:
                        annotation = language.Tensor(
                            *(self.static(args[key]) for key in ("shape", "dtype", "data", "scope"))
                        )
                    except (TypeError, ValueError) as exc:
                        self.fail(param, str(exc))
                else:
                    if node is None:
                        self.fail(param, "Every parameter needs T.Tensor(shape, dtype) or a scalar dtype")
                    annotation = self.static(node)
                    if not isinstance(annotation, str):
                        self.fail(param, "Scalar parameters need a dtype such as T.int32 or T.float32")
            if param.arg.startswith("_nt_"):
                self.fail(param, "Names starting with _nt_ are reserved")
            if isinstance(annotation, TensorType):
                parameter = Buffer(
                    param.arg,
                    annotation,
                    strides=tuple(
                        prod(annotation.shape[index + 1 :]) for index in range(len(annotation.shape))
                    ),
                    source_scope="global",
                )
                self.buffers[param.arg] = parameter
            else:
                parameter = ScalarParameter(param.arg, annotation)
                self.variables.add(param.arg)
                self.parameter_types[param.arg] = Expr("parameter", value=parameter.dtype)
            self.value_frames[param.arg] = self.frames[-1]
            parameters.append(parameter)
        parameters = tuple(parameters)
        body = fn.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if (
            not body
            or not isinstance(body[-1], ast.With)
            or len(body[-1].items) != 1
            or any(isinstance(node, ast.With) for node in body[:-1])
        ):
            self.fail(fn, "A kernel must end with one top-level with T.Kernel(...) block")
        self.before_launch = True
        prefix = self.statements(body[:-1])
        for statement in prefix:
            if statement.op == "evaluate":
                self.check_prelude_expr(statement.args[0], fn)
            elif statement.op != "pass":
                self.fail(fn, "Statements before T.Kernel require pure construction-time bindings")
        self.before_launch = False
        launch = body[-1]
        item = launch.items[0]
        call = item.context_expr
        if self.call_name(call) != "Kernel":
            self.fail(call, "Expected T.Kernel")
        grid = tuple(self.static(a) for a in call.args) or (1,)
        if not 1 <= len(grid) <= 3 or any(type(x) is not int or x <= 0 for x in grid):
            self.fail(call, "Kernel grid needs one to three positive static dimensions")
        if grid[0] > 2**31 - 1 or any(x > 65535 for x in grid[1:]):
            self.fail(call, "Kernel grid exceeds CUDA grid dimension limits")
        self.threads = self.keywords(call, {"threads"}).get("threads", 128)
        Partition((1,), self.threads)
        targets = (
            item.optional_vars.elts
            if isinstance(item.optional_vars, (ast.Tuple, ast.List))
            else [item.optional_vars]
        )
        if item.optional_vars is None:
            targets = [ast.Name(id=self.fresh(f"block{axis}")) for axis in range(len(grid))]
        if len(targets) != len(grid) or any(not isinstance(t, ast.Name) for t in targets):
            self.fail(launch, "T.Kernel needs one block variable per grid dimension")
        source_names = tuple(t.id for t in targets)
        if len(set(source_names)) != len(source_names) or any(
            v in self.buffers or v in self.variables or v.startswith("_nt_") for v in source_names
        ):
            self.fail(launch, "Duplicate or reserved block variable")
        block_vars = tuple(self.fresh(name) if name in self.values else name for name in source_names)
        self.variables.update(block_vars)
        with self.lexical_scope():
            for name, canonical in zip(source_names, block_vars):
                self.values[name] = Expr("var", value=canonical)
                self.value_frames[name] = self.frames[-1]
            statements = self.statements(launch.body)
        if not statements:
            self.fail(fn, "A kernel requires statements")
        shared_bytes = 0
        for buffer in self.allocated:
            if buffer.space == "shared":
                shared_bytes = language.ceildiv(shared_bytes, 16) * 16
                shared_bytes += prod(buffer.type.shape) * DTYPES[buffer.type.dtype]
        if shared_bytes > 48 * 1024:
            self.fail(fn, "This version supports at most 48 KiB of shared memory per block")
        return Kernel(
            fn.name,
            parameters,
            tuple(self.allocated),
            grid,
            block_vars,
            self.threads,
            statements,
            self.source,
        )


def parse(program: language.PrimFunc) -> Kernel:
    if not isinstance(program, language.PrimFunc):
        raise TypeError("Expected a function decorated with @ntilang.language.prim_func")
    return Parser(program).parse()
