"""Restricted Python AST frontend; unsupported syntax is a compilation error."""

from __future__ import annotations

import ast
import builtins
import inspect
import operator
import textwrap
from math import prod

from . import language
from .ir import (
    DTYPES,
    Buffer,
    CompileError,
    Expr,
    Kernel,
    Partition,
    Region,
    SourceLocation,
    Statement,
    TensorType,
)
from .validation import affine

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
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}


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

    def location(self, node):
        return SourceLocation(self.filename, self.first_line + node.lineno - 1, node.col_offset)

    def fail(self, node, message):
        raise CompileError(message, self.location(node))

    def call_name(self, node):
        if not isinstance(node, ast.Call):
            self.fail(node, "Expected a language operation")
        fn = node.func
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            if self.constants.get(fn.value.id) is language:
                return fn.attr
        if isinstance(fn, ast.Name):
            value = self.constants.get(fn.id, getattr(builtins, fn.id, None))
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
        if isinstance(node, ast.Constant) and (
            type(node.value) in (int, float, str, bool) or node.value is None
        ):
            return node.value
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
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if self.constants.get(node.value.id) is language and node.attr in language.DTYPE_NAMES:
                return language.DTYPE_NAMES[node.attr]
        if isinstance(node, ast.Call) and self.call_name(node) in ("ceildiv", "cdiv", "align_up"):
            name = self.call_name(node)
            parameters = ["x", "y"] if name == "align_up" else ["lhs", "rhs", "span"]
            args = self.bind_call(node, parameters, {} if name == "align_up" else {"span": None})
            try:
                operation = language.align_up if name == "align_up" else language.ceildiv
                return operation(*(self.static(args[key]) for key in parameters))
            except (TypeError, ValueError) as exc:
                self.fail(node, str(exc))
        self.fail(node, "Expected a static specialization constant")

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

    def static_loop(self, call, name):
        """Bind the upstream static serial/unroll signatures and scheduling hints."""
        unrolled = name in ("unroll", "Unroll")
        parameters = ["start", "stop", "step", "annotations"]
        defaults = {"stop": None, "step": None, "annotations": None}
        if unrolled:
            parameters += ["explicit", "unroll_factor"]
            defaults.update(explicit=False, unroll_factor=None)
        if len(call.args) > 3:
            self.fail(call, "Loop scheduling arguments must be keyword-only")
        values = {
            key: self.static(value) for key, value in self.bind_call(call, parameters, defaults).items()
        }
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
        return tuple(self.expr(p) for p in parts)

    def expr(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
            return Expr("const", value=node.value)
        if isinstance(node, ast.Name):
            if node.id in self.variables:
                return Expr("var", value=node.id)
            return Expr("const", value=self.static(node))
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            name = node.value.id
            if name not in self.buffers:
                self.fail(node, f"Unknown buffer {name}")
            buf = self.buffers[name]
            if buf.space != "global" and name not in self.initialized:
                self.fail(node, f"Buffer {name} is read before initialization")
            indices = self.indices(node.slice)
            if len(indices) != len(buf.type.shape):
                self.fail(node, f"Buffer {name} expects {len(buf.type.shape)} indices")
            return Expr("load", indices, name)
        if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
            return Expr(BINOPS[type(node.op)], (self.expr(node.left), self.expr(node.right)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not, ast.Invert)):
            op = {ast.USub: "neg", ast.UAdd: "pos", ast.Not: "not", ast.Invert: "invert"}[type(node.op)]
            operand = self.expr(node.operand)
            if op in ("neg", "pos") and operand.op == "const" and type(operand.value) in (int, float):
                return Expr("const", value=-operand.value if op == "neg" else operand.value)
            return Expr(op, (operand,))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in COMPARISONS:
            return Expr(
                COMPARISONS[type(node.ops[0])], (self.expr(node.left), self.expr(node.comparators[0]))
            )
        if isinstance(node, ast.BoolOp):
            return Expr(
                "and" if isinstance(node.op, ast.And) else "or", tuple(self.expr(v) for v in node.values)
            )
        if isinstance(node, ast.Call):
            name = self.call_name(node)
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
                "exp": 1,
                "exp2": 1,
                "sqrt": 1,
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
        if isinstance(node, ast.Name) and node.id in self.buffers:
            return node.id
        self.fail(node, "Expected a declared buffer")

    def region_spec(self, node):
        if isinstance(node, ast.Name):
            name = self.buffer_name(node)
            shape = self.buffers[name].type.shape
            return name, tuple(Expr("const", value=0) for _ in shape), shape
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
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
            and isinstance(args["dst"], ast.Name)
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

    def statement(self, node, *, parallel=False, nested=False):
        loc = self.location(node)
        if isinstance(node, ast.If):
            condition = self.expr(node.test)
            before_vars = self.variables.copy()
            before_initialized = self.initialized.copy()
            then_body = tuple(self.statement(n, parallel=parallel, nested=True) for n in node.body)
            then_vars, then_initialized = self.variables.copy(), self.initialized.copy()
            self.variables = before_vars
            self.initialized = before_initialized
            else_body = tuple(self.statement(n, parallel=parallel, nested=True) for n in node.orelse)
            self.variables.intersection_update(then_vars)
            self.initialized.intersection_update(then_initialized)
            return Statement("if", (condition, then_body, else_body), loc)
        if isinstance(node, ast.Pass):
            return Statement("pass", (), loc)
        if isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Subscript) or type(node.op) not in BINOPS:
                self.fail(node, "Augmented assignment currently requires a tensor element")
            assignment = ast.Assign(
                targets=[node.target], value=ast.BinOp(left=node.target, op=node.op, right=node.value)
            )
            ast.copy_location(assignment, node)
            ast.fix_missing_locations(assignment)
            return self.statement(assignment, parallel=parallel, nested=nested)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                name = self.call_name(node.value)
                if name in ("alloc_shared", "alloc_fragment"):
                    if nested:
                        self.fail(node, "Allocate buffers directly inside T.Kernel, before loops")
                    if (
                        target.id in self.variables
                        or target.id in self.buffers
                        or target.id.startswith("_nt_")
                    ):
                        self.fail(node, f"Duplicate or reserved name {target.id}")
                    if len(node.value.args) != 2 or node.value.keywords:
                        self.fail(node, "Allocation requires (shape, dtype)")
                    typ = TensorType(tuple(self.static(node.value.args[0])), self.static(node.value.args[1]))
                    buf = Buffer(target.id, typ, "shared" if name == "alloc_shared" else "fragment")
                    self.buffers[buf.name] = buf
                    self.allocated.append(buf)
                    return Statement("alloc", (buf.name,), loc)
            value = self.expr(node.value)
            if isinstance(target, ast.Name):
                if target.id in self.variables or target.id in self.buffers or target.id.startswith("_nt_"):
                    self.fail(node, f"Cannot assign to {target.id}")
                self.variables.add(target.id)
                return Statement("let", (target.id, value), loc)
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                name = self.buffer_name(target.value)
                if not parallel:
                    self.fail(node, "Scalar tensor stores require a T.Parallel loop")
                idx = self.indices(target.slice)
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
            name = self.call_name(node.iter)
            if name not in ("Parallel", "serial", "Serial", "Pipelined", "unroll", "Unroll"):
                self.fail(node, f"Unsupported loop {name}")
            if parallel and name == "Parallel":
                self.fail(node, "Nested T.Parallel loops are not supported; use T.Parallel(M, N)")
            annotations = ()
            if name in ("serial", "Serial", "unroll", "Unroll"):
                extents, annotations = self.static_loop(node.iter, name)
            else:
                kw = self.keywords(node.iter, {"num_stages"} if name == "Pipelined" else set())
                if kw.get("num_stages", 1) not in (0, 1):
                    self.fail(node, "Asynchronous multi-stage pipelines are not implemented; use T.serial")
                extents = tuple(self.static(a) for a in node.iter.args)
            targets = node.target.elts if isinstance(node.target, (ast.Tuple, ast.List)) else [node.target]
            if not targets or any(not isinstance(t, ast.Name) for t in targets):
                self.fail(node, "Loop targets must be names")
            names = tuple(t.id for t in targets)
            if any(n in self.variables or n in self.buffers or n.startswith("_nt_") for n in names):
                self.fail(node, "Loop variables must have unique, non-reserved names")
            if name == "Parallel":
                Partition(extents, self.threads)
                if len(names) != len(extents):
                    self.fail(node, "T.Parallel needs one variable per extent")
            else:
                if (
                    len(names) != 1
                    or len(extents) not in (1, 2, 3)
                    or any(type(x) is not int for x in extents)
                ):
                    self.fail(
                        node, "Serial/unroll loops accept static stop, (start, stop), or (start, stop, step)"
                    )
                extents = (
                    (0, extents[0], 1)
                    if len(extents) == 1
                    else (*extents, 1)
                    if len(extents) == 2
                    else extents
                )
                if extents[2] == 0:
                    self.fail(node, "Loop step must be nonzero")
                if any(x < -(2**31) or x > 2**31 - 1 for x in extents):
                    self.fail(node, "Loop bounds must fit signed 32-bit integers")
                if len(range(*extents)) > 2**31 - 1:
                    self.fail(node, "Loop iteration count exceeds signed 32-bit indexing")
            if node.orelse:
                self.fail(node, "Loop else clauses are not supported")
            old_vars = self.variables.copy()
            old_initialized = self.initialized.copy()
            old_parallel = self.parallel_context
            self.variables.update(names)
            if name == "Parallel":
                self.parallel_context = (extents, names)
            body = tuple(
                self.statement(n, parallel=parallel or name == "Parallel", nested=True) for n in node.body
            )
            self.variables = old_vars
            self.parallel_context = old_parallel
            if name != "Parallel" and not range(*extents):
                self.initialized = old_initialized
            kind = (
                "parallel" if name == "Parallel" else "unroll" if name in ("unroll", "Unroll") else "serial"
            )
            return Statement(kind, (names, extents, body), loc, annotations)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            name = self.call_name(call)
            if name == "copy":
                return self.copy_statement(call, node, parallel, nested)
            if parallel:
                self.fail(node, "Collective tile operations cannot appear inside T.Parallel")
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
            self.fail(fn, "Kernel parameters must be positional tensors without defaults")
        for param in fn.args.args:
            annotation = self.function.__annotations__.get(param.arg)
            if not isinstance(annotation, TensorType):
                node = param.annotation
                if (
                    not isinstance(node, ast.Call)
                    or self.call_name(node) != "Tensor"
                    or len(node.args) != 2
                    or node.keywords
                ):
                    self.fail(param, "Every parameter needs T.Tensor(shape, dtype)")
                annotation = TensorType(tuple(self.static(node.args[0])), self.static(node.args[1]))
            if param.arg.startswith("_nt_"):
                self.fail(param, "Names starting with _nt_ are reserved")
            self.buffers[param.arg] = Buffer(param.arg, annotation)
        parameters = tuple(self.buffers.values())
        body = fn.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.With) or len(body[0].items) != 1:
            self.fail(fn, "A kernel must contain one top-level with T.Kernel(...) block")
        launch = body[0]
        item = launch.items[0]
        call = item.context_expr
        if self.call_name(call) != "Kernel":
            self.fail(call, "Expected T.Kernel")
        grid = tuple(self.static(a) for a in call.args)
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
        if len(targets) != len(grid) or any(not isinstance(t, ast.Name) for t in targets):
            self.fail(launch, "T.Kernel needs one block variable per grid dimension")
        block_vars = tuple(t.id for t in targets)
        if len(set(block_vars)) != len(block_vars) or any(
            v in self.buffers or v.startswith("_nt_") for v in block_vars
        ):
            self.fail(launch, "Duplicate or reserved block variable")
        self.variables.update(block_vars)
        statements = tuple(self.statement(n) for n in launch.body)
        if not statements or not parameters:
            self.fail(fn, "A kernel requires tensor parameters and statements")
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
