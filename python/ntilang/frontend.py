"""Restricted Python AST frontend; unsupported syntax is a compilation error."""

from __future__ import annotations

import ast
import builtins
import inspect
import operator
import textwrap
from math import prod

from . import language
from .ir import Buffer, CompileError, Expr, Kernel, Partition, Region, SourceLocation, Statement, TensorType

BINOPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
}
COMPARISONS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
STATIC_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
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
            if value is builtins.range:
                return "serial"
            if callable(value) and value in language._MARKER_NAMES:
                return language._MARKER_NAMES[value]
            if value is language.ceildiv:
                return "ceildiv"
            if value is language.Tensor:
                return "Tensor"
        self.fail(node, "Only ntilang.language operations are supported in kernels")

    def static(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, str, bool):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.constants:
            value = self.constants[node.id]
            if type(value) in (int, float, str, bool, tuple):
                return value
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.static(x) for x in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self.static(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in STATIC_OPS:
            try:
                return STATIC_OPS[type(node.op)](self.static(node.left), self.static(node.right))
            except (TypeError, ZeroDivisionError) as exc:
                self.fail(node, f"Invalid static expression: {exc}")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if self.constants.get(node.value.id) is language and node.attr in (
                "float16",
                "bfloat16",
                "float32",
                "int32",
            ):
                return node.attr
        if (
            isinstance(node, ast.Call)
            and self.call_name(node) == "ceildiv"
            and len(node.args) == 2
            and not node.keywords
        ):
            try:
                return language.ceildiv(*(self.static(x) for x in node.args))
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

    def indices(self, node):
        parts = node.elts if isinstance(node, ast.Tuple) else [node]
        if any(isinstance(p, ast.Slice) for p in parts):
            self.fail(node, "Use a tile origin such as A[row, col] in T.copy; slices are not supported yet")
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
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
            op = "neg" if isinstance(node.op, ast.USub) else "pos" if isinstance(node.op, ast.UAdd) else "not"
            return Expr(op, (self.expr(node.operand),))
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
            if name == "ceildiv":
                return Expr("const", value=self.static(node))
            arity = {"exp": 1, "exp2": 1, "sqrt": 1, "maximum": 2, "minimum": 2, "cast": 2}
            if name in arity and len(node.args) == arity[name] and not node.keywords:
                if name == "cast":
                    dtype = self.static(node.args[1])
                    TensorType((1,), dtype)
                    return Expr("cast", (self.expr(node.args[0]),), dtype)
                return Expr(name, tuple(self.expr(x) for x in node.args))
        self.fail(node, f"Unsupported expression: {ast.dump(node, include_attributes=False)}")

    def buffer_name(self, node):
        if isinstance(node, ast.Name) and node.id in self.buffers:
            return node.id
        self.fail(node, "Expected a declared buffer")

    def region(self, node, shape):
        if isinstance(node, ast.Name):
            name = self.buffer_name(node)
            origin = tuple(Expr("const", value=0) for _ in shape)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            name = self.buffer_name(node.value)
            origin = self.indices(node.slice)
        else:
            self.fail(node, "Expected a whole tile or a global tensor tile origin")
        buf = self.buffers[name]
        if len(origin) != len(buf.type.shape) or len(shape) != len(origin):
            self.fail(node, "Copy rank mismatch")
        if buf.space != "global" and (
            shape != buf.type.shape or any(x != Expr("const", value=0) for x in origin)
        ):
            self.fail(node, "Copies of partial shared or fragment buffers are not supported")
        return Region(name, origin, shape)

    def statement(self, node, *, parallel=False, nested=False):
        loc = self.location(node)
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
                if self.buffers[name].space == "fragment":
                    shape, names = self.parallel_context
                    expected = tuple(Expr("var", value=n) for n in names)
                    if self.buffers[name].type.shape != shape or idx != expected:
                        self.fail(node, "Fragment stores require the matching T.Parallel shape and indices")
                    self.initialized.add(name)
                elif self.buffers[name].space != "global":
                    self.fail(node, "Shared element stores require a thread-ownership analysis")
                return Statement("store", (name, idx, value), loc)
        if isinstance(node, ast.For):
            name = self.call_name(node.iter)
            if name not in ("Parallel", "serial", "Serial", "Pipelined", "unroll", "Unroll"):
                self.fail(node, f"Unsupported loop {name}")
            if parallel and name == "Parallel":
                self.fail(node, "Nested T.Parallel loops are not supported; use T.Parallel(M, N)")
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
            return Statement(kind, (names, extents, body), loc)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            name = self.call_name(call)
            if parallel:
                self.fail(node, "Collective tile operations cannot appear inside T.Parallel")
            if name in ("clear", "fill"):
                if len(call.args) != (1 if name == "clear" else 2) or call.keywords:
                    self.fail(node, "T.clear(buffer) or T.fill(buffer, value) expected")
                buf = self.buffer_name(call.args[0])
                if self.buffers[buf].space == "global":
                    self.fail(node, "Fill applies to shared or fragment tiles")
                value = Expr("const", value=0) if name == "clear" else self.expr(call.args[1])
                self.initialized.add(buf)
                return Statement("fill", (buf, value), loc)
            if name == "copy":
                if len(call.args) != 2 or call.keywords:
                    self.fail(node, "T.copy expects exactly source and destination")
                tiles = [
                    self.buffers[x.id]
                    for x in call.args
                    if isinstance(x, ast.Name)
                    and x.id in self.buffers
                    and self.buffers[x.id].space != "global"
                ]
                if not tiles:
                    self.fail(node, "T.copy requires a shared or fragment tile to determine the extent")
                shape = tiles[0].type.shape
                src, dst = (self.region(x, shape) for x in call.args)
                if self.buffers[src.buffer].space != "global" and src.buffer not in self.initialized:
                    self.fail(node, f"Buffer {src.buffer} is read before initialization")
                self.initialized.add(dst.buffer)
                return Statement("copy", (src, dst), loc)
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
                shared_bytes += (
                    prod(buffer.type.shape)
                    * {"float16": 2, "bfloat16": 2, "float32": 4, "int32": 4}[buffer.type.dtype]
                )
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
