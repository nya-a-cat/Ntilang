"""Conservative checks for the initial, statically shaped language subset."""

from __future__ import annotations

from .ir import DTYPES, CompileError, Expr, Kernel, ScalarParameter, integer_limits
from .scalar import (
    BINARY_MATH_OPS,
    BINARY_NUMERIC_OPS,
    BITWISE_OPS,
    CHOICE_OPS,
    INTEGER_DIVISION_OPS,
    ROUNDING_OPS,
    UNARY_MATH_OPS,
    expression_dtype,
)

INT_MIN, INT_MAX = -(2**31), 2**31 - 1


def resolved_dtype(expr, bounds, definitions, buffers, *, ignore_predicates=False):
    def value_expression(value):
        if ignore_predicates and value.op in CHOICE_OPS:
            value = Expr("phi", value.args[1:])
        return Expr(value.op, tuple(value_expression(arg) for arg in value.args), value.value)

    class VariableTypes(dict):
        def __missing__(self, name):
            if name not in definitions:
                raise CompileError(f"Scalar {name} is not defined in this scope")
            self[name] = expression_dtype(value_expression(definitions[name]), buffers, self)
            return self[name]

    return expression_dtype(
        value_expression(expr), buffers, VariableTypes({name: "int32" for name in bounds})
    )


def predicate_bounds(condition, truth, bounds, definitions, buffers=None):
    """Refine an integer interval for a necessary single-variable predicate."""
    buffers = {} if buffers is None else buffers
    if condition.op == "const" and type(condition.value) is bool:
        return bounds.copy() if condition.value == truth else None
    if condition.op == "var" and condition.value in definitions:
        return predicate_bounds(definitions[condition.value], truth, bounds, definitions, buffers)
    if condition.op == "not":
        return predicate_bounds(condition.args[0], not truth, bounds, definitions, buffers)
    if condition.op == ("and" if truth else "or"):
        result = bounds.copy()
        for child in condition.args:
            result = predicate_bounds(child, truth, result, definitions, buffers)
            if result is None:
                break
        return result
    relation = condition.op
    if relation not in ("<", "<=", ">", ">=", "==", "!="):
        return bounds.copy()
    difference = Expr("-", condition.args)
    try:
        operand_bounds = [interval(arg, bounds, definitions, buffers) for arg in condition.args]
        dtype = resolved_dtype(difference, bounds, definitions, buffers, ignore_predicates=True)
        if dtype.startswith("uint") and any(low < 0 for low, _ in operand_bounds):
            return bounds.copy()
        constant, coefficients = affine(difference, definitions, bounds, buffers)
    except CompileError:
        return bounds.copy()
    if not truth:
        relation = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}[relation]
    if not coefficients:
        holds = {
            "<": constant < 0,
            "<=": constant <= 0,
            ">": constant > 0,
            ">=": constant >= 0,
            "==": constant == 0,
            "!=": constant != 0,
        }[relation]
        return bounds.copy() if holds else None
    if len(coefficients) != 1:
        return bounds.copy()
    name, coefficient = next(iter(coefficients.items()))
    if name not in bounds:
        return bounds.copy()
    if coefficient < 0:
        coefficient, constant = -coefficient, -constant
        relation = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}[relation]
    low, high = bounds[name]
    if relation == "<":
        high = min(high, (-constant - 1) // coefficient)
    elif relation == "<=":
        high = min(high, -constant // coefficient)
    elif relation == ">":
        low = max(low, -constant // coefficient + 1)
    elif relation == ">=":
        low = max(low, -(constant // coefficient))
    elif -constant % coefficient:
        return None if relation == "==" else bounds.copy()
    else:
        value = -constant // coefficient
        if relation == "==":
            low, high = max(low, value), min(high, value)
        elif value == low:
            low += 1
        elif value == high:
            high -= 1
    return None if low > high else {**bounds, name: (low, high)}


def interval(expr, bounds, definitions, buffers=None):
    """Bound every intermediate operation in an integer index expression."""
    buffers = {} if buffers is None else buffers
    if expr.op == "const" and type(expr.value) is int:
        result = (expr.value, expr.value)
    elif expr.op == "phi" or expr.op in CHOICE_OPS:
        alternatives = []
        for index, branch in enumerate(expr.args if expr.op == "phi" else expr.args[1:]):
            branch_bounds = (
                predicate_bounds(expr.args[0], index == 0, bounds, definitions, buffers)
                if expr.op == "if_then_else"
                else bounds
            )
            if branch_bounds is not None:
                alternatives.append(interval(branch, branch_bounds, definitions, buffers))
        result = (min(x[0] for x in alternatives), max(x[1] for x in alternatives))
        dtype = resolved_dtype(expr, bounds, definitions, buffers, ignore_predicates=True)
        low, high = integer_limits(dtype)
        if result[0] < low or result[1] > high:
            raise CompileError("A conditional index conversion can change the represented value")
    elif expr.op == "var":
        if expr.value in definitions:
            return interval(definitions[expr.value], bounds, definitions, buffers)
        if expr.value not in bounds:
            raise CompileError("Indices must use integer block/loop variables and static constants")
        result = bounds[expr.value]
    elif expr.op in ("mutable", "load", "parameter"):
        if expr.op == "load" and expr.value not in buffers:
            raise CompileError("Integer data bounds require buffer dtype information")
        dtype = buffers[expr.value].type.dtype if expr.op == "load" else expr.value
        if not dtype.startswith(("int", "uint")):
            raise CompileError("Data-dependent indices require an integer dtype")
        result = integer_limits(dtype)
    elif expr.op == "cast" and (expr.value.startswith(("int", "uint")) or expr.value == "bool"):
        result = interval(expr.args[0], bounds, definitions, buffers)
        low, high = integer_limits(expr.value)
        if result[0] < low or result[1] > high:
            raise CompileError("An integer index cast can change the represented value")
    elif expr.op in ROUNDING_OPS:
        result = interval(expr.args[0], bounds, definitions, buffers)
    elif expr.op == "abs":
        low, high = interval(expr.args[0], bounds, definitions, buffers)
        dtype = resolved_dtype(expr, bounds, definitions, buffers, ignore_predicates=True)
        type_low, type_high = integer_limits(dtype)
        if dtype.startswith("int") and low == type_low:
            # TIR's integer Select preserves the signed-minimum bit pattern.
            result = (type_low, type_low) if high == low else (type_low, type_high)
        else:
            result = (0 if low <= 0 <= high else min(abs(low), abs(high)), max(abs(low), abs(high)))
    elif expr.op in ("neg", "pos", "invert"):
        low, high = interval(expr.args[0], bounds, definitions, buffers)
        result = (-high, -low) if expr.op == "neg" else (~high, ~low) if expr.op == "invert" else (low, high)
        if expr.op == "invert":
            dtype = resolved_dtype(expr, bounds, definitions, buffers, ignore_predicates=True)
            if dtype.startswith("uint"):
                mask = integer_limits(dtype)[1]
                result = (mask - high, mask - low)
    elif expr.op in ("&", "|", "^", "<<", ">>"):
        a, b = (interval(x, bounds, definitions, buffers) for x in expr.args)
        dtype = resolved_dtype(expr, bounds, definitions, buffers, ignore_predicates=True)
        type_low, type_high = integer_limits(dtype)
        if dtype.startswith("uint"):

            def unsigned(values):
                low, high = values
                if high < 0:
                    return (low + type_high + 1, high + type_high + 1)
                return (0, type_high) if low < 0 else values

            a, b = unsigned(a), unsigned(b)
        if expr.op in ("<<", ">>"):
            if b[0] < 0 or b[1] >= DTYPES[dtype] * 8:
                raise CompileError("Index shift counts must be within the promoted integer width")
            operation = (lambda x, y: x << y) if expr.op == "<<" else (lambda x, y: x >> y)
            endpoints = [operation(x, y) for x in a for y in b]
            result = (min(endpoints), max(endpoints))
            if result[0] < type_low or result[1] > type_high:
                raise CompileError("An index shift can overflow its integer dtype")
        elif expr.op == "&" and (a[0] >= 0 or b[0] >= 0):
            result = (0, min(high for low, high in (a, b) if low >= 0))
        elif a[0] >= 0 and b[0] >= 0:
            result = (0, (1 << max(a[1].bit_length(), b[1].bit_length())) - 1)
        else:
            result = (INT_MIN, INT_MAX)
    elif expr.op in INTEGER_DIVISION_OPS:
        a, b = (interval(x, bounds, definitions, buffers) for x in expr.args)
        dtype = resolved_dtype(expr, bounds, definitions, buffers, ignore_predicates=True)
        type_low, type_high = integer_limits(dtype)
        if dtype.startswith("uint") and (a[0] < 0 or b[0] < 0):
            raise CompileError("An unsigned index operand conversion can change the represented value")
        if b[0] <= 0 <= b[1]:
            raise CompileError("An index divisor can be zero")
        if expr.op == "ceildiv":
            total = (a[0] + b[0], a[1] + b[1])
            if total[0] < type_low or total[1] > type_high or total[0] - 1 < type_low:
                raise CompileError("The ceildiv numerator can overflow its integer dtype")
            a = (total[0] - 1, total[1] - 1)
        if dtype.startswith("int") and a[0] <= type_low <= a[1] and b[0] <= -1 <= b[1]:
            raise CompileError("Signed minimum divided by -1 can overflow the integer dtype")
        if expr.op in ("//", "ceildiv", "truncdiv"):

            def quotient(x, y):
                if expr.op == "truncdiv":
                    return (abs(x) // abs(y)) * (-1 if (x < 0) != (y < 0) else 1)
                return x // y

            endpoints = [quotient(x, y) for x in a for y in b]
            result = (min(endpoints), max(endpoints))
        elif a[0] == a[1] and b[0] == b[1]:
            if expr.op == "%":
                result = (a[0] % b[0],) * 2
            else:
                remainder = abs(a[0]) % abs(b[0]) * (-1 if a[0] < 0 else 1)
                result = (remainder,) * 2
        elif expr.op == "%":
            result = (0, b[1] - 1) if b[0] > 0 else (b[0] + 1, 0)
        else:
            magnitude = max(abs(b[0]), abs(b[1])) - 1
            result = (max(a[0], -magnitude) if a[0] < 0 else 0, min(a[1], magnitude) if a[1] > 0 else 0)
    elif expr.op in ("min", "max", "minimum", "maximum"):
        a, b = (interval(x, bounds, definitions, buffers) for x in expr.args)
        dtype = resolved_dtype(expr, bounds, definitions, buffers, ignore_predicates=True)
        if dtype.startswith("uint") and (a[0] < 0 or b[0] < 0):
            raise CompileError("An unsigned index operand conversion can change the represented value")
        operation = min if expr.op in ("min", "minimum") else max
        result = tuple(operation(left, right) for left, right in zip(a, b))
    elif expr.op in ("+", "-", "*"):
        a, b = (interval(x, bounds, definitions, buffers) for x in expr.args)
        if expr.op == "+":
            result = (a[0] + b[0], a[1] + b[1])
        elif expr.op == "-":
            result = (a[0] - b[1], a[1] - b[0])
        elif expr.op == "*":
            products = [x * y for x in a for y in b]
            result = (min(products), max(products))
    else:
        raise CompileError("Data-dependent or non-integer indexing is not supported")
    if result[0] < INT_MIN or result[1] > INT_MAX:
        raise CompileError("An index expression can overflow signed 32-bit arithmetic")
    if expr.op in INTEGER_DIVISION_OPS | {"+", "-", "*", "neg", "pos"}:
        dtype = resolved_dtype(expr, bounds, definitions, buffers, ignore_predicates=True)
        low, high = integer_limits(dtype)
        if result[0] < low or result[1] > high:
            raise CompileError("An index expression can overflow its integer dtype")
    return result


def affine(expr, definitions, bounds=None, buffers=None):
    """Return constant and integer coefficients, or reject a non-affine expression."""
    if expr.op == "const" and type(expr.value) is int:
        return expr.value, {}
    if expr.op == "phi" or expr.op in CHOICE_OPS:
        alternatives = [
            affine(x, definitions, bounds, buffers)
            for x in (expr.args if expr.op == "phi" else expr.args[1:])
        ]
        if all(x == alternatives[0] for x in alternatives):
            return alternatives[0]
        raise CompileError("Global writes require branch-independent affine ownership")
    if expr.op == "var":
        if expr.value in definitions:
            return affine(definitions[expr.value], definitions, bounds, buffers)
        return 0, {expr.value: 1}
    if expr.op == "cast" and (expr.value.startswith(("int", "uint")) or expr.value == "bool"):
        interval(expr, {} if bounds is None else bounds, definitions, buffers)
        return affine(expr.args[0], definitions, bounds, buffers)
    if expr.op in ("neg", "pos"):
        const, coeff = affine(expr.args[0], definitions, bounds, buffers)
        sign = -1 if expr.op == "neg" else 1
        return sign * const, {n: sign * v for n, v in coeff.items()}
    if expr.op == "<<":
        shift, coefficients = affine(expr.args[1], definitions, bounds, buffers)
        if not coefficients and 0 <= shift < 32:
            interval(expr, {} if bounds is None else bounds, definitions, buffers)
            return affine(
                Expr("*", (expr.args[0], Expr("const", value=1 << shift))), definitions, bounds, buffers
            )
    if expr.op in ("//", "truncdiv", "ceildiv"):
        divisor, variables = affine(expr.args[1], definitions, bounds, buffers)
        constant, coefficients = affine(expr.args[0], definitions, bounds, buffers)
        if not variables and divisor and all(value % divisor == 0 for value in coefficients.values()):
            if expr.op == "truncdiv" and constant % divisor:
                raise CompileError("Truncating division needs an exactly divisible affine numerator")
            interval(expr, {} if bounds is None else bounds, definitions, buffers)
            if expr.op == "ceildiv":
                constant += divisor - 1
            return constant // divisor, {name: value // divisor for name, value in coefficients.items()}
    if expr.op in ("+", "-", "*"):
        (ac, av), (bc, bv) = (affine(x, definitions, bounds, buffers) for x in expr.args)
        if expr.op in ("+", "-"):
            sign = -1 if expr.op == "-" else 1
            out = av.copy()
            for n, v in bv.items():
                out[n] = out.get(n, 0) + sign * v
            return ac + sign * bc, {n: v for n, v in out.items() if v}
        if not av:
            return ac * bc, {n: ac * v for n, v in bv.items()}
        if not bv:
            return ac * bc, {n: bc * v for n, v in av.items()}
    raise CompileError("Global writes require affine indices whose ownership can be checked")


def check_ownership(indices, bounds, definitions, buffers=None):
    """Prove injectivity with separated integer strides in each coordinate.

    For sorted nonzero coefficients, each larger coefficient must exceed the
    complete span of all smaller coefficients. An equality of coordinates then
    forces each contributing variable to agree. Every active variable must be
    recoverable from at least one output coordinate. This sufficient condition
    deliberately rejects affine maps outside the supported proof rule.
    """
    active = {name for name, (low, high) in bounds.items() if high > low}
    recovered = set()
    for index in indices:
        interval(index, bounds, definitions, buffers)
        _, coefficients = affine(index, definitions, bounds, buffers)
        ordered = sorted((abs(c), n) for n, c in coefficients.items() if c and n in active)
        span = 0
        for coefficient, name in ordered:
            if coefficient <= span:
                break
            low, high = bounds[name]
            span += coefficient * (high - low)
        else:
            recovered.update(name for _, name in ordered)
    if recovered != active:
        raise CompileError(
            "Cannot prove unique global write ownership for variables: "
            + ", ".join(sorted(active - recovered))
            + ". Use disjoint tiles such as output[block * tile_size + i]."
        )


def validate(kernel: Kernel):
    buffers = kernel.buffer_map
    reads, writes = set(), {}
    initial_bounds = {name: (0, size - 1) for name, size in zip(kernel.block_vars, kernel.grid)}

    def runtime_value(expr, definitions):
        if expr.op in ("load", "mutable", "parameter"):
            return True
        if expr.op == "var" and expr.value in definitions:
            return runtime_value(definitions[expr.value], definitions)
        return any(runtime_value(arg, definitions) for arg in expr.args)

    def expression(expr, bounds, definitions):
        if expr.op in BITWISE_OPS | BINARY_NUMERIC_OPS | BINARY_MATH_OPS | CHOICE_OPS | UNARY_MATH_OPS | {
            "and",
            "or",
            "not",
            "pow_integer",
        }:
            dtype = resolved_dtype(expr, bounds, definitions, buffers)
            if expr.op in ("<<", ">>"):
                try:
                    low, high = interval(expr.args[1], bounds, definitions, buffers)
                except CompileError:
                    # Data-dependent counts retain the upstream valid-count precondition.
                    pass
                else:
                    if (low < 0 or high >= DTYPES[dtype] * 8) and (
                        not runtime_value(expr.args[1], definitions) or high < 0 or low >= DTYPES[dtype] * 8
                    ):
                        raise CompileError(
                            "Shift count must be nonnegative and smaller than the promoted integer width"
                        )
        if expr.op == "if_then_else":
            expression(expr.args[0], bounds, definitions)
            for truth, branch in zip((True, False), expr.args[1:]):
                branch_bounds = predicate_bounds(expr.args[0], truth, bounds, definitions, buffers)
                if branch_bounds is not None:
                    expression(branch, branch_bounds, definitions)
            return
        if expr.op in INTEGER_DIVISION_OPS:
            operands = []
            for arg in expr.args:
                try:
                    operands.append(interval(arg, bounds, definitions, buffers))
                except CompileError:
                    operands.append(None)
            left, right = operands
            if right == (0, 0):
                raise CompileError("Integer division requires a nonzero divisor")
            if (
                expr.op == "ceildiv"
                and left is not None
                and right is not None
                and not runtime_value(expr, definitions)
            ):
                interval(expr, bounds, definitions, buffers)
            elif left is not None and right == (-1, -1):
                low = integer_limits(dtype)[0]
                if dtype.startswith("int") and left == (low, low):
                    raise CompileError("Signed minimum divided by -1 overflows the integer dtype")
        if expr.op == "load":
            if buffers[expr.value].space == "global":
                reads.add(expr.value)
            for index in expr.args:
                interval(index, bounds, definitions, buffers)
        for arg in expr.args:
            expression(arg, bounds, definitions)

    def record_write(name, indices, bounds, definitions, in_serial, path):
        if in_serial:
            raise CompileError("Write global outputs after serial tile accumulation loops")
        check_ownership(indices, bounds, definitions, buffers)
        mapping = (tuple(affine(x, definitions, bounds, buffers) for x in indices), bounds)
        footprint = tuple(
            (max(0, low), min(size - 1, high))
            for (low, high), size in zip(
                (interval(x, bounds, definitions, buffers) for x in indices), buffers[name].type.shape
            )
        )
        for previous_path, previous_mapping, previous_footprint in writes.get(name, []):
            disjoint = any(
                high < old_low or old_high < low
                for (low, high), (old_low, old_high) in zip(footprint, previous_footprint)
            )
            if disjoint:
                continue
            exclusive = any(
                key in previous_path and previous_path[key] != value for key, value in path.items()
            )
            if not exclusive or mapping != previous_mapping:
                raise CompileError(
                    f"Output {name} has multiple write sites without disjoint regions or exclusive branches and identical ownership"
                )
        writes.setdefault(name, []).append((path.copy(), mapping, footprint))

    def statements(body, bounds, definitions, in_serial=False, path=None):
        path = {} if path is None else path
        for stmt in body:
            op, args = stmt.op, stmt.args
            try:
                if op == "if":
                    condition, then_body, else_body = args
                    expression(condition, bounds, definitions)
                    then_defs, else_defs = definitions.copy(), definitions.copy()
                    statements(then_body, bounds, then_defs, in_serial, {**path, id(stmt): True})
                    statements(else_body, bounds, else_defs, in_serial, {**path, id(stmt): False})
                    for name in then_defs.keys() & else_defs.keys():
                        # Branch-local aliases must be resolved before joining:
                        # their definitions need not survive after the branch.
                        def resolve(expr, local):
                            if expr.op == "var" and expr.value in local:
                                return resolve(local[expr.value], local)
                            return Expr(expr.op, tuple(resolve(x, local) for x in expr.args), expr.value)

                        left, right = resolve(then_defs[name], then_defs), resolve(else_defs[name], else_defs)
                        definitions[name] = left if left == right else Expr("phi", (left, right))
                elif op == "let":
                    expression(args[1], bounds, definitions)
                    # Assignments use fresh names in the frontend, avoiding cycles.
                    definitions[args[0]] = args[1]
                elif op == "declare":
                    expression(args[2], bounds, definitions)
                    definitions[args[0]] = Expr("mutable", value=args[1])
                elif op == "assign":
                    expression(args[1], bounds, definitions)
                elif op == "while":
                    expression(args[0], bounds, definitions)
                    if resolved_dtype(args[0], bounds, definitions, buffers) != "bool":
                        raise CompileError("While conditions require Boolean expressions")
                    if args[0] == Expr("const", value=False):
                        continue
                    statements(args[1], bounds, definitions.copy(), True, path)
                elif op == "store":
                    name, indices, value = args
                    expression(value, bounds, definitions)
                    for index in indices:
                        expression(index, bounds, definitions)
                        interval(index, bounds, definitions, buffers)
                    if buffers[name].space == "global":
                        record_write(name, indices, bounds, definitions, in_serial, path)
                elif op == "copy":
                    src, dst = args
                    if buffers[src.buffer].space == "global":
                        reads.add(src.buffer)
                    tile_names = tuple(f"_nt_copy_coordinate_{i}" for i in range(len(src.shape)))
                    tile_bounds = {**bounds, **{n: (0, size - 1) for n, size in zip(tile_names, src.shape)}}
                    for region in (src, dst):
                        indices = tuple(
                            Expr(
                                "+",
                                (
                                    origin,
                                    Expr("const", value=0)
                                    if axis is None
                                    else Expr("var", value=tile_names[axis]),
                                ),
                            )
                            for origin, axis in zip(region.origin, region.axes)
                        )
                        for index in indices:
                            expression(index, tile_bounds, definitions)
                            interval(index, tile_bounds, definitions, buffers)
                        if region is dst and buffers[dst.buffer].space == "global":
                            record_write(dst.buffer, indices, tile_bounds, definitions, in_serial, path)
                elif op == "fill":
                    expression(args[1], bounds, definitions)
                elif op in ("parallel", "serial", "unroll"):
                    names, extent, inner = args
                    inner_bounds = bounds.copy()
                    if op == "parallel":
                        inner_bounds.update({n: (0, s - 1) for n, s in zip(names, extent)})
                    elif all(type(value) is int for value in extent):
                        domain = range(*extent)
                        if not domain:
                            continue
                        inner_bounds[names[0]] = (min(domain[0], domain[-1]), max(domain[0], domain[-1]))
                    else:
                        domains = []
                        for value in extent[:2]:
                            value = Expr("const", value=value) if type(value) is int else value
                            expression(value, bounds, definitions)
                            dtype = resolved_dtype(value, bounds, definitions, buffers)
                            if not dtype.startswith(("int", "uint")):
                                raise CompileError("Dynamic loop bounds require integer expressions")
                            domains.append(interval(value, bounds, definitions, buffers))
                        start, stop = domains
                        step = extent[2]
                        low, high = (start[0], stop[1] - 1) if step > 0 else (stop[0] + 1, start[1])
                        if low > high:
                            continue
                        maximum_trips = (high - low) // abs(step) + 1
                        if maximum_trips > INT_MAX:
                            raise CompileError(
                                "Dynamic loop iteration count can exceed signed 32-bit indexing"
                            )
                        inner_bounds[names[0]] = (low, high)
                    statements(inner, inner_bounds, definitions.copy(), in_serial or op != "parallel", path)
            except CompileError as exc:
                if exc.location is not None:
                    raise
                raise CompileError(str(exc), stmt.location) from exc

    parameter_definitions = {
        p.name: Expr("parameter", value=p.dtype) for p in kernel.parameters if isinstance(p, ScalarParameter)
    }
    statements(kernel.body, initial_bounds, parameter_definitions)
    if reads & writes.keys():
        raise CompileError(
            "Global parameters cannot be both read and written in this version: "
            + ", ".join(sorted(reads & writes.keys()))
        )
    if not writes:
        raise CompileError("A kernel must write at least one global output")
