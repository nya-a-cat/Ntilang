"""Conservative checks for the initial, statically shaped language subset."""

from __future__ import annotations

from .ir import CompileError, Expr, Kernel, integer_limits

INT_MIN, INT_MAX = -(2**31), 2**31 - 1


def interval(expr, bounds, definitions):
    """Bound every intermediate operation in an integer index expression."""
    if expr.op == "const" and type(expr.value) is int:
        result = (expr.value, expr.value)
    elif expr.op == "phi":
        alternatives = [interval(x, bounds, definitions) for x in expr.args]
        result = (min(x[0] for x in alternatives), max(x[1] for x in alternatives))
    elif expr.op == "var":
        if expr.value in definitions:
            return interval(definitions[expr.value], bounds, definitions)
        if expr.value not in bounds:
            raise CompileError("Indices must use integer block/loop variables and static constants")
        result = bounds[expr.value]
    elif expr.op == "cast" and (expr.value.startswith(("int", "uint")) or expr.value == "bool"):
        result = interval(expr.args[0], bounds, definitions)
        low, high = integer_limits(expr.value)
        if result[0] < low or result[1] > high:
            raise CompileError("An integer index cast can change the represented value")
    elif expr.op in ("neg", "pos"):
        low, high = interval(expr.args[0], bounds, definitions)
        result = (-high, -low) if expr.op == "neg" else (low, high)
    elif expr.op in ("+", "-", "*", "//", "%"):
        a, b = (interval(x, bounds, definitions) for x in expr.args)
        if expr.op == "+":
            result = (a[0] + b[0], a[1] + b[1])
        elif expr.op == "-":
            result = (a[0] - b[1], a[1] - b[0])
        elif expr.op == "*":
            products = [x * y for x in a for y in b]
            result = (min(products), max(products))
        elif b[0] != b[1] or b[0] <= 0:
            raise CompileError("Index division/remainder requires a positive static divisor")
        elif expr.op == "//":
            # CuTe integer division truncates toward zero; reject negative cases.
            if a[0] < 0:
                raise CompileError("Index division requires a nonnegative dividend")
            result = (a[0] // b[0], a[1] // b[0])
        else:
            if a[0] < 0:
                raise CompileError("Index remainder requires a nonnegative dividend")
            result = (0, b[0] - 1)
    else:
        raise CompileError("Data-dependent or non-integer indexing is not supported")
    if result[0] < INT_MIN or result[1] > INT_MAX:
        raise CompileError("An index expression can overflow signed 32-bit arithmetic")
    return result


def affine(expr, definitions):
    """Return constant and integer coefficients, or reject a non-affine expression."""
    if expr.op == "const" and type(expr.value) is int:
        return expr.value, {}
    if expr.op == "phi":
        alternatives = [affine(x, definitions) for x in expr.args]
        if all(x == alternatives[0] for x in alternatives):
            return alternatives[0]
        raise CompileError("Global writes require branch-independent affine ownership")
    if expr.op == "var":
        if expr.value in definitions:
            return affine(definitions[expr.value], definitions)
        return 0, {expr.value: 1}
    if expr.op == "cast" and (expr.value.startswith(("int", "uint")) or expr.value == "bool"):
        return affine(expr.args[0], definitions)
    if expr.op in ("neg", "pos"):
        const, coeff = affine(expr.args[0], definitions)
        sign = -1 if expr.op == "neg" else 1
        return sign * const, {n: sign * v for n, v in coeff.items()}
    if expr.op in ("+", "-", "*"):
        (ac, av), (bc, bv) = (affine(x, definitions) for x in expr.args)
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


def check_ownership(indices, bounds, definitions):
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
        interval(index, bounds, definitions)
        _, coefficients = affine(index, definitions)
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

    def expression(expr, bounds, definitions):
        if expr.op in ("//", "%"):
            interval(expr, bounds, definitions)
        if expr.op == "load":
            if buffers[expr.value].space == "global":
                reads.add(expr.value)
            for index in expr.args:
                interval(index, bounds, definitions)
        for arg in expr.args:
            expression(arg, bounds, definitions)

    def record_write(name, indices, bounds, definitions, in_serial, path):
        if in_serial:
            raise CompileError("Write global outputs after serial tile accumulation loops")
        check_ownership(indices, bounds, definitions)
        mapping = (tuple(affine(x, definitions) for x in indices), bounds)
        for previous_path, previous_mapping in writes.get(name, []):
            exclusive = any(
                key in previous_path and previous_path[key] != value for key, value in path.items()
            )
            if not exclusive or mapping != previous_mapping:
                raise CompileError(
                    f"Output {name} has multiple write sites without exclusive branches and identical ownership"
                )
        writes.setdefault(name, []).append((path.copy(), mapping))

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
                elif op == "store":
                    name, indices, value = args
                    expression(value, bounds, definitions)
                    for index in indices:
                        interval(index, bounds, definitions)
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
                            Expr("+", (origin, Expr("var", value=n)))
                            for origin, n in zip(region.origin, tile_names)
                        )
                        for index in indices:
                            interval(index, tile_bounds, definitions)
                        if region is dst and buffers[dst.buffer].space == "global":
                            record_write(dst.buffer, indices, tile_bounds, definitions, in_serial, path)
                elif op == "fill":
                    expression(args[1], bounds, definitions)
                elif op in ("parallel", "serial", "unroll"):
                    names, extent, inner = args
                    inner_bounds = bounds.copy()
                    if op == "parallel":
                        inner_bounds.update({n: (0, s - 1) for n, s in zip(names, extent)})
                    else:
                        domain = range(*extent)
                        if not domain:
                            continue
                        inner_bounds[names[0]] = (min(domain[0], domain[-1]), max(domain[0], domain[-1]))
                    statements(inner, inner_bounds, definitions.copy(), in_serial or op != "parallel", path)
            except CompileError as exc:
                if exc.location is not None:
                    raise
                raise CompileError(str(exc), stmt.location) from exc

    statements(kernel.body, initial_bounds, {})
    if reads & writes.keys():
        raise CompileError(
            "Global parameters cannot be both read and written in this version: "
            + ", ".join(sorted(reads & writes.keys()))
        )
    if not writes:
        raise CompileError("A kernel must write at least one global output")
