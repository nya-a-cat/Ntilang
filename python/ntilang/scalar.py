"""Scalar type inference shared by source generation and IR evaluation."""

from .ir import DTYPES, CompileError

BITWISE_OPS = frozenset(("&", "|", "^", "invert", "<<", ">>"))


def promote(left, right):
    if left == right:
        return left
    if left == "bool":
        return right
    if right == "bool":
        return left
    left_float = left.startswith(("float", "bfloat"))
    right_float = right.startswith(("float", "bfloat"))
    lw, rw = DTYPES[left] * 8, DTYPES[right] * 8
    if left_float and right_float:
        return f"float{max(lw, rw)}"
    if left_float or right_float:
        float_type, fw, iw = (left, lw, rw) if left_float else (right, rw, lw)
        return float_type if fw > iw else f"float{iw}"
    if left.startswith("uint") != right.startswith("uint"):
        unsigned, signed = (left, right) if left.startswith("uint") else (right, left)
        return unsigned if DTYPES[unsigned] >= DTYPES[signed] else signed
    return left if lw >= rw else right


def expression_dtype(expr, buffers, variables):
    if expr.op == "const":
        if type(expr.value) is int and not -(2**31) <= expr.value < 2**31:
            if not -(2**63) <= expr.value < 2**63:
                raise CompileError("Bare scalar integer constants must fit signed 64-bit arithmetic")
            return "int64"
        return {bool: "bool", int: "int32", float: "float32"}[type(expr.value)]
    if expr.op == "var":
        return variables[expr.value]
    if expr.op == "load":
        return buffers[expr.value].type.dtype
    if expr.op == "cast":
        return expr.value
    if expr.op in ("<", "<=", ">", ">=", "==", "!=", "and", "or", "not"):
        return "bool"
    if expr.op in BITWISE_OPS:
        types = [expression_dtype(arg, buffers, variables) for arg in expr.args]
        if any(not (dtype.startswith(("int", "uint")) or dtype == "bool") for dtype in types):
            raise CompileError("Bitwise operations require integer or Boolean operands")
        if expr.op in ("<<", ">>") and "bool" in types:
            raise CompileError("Shift operations require integer operands, excluding Boolean")
    result = expression_dtype(expr.args[0], buffers, variables)
    for arg in expr.args[1:]:
        result = promote(result, expression_dtype(arg, buffers, variables))
    return (
        "float32" if expr.op == "/" and (result.startswith(("int", "uint")) or result == "bool") else result
    )


def body_types(body, types, buffers):
    types = types.copy()
    for stmt in body:
        if stmt.op == "let":
            types[stmt.args[0]] = expression_dtype(stmt.args[1], buffers, types)
        elif stmt.op == "if":
            then_types = body_types(stmt.args[1], types, buffers)
            else_types = body_types(stmt.args[2], types, buffers)
            for name in then_types.keys() & else_types.keys():
                types[name] = promote(then_types[name], else_types[name])
    return types
