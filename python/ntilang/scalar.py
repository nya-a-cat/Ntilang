"""Scalar type inference shared by source generation and IR evaluation."""

from .ir import DTYPES, CompileError, integer_limits

BITWISE_OPS = frozenset(("&", "|", "^", "invert", "<<", ">>"))
INTEGER_DIVISION_OPS = frozenset(("//", "%", "truncdiv", "truncmod", "ceildiv"))
BINARY_NUMERIC_OPS = (
    frozenset(("+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!=", "maximum", "minimum", "max", "min"))
    | INTEGER_DIVISION_OPS
)


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
        return left if left_float else right
    if left.startswith("uint") != right.startswith("uint"):
        unsigned, signed = (left, right) if left.startswith("uint") else (right, left)
        return unsigned if DTYPES[unsigned] >= DTYPES[signed] else signed
    return left if lw >= rw else right


def operand_dtype(expr, buffers, variables):
    """Match TIR numeric operands, including context-typed bitwise literals."""
    types = [expression_dtype(arg, buffers, variables) for arg in expr.args]
    if expr.op in BITWISE_OPS and len(types) == 2:
        # The upstream bit-operation FFI creates a Python integer literal in
        # the other operand's dtype before BinaryOpMatchTypes runs.
        literals = [arg.op == "const" and type(arg.value) is int for arg in expr.args]
        if literals[0] != literals[1]:
            index = 0 if literals[0] else 1
            target = types[1 - index]
            if target.startswith(("int", "uint")) or target == "bool":
                low, high = integer_limits(target)
                if not low <= expr.args[index].value <= high:
                    raise CompileError(f"Bitwise integer literal must fit the other operand's {target} dtype")
                types[index] = target
    result = types[0]
    for dtype in types[1:]:
        result = promote(result, dtype)
    return result


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
    if expr.op in ("and", "or", "not"):
        if any(expression_dtype(arg, buffers, variables) != "bool" for arg in expr.args):
            raise CompileError("Logical operations require Boolean operands")
        return "bool"
    if expr.op in BITWISE_OPS:
        types = [expression_dtype(arg, buffers, variables) for arg in expr.args]
        if any(not (dtype.startswith(("int", "uint")) or dtype == "bool") for dtype in types):
            raise CompileError("Bitwise operations require integer or Boolean operands")
        if expr.op in ("<<", ">>") and "bool" in types:
            raise CompileError("Shift operations require integer operands, excluding Boolean")
    if expr.op in INTEGER_DIVISION_OPS:
        if any(
            not expression_dtype(arg, buffers, variables).startswith(("int", "uint")) for arg in expr.args
        ):
            raise CompileError("Integer division and remainder require integer operands, excluding Boolean")
    result = operand_dtype(expr, buffers, variables)
    if expr.op in ("<", "<=", ">", ">=", "==", "!="):
        return "bool"
    if expr.op == "/" and (result.startswith(("int", "uint")) or result == "bool"):
        raise CompileError(
            "Integer '/' is ambiguous; use explicit integer division or cast to a floating dtype"
        )
    return result


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
