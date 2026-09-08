"""Scalar type inference shared by source generation and IR evaluation."""

from .ir import DTYPES, CompileError, integer_limits

BITWISE_OPS = frozenset(("&", "|", "^", "invert", "<<", ">>"))
INTEGER_DIVISION_OPS = frozenset(("//", "%", "truncdiv", "truncmod", "ceildiv"))
CHOICE_OPS = frozenset(("select", "if_then_else"))
ROUNDING_OPS = frozenset(("floor", "ceil", "trunc", "round", "round_away", "nearbyint"))
CLASSIFICATION_OPS = frozenset(("isnan", "isinf", "isfinite"))
TRANSCENDENTAL_OPS = frozenset(
    (
        "exp",
        "exp2",
        "exp10",
        "log",
        "log2",
        "log10",
        "log1p",
        "sqrt",
        "rsqrt",
        "erf",
        "sigmoid",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
    )
)
UNARY_MATH_OPS = ROUNDING_OPS | CLASSIFICATION_OPS | TRANSCENDENTAL_OPS | {"abs"}
BINARY_MATH_OPS = frozenset(("pow", "fmod", "atan2", "copysign", "hypot", "nextafter", "ldexp"))
FAST_MATH_OPS = frozenset(
    ("__exp", "__exp10", "__log", "__log2", "__log10", "__sin", "__cos", "__tan", "fast_rcp")
)
IEEE_MATH_OPS = {
    "ieee_add": ("add", 2),
    "ieee_sub": ("sub", 2),
    "ieee_mul": ("mul", 2),
    "ieee_fmaf": ("fma", 3),
    "ieee_frcp": ("rcp", 1),
    "ieee_fsqrt": ("sqrt", 1),
    "ieee_frsqrt": ("rsqrt", 1),
    "ieee_fdiv": ("div", 2),
    "fma": ("fma", 3),
    "fmul": ("mul", 2),
}
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
    if expr.op in BINARY_MATH_OPS - {"pow"}:
        return types[0]
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
    if expr.op in ("cast", "mutable", "parameter"):
        return expr.value
    if expr.op == "pow_integer":
        return expression_dtype(expr.args[0], buffers, variables)
    if expr.op in FAST_MATH_OPS:
        dtype = expression_dtype(expr.args[0], buffers, variables)
        if expr.op == "fast_rcp" and dtype != "float32":
            raise CompileError("T.fast_rcp requires a scalar float32 input in the pinned CUDA lowering")
        if dtype not in ("float16", "bfloat16", "float32", "float64"):
            raise CompileError(f"T.{expr.op} requires floating inputs")
        return dtype
    if expr.op in IEEE_MATH_OPS:
        types = [expression_dtype(arg, buffers, variables) for arg in expr.args]
        dtype = types[0]
        if dtype not in ("float16", "bfloat16", "float32", "float64"):
            raise CompileError(f"T.{expr.op} requires a floating first operand")
        if expr.op in ("fma", "fmul") and any(typ != dtype for typ in types):
            raise CompileError(f"T.{expr.op} requires identical floating operand dtypes")
        if dtype in ("float16", "bfloat16") and expr.value != "rn":
            raise CompileError(f"T.{expr.op} supports only rn rounding for {dtype}")
        if expr.op == "ieee_frsqrt" and dtype == "float64":
            raise CompileError("T.ieee_frsqrt does not support float64 in the pinned CUDA lowering")
        return dtype
    if expr.op in BINARY_MATH_OPS:
        dtype = operand_dtype(expr, buffers, variables)
        if expr.op in ("hypot", "nextafter", "ldexp") and dtype not in ("float32", "float64"):
            raise CompileError(
                f"T.{expr.op} requires a float32 or float64 first operand in the pinned CUDA lowering"
            )
        if not dtype.startswith("float") and not (dtype == "bfloat16" and expr.op != "pow"):
            raise CompileError(f"T.{expr.op} requires a floating result dtype")
        return dtype
    if expr.op in UNARY_MATH_OPS:
        dtype = expression_dtype(expr.args[0], buffers, variables)
        if expr.op in TRANSCENDENTAL_OPS:
            if expr.op == "exp" and dtype.startswith(("int", "uint")):
                return "float32"
            if not dtype.startswith(("float", "bfloat")):
                raise CompileError(f"T.{expr.op} requires floating inputs in the CUDA lowering")
            return dtype
        if expr.op in CLASSIFICATION_OPS:
            if dtype == "bfloat16":
                raise CompileError("The pinned TIR classification operators do not accept bfloat16")
            return "bool"
        return dtype
    if expr.op in CHOICE_OPS:
        condition, true_value, false_value = expr.args
        if expression_dtype(condition, buffers, variables) != "bool":
            raise CompileError("Conditional expressions require a Boolean condition")
        left = expression_dtype(true_value, buffers, variables)
        right = expression_dtype(false_value, buffers, variables)
        if expr.op == "select" and left != right:
            raise CompileError("T.Select requires identical true and false value dtypes")
        return promote(left, right)
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


def constant_integer(expr, bindings):
    """Resolve integer literals and immutable aliases without reading runtime state."""

    def resolve(value):
        if value.op == "var":
            return resolve(bindings[value.value]) if value.value in bindings else None
        if value.op == "const" and type(value.value) in (int, bool):
            return int(value.value), expression_dtype(value, {}, {})
        if value.op == "cast" and value.value.startswith(("int", "uint")):
            item = resolve(value.args[0])
            if item is None:
                return None
            result, dtype = item[0], value.value
        elif value.op in ("+", "-", "*"):
            parts = [resolve(arg) for arg in value.args]
            if any(part is None for part in parts):
                return None
            (left, lt), (right, rt) = parts
            dtype = promote(lt, rt)
            if dtype == "bool":
                return None
            result = left + right if value.op == "+" else left - right if value.op == "-" else left * right
        else:
            return None
        low, high = integer_limits(dtype)
        return (result - low) % (high - low + 1) + low, dtype

    result = resolve(expr)
    return None if result is None else result[0]


def body_types(body, types, buffers):
    types = types.copy()
    for stmt in body:
        if stmt.op == "declare":
            types[stmt.args[0]] = stmt.args[1]
        elif stmt.op == "let":
            types[stmt.args[0]] = expression_dtype(stmt.args[1], buffers, types)
        elif stmt.op == "if":
            then_types = body_types(stmt.args[1], types, buffers)
            else_types = body_types(stmt.args[2], types, buffers)
            for name in then_types.keys() & else_types.keys():
                types[name] = promote(then_types[name], else_types[name])
    return types
