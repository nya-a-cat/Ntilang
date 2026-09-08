"""Exact scalar arithmetic oracle for explicit floating-point rounding.

Finite arithmetic uses rational integers and rounds once to the destination
format. Square roots use integer comparisons against rounding midpoints.
Low-precision CUDA reciprocal, root, and division instructions are approximate;
this oracle supplies ideal mathematical values for those operations.
"""

import math
from fractions import Fraction

FORMATS = {
    "float16": (11, -14, 15),
    "bfloat16": (8, -126, 127),
    "float32": (24, -126, 127),
    "float64": (53, -1022, 1023),
}


def _negative(value):
    return math.copysign(1.0, value) < 0


def _log2(numerator, denominator):
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        return exponent - (numerator < denominator << exponent)
    return exponent - (numerator << -exponent < denominator)


def _scale(numerator, denominator, exponent):
    return (numerator, denominator << exponent) if exponent >= 0 else (numerator << -exponent, denominator)


def _rounded(numerator, denominator, negative, mode, root):
    if root:
        quotient = math.isqrt(numerator // denominator)
        remainder = numerator - quotient * quotient * denominator
        midpoint = 4 * numerator - (2 * quotient + 1) ** 2 * denominator
    else:
        quotient, remainder = divmod(numerator, denominator)
        midpoint = 2 * remainder - denominator
    if mode == "rn":
        increment = midpoint > 0 or (midpoint == 0 and quotient % 2 == 1)
    else:
        increment = remainder != 0 and ((mode == "ru" and not negative) or (mode == "rd" and negative))
    return quotient + increment


def round_exact(value, dtype, mode="rn", *, negative_zero=False, root=False):
    """Round a Fraction (or its nonnegative square root) without double rounding."""
    precision, minimum, maximum = FORMATS[dtype]
    if mode not in ("rn", "rz", "ru", "rd"):
        raise ValueError("Unknown floating-point rounding mode")
    if not value:
        return -0.0 if negative_zero else 0.0
    negative = value < 0
    if root and negative:
        return math.nan
    numerator, denominator = abs(value.numerator), value.denominator
    exponent = _log2(numerator, denominator)
    if root:
        exponent //= 2
    unit = max(exponent, minimum) - precision + 1
    n, d = _scale(numerator, denominator, unit * (2 if root else 1))
    significand = _rounded(n, d, negative, mode, root)
    if significand and significand.bit_length() - 1 + unit > maximum:
        infinity = mode == "rn" or (mode == "ru" and not negative) or (mode == "rd" and negative)
        result = math.inf if infinity else math.ldexp((1 << precision) - 1, maximum - precision + 1)
    else:
        result = math.ldexp(significand, unit)
    return -result if negative else result


def evaluate(operation, values, dtype, mode="rn"):
    """Evaluate already destination-typed operands with one final rounding."""
    values = tuple(map(float, values))
    if any(math.isnan(value) for value in values):
        return math.nan
    x = values[0]
    if operation in ("sqrt", "rsqrt"):
        if x < 0:
            return math.nan
        if x == 0:
            return x if operation == "sqrt" else math.copysign(math.inf, x)
        if math.isinf(x):
            return x if operation == "sqrt" else 0.0
        exact = Fraction(x)
        return round_exact(exact if operation == "sqrt" else 1 / exact, dtype, mode, root=True)
    if operation == "rcp":
        return evaluate("div", (1.0, x), dtype, mode)
    y = values[1]
    if operation == "sub":
        y = -y
        operation = "add"
    if operation == "add":
        if math.isinf(x) or math.isinf(y):
            return x + y
        negative_zero = _negative(x) if _negative(x) == _negative(y) else mode == "rd"
        return round_exact(Fraction(x) + Fraction(y), dtype, mode, negative_zero=negative_zero)
    negative = _negative(x) != _negative(y)
    sign = -1.0 if negative else 1.0
    if operation == "div":
        if (x == 0 and y == 0) or (math.isinf(x) and math.isinf(y)):
            return math.nan
        if math.isinf(x) or y == 0:
            return math.copysign(math.inf, sign)
        if x == 0 or math.isinf(y):
            return math.copysign(0.0, sign)
        return round_exact(Fraction(x) / Fraction(y), dtype, mode)
    if (x == 0 and math.isinf(y)) or (math.isinf(x) and y == 0):
        return math.nan
    if operation == "mul":
        if math.isinf(x) or math.isinf(y):
            return math.copysign(math.inf, sign)
        return round_exact(Fraction(x) * Fraction(y), dtype, mode, negative_zero=negative)
    if operation != "fma":
        raise ValueError(f"Unknown floating operation {operation}")
    z = values[2]
    if math.isinf(x) or math.isinf(y):
        return math.copysign(math.inf, sign) + z
    if math.isinf(z):
        return z
    negative_zero = negative if negative == _negative(z) else mode == "rd"
    return round_exact(Fraction(x) * Fraction(y) + Fraction(z), dtype, mode, negative_zero=negative_zero)
