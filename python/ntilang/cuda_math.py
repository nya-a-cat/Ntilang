"""Scalar PTX plans for the CUDA half/bfloat16 math intrinsic contracts.

Instruction choices follow CUDA 12.9's device headers. This module emits source;
it imports neither CUDA nor CuTe. Approximate instructions remain approximate.
"""

import struct


def low_precision_asm(operation, dtype, target):
    """Return a scoped PTX expression with a b16 output and b16 inputs."""
    if operation.startswith("unary_"):
        operation = operation[6:]
        if operation in ("sin", "cos"):
            return _half_trigonometric_asm(operation)
        return _native_unary_asm(operation, dtype)
    suffix = "f16" if dtype == "float16" else "bf16"
    if operation == "tanh" and suffix == "f16":
        return "{ tanh.approx.f16 $0, $1; }"
    if operation in ("add", "sub", "mul", "fma"):
        if suffix == "bf16" and int(target[3:].rstrip("a")) < 90 and operation != "fma":
            # SM80/86/87/89 implement scalar bfloat16 arithmetic with native FMA.
            expressions = {
                "add": "$1, one, $2",
                "sub": "$2, minus_one, $1",
                "mul": "$1, $2, negative_zero",
            }
            return (
                "{ .reg .b16 one, minus_one, negative_zero; "
                "mov.b16 one, 0x3f80; mov.b16 minus_one, 0xbf80; mov.b16 negative_zero, 0x8000; "
                f"fma.rn.bf16 $0, {expressions[operation]}; }}"
            )
        operands = "$1, $2, $3" if operation == "fma" else "$1, $2"
        return f"{{ {operation}.rn.{suffix} $0, {operands}; }}"
    lines = ["{", ".reg .f32 a, result;", f"cvt.f32.{suffix} a, $1;"]
    if operation != "div":
        modifier = ".ftz" if suffix == "f16" else ""
        lines += [f"{operation}.approx{modifier}.f32 result, a;"]
    elif suffix == "f16":
        # CUDA half division refines small, nonzero rounded quotients using
        # two fused operations. The threshold is the positive half bit pattern 143.
        lines += [
            ".reg .f32 b, reciprocal, minus_b, error;",
            ".reg .b16 rounded, magnitude;",
            ".reg .pred nonzero, small, refine;",
            "cvt.f32.f16 b, $2;",
            "rcp.approx.ftz.f32 reciprocal, b;",
            "mul.rn.f32 result, reciprocal, a;",
            "cvt.rn.f16.f32 rounded, result;",
            "and.b16 magnitude, rounded, 0x7fff;",
            "setp.gt.u16 nonzero, magnitude, 0;",
            "setp.lt.u16 small, magnitude, 143;",
            "and.pred refine, nonzero, small;",
            "neg.f32 minus_b, b;",
            "@refine fma.rn.f32 error, minus_b, result, a;",
            "@refine fma.rn.f32 result, reciprocal, error, result;",
        ]
    else:
        # Scaling a large denominator keeps the approximate division in range;
        # the final FMA retains subnormals and the sign of a zero result.
        lines += [
            ".reg .f32 b, magnitude;",
            ".reg .pred scale;",
            "cvt.f32.bf16 b, $2;",
            "abs.f32 magnitude, b;",
            "setp.ge.f32 scale, magnitude, 0f7e800000;",
            "@scale mul.rn.f32 b, b, 0f3e800000;",
            "div.approx.f32 result, a, b;",
            "@scale fma.rn.f32 result, result, 0f3e800000, 0f80000000;",
        ]
    lines += [f"cvt.rn.{suffix}.f32 $0, result;", "}"]
    return "\n".join(lines)


def _native_unary_asm(operation, dtype):
    """CUDA half/BF16 logarithms and exponentials, including correction points."""
    suffix = "f16" if dtype == "float16" else "bf16"
    ftz = ".ftz" if suffix == "f16" else ""
    lines = ["{", ".reg .f32 value;", ".reg .b16 rounded;", f"cvt.f32.{suffix} value, $1;"]
    if operation in ("exp", "exp2", "exp10"):
        if operation != "exp2":
            coefficient = (
                ("0f3fb8aa3b" if suffix == "f16" else "0f3fb8aa3c") if operation == "exp" else "0f40549a78"
            )
            if suffix == "f16":
                lines.append(f"fma.rn.f32 value, value, {coefficient}, 0f80000000;")
            else:
                lines.append(f"mul.rn.f32 value, value, {coefficient};")
        lines.append(f"ex2.approx{ftz}.f32 value, value;")
        if operation == "exp2" and suffix == "f16":
            lines.append("fma.rn.f32 value, value, 0f33800000, value;")
    else:
        lines.append(f"lg2.approx{ftz}.f32 value, value;")
        if operation in ("log", "log10"):
            coefficient = "0f3f317218" if operation == "log" else "0f3e9a209b"
            lines.append(f"mul.rn.f32 value, value, {coefficient};")
    lines.append(f"cvt.rn.{suffix}.f32 rounded, value;")
    if suffix == "f16":
        # Input/result bit patterns and representable corrections from CUDA
        # 12.9. Comparisons use only finite, nonzero patterns.
        corrections = {
            "exp": ((0x1F79, 0x9400), (0x25CF, 0x9400), (0xC13B, 0x0400), (0xC1EF, 0x0200)),
            "exp10": (
                (0x34DE, 0x9800),
                (0x9766, 0x9000),
                (0x9972, 0x1000),
                (0xA5C4, 0x1000),
                (0xBF0A, 0x8100),
            ),
            "log2": ((0xA2E2, 0x8080), (0xBF46, 0x9400)),
            "log": ((0x160D, 0x9C00), (0x3BFE, 0x8010), (0x3C0B, 0x8080), (0x6051, 0x1C00)),
            "log10": ((0x338F, 0x1000), (0x33F8, 0x9000), (0x57E1, 0x9800), (0x719D, 0x9C00)),
        }.get(operation, ())
        key = "rounded" if operation == "log2" else "$1"
        lines += _half_corrections(key, corrections)
    elif operation == "exp10":
        lines += [".reg .pred patch;", "setp.eq.u16 patch, $1, 0xbc95;", "@patch mov.b16 rounded, 0x3f75;"]
    lines += ["mov.b16 $0, rounded;", "}"]
    return "\n".join(lines)


def _half_corrections(key, corrections):
    if not corrections:
        return []
    lines = [".reg .pred patch;", ".reg .b16 correction;"]
    for pattern, delta in corrections:
        lines += [
            f"setp.eq.u16 patch, {key}, {pattern};",
            f"mov.b16 correction, {delta};",
            "@patch add.rn.f16 rounded, rounded, correction;",
        ]
    return lines


def _f32_literal(value):
    return "0f" + struct.pack(">f", value).hex()


def _half_trigonometric_asm(operation):
    """Emit the CUDA 12.9 half-range reduction and sine/cosine polynomial."""
    lines = [
        "{",
        ".reg .f32 input_value, reduced, rounded_index, index_value, square, result;",
        ".reg .f32 c8, c6, c4, c2, linear, constant;",
        ".reg .u32 quadrant, bit;",
        ".reg .b16 rounded, input_magnitude, sign;",
        ".reg .pred cosine, negative;",
        "cvt.f32.f16 input_value, $1;",
        f"fma.rn.f32 rounded_index, input_value, {_f32_literal(0.636619772)}, {_f32_literal(12582912)};",
        "mov.b32 quadrant, rounded_index;",
        f"sub.rn.f32 index_value, rounded_index, {_f32_literal(12582912)};",
        f"fma.rn.f32 reduced, index_value, {_f32_literal(-1.5707962512969971)}, input_value;",
        f"fma.rn.f32 reduced, index_value, {_f32_literal(-7.5497894158615964e-8)}, reduced;",
    ]
    if operation == "cos":
        lines += ["and.b32 quadrant, quadrant, 3;", "add.u32 quadrant, quadrant, 1;"]
    lines += [
        "and.b32 bit, quadrant, 1;",
        "setp.ne.u32 cosine, bit, 0;",
        "mul.rn.f32 square, reduced, reduced;",
    ]
    coefficients = (
        ("c8", 2.44331571e-5, -1.95152959e-4),
        ("c6", -1.38873163e-3, 8.33216087e-3),
        ("c4", 4.16666457e-2, -1.66666546e-1),
        ("c2", -0.5, 0),
    )
    for name, cosine_value, sine_value in coefficients:
        lines.append(f"selp.f32 {name}, {_f32_literal(cosine_value)}, {_f32_literal(sine_value)}, cosine;")
    lines += [
        "selp.f32 linear, square, reduced, cosine;",
        "selp.f32 constant, 0f3f800000, reduced, cosine;",
        "fma.rn.f32 result, c8, square, c6;",
        "fma.rn.f32 result, result, square, c4;",
        "fma.rn.f32 result, result, square, c2;",
        "fma.rn.f32 result, result, linear, constant;",
        "and.b32 bit, quadrant, 2;",
        "setp.ne.u32 negative, bit, 0;",
        "@negative neg.f32 result, result;",
        "cvt.rn.f16.f32 rounded, result;",
        "and.b16 input_magnitude, $1, 0x7fff;",
    ]
    if operation == "sin":
        lines += ["and.b16 sign, rounded, 0x8000;", "and.b16 rounded, rounded, 0x7fff;"]
        lines += _half_corrections("input_magnitude", ((0x32B3, 0x0800), (0x5CB0, 0x9000)))
        lines += ["or.b16 rounded, rounded, sign;"]
    else:
        lines += _half_corrections("input_magnitude", ((0x2B7C, 0x1000),))
    lines += ["mov.b16 $0, rounded;", "}"]
    return "\n".join(lines)
