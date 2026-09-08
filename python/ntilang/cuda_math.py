"""Scalar PTX plans for the CUDA half/bfloat16 math intrinsic contracts.

Instruction choices follow CUDA 12.9's device headers. This module emits source;
it imports neither CUDA nor CuTe. Approximate instructions remain approximate.
"""


def low_precision_asm(operation, dtype, target):
    """Return a scoped PTX expression with a b16 output and b16 inputs."""
    suffix = "f16" if dtype == "float16" else "bf16"
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
