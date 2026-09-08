"""Host scalar validation without a CUDA or array-library dependency."""

from numbers import Integral, Real

from .ir import ScalarParameter, integer_limits


def normalize_scalar(value, parameter: ScalarParameter):
    """Preserve logical values while checking the declared kernel input domain."""
    name, dtype = parameter.name, parameter.dtype
    if dtype == "bool":
        if type(value) is not bool:
            raise TypeError(f"{name} requires a Python bool")
        return value
    if dtype.startswith(("int", "uint")):
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(f"{name} requires an integer scalar for {dtype}")
        value = int(value)
        low, high = integer_limits(dtype)
        if not low <= value <= high:
            raise ValueError(f"{name}: {value} is outside the {dtype} range [{low}, {high}]")
        return value
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} requires a real scalar for {dtype}")
    return float(value)


def scalar_ffi_argument(value, parameter: ScalarParameter):
    """TVM FFI carries integer scalars in a signed 64-bit payload."""
    value = normalize_scalar(value, parameter)
    if parameter.dtype == "uint64" and value >= 2**63:
        return value - 2**64
    return value
