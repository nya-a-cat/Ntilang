"""CUDA diagnostic formatting shared by source generation and the reference."""

# Names and value conversions follow the pinned CUDA debug.h PrintTraits.
# The 64-bit integer names use the Linux CUDA compiler's LP64 typedefs.
PRINT_TYPES = {
    "bool": ("bool", None),
    "int8": ("signed char", "%d"),
    "uint8": ("unsigned char", "%u"),
    "int16": ("short", "%d"),
    "uint16": ("unsigned short", "%u"),
    "int32": ("int", "%d"),
    "uint32": ("uint", "%u"),
    "int64": ("long", "%lld"),
    "uint64": ("ulong", "%llu"),
    "float16": ("half_t", "%f"),
    "bfloat16": ("bfloat16_t", "%f"),
    "float32": ("float", "%f"),
    "float64": ("double", "%f"),
}


def print_format(dtype=None, buffer=None, *, boolean=None):
    result = "msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, 0, 0)"
    if dtype is not None:
        label, specifier = PRINT_TYPES[dtype]
        if buffer is not None:
            result += ": buffer=%s, index=%d, "
            if dtype == "uint16":
                label = "uint16_t"
        else:
            result += ": "
        if dtype == "bool":
            specifier = "true" if boolean else "false"
        result += f"dtype={label} value={specifier}"
    return result + "\n"


def reference_print(message, block, thread, value=None, dtype=None, buffer=None, index=None):
    """Render one diagnostic event without imposing device-wide output ordering."""
    message = message.split("\0", 1)[0]
    result = f"msg='{message}' BlockIdx=({', '.join(map(str, block))}), ThreadIdx=({thread}, 0, 0)"
    if dtype is not None:
        label = PRINT_TYPES[dtype][0]
        if buffer is None:
            result += ": "
        else:
            buffer = buffer.split("\0", 1)[0]
            result += f": buffer={buffer}, index={index}, "
            if dtype == "uint16":
                label = "uint16_t"
        text = (
            ("true" if value else "false")
            if dtype == "bool"
            else f"{float(value):f}"
            if dtype.startswith(("float", "bfloat"))
            else str(int(value))
        )
        result += f"dtype={label} value={text}"
    print(result)
