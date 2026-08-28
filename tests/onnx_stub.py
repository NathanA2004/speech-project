"""Build a tiny Silero-v4-shaped ONNX graph without the `onnx` package.

Inputs:  input [1, samples], h [2,1,64], c [2,1,64], sr int64 scalar
Outputs: output [1,1] speech probability, hn, cn

Probability is clip(rms(input) * 8, 0, 1) so silence stays near 0 and
 energetic synthetic tones/noise go high. LSTM states are Identity-passed
 through so the detector can thread h/c across frames.
"""

from __future__ import annotations

import struct
from pathlib import Path


def _varint(n: int) -> bytes:
    if n < 0:
        n &= (1 << 64) - 1
    out = bytearray()
    while n > 0x7F:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _ld(field: int, payload: bytes) -> bytes:
    return _key(field, 2) + _varint(len(payload)) + payload


def _var(field: int, n: int) -> bytes:
    return _key(field, 0) + _varint(n)


def _str(field: int, s: str) -> bytes:
    return _ld(field, s.encode("utf-8"))


def _attr_int(name: str, value: int) -> bytes:
    return _str(1, name) + _var(3, value) + _var(20, 2)


def _attr_ints(name: str, values: list[int]) -> bytes:
    body = _str(1, name)
    for v in values:
        body += _var(8, v)
    return body + _var(20, 7)


def _dim_value(n: int) -> bytes:
    return _ld(1, _var(1, n))


def _dim_param(name: str) -> bytes:
    return _ld(1, _str(2, name))


def _tensor_type(elem: int, dims: list[tuple[str, int | str]]) -> bytes:
    shape = b""
    for kind, val in dims:
        if kind == "value":
            shape += _dim_value(int(val))
        else:
            shape += _dim_param(str(val))
    tensor = _var(1, elem)
    if dims:
        tensor += _ld(2, shape)
    return _ld(2, _ld(1, tensor))


def _value_info(name: str, elem: int, dims: list[tuple[str, int | str]]) -> bytes:
    return _str(1, name) + _tensor_type(elem, dims)


def _node(op: str, inputs: list[str], outputs: list[str], name: str, attrs: list[bytes] | None = None) -> bytes:
    body = b"".join(_str(1, i) for i in inputs)
    body += b"".join(_str(2, o) for o in outputs)
    body += _str(3, name)
    body += _str(4, op)
    if attrs:
        body += b"".join(_ld(6, a) for a in attrs)
    return body


def _float_scalar(name: str, value: float) -> bytes:
    raw = struct.pack("<f", value)
    # TensorProto: data_type FLOAT=1, name, raw_data
    return _var(2, 1) + _str(8, name) + _ld(9, raw)


def write_silero_vad_stub(path: str | Path) -> Path:
    """Write a local ONNX stub that matches Silero VAD v4 I/O names."""
    FLOAT, INT64 = 1, 7
    nodes = [
        _node("Mul", ["input", "input"], ["sq"], "sq"),
        _node(
            "ReduceMean",
            ["sq"],
            ["mean"],
            "mean",
            [_attr_ints("axes", [1]), _attr_int("keepdims", 1)],
        ),
        _node("Sqrt", ["mean"], ["rms"], "rms"),
        _node("Mul", ["rms", "scale"], ["scaled"], "scale_rms"),
        _node("Clip", ["scaled", "clip_min", "clip_max"], ["output"], "clip"),
        _node("Identity", ["h"], ["hn"], "pass_h"),
        _node("Identity", ["c"], ["cn"], "pass_c"),
    ]
    initializers = [
        _float_scalar("scale", 8.0),
        _float_scalar("clip_min", 0.0),
        _float_scalar("clip_max", 1.0),
    ]
    inputs = [
        _value_info("input", FLOAT, [("value", 1), ("param", "samples")]),
        _value_info("h", FLOAT, [("value", 2), ("value", 1), ("value", 64)]),
        _value_info("c", FLOAT, [("value", 2), ("value", 1), ("value", 64)]),
        _value_info("sr", INT64, []),
    ]
    outputs = [
        _value_info("output", FLOAT, [("value", 1), ("value", 1)]),
        _value_info("hn", FLOAT, [("value", 2), ("value", 1), ("value", 64)]),
        _value_info("cn", FLOAT, [("value", 2), ("value", 1), ("value", 64)]),
    ]
    graph = _str(2, "silero_vad_energy_stub")
    graph += b"".join(_ld(1, n) for n in nodes)
    graph += b"".join(_ld(5, t) for t in initializers)
    graph += b"".join(_ld(11, i) for i in inputs)
    graph += b"".join(_ld(12, o) for o in outputs)
    # ir_version=8, opset 13 (ai.onnx default domain)
    model = _var(1, 8) + _ld(7, graph) + _ld(8, _var(2, 13))
    out = Path(path)
    out.write_bytes(model)
    return out
