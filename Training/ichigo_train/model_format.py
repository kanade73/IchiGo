""".ichigo model directory v1 (docs/spec/01-network.md §6).

    model.ichigo/
      manifest.json   JSON, no NaN/Infinity
      wiring.i32      int32 little-endian [L, C, 2, 4]  (bank, channel, dx, dy)
      gates.u8        uint8 [L, C]  gate ids 0..15 (truth-table encoding, row = 2a+b)
      heads.f32       float32 little-endian, 14 tensors at the byte offsets listed in the manifest

Every check that the Swift loader performs (file names, sizes, sha256, ranges, overlaps,
alignment, trailing bytes) is mirrored in ``read_model`` so Python and Swift reject the same
inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from .model import HEAD_TENSOR_NAMES, HEAD_VERSION, SUPPORTED_HEAD_VERSIONS, head_shapes
from .wiring import INPUT_CHANNELS, validate_wiring

FORMAT = "ichigo.logic"
VERSION = 1
FEATURE_VERSION = 1
RULES_ID = "cgos-area-psk-v1"
GATE_ENCODING = "truth-table-lsb-2a-plus-b"
FILE_NAMES = ("wiring.i32", "gates.u8", "heads.f32")
MIN_CHANNELS, MAX_CHANNELS = 16, 4096
MIN_LAYERS, MAX_LAYERS = 1, 64
MIN_DILATION, MAX_DILATION = 1, 19
MAX_PAYLOAD_BYTES = 512 * 1024 * 1024
SUPPORTED_BOARD_SIZES = (9, 19)


class ModelFormatError(ValueError):
    pass


@dataclass
class LoadedModel:
    manifest: dict
    wiring: np.ndarray  # int32 [L,C,2,4]
    gates: np.ndarray  # uint8 [L,C]
    heads: dict[str, np.ndarray]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_nonfinite(obj, path="manifest"):
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ModelFormatError(f"{path}: non-finite number")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _reject_nonfinite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _reject_nonfinite(v, f"{path}[{i}]")


def build_manifest(
    *,
    board_sizes: list[int],
    channels: int,
    dilations: list[int],
    wiring_bytes: bytes,
    gates_bytes: bytes,
    heads_bytes: bytes,
    head_tensors: list[dict],
    training_provenance: dict,
    calibration_temperature: float = 1.0,
    head_version: int = HEAD_VERSION,
) -> dict:
    return {
        "format": FORMAT,
        "version": VERSION,
        "featureVersion": FEATURE_VERSION,
        "headVersion": int(head_version),
        "boardSizes": list(board_sizes),
        "rulesId": RULES_ID,
        "channels": int(channels),
        "layers": len(dilations),
        "dilations": [int(d) for d in dilations],
        "gateEncoding": GATE_ENCODING,
        "layout": "NHWC",
        "endianness": "little",
        "valuePerspective": "to-move",
        "wdlOrder": ["win", "draw", "loss"],
        "calibrationTemperature": float(calibration_temperature),
        "files": {
            "wiring.i32": {"byteLength": len(wiring_bytes), "sha256": sha256_bytes(wiring_bytes)},
            "gates.u8": {"byteLength": len(gates_bytes), "sha256": sha256_bytes(gates_bytes)},
            "heads.f32": {"byteLength": len(heads_bytes), "sha256": sha256_bytes(heads_bytes)},
        },
        "headTensors": head_tensors,
        "trainingProvenance": training_provenance,
    }


def pack_heads(heads: dict[str, np.ndarray], channels: int, head_version: int = HEAD_VERSION) -> tuple[bytes, list[dict]]:
    """Concatenate the 14 head tensors in HEAD_TENSOR_NAMES order (row-major float32 LE)."""
    shapes = head_shapes(channels, head_version)
    chunks: list[bytes] = []
    entries: list[dict] = []
    offset = 0
    for name in HEAD_TENSOR_NAMES:
        arr = np.ascontiguousarray(np.asarray(heads[name], dtype="<f4"))
        if arr.shape != shapes[name]:
            raise ModelFormatError(f"head tensor {name} has shape {arr.shape}, expected {shapes[name]}")
        if not np.all(np.isfinite(arr)):
            raise ModelFormatError(f"head tensor {name} contains non-finite values")
        raw = arr.tobytes()
        entries.append({"name": name, "shape": list(arr.shape), "byteOffset": offset, "byteLength": len(raw)})
        chunks.append(raw)
        offset += len(raw)
    return b"".join(chunks), entries


def serialize_model(wiring: np.ndarray, gates: np.ndarray, heads: dict[str, np.ndarray], dilations: list[int],
                    board_sizes: list[int], training_provenance: dict, calibration_temperature: float = 1.0,
                    head_version: int = HEAD_VERSION) -> dict[str, bytes]:
    """Returns ``{filename: bytes}`` for the four files, after validating everything."""
    wiring = np.ascontiguousarray(np.asarray(wiring, dtype="<i4"))
    gates = np.ascontiguousarray(np.asarray(gates, dtype=np.uint8))
    validate_wiring(wiring, dilations)
    L, C = wiring.shape[:2]
    if gates.shape != (L, C):
        raise ModelFormatError(f"gates must be [L,C]={L,C}, got {gates.shape}")
    if gates.max() > 15:
        raise ModelFormatError("gate id > 15")
    _check_dims(C, dilations, board_sizes)
    heads_bytes, entries = pack_heads(heads, C, head_version)
    wiring_bytes = wiring.tobytes()
    gates_bytes = gates.tobytes()
    manifest = build_manifest(
        board_sizes=board_sizes, channels=C, dilations=dilations, wiring_bytes=wiring_bytes,
        gates_bytes=gates_bytes, heads_bytes=heads_bytes, head_tensors=entries,
        training_provenance=training_provenance, calibration_temperature=calibration_temperature, head_version=head_version,
    )
    _reject_nonfinite(manifest)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    return {"manifest.json": manifest_bytes, "wiring.i32": wiring_bytes, "gates.u8": gates_bytes, "heads.f32": heads_bytes}


def _check_dims(C: int, dilations: list[int], board_sizes: list[int]) -> None:
    if not MIN_CHANNELS <= C <= MAX_CHANNELS:
        raise ModelFormatError(f"channels {C} outside [{MIN_CHANNELS},{MAX_CHANNELS}]")
    if not MIN_LAYERS <= len(dilations) <= MAX_LAYERS:
        raise ModelFormatError(f"layers {len(dilations)} outside [{MIN_LAYERS},{MAX_LAYERS}]")
    for d in dilations:
        if not MIN_DILATION <= int(d) <= MAX_DILATION:
            raise ModelFormatError(f"dilation {d} outside [{MIN_DILATION},{MAX_DILATION}]")
    if not board_sizes or any(s not in SUPPORTED_BOARD_SIZES for s in board_sizes):
        raise ModelFormatError(f"boardSizes must be a non-empty subset of {SUPPORTED_BOARD_SIZES}")


def write_model_dir(path: str, files: dict[str, bytes]) -> None:
    os.makedirs(path, exist_ok=False)
    for name, data in files.items():
        with open(os.path.join(path, name), "wb") as f:
            f.write(data)


def _read_file_checked(dirpath: str, name: str, expected: dict) -> bytes:
    full = os.path.join(dirpath, name)
    if os.path.islink(full):
        raise ModelFormatError(f"{name}: symlinks are rejected")
    if not os.path.isfile(full):
        raise ModelFormatError(f"{name}: missing")
    with open(full, "rb") as f:
        data = f.read()
    if len(data) != int(expected["byteLength"]):
        raise ModelFormatError(f"{name}: byteLength {len(data)} != manifest {expected['byteLength']}")
    if sha256_bytes(data) != expected["sha256"]:
        raise ModelFormatError(f"{name}: sha256 mismatch")
    return data


def read_model(path: str) -> LoadedModel:
    """Load and fully validate a ``.ichigo`` directory. Raises ModelFormatError."""
    if os.path.islink(path) or not os.path.isdir(path):
        raise ModelFormatError(f"{path}: not a directory (symlinks rejected)")
    mpath = os.path.join(path, "manifest.json")
    if os.path.islink(mpath) or not os.path.isfile(mpath):
        raise ModelFormatError("manifest.json missing")
    with open(mpath, "rb") as f:
        raw = f.read()
    try:
        manifest = json.loads(raw.decode("utf-8"), parse_constant=_bad_constant)
    except (ValueError, UnicodeDecodeError) as e:
        raise ModelFormatError(f"manifest.json: {e}") from e
    validate_manifest(manifest)
    files = manifest["files"]
    wiring_b = _read_file_checked(path, "wiring.i32", files["wiring.i32"])
    gates_b = _read_file_checked(path, "gates.u8", files["gates.u8"])
    heads_b = _read_file_checked(path, "heads.f32", files["heads.f32"])
    L, C = manifest["layers"], manifest["channels"]
    wiring = np.frombuffer(wiring_b, dtype="<i4").reshape(L, C, 2, 4).astype(np.int32)
    gates = np.frombuffer(gates_b, dtype=np.uint8).reshape(L, C).copy()
    validate_wiring(wiring, manifest["dilations"])
    if gates.max() > 15:
        raise ModelFormatError("gate id > 15")
    heads = {}
    for entry in manifest["headTensors"]:
        n = int(np.prod(entry["shape"]))
        heads[entry["name"]] = np.frombuffer(heads_b, dtype="<f4", count=n, offset=entry["byteOffset"]).reshape(entry["shape"]).astype(np.float32)
        if not np.all(np.isfinite(heads[entry["name"]])):
            raise ModelFormatError(f"head tensor {entry['name']} contains non-finite values")
    return LoadedModel(manifest=manifest, wiring=wiring, gates=gates, heads=heads)


def _bad_constant(name: str):
    raise ModelFormatError(f"manifest.json: {name} is not allowed")


REQUIRED_KEYS = {
    "format", "version", "featureVersion", "headVersion", "boardSizes", "rulesId", "channels", "layers",
    "dilations", "gateEncoding", "layout", "endianness", "valuePerspective", "wdlOrder",
    "calibrationTemperature", "files", "headTensors", "trainingProvenance",
}


def validate_manifest(m: dict) -> None:
    if not isinstance(m, dict):
        raise ModelFormatError("manifest is not an object")
    missing = REQUIRED_KEYS - set(m)
    if missing:
        raise ModelFormatError(f"manifest missing keys: {sorted(missing)}")
    _reject_nonfinite(m)
    if m["format"] != FORMAT:
        raise ModelFormatError(f"unknown format {m['format']!r}")
    if m["version"] != VERSION:
        raise ModelFormatError(f"unsupported version {m['version']}")
    if m["featureVersion"] != FEATURE_VERSION:
        raise ModelFormatError(f"unsupported featureVersion {m['featureVersion']}")
    if m["headVersion"] not in SUPPORTED_HEAD_VERSIONS:
        raise ModelFormatError(f"unsupported headVersion {m['headVersion']}")
    if m["rulesId"] != RULES_ID:
        raise ModelFormatError(f"unsupported rulesId {m['rulesId']!r}")
    if m["gateEncoding"] != GATE_ENCODING or m["layout"] != "NHWC" or m["endianness"] != "little":
        raise ModelFormatError("unsupported encoding/layout/endianness")
    if m["valuePerspective"] != "to-move" or m["wdlOrder"] != ["win", "draw", "loss"]:
        raise ModelFormatError("unsupported value perspective or wdl order")
    if not (isinstance(m["calibrationTemperature"], (int, float)) and m["calibrationTemperature"] > 0):
        raise ModelFormatError("calibrationTemperature must be positive")
    C, L, dil = m["channels"], m["layers"], m["dilations"]
    if not isinstance(C, int) or not isinstance(L, int) or not isinstance(dil, list) or len(dil) != L:
        raise ModelFormatError("channels/layers/dilations inconsistent")
    _check_dims(C, dil, m["boardSizes"])
    # byte sizes with explicit overflow-style guards
    wiring_len = L * C * 2 * 4 * 4
    gates_len = L * C
    if wiring_len + gates_len > MAX_PAYLOAD_BYTES:
        raise ModelFormatError("payload too large")
    files = m["files"]
    if not isinstance(files, dict) or set(files) != set(FILE_NAMES):
        raise ModelFormatError(f"files must list exactly {FILE_NAMES}")
    for name in FILE_NAMES:
        e = files[name]
        if not isinstance(e, dict) or not isinstance(e.get("byteLength"), int) or e["byteLength"] < 0:
            raise ModelFormatError(f"files.{name}.byteLength invalid")
        sha = e.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            raise ModelFormatError(f"files.{name}.sha256 invalid")
    if files["wiring.i32"]["byteLength"] != wiring_len:
        raise ModelFormatError("wiring.i32 byteLength does not match L*C*2*4*4")
    if files["gates.u8"]["byteLength"] != gates_len:
        raise ModelFormatError("gates.u8 byteLength does not match L*C")
    heads_len = files["heads.f32"]["byteLength"]
    if wiring_len + gates_len + heads_len > MAX_PAYLOAD_BYTES:
        raise ModelFormatError("payload too large")
    entries = m["headTensors"]
    if not isinstance(entries, list):
        raise ModelFormatError("headTensors must be a list")
    shapes = head_shapes(C, m["headVersion"])
    seen = set()
    spans = []
    total = 0
    for e in entries:
        name = e.get("name")
        if name not in shapes:
            raise ModelFormatError(f"unknown head tensor {name!r}")
        if name in seen:
            raise ModelFormatError(f"duplicate head tensor {name}")
        seen.add(name)
        if list(e.get("shape", [])) != list(shapes[name]):
            raise ModelFormatError(f"head tensor {name} shape {e.get('shape')} != {shapes[name]}")
        off, ln = e.get("byteOffset"), e.get("byteLength")
        if not isinstance(off, int) or not isinstance(ln, int) or off < 0 or ln < 0:
            raise ModelFormatError(f"head tensor {name}: invalid offset/length")
        if off % 4 != 0:
            raise ModelFormatError(f"head tensor {name}: byteOffset not 4-byte aligned")
        if ln != 4 * int(np.prod(shapes[name])):
            raise ModelFormatError(f"head tensor {name}: byteLength != 4*prod(shape)")
        if off + ln > heads_len:
            raise ModelFormatError(f"head tensor {name}: exceeds heads.f32")
        spans.append((off, off + ln))
        total += ln
    if seen != set(HEAD_TENSOR_NAMES):
        raise ModelFormatError(f"missing head tensors: {sorted(set(HEAD_TENSOR_NAMES) - seen)}")
    spans.sort()
    for (a0, a1), (b0, _) in zip(spans, spans[1:]):
        if b0 < a1:
            raise ModelFormatError("head tensors overlap")
    if total != heads_len:
        raise ModelFormatError("heads.f32 has extra bytes not covered by headTensors")
    if not isinstance(m["trainingProvenance"], dict):
        raise ModelFormatError("trainingProvenance must be an object")
