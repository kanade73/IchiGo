import json
import os

import numpy as np
import pytest
import torch

from ichigo_train import model_format as MF
from ichigo_train.export import export_model, load_checkpoint, model_from_loaded, save_checkpoint
from ichigo_train.model import build_model


@pytest.fixture
def model():
    m = build_model("tiny")
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        m.theta.add_(torch.randn(m.theta.shape, generator=g) * 2)
    return m


def _inputs(S=9, B=2):
    sp = torch.from_numpy(np.random.default_rng(7).integers(0, 2, size=(B, S, S, 32)).astype(np.uint8))
    return sp, torch.zeros(B, 4)


def test_roundtrip_hard_output_identical(model, tmp_path):
    out = tmp_path / "m.ichigo"
    manifest = export_model(model, str(out), [9], {"runId": "t"})
    assert set(manifest["files"]) == set(MF.FILE_NAMES)
    loaded = MF.read_model(str(out))
    m2 = model_from_loaded(loaded)
    sp, g = _inputs()
    a, b = model.forward_hard(sp, g), m2.forward_hard(sp, g)
    for k in a:
        assert torch.equal(a[k], b[k]), k
    assert np.array_equal(loaded.gates, model.hard_gates())


def test_overwrite_control_and_no_partial_output(model, tmp_path):
    out = tmp_path / "m.ichigo"
    export_model(model, str(out), [9])
    with pytest.raises(FileExistsError):
        export_model(model, str(out), [9])
    export_model(model, str(out), [9], overwrite=True)
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".export-")]
    # a failing export must not leave a partial directory behind
    bad = build_model("tiny")
    with torch.no_grad():
        bad.heads["Wpass"].fill_(float("nan"))
    with pytest.raises(MF.ModelFormatError):
        export_model(bad, str(tmp_path / "bad.ichigo"), [9])
    assert not (tmp_path / "bad.ichigo").exists()
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".export-")]


def test_checkpoint_roundtrip(model, tmp_path):
    ck = tmp_path / "c.pt"
    save_checkpoint(model, str(ck))
    m2 = load_checkpoint(str(ck))
    assert np.array_equal(m2.wiring_numpy(), model.wiring_numpy())
    assert torch.equal(m2.theta, model.theta)


def _corrupt(out, fn):
    p = os.path.join(out, "manifest.json")
    with open(p) as f:
        m = json.load(f)
    fn(m)
    with open(p, "w") as f:
        json.dump(m, f)


@pytest.mark.parametrize("mutate", [
    lambda m: m.__setitem__("version", 2),
    lambda m: m.__setitem__("featureVersion", 2),
    lambda m: m.__setitem__("rulesId", "japanese"),
    lambda m: m["headTensors"].pop(),
    lambda m: m["headTensors"].append(dict(m["headTensors"][0])),
    lambda m: m["headTensors"][1].__setitem__("byteOffset", m["headTensors"][0]["byteOffset"]),
    lambda m: m["headTensors"][0].__setitem__("byteOffset", 2),
    lambda m: m.__setitem__("dilations", m["dilations"][:-1]),
    lambda m: m.__setitem__("channels", 8),
    lambda m: m["files"].__setitem__("extra.bin", {"byteLength": 0, "sha256": "0" * 64}),
    lambda m: m["files"]["heads.f32"].__setitem__("byteLength", m["files"]["heads.f32"]["byteLength"] + 4),
])
def test_manifest_rejections(model, tmp_path, mutate):
    out = str(tmp_path / "m.ichigo")
    export_model(model, out, [9])
    _corrupt(out, mutate)
    with pytest.raises(MF.ModelFormatError):
        MF.read_model(out)


def test_payload_rejections(model, tmp_path):
    out = str(tmp_path / "m.ichigo")
    export_model(model, out, [9])
    # truncation
    p = os.path.join(out, "heads.f32")
    data = open(p, "rb").read()
    open(p, "wb").write(data[:-4])
    with pytest.raises(MF.ModelFormatError, match="byteLength"):
        MF.read_model(out)
    open(p, "wb").write(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(MF.ModelFormatError, match="sha256"):
        MF.read_model(out)
    open(p, "wb").write(data)
    # gate > 15 with sha fixed up
    gp = os.path.join(out, "gates.u8")
    gd = bytearray(open(gp, "rb").read()); gd[0] = 16
    open(gp, "wb").write(bytes(gd))
    _corrupt(out, lambda m: m["files"]["gates.u8"].__setitem__("sha256", MF.sha256_bytes(bytes(gd))))
    with pytest.raises(MF.ModelFormatError, match="gate"):
        MF.read_model(out)


def test_nan_in_manifest_rejected(model, tmp_path):
    out = str(tmp_path / "m.ichigo")
    export_model(model, out, [9])
    p = os.path.join(out, "manifest.json")
    s = open(p).read().replace('"calibrationTemperature": 1.0', '"calibrationTemperature": NaN')
    open(p, "w").write(s)
    with pytest.raises(MF.ModelFormatError):
        MF.read_model(out)


def test_symlink_rejected(model, tmp_path):
    out = str(tmp_path / "m.ichigo")
    export_model(model, out, [9])
    p = os.path.join(out, "gates.u8")
    os.rename(p, p + ".real")
    os.symlink(p + ".real", p)
    with pytest.raises(MF.ModelFormatError, match="symlink"):
        MF.read_model(out)


def test_cli_init_export_inspect_roundtrip(tmp_path):
    import subprocess, sys
    ck = tmp_path / "tiny.pt"
    out = tmp_path / "tiny.ichigo"
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-m", "ichigo_train", "init-model", "--profile", "tiny", "--out", str(ck)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-m", "ichigo_train", "export", "--checkpoint", str(ck), "--out", str(out), "--board-sizes", "9", "19"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-m", "ichigo_train", "export", "--checkpoint", str(ck), "--out", str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 2  # exists, no --overwrite
    r = subprocess.run([sys.executable, "-m", "ichigo_train", "inspect", "--model", str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and '"boardSizes"' in r.stdout
    m = MF.read_model(str(out))
    assert m.manifest["boardSizes"] == [9, 19]


def test_small_and_base_profiles_forward_both_sizes():
    from ichigo_train.wiring import ModelSpec, generate_wiring, validate_wiring
    for prof in ("small", "base"):
        w = generate_wiring(ModelSpec.from_profile(prof))
        validate_wiring(w.wiring, w.dilations)
    m = build_model("small")
    for S in (9, 19):
        sp = torch.from_numpy(np.random.default_rng(S).integers(0, 2, size=(1, S, S, 32)).astype(np.uint8))
        out = m.forward_hard(sp, torch.zeros(1, 4))
        assert out["policy_logits"].shape == (1, S * S + 1)


def test_failed_rename_keeps_old_model(model, tmp_path, monkeypatch):
    """A failure while promoting the freshly written model must leave the old one intact."""
    out = str(tmp_path / "m.ichigo")
    export_model(model, out, [9], {"runId": "old"})
    old_manifest = MF.read_model(out).manifest
    real_rename = os.rename

    # (1) every rename fails: the old model is never moved aside.
    def all_fail(src, dst, *a, **k):
        raise OSError("injected rename failure")

    monkeypatch.setattr(os, "rename", all_fail)
    with pytest.raises(OSError, match="injected"):
        export_model(model, out, [9], {"runId": "new"}, overwrite=True)
    monkeypatch.setattr(os, "rename", real_rename)
    assert MF.read_model(out).manifest == old_manifest
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".export-")]

    # (2) only the promotion of the new model fails: the backup must be restored.
    def promote_fails(src, dst, *a, **k):
        if os.path.basename(src) == "model":
            raise OSError("injected rename failure")
        return real_rename(src, dst, *a, **k)

    monkeypatch.setattr(os, "rename", promote_fails)
    with pytest.raises(OSError, match="injected"):
        export_model(model, out, [9], {"runId": "new"}, overwrite=True)
    monkeypatch.setattr(os, "rename", real_rename)
    assert MF.read_model(out).manifest == old_manifest
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".export-")]


@pytest.mark.parametrize("hv", [1, 2])
def test_head_versions_roundtrip_and_shapes(tmp_path, hv):
    from ichigo_train.model import global_input_size, head_shapes
    m = build_model("tiny", head_version=hv)
    assert tuple(m.heads["Wglobal"].shape) == (global_input_size(64, hv), 128)
    assert global_input_size(64, 1) == 132 and global_input_size(64, 2) == 197
    out = tmp_path / f"m{hv}.ichigo"
    export_model(m, str(out), [9])
    loaded = MF.read_model(str(out))
    assert loaded.manifest["headVersion"] == hv
    m2 = model_from_loaded(loaded)
    assert m2.head_version == hv
    sp, g = _inputs()
    a, b = m.forward_hard(sp, g), m2.forward_hard(sp, g)
    assert torch.equal(a["wdl_logits"], b["wdl_logits"])
    # a v2 manifest with v1-shaped Wglobal must be rejected
    _corrupt(str(out), lambda mm: mm.__setitem__("headVersion", 3 - hv))
    with pytest.raises(MF.ModelFormatError):
        MF.read_model(str(out))
    with pytest.raises(ValueError):
        build_model("tiny", head_version=3)
