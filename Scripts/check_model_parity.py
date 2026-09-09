#!/usr/bin/env python3
"""Exported .ichigo: Python hard model vs Swift scalar backend on real dataset positions.
Every logic layer must be bit-identical; heads within docs/spec/05-validation.md §3; post-processing 1e-5.
Writes a JSON report. Usage: check_model_parity.py --model M.ichigo --data DATASET --split validation --count 8 [--ichigo .build/release/ichigo]
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Training"))
from ichigo_train.data_loader import load_split_arrays  # noqa: E402
from ichigo_train.export import model_from_loaded  # noqa: E402
from ichigo_train.model import postprocess  # noqa: E402
from ichigo_train.model_format import read_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--split", default="validation")
ap.add_argument("--count", type=int, default=8); ap.add_argument("--ichigo", default=".build/release/ichigo"); ap.add_argument("--out", default=None)
ap.add_argument("--backend", choices=["cpu", "cpu-packed", "auto"], default="cpu")
a = ap.parse_args()
loaded = read_model(a.model)
model = model_from_loaded(loaded)
fixture = os.path.exists(os.path.join(a.data, "fixture.npz")) and not os.path.exists(os.path.join(a.data, "manifest.json"))
arrays = load_split_arrays(a.data, a.split, fixture, a.count)
S = arrays["spatial"].shape[1]
if not os.path.exists(a.ichigo):
    subprocess.run(["swift", "build", "-c", "release", "--product", "ichigo"], check=True)
report = {"model": a.model, "count": int(arrays["spatial"].shape[0]), "samples": [], "layerBitExact": True, "headsWithinTolerance": True}
sp = torch.from_numpy(arrays["spatial"])
layers = model.hard_layers(sp)
with torch.no_grad():
    raw = model.heads_forward(layers[-1].to(torch.float32), torch.from_numpy(arrays["global"]))
post = postprocess(raw["policy_logits"].numpy(), arrays["legal"], raw["wdl_logits"].numpy())
with tempfile.TemporaryDirectory() as td:
    for b in range(arrays["spatial"].shape[0]):
        pos = os.path.join(td, f"p{b}.json")
        json.dump({"boardSize": S, "spatial": arrays["spatial"][b].reshape(-1).tolist(), "global": arrays["global"][b].tolist(), "legal": arrays["legal"][b].tolist()}, open(pos, "w"))
        r = subprocess.run([a.ichigo, "eval", "--model", a.model, "--position", pos, "--backend", a.backend, "--dump-layers", os.path.join(td, f"l{b}.bin")], capture_output=True, text=True)
        if r.returncode != 0:
            print("ichigo eval failed", r.stderr); sys.exit(1)
        out = json.loads(r.stdout)
        swift_layers = np.fromfile(os.path.join(td, f"l{b}.bin"), dtype=np.uint8)
        py_layers = np.concatenate([l[b].numpy().reshape(-1) for l in layers])
        bit_exact = bool(np.array_equal(swift_layers, py_layers))
        def close(g, ref, ta, tr):
            g, ref = np.asarray(g, np.float64), np.asarray(ref, np.float64)
            return bool(np.all(np.abs(g - ref) <= ta + tr * np.abs(ref)))
        heads_ok = all([
            close(out["raw"]["policyLogits"], raw["policy_logits"][b].numpy(), 1e-4, 1e-4),
            close(out["raw"]["wdlLogits"], raw["wdl_logits"][b].numpy(), 1e-4, 1e-4),
            close([out["raw"]["scoreMean"]], [raw["score_mean"][b].item()], 1e-4, 1e-4),
            close(out["raw"]["ownership"], raw["ownership"][b].numpy(), 1e-4, 1e-4),
            close(out["evaluation"]["policy"], post["policy"][b], 1e-5, 0),
            close(out["evaluation"]["expectedResult"], post["expected_result"][b], 1e-5, 0),
        ])
        max_head_diff = float(np.max(np.abs(np.asarray(out["raw"]["policyLogits"]) - raw["policy_logits"][b].numpy())))
        report["samples"].append({"index": b, "layerBitExact": bit_exact, "headsWithinTolerance": heads_ok, "maxPolicyLogitDiff": max_head_diff, "modelHash": out["modelHash"]})
        report["layerBitExact"] &= bit_exact; report["headsWithinTolerance"] &= heads_ok
        print(f"sample {b}: layers {'bit-exact' if bit_exact else 'MISMATCH'}, heads {'ok' if heads_ok else 'MISMATCH'} (max policy logit diff {max_head_diff:.2e})")
print("model parity:", "PASS" if report["layerBitExact"] and report["headsWithinTolerance"] else "FAIL")
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(report, open(a.out, "w"), indent=2)
sys.exit(0 if report["layerBitExact"] and report["headsWithinTolerance"] else 1)
