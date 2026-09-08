#!/usr/bin/env python3
"""Python hard model -> `ichigo eval` parity for every sample of a parity fixture (T10).

Usage: python Scripts/check_eval_parity.py [--fixture Tests/Fixtures/parity/tiny-9] [--ichigo .build/debug/ichigo]
Compares raw head outputs (tolerance docs/spec/05-validation.md §3) and post-processed values (1e-5).
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fixture", default="Tests/Fixtures/parity/tiny-9")
ap.add_argument("--ichigo", default=None)
a = ap.parse_args()
fx = a.fixture
inputs = json.load(open(os.path.join(fx, "inputs.json")))
S, B = inputs["boardSize"], inputs["batch"]
spatial = np.fromfile(os.path.join(fx, "spatial.u8"), dtype=np.uint8).reshape(B, S * S * 32)
glob = np.fromfile(os.path.join(fx, "global.f32"), dtype="<f4").reshape(B, 4)
legal = np.fromfile(os.path.join(fx, "legal.u8"), dtype=np.uint8).reshape(B, S * S + 1)
expected = json.load(open(os.path.join(fx, "expected.json")))
ichigo = a.ichigo or (["swift", "run", "-q", "ichigo"])
if isinstance(ichigo, str):
    ichigo = [ichigo]

def close(got, ref, tol_abs, tol_rel):
    got, ref = np.asarray(got, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    return got.shape == ref.shape and np.all(np.abs(got - ref) <= tol_abs + tol_rel * np.abs(ref))

ok = True
with tempfile.TemporaryDirectory() as td:
    for b in range(B):
        pos = os.path.join(td, f"pos{b}.json")
        json.dump({"schemaVersion": 1, "boardSize": S, "spatial": spatial[b].tolist(), "global": glob[b].tolist(), "legal": legal[b].tolist()}, open(pos, "w"))
        r = subprocess.run(ichigo + ["eval", "--model", os.path.join(fx, "model.ichigo"), "--position", pos, "--backend", "cpu"], capture_output=True, text=True)
        if r.returncode != 0:
            print("ichigo eval failed:", r.stderr); sys.exit(1)
        out = json.loads(r.stdout)
        raw, ev = out["raw"], out["evaluation"]
        checks = [
            ("policyLogits", close(raw["policyLogits"], expected["policyLogits"][b], 1e-4, 1e-4)),
            ("wdlLogits", close(raw["wdlLogits"], expected["wdlLogits"][b], 1e-4, 1e-4)),
            ("scoreMean", close([raw["scoreMean"]], [expected["scoreMean"][b]], 1e-4, 1e-4)),
            ("ownership", close(raw["ownership"], expected["ownership"][b], 1e-4, 1e-4)),
            ("policy", close(ev["policy"], expected["policy"][b], 1e-5, 0)),
            ("wdl", close(ev["winDrawLoss"], expected["wdl"][b], 1e-5, 0)),
            ("expectedResult", close([ev["expectedResult"]], [expected["expectedResult"][b]], 1e-5, 0)),
        ]
        for name, good in checks:
            if not good:
                ok = False
                print(f"sample {b}: {name} MISMATCH")
        print(f"sample {b}: {'ok' if all(g for _, g in checks) else 'FAIL'} (modelHash {out['modelHash'][:12]})")
print("eval parity:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
