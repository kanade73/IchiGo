#!/usr/bin/env python3
"""Overfit acceptance (docs/spec/02-training.md §7).

Soft criterion: within maxSteps (2000) some soft validation reaches policy top1 >= 0.90 and
expected-result MAE <= 0.10 at the same validation point (read from the run's validation JSONs).
Hard criterion: the best-hard checkpoint (all gates argmax, 0/1 inputs) reaches top1 >= 0.80 and
MAE <= 0.15 (read from the hard evaluation report).
"""
import argparse, glob, json, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--hard", required=True); ap.add_argument("--max-steps", type=int, default=2000)
ap.add_argument("--out", default=None)
a = ap.parse_args()
vals = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.run, "validation", "step-*.json")))]
soft_ok = [v for v in vals if v["step"] <= a.max_steps and v["soft"]["policyTop1"] >= 0.90 and v["soft"]["expectedMAE"] <= 0.10]
best_soft = max(vals, key=lambda v: (v["soft"]["policyTop1"], -v["soft"]["expectedMAE"]))
hard = json.load(open(a.hard))
checks = [
    ("soft: some validation <= %d steps with top1>=0.90 and MAE<=0.10" % a.max_steps, bool(soft_ok),
     f"first at step {soft_ok[0]['step']} (top1 {soft_ok[0]['soft']['policyTop1']:.3f}, MAE {soft_ok[0]['soft']['expectedMAE']:.3f})" if soft_ok else f"best soft top1 {best_soft['soft']['policyTop1']:.3f} MAE {best_soft['soft']['expectedMAE']:.3f}"),
    ("hard: best-hard checkpoint top1 >= 0.80", hard["policyTop1"] >= 0.80, f"{hard['policyTop1']:.4f} at step {hard['step']}"),
    ("hard: best-hard checkpoint MAE <= 0.15", hard["expectedMAE"] <= 0.15, f"{hard['expectedMAE']:.4f}"),
]
ok = True
for name, good, detail in checks:
    print(f"{'PASS' if good else 'FAIL'} {name}: {detail}"); ok &= good
print("overfit gate:", "PASS" if ok else "FAIL")
if a.out:
    json.dump({"pass": ok, "checks": [{"name": n, "pass": g, "detail": d} for n, g, d in checks], "hardStep": hard["step"], "hardTop1": hard["policyTop1"], "hardMAE": hard["expectedMAE"]}, open(a.out, "w"), indent=2)
sys.exit(0 if ok else 1)
