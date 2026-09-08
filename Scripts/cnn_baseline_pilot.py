#!/usr/bin/env python3
"""Diagnostic CNN baseline (docs/spec/04-tasks.md T33): 32->64 3x3 stem, 4 residual blocks (64ch,
2 conv each), the SAME local/global heads (headVersion 2 layout) and the same 5 losses/masks as the
logic model. Purpose: decide whether the data/labels allow value learning at pilot scale.
NOT a production model. Usage: cnn_baseline_pilot.py --data DIR --steps 2000 --out reports/pilot/cnn-baseline.json
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Training"))
from ichigo_train import losses as L
from ichigo_train.data_loader import load_split_arrays, to_tensors, augment_d4
from ichigo_train.metrics import evaluate_split

class Block(nn.Module):
    def __init__(s, c): super().__init__(); s.c1 = nn.Conv2d(c, c, 3, padding=1); s.c2 = nn.Conv2d(c, c, 3, padding=1)
    def forward(s, x): return F.relu(x + s.c2(F.relu(s.c1(x))))

class CNN(nn.Module):
    def __init__(s, C=64):
        super().__init__(); s.stem = nn.Conv2d(32, C, 3, padding=1); s.blocks = nn.Sequential(*[Block(C) for _ in range(4)])
        s.Wlocal = nn.Linear(3 * C + 4, 64); s.Wpolicy = nn.Linear(64, 1); s.Wowner = nn.Linear(64, 1)
        s.Wglobal = nn.Linear(2 * C + 64 + 1 + 4, 128); s.Wpass = nn.Linear(128, 1); s.Wwdl = nn.Linear(128, 3); s.Wscore = nn.Linear(128, 1)
    def forward(s, spatial, glob, **kw):
        x = spatial.float().permute(0, 3, 1, 2); h = s.blocks(F.relu(s.stem(x))).permute(0, 2, 3, 1)  # NHWC
        B, S, _, C = h.shape; m = h.mean((1, 2)); v = h.amax((1, 2)); ls = torch.cat([m, v, glob], 1)
        u = torch.cat([h, ls.view(B, 1, 1, -1).expand(B, S, S, 2 * C + 4)], 3); z = F.relu(s.Wlocal(u))
        pol = s.Wpolicy(z).reshape(B, S * S); own = torch.tanh(s.Wowner(z)).reshape(B, S * S)
        g = torch.cat([m, v, z.mean((1, 2)), own.mean(1, keepdim=True), glob], 1); zg = F.relu(s.Wglobal(g))
        return {"policy_logits": torch.cat([pol, s.Wpass(zg)], 1), "wdl_logits": s.Wwdl(zg), "score_mean": s.Wscore(zg).reshape(B), "ownership": own}
    def forward_hard(s, spatial, glob): return s.forward(spatial, glob)
    def eval(s): return super().eval()

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--out", required=True); ap.add_argument("--val-every", type=int, default=250); a = ap.parse_args()
torch.manual_seed(0); rng = np.random.default_rng(0)
tr = load_split_arrays(a.data, "train"); va = load_split_arrays(a.data, "validation", limit=1024); S = tr["spatial"].shape[1]
model = CNN(); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4); N = tr["spatial"].shape[0]
hist = []; t0 = time.time()
for step in range(1, a.steps + 1):
    idx = rng.integers(0, N, size=a.batch); b = augment_d4({k: v[idx] for k, v in tr.items()}, S, rng); t = to_tensors(b, torch.device("cpu"))
    out = model(t["spatial"], t["global"]); r = L.compute_losses(out, t, S); opt.zero_grad(); r["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % a.val_every == 0 or step == a.steps:
        model.eval()
        with torch.no_grad(): res = evaluate_split(model, va, S, "hard", 1.0, 0, torch.device("cpu"))
        model.train(); rec = {"step": step, "trainExpectedLoss": r["expected_result"].item(), "trainScoreLoss": r["score"].item(), **{k: res[k] for k in ["total", "policy", "expected_result", "score", "ownership", "policyTop1", "expectedMAE", "expectedBrier", "scoreMAEPoints", "ownershipMSE"]}, "elapsed": time.time() - t0}
        hist.append(rec); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
json.dump({"model": "cnn-baseline-64ch-4res", "data": a.data, "steps": a.steps, "history": hist}, open(a.out, "w"), indent=2)
