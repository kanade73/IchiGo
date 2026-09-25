"""Small-scale checks for the phase 20 aggregation layers (Training/ichigo_train/aggregation.py).

  fixture  first N positions of a dataset split -> <out>/fixture.npz, for fixtureMode overfit runs
           (train == validation: can the network fit a small set in hard mode?)
  outputs  hard-mode outputs of a checkpoint on a positions JSONL (`ichigo features` rows), in the
           `ichigo eval-batch` JSONL shape, for Scripts/death_bench.py score (the .ichigo format
           cannot hold aggregation layers yet)
  diag     activation / gradient / saturation statistics of every aggregation layer, from a fresh
           model (--config) or a checkpoint (--checkpoint), on N validation positions:
             component-or  per component-size bucket: mean input, mean pooled output, share of
                           outputs >0.99 / <0.01, mean |d loss / d input|; hard: share of pooled
                           channels that are constant (all 0 or all 1) within the bucket
             threshold     firing rate, share of constant nodes, mean |tanh w|, soft/hard agreement
                           of each node on the same hard input bits

Run from the repo root with the Training venv:
  uv run --project Training python Scripts/aggregation_check.py diag --config C.json --data D --device cuda
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Training"))
from ichigo_train import losses as L  # noqa: E402
from ichigo_train.checkpoint import load_checkpoint, model_from_checkpoint  # noqa: E402
from ichigo_train.config import load_config  # noqa: E402
from ichigo_train.data_loader import load_split_arrays  # noqa: E402
from ichigo_train.metrics import to_tensors  # noqa: E402
from ichigo_train.model_factory import build_model_from_config  # noqa: E402

BUCKETS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 10 ** 6)]


def cmd_fixture(a):
    arrays = load_split_arrays(a.data, a.split, limit=a.n)
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "fixture.npz"), **arrays)
    print(json.dumps({"positions": int(arrays["spatial"].shape[0]), "out": a.out}))


def cmd_outputs(a):
    from ichigo_train.model import postprocess
    dev = torch.device(a.device)
    ck = load_checkpoint(a.checkpoint)
    model = model_from_checkpoint(ck).to(dev).eval()
    rows = [json.loads(l) for l in open(a.positions)]
    fv = int(ck.get("featureVersion", 1))
    if any(int(r.get("featureVersion", 1)) != fv for r in rows):
        raise SystemExit(f"positions featureVersion differs from the checkpoint's ({fv})")
    with open(a.out, "w") as out:
        for i in range(0, len(rows), a.batch):
            chunk = rows[i:i + a.batch]
            S = chunk[0]["boardSize"]
            sp = torch.tensor(np.array([r["spatial"] for r in chunk], dtype=np.uint8).reshape(len(chunk), S, S, 32), device=dev)
            gl = torch.tensor(np.array([r["global"] for r in chunk], dtype=np.float32), device=dev)
            with torch.no_grad():
                o = model.forward_hard(sp, gl)
            post = postprocess(o["policy_logits"].cpu().numpy(), np.array([r["legal"] for r in chunk]), o["wdl_logits"].cpu().numpy())
            for j, r in enumerate(chunk):
                out.write(json.dumps({"positionId": r["positionId"], "ownership": o["ownership"][j].cpu().tolist(),
                                      "expectedResult": float(post["expected_result"][j]), "policy": post["policy"][j].tolist(),
                                      "scoreMean": float(o["score_mean"][j]), "winDrawLoss": post["wdl"][j].tolist()}) + "\n")
    print(json.dumps({"positions": len(rows), "out": a.out}))


def bucket_of(sizes):
    out = torch.full_like(sizes, -1)
    for i, (lo, hi) in enumerate(BUCKETS):
        out[(sizes >= lo) & (sizes <= hi)] = i
    return out


def cmd_diag(a):
    dev = torch.device(a.device)
    if a.checkpoint:
        ck = load_checkpoint(a.checkpoint)
        model = model_from_checkpoint(ck).to(dev)
        S = ck["config"]["boardSize"]
    else:
        _, cfg = load_config(a.config)
        torch.manual_seed(cfg["seed"])
        model = build_model_from_config(cfg).to(dev)
        S = cfg["boardSize"]
    agg = model.agg
    if agg is None:
        raise SystemExit("model has no aggregation layers")
    kind = agg.cfg["type"]
    arrays = load_split_arrays(a.data, "validation", limit=a.n)
    n = arrays["spatial"].shape[0]
    acc = {}

    def add(key, value, count=1):
        s, c = acc.get(key, (0.0, 0))
        v = value.detach() if torch.is_tensor(value) else value
        acc[key] = (s + float(v), c + count)

    hard_bucket_rates = {}
    for i in range(0, n, a.batch):
        b = to_tensors({k: v[i:i + a.batch] for k, v in arrays.items()}, dev)
        # soft pass with gradients
        agg.capture = {}
        model.zero_grad(set_to_none=True)
        out = model(b["spatial"], b["global"], tau=a.tau, frozen_prefix=a.frozen)
        L.compute_losses(out, b, S)["total"].backward()
        soft_cap = agg.capture
        # hard pass
        agg.capture = {}
        with torch.no_grad():
            model.forward_hard(b["spatial"], b["global"])
        hard_cap = agg.capture
        agg.capture = None
        for layer, c in soft_cap.items():
            if kind == "component-or":
                k = agg.cfg["channels"]
                labels, member = c["labels"], c["member"]
                sizes = torch.zeros_like(labels).scatter_add_(1, labels, torch.ones_like(labels)).gather(1, labels)
                bk = bucket_of(sizes)
                x = c["in"][..., :k].reshape(labels.shape[0], -1, k)
                full_g = c["in"].grad if c["in"].grad is not None else torch.zeros_like(c["in"])
                g = full_g[..., :k].reshape(labels.shape[0], -1, k)
                g_ref = full_g[..., k:].reshape(labels.shape[0], -1, full_g.shape[-1] - k)
                y = c["out"].reshape(labels.shape[0], -1, k)
                yh = hard_cap[layer]["out"].reshape(labels.shape[0], -1, k)
                xh = hard_cap[layer]["in"][..., :k].to(torch.float32).reshape(labels.shape[0], -1, k)
                for bi in range(len(BUCKETS)):
                    m = member & (bk == bi)
                    cnt = int(m.sum())
                    if not cnt:
                        continue
                    sel = m.unsqueeze(-1).expand_as(x)
                    add((layer, bi, "points"), cnt, 0)
                    add((layer, bi, "x"), x[sel].sum(), cnt * k)
                    add((layer, bi, "y"), y[sel].sum(), cnt * k)
                    add((layer, bi, "sat1"), (y[sel] > 0.99).sum(), cnt * k)
                    add((layer, bi, "sat0"), (y[sel] < 0.01).sum(), cnt * k)
                    add((layer, bi, "grad"), g[sel].abs().sum(), cnt * k)
                    if g_ref.shape[-1]:
                        add((layer, bi, "gradRef"), (g_ref * m.unsqueeze(-1)).abs().sum(), cnt * g_ref.shape[-1])
                    ones = (yh * m.unsqueeze(-1)).sum(dim=(0, 1))  # per channel
                    ones_in = (xh * m.unsqueeze(-1)).sum(dim=(0, 1))
                    hb = hard_bucket_rates.setdefault((layer, bi), [torch.zeros(k, device=dev), 0, torch.zeros(k, device=dev)])
                    hb[0] += ones
                    hb[1] += cnt
                    hb[2] += ones_in
            else:
                node = model.agg.layers[str(layer)]
                hin = hard_cap[layer]["in"].to(torch.float32)
                hard_out = hard_cap[layer]["out"]
                with torch.no_grad():
                    soft_on_bits = node(hin, a.tau, hard=False)
                cnt = hard_out.numel()
                add((layer, "fire"), hard_out.sum(), cnt)
                add((layer, "agree"), ((soft_on_bits > 0.5).float() == hard_out).sum(), cnt)
                add((layer, "gap"), (soft_on_bits - hard_out).abs().sum(), cnt)
                per_node = hard_out.reshape(-1, hard_out.shape[-1]).sum(0)
                hb = hard_bucket_rates.setdefault((layer, "nodes"), [torch.zeros_like(per_node), 0])
                hb[0] += per_node
                hb[1] += hard_out.reshape(-1, hard_out.shape[-1]).shape[0]
                if node.weight.grad is not None:
                    add((layer, "wgrad"), node.weight.grad.abs().mean(), 1)
    report = {"kind": kind, "config": agg.cfg, "positions": n, "tau": a.tau, "frozenPrefix": a.frozen, "layers": {}}
    for layer in sorted(agg.after):
        rows = {}
        if kind == "component-or":
            for bi, (lo, hi) in enumerate(BUCKETS):
                if (layer, bi, "x") not in acc:
                    continue
                name = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+")
                r = {key: acc[(layer, bi, key)][0] / acc[(layer, bi, key)][1] for key in ("x", "y", "sat1", "sat0", "grad", "gradRef")
                     if (layer, bi, key) in acc}
                r["points"] = int(acc[(layer, bi, "points")][0])
                ones, cnt, ones_in = hard_bucket_rates[(layer, bi)]
                rate = (ones / cnt).cpu().numpy()
                rate_in = (ones_in / cnt).cpu().numpy()
                r["hardOneRate"] = float(rate.mean())
                r["hardConstantChannels"] = float(((rate < 0.01) | (rate > 0.99)).mean())
                r["hardConstantInputChannels"] = float(((rate_in < 0.01) | (rate_in > 0.99)).mean())
                rows[name] = r
        else:
            node = model.agg.layers[str(layer)]
            ones, cnt = hard_bucket_rates[(layer, "nodes")]
            rate = (ones / cnt).cpu().numpy()
            rows = {key: acc[(layer, key)][0] / acc[(layer, key)][1] for key in ("fire", "agree", "gap") if (layer, key) in acc}
            rows["constantNodes"] = float(((rate < 0.01) | (rate > 0.99)).mean())
            rows["meanAbsTanhW"] = float(torch.tanh(node.weight.detach()).abs().mean())
            if (layer, "wgrad") in acc:
                rows["weightGrad"] = acc[(layer, "wgrad")][0] / acc[(layer, "wgrad")][1]
        report["layers"][str(layer)] = rows
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fixture")
    f.add_argument("--data", required=True)
    f.add_argument("--split", default="validation")
    f.add_argument("--n", type=int, default=512)
    f.add_argument("--out", required=True)
    o = sub.add_parser("outputs")
    o.add_argument("--checkpoint", required=True)
    o.add_argument("--positions", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--batch", type=int, default=256)
    o.add_argument("--device", default="cpu")
    d = sub.add_parser("diag")
    src = d.add_mutually_exclusive_group(required=True)
    src.add_argument("--config")
    src.add_argument("--checkpoint")
    d.add_argument("--data", required=True)
    d.add_argument("--n", type=int, default=1024)
    d.add_argument("--batch", type=int, default=256)
    d.add_argument("--tau", type=float, default=1.0)
    d.add_argument("--frozen", type=int, default=0)
    d.add_argument("--device", default="cpu")
    d.add_argument("--out")
    a = ap.parse_args()
    {"fixture": cmd_fixture, "outputs": cmd_outputs, "diag": cmd_diag}[a.cmd](a)


if __name__ == "__main__":
    main()
