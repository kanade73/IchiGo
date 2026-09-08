"""Training loop (docs/spec/02-training.md §5–§7/§10, T15/T16/T26).

Optimizer-step based loop with gradient accumulation, D4 augmentation, soft+hard validation,
prefix discretisation, best-hard selection, CSV/JSON metrics, resumable checkpoints and SIGINT
handling. Single-server multi-GPU DDP (T26, docs/spec/02-training.md §7/§10 "4GPU検収の具体条件"):
``distributed.py`` detects torchrun's env vars; with world_size==1 this is exactly the original
single-process loop (no process group, no DDP wrapper). With world_size>1 the (unwrapped) model is
wrapped in ``torch.nn.parallel.DistributedDataParallel`` for the forward/backward pass only --
``self.model`` always stays the raw LogicNet used for validation, checkpointing and export, per
the spec note that rank0-only validation must run on the unwrapped module to avoid a DDP collective
wait. Loss normalisation for masked targets is global-effective-batch-aware (accumulation
microbatches and, under DDP, all ranks): see losses.py's module docstring and distributed.py.
Only rank 0 writes run artifacts (metrics.csv, training.log, validation/*.json, checkpoints,
run-summary.json); other ranks call the same barrier()/broadcast_object() sync points but never
open those files. Rank failure: any exception is logged and re-raised so the process exits
non-zero; torchrun's elastic agent SIGTERMs the remaining ranks as soon as one exits non-zero, and
the NCCL/gloo process group is given an explicit timeout (distributed.DEFAULT_TIMEOUT) as a
backstop in case a peer hangs in a collective instead of exiting. The process group is only ever
torn down (``distributed.cleanup()``) on the fully-synchronized success path -- calling
``destroy_process_group()`` from an error path can itself hang if not every rank calls it, so error
paths just let the process die and rely on the OS + torchrun to reclaim resources.

Exit codes: 0 ok, 2 config/schema error, 3 input/device error, 4 non-finite loss/gradient.
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from . import distributed as D
from . import losses as L
from .checkpoint import load_checkpoint, model_from_checkpoint, save_checkpoint, set_rng_state
from .config import ConfigError, load_config
from .data_loader import ShardSampler, augment_d4, load_split_arrays, to_tensors
from .dataset import manifest_hash
from .discretize import PrefixSchedule
from .metrics import CSV_FIELDS, MetricsCSV, evaluate_split, gate_statistics, write_json
from .model_factory import build_model_from_config, model_spec_from_config
from .optim import build_optimizer, build_scheduler, clip_gradients

EXIT_OK, EXIT_CONFIG, EXIT_INPUT, EXIT_TRAINING = 0, 2, 3, 4
FEATURE_VERSION = 1


class Interrupted(Exception):
    pass


def _code_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)).decode().strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment_json(device) -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
            "device": str(device), "cuda": torch.version.cuda, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            "codeRevision": _code_revision()}


def resolve_accumulation(effective_batch: int, micro_batch: int, world_size: int) -> int:
    """N GPUs x microBatch x accumulation = effectiveBatch (docs/spec/02-training.md §7). Rejects
    a configuration that does not divide evenly (config error, exit code 2)."""
    denom = micro_batch * world_size
    if effective_batch % denom != 0:
        raise ConfigError(
            f"effectiveBatch ({effective_batch}) must be divisible by microBatch*worldSize "
            f"({micro_batch}*{world_size}={denom})")
    return effective_batch // denom


class Trainer:
    def __init__(self, config_path: str, resume: str | None):
        self.raw_cfg, self.cfg = load_config(config_path)
        cfg = self.cfg
        D.init(cfg["device"])
        self.rank, self.world_size, self.local_rank = D.rank(), D.world_size(), D.local_rank()
        self.is_main = D.is_main()
        self.accum = resolve_accumulation(cfg["effectiveBatch"], cfg["microBatch"], self.world_size)
        self.out = cfg["out"]
        if self.is_main:
            os.makedirs(os.path.join(self.out, "validation"), exist_ok=True)
            self.log = open(os.path.join(self.out, "training.log"), "a")
        else:
            self.log = None
        self.device = D.resolve_device(cfg["device"])
        if cfg["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        self.fixture = bool(cfg["fixtureMode"])
        self.dataset_hash = "fixture:" + _sha_file(os.path.join(cfg["data"], "fixture.npz")) if self.fixture else manifest_hash(cfg["data"])
        self.S = cfg["boardSize"]
        self.step = 0
        self.warnings: list[dict] = []
        self.best_hard = None  # (loss, step)
        self.interrupted = False
        if resume:
            ck = load_checkpoint(resume)
            self._check_resume(ck)
            self.model = model_from_checkpoint(ck).to(self.device)
        else:
            self.model = build_model_from_config(cfg).to(self.device)
        self.schedule = PrefixSchedule(cfg["maxSteps"], self.model.spec.layers, cfg["tauFinal"])
        self.opt = build_optimizer(self.model, cfg["gateLearningRate"], cfg["headLearningRate"], cfg["headWeightDecay"])
        self.sched = build_scheduler(self.opt, cfg["maxSteps"])
        if self.world_size > 1:
            # find_unused_parameters=True: the prefix-freeze schedule can put theta entirely
            # outside the forward graph (heads-only stage), which varies step to step.
            ddp_kwargs = {"find_unused_parameters": True}
            if cfg["device"] == "cuda":
                ddp_kwargs["device_ids"] = [self.local_rank]
                ddp_kwargs["output_device"] = self.local_rank
            self.ddp_model = DistributedDataParallel(self.model, **ddp_kwargs)
        else:
            self.ddp_model = self.model
        self.sampler = ShardSampler(cfg["data"], cfg["seed"], self.fixture, rank=self.rank, world_size=self.world_size)
        self.aug_rng = np.random.Generator(np.random.PCG64([cfg["seed"], 7]))
        torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"] % (2 ** 32))
        if resume:
            self.opt.load_state_dict(ck["optimizer"])
            self.sched.load_state_dict(ck["scheduler"])
            self.step = ck["step"]
            self.sampler.load_state(ck["sampler"])
            self.aug_rng.bit_generator.state = ck["augRng"]
            set_rng_state(ck["rng"])
            self.best_hard = tuple(ck["extra"]["bestHard"]) if ck["extra"].get("bestHard") else None
            self.warnings = ck["extra"].get("warnings", [])
            self._log(f"resumed from {resume} at step {self.step}")
        self.val_arrays = load_split_arrays(cfg["data"], "validation", self.fixture, cfg["maxValidationPositions"])
        if self.is_main:
            self.csv = MetricsCSV(os.path.join(self.out, "metrics.csv"), resume=bool(resume))
            write_json(os.path.join(self.out, "config.json"), self.raw_cfg)
            write_json(os.path.join(self.out, "resolved-config.json"), dict(cfg, datasetHash=self.dataset_hash, accumulation=self.accum, worldSize=self.world_size))
            write_json(os.path.join(self.out, "environment.json"), environment_json(self.device))
            np.savez(os.path.join(self.out, "wiring.npz"), wiring=self.model.wiring_numpy(), dilations=np.array(self.model.dilations))
        else:
            self.csv = None
        self.prev_hard_loss = None
        self.last_frozen = self.schedule.state_at(self.step).frozen_prefix
        self.t_start = time.monotonic()
        self.val_time = 0.0

    def _check_resume(self, ck):
        if ck["featureVersion"] != FEATURE_VERSION:
            raise ConfigError("checkpoint featureVersion mismatch")
        if ck["datasetHash"] != self.dataset_hash:
            raise ConfigError("checkpoint dataset hash does not match the configured dataset; start a new run")
        if self.cfg.get("modelType", "logic") == "cnn-baseline":
            return  # no fixed wiring to check for the diagnostic CNN baseline (T33)
        spec = model_spec_from_config(self.cfg)
        from .wiring import generate_wiring
        if not np.array_equal(ck["wiring"], generate_wiring(spec).wiring):
            raise ConfigError("checkpoint wiring does not match the configured profile/seed")

    def _log(self, msg: str):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} rank={self.rank} step={self.step} {msg}"
        if self.log is not None:
            self.log.write(line + "\n"); self.log.flush()
        print(line, file=sys.stderr, flush=True)

    # ---- one optimizer step ----
    def train_step(self) -> dict:
        cfg = self.cfg
        st = self.schedule.state_at(self.step)
        model = self.model
        fmodel = self.ddp_model
        model.train()
        self.opt.zero_grad(set_to_none=True)
        frozen_theta = model.theta.detach()[: st.frozen_prefix].clone() if st.frozen_prefix else None
        all_theta = model.theta.detach().clone() if st.heads_only else None

        # Draw every microbatch of this optimizer step up front so the per-term valid-weight
        # denominator can be computed over the whole (this-rank) accumulation window, then
        # all-reduced across ranks -- see losses.py's module docstring for why normalising each
        # microbatch by its own local sum (and averaging by 1/accumulation) is wrong whenever the
        # valid mask differs between microbatches.
        batches = []
        for _ in range(self.accum):
            b = self.sampler.next_batch(cfg["microBatch"])
            if cfg["augmentation"] == "d4":
                b = augment_d4(b, self.S, self.aug_rng)
            batches.append(to_tensors(b, self.device))

        local_sums = torch.zeros(len(L.TERM_NAMES), dtype=torch.float32, device=self.device)
        for t in batches:
            ws = L.local_weight_sums(t)
            for i, name in enumerate(L.TERM_NAMES):
                local_sums[i] = local_sums[i] + ws[name]
        D.all_reduce_sum(local_sums)
        global_sums = {name: local_sums[i] for i, name in enumerate(L.TERM_NAMES)}

        totals = {k: 0.0 for k in ["total", "policy", "expected_result", "wdl", "score", "ownership"]}
        for t in batches:
            out = fmodel(t["spatial"], t["global"], tau=st.tau, frozen_prefix=st.frozen_prefix)
            r = L.compute_losses(out, t, self.S, weight_denominators=global_sums, numerator_scale=float(self.world_size))
            loss = r["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {self.step}")
            loss.backward()
            for k in totals:
                totals[k] += r[k].item()
        # totals currently sum, over this rank's microbatches, world_size * NUM_local/DEN_global.
        # All-reduce (sum across ranks) then divide by world_size once more turns that into the
        # true global weighted-mean loss NUM_global/DEN_global for reporting (a no-op algebraically
        # when world_size==1, since numerator_scale was 1 there).
        tvec = torch.tensor([totals[k] for k in totals], dtype=torch.float64, device=self.device)
        D.all_reduce_sum(tvec)
        totals = {k: v / self.world_size for k, v in zip(totals, tvec.tolist())}

        if st.heads_only:
            model.theta.grad = None
        elif st.frozen_prefix and model.theta.grad is not None:
            model.theta.grad[: st.frozen_prefix].zero_()
        gnorm = clip_gradients(model, cfg["gradClipNorm"])
        if not np.isfinite(gnorm):
            raise FloatingPointError(f"non-finite gradient norm at step {self.step}")
        layer_grad = [float(model.theta.grad[l].norm()) for l in range(model.spec.layers)] if model.theta.grad is not None else [0.0] * model.spec.layers
        self.opt.step()
        self.sched.step()
        with torch.no_grad():
            if all_theta is not None:
                model.theta.copy_(all_theta)
            elif frozen_theta is not None:
                model.theta[: st.frozen_prefix].copy_(frozen_theta)
        self.step += 1
        lrs = [g["lr"] for g in self.opt.param_groups]
        return dict(totals, gradNorm=gnorm, tau=st.tau, frozenPrefix=st.frozen_prefix, headsOnly=st.heads_only, lrGates=lrs[0], lrHeads=lrs[1], layerGradNorm=layer_grad)

    # ---- validation ----
    def validate(self, tag: str = "") -> dict:
        t0 = time.monotonic()
        D.barrier()
        rec = None
        if self.is_main:
            st = self.schedule.state_at(self.step)
            # rank0 evaluates the unwrapped module (not the DDP wrapper): a DDP forward here would
            # need every rank's participation and hang, since only rank0 runs validation.
            soft = evaluate_split(self.model, self.val_arrays, self.S, "soft", st.tau, st.frozen_prefix, self.device)
            hard = evaluate_split(self.model, self.val_arrays, self.S, "hard", st.tau, st.frozen_prefix, self.device)
            sample = to_tensors({k: v[:32] for k, v in self.val_arrays.items()}, self.device)["spatial"]
            stats = gate_statistics(self.model, sample)
            rec = {"step": self.step, "tau": st.tau, "frozenPrefix": st.frozen_prefix, "headsOnly": st.heads_only, "stage": st.stage,
                   "soft": soft, "hard": hard, "softHardPolicyCEDiff": hard["policy"] - soft["policy"], "gates": stats, "tag": tag,
                   "elapsedSeconds": time.monotonic() - self.t_start}
            write_json(os.path.join(self.out, "validation", f"step-{self.step:07d}{('-' + tag) if tag else ''}.json"), rec)
            for mode, r in (("soft", soft), ("hard", hard)):
                self.csv.write({"step": self.step, "phase": "validation", "mode": mode, "tau": st.tau, "frozenPrefix": st.frozen_prefix, "headsOnly": st.heads_only,
                                **{k: r[k] for k in ["total", "policy", "expected_result", "wdl", "score", "ownership", "policyTop1", "expectedMAE", "expectedBrier", "scoreMAEPoints", "ownershipMSE"]},
                                "softHardPolicyCEDiff": rec["softHardPolicyCEDiff"], "elapsedSeconds": rec["elapsedSeconds"]})
            self._log(f"validation soft total={soft['total']:.4f} top1={soft['policyTop1']:.3f} mae={soft['expectedMAE']:.3f} | hard total={hard['total']:.4f} top1={hard['policyTop1']:.3f} mae={hard['expectedMAE']:.3f} frozen={st.frozen_prefix}")
            # best-hard selection: minimal hard total, ties -> earliest step
            if self.best_hard is None or hard["total"] < self.best_hard[0]:
                self.best_hard = (hard["total"], self.step)
                self._save("checkpoint-best-hard.pt")
        rec = D.broadcast_object(rec, src=0)
        D.barrier()
        self.val_time += time.monotonic() - t0
        return rec

    def _save(self, name: str):
        if not self.is_main:
            return
        st = self.schedule.state_at(self.step)
        save_checkpoint(os.path.join(self.out, name), self.model, self.opt, self.sched, self.step, st.frozen_prefix, self.sampler.state(),
                        self.aug_rng, self.cfg, self.dataset_hash, FEATURE_VERSION, {"bestHard": self.best_hard, "warnings": self.warnings})

    def run(self) -> int:
        cfg = self.cfg
        prev = signal.getsignal(signal.SIGINT)

        def on_sigint(sig, frame):
            self.interrupted = True
            self._log("SIGINT received; will checkpoint at the next optimizer-step boundary")
        signal.signal(signal.SIGINT, on_sigint)
        try:
            if self.step == 0:
                self.validate("initial")
            step_times = []
            while self.step < cfg["maxSteps"]:
                before_state = self.schedule.state_at(self.step)
                t0 = time.monotonic()
                r = self.train_step()
                step_times.append(time.monotonic() - t0)
                if self.is_main and (self.step % 10 == 0 or self.step == 1):
                    self.csv.write({"step": self.step, "phase": "train", "mode": "soft", **{k: r[k] for k in CSV_FIELDS if k in r}, "elapsedSeconds": time.monotonic() - self.t_start})
                after_state = self.schedule.state_at(self.step)
                if after_state.frozen_prefix != before_state.frozen_prefix:
                    self._on_freeze(before_state.frozen_prefix, after_state.frozen_prefix)
                if self.step % cfg["validationInterval"] == 0 or self.step == cfg["maxSteps"]:
                    self.validate()
                if self.step % cfg["checkpointInterval"] == 0 or self.step == cfg["maxSteps"]:
                    D.barrier()
                    self._save("checkpoint-latest.pt")
                    D.barrier()
                if self.interrupted:
                    D.barrier()
                    self._save("checkpoint-latest.pt")
                    D.barrier()
                    self._log("checkpoint saved after SIGINT; exiting")
                    break
            D.barrier()
            self._summary(step_times)
            D.cleanup()
            return EXIT_OK
        except FloatingPointError as e:
            # Only this rank necessarily observes the non-finite value; other healthy ranks will
            # hang on their next collective (DDP backward, or a barrier/all-reduce here) until the
            # process-group timeout fires, at which point they take the except-Exception path
            # below and also exit non-zero. Do not attempt run-summary.json or destroy_process_group
            # here: both are collectives that this rank cannot assume its peers will join.
            self._log(f"training abort (non-finite loss/gradient): {e}")
            return EXIT_TRAINING
        except Exception as e:
            self._log(f"training aborted by exception: {e!r}")
            raise
        finally:
            signal.signal(signal.SIGINT, prev)
            if self.is_main:
                self.csv.close()
                self.log.close()

    def _on_freeze(self, old: int, new: int):
        """Hard-validation before/after a layer freeze; warn on >threshold worsening. ``rec`` is
        the (broadcasted) validation record, identical on every rank, so this needs no extra
        rank-guarding beyond ``validate()``'s own -- only rank0's warnings/log end up on disk."""
        rec = self.validate(f"freeze-{new}")
        hard = rec["hard"]["total"]
        if self.prev_hard_loss is not None and hard > self.prev_hard_loss * (1 + self.cfg["freezeWarningThreshold"]):
            w = {"step": self.step, "frozenPrefix": new, "hardBefore": self.prev_hard_loss, "hardAfter": hard,
                 "message": "hard validation loss worsened by more than the threshold after freezing; rerun from the previous checkpoint with a doubled freeze interval (not automated)"}
            self.warnings.append(w)
            self._log(f"WARNING: {w['message']} ({self.prev_hard_loss:.4f} -> {hard:.4f})")
        self.prev_hard_loss = hard

    def _summary(self, step_times):
        elapsed = time.monotonic() - self.t_start
        local_step_time = float(np.mean(step_times)) if step_times else None
        if self.cfg["device"] == "cuda":
            device_name = torch.cuda.get_device_name(self.local_rank)
            peak_mem = int(torch.cuda.max_memory_allocated(self.device))
        else:
            device_name = platform.processor() or "cpu"
            peak_mem = None
        local_perf = {"rank": self.rank, "device": device_name, "secondsPerOptimizerStep": local_step_time, "peakMemoryBytes": peak_mem}
        gathered = D.gather_to_main(local_perf, dst=0)
        if not self.is_main:
            return
        per_rank = sorted(gathered, key=lambda d: d["rank"])
        rank_step_times = [d["secondsPerOptimizerStep"] for d in per_rank if d["secondsPerOptimizerStep"] is not None]
        # Worst (slowest) rank bounds DDP throughput since every step synchronises via backward().
        global_step_time = max(rank_step_times) if rank_step_times else None
        samples_per_second = (self.cfg["effectiveBatch"] / global_step_time) if global_step_time else None
        throughput = {"worldSize": self.world_size, "perRank": per_rank,
                      "secondsPerOptimizerStep": global_step_time, "samplesPerSecond": samples_per_second}
        ref_path = self.cfg.get("throughputReference")
        if ref_path:
            try:
                ref = json.load(open(ref_path))
                ref_sps = ref.get("throughput", {}).get("samplesPerSecond")
                if ref_sps and samples_per_second is not None:
                    ratio = samples_per_second / ref_sps
                    throughput["throughputReferenceRun"] = ref_path
                    throughput["throughputRatio"] = ratio
                    throughput["scalingEfficiency"] = ratio / self.world_size
            except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as e:
                self.warnings.append({"step": self.step, "message": f"could not read throughputReference {ref_path}: {e}"})
        summary = {
            "runId": self.cfg["runId"], "stepsTrained": self.step, "maxSteps": self.cfg["maxSteps"], "completed": self.step >= self.cfg["maxSteps"],
            "bestHard": {"hardTotal": self.best_hard[0], "step": self.best_hard[1]} if self.best_hard else None,
            "bestHardCheckpoint": os.path.join(self.out, "checkpoint-best-hard.pt"),
            "latestCheckpoint": os.path.join(self.out, "checkpoint-latest.pt"),
            "warnings": self.warnings, "interrupted": self.interrupted,
            "timing": {"elapsedSeconds": elapsed, "validationSeconds": self.val_time,
                       "secondsPerOptimizerStep": global_step_time,
                       "estimatedRemainingSeconds": (global_step_time * (self.cfg["maxSteps"] - self.step)) if global_step_time else None},
            "worldSize": self.world_size, "throughput": throughput,
            "datasetHash": self.dataset_hash, "frozenPrefix": self.schedule.state_at(self.step).frozen_prefix,
            "finalGatesAllFrozen": self.schedule.state_at(self.step).heads_only,
        }
        write_json(os.path.join(self.out, "run-summary.json"), summary)
        self._log(f"run summary written ({elapsed:.0f}s, best hard {self.best_hard})")


def _sha_file(path: str) -> str:
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(config_path: str, resume: str | None) -> int:
    try:
        trainer = Trainer(config_path, resume)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    except (OSError, ValueError, RuntimeError) as e:
        print(f"input/device error: {e}", file=sys.stderr)
        return EXIT_INPUT
    return trainer.run()
