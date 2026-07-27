"""Matched baseline: what does a NORMAL-init net get on the exact split the grok runs on?

Same pipeline / split / subsample seeds / balanced 8000-sample val as the grok runs
(run_grok_pilot defaults: train_subset=100 subsample_seed=42, val_subset=8000, seed=100,
arch [200,100,70], adamw, clean labels). Only init_scale=1.0 (no Omnigrok large init), over a
few short regimes. Answers: what a vanilla net reaches on this split, and how fast.

Nothing here actually early-stops — ``summarize`` reports max(val_accuracy) over the whole logged
trajectory, i.e. what a *perfect* early-stopping oracle peeking at the val set would get. That is
an upper bound, not a realistic early-stopping result. It is the same raw-maximum statistic
grok_summary reports as ``peak_val``, so the two are directly comparable, but neither should be
quoted without the noise floor: 1σ ≈ 0.009 on this subsample (effective N ~3000 after RMS-window
overlap), so differences under ~0.03 are not separable.
"""
import json
import numpy as np
from grokking import run_grok_pilot

LOG = "baseline_out.log"

def logline(s):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")

open(LOG, "w", encoding="utf-8").close()
logline("CHANCE = 1/8 = 0.125 (balanced 8000-row val subsample) ; "
        "grok headline run (P11 config, init10x, 450k) = 0.583 final / 0.595 raw peak")

CONFIGS = [
    {"init_scale": 1.0, "lr": 1e-3, "wd": 0.0,  "epochs": 3000, "log_every": 25},  # pure vanilla
    {"init_scale": 1.0, "lr": 1e-3, "wd": 1e-2, "epochs": 3000, "log_every": 25},  # light wd
    {"init_scale": 1.0, "lr": 3e-3, "wd": 1e-2, "epochs": 3000, "log_every": 25},  # faster lr
    {"init_scale": 1.0, "lr": 1e-4, "wd": 0.15, "epochs": 6000, "log_every": 50},  # grok regime, normal init
]
# Measured (this machine, under the keras.utils.set_random_seed fix — pre-fix runs of this file
# gave a different answer every time): the fast lr=1e-3 configs top out at 0.5610 / 0.5592, both at
# epoch 125, and lr=3e-3 at 0.5506 by epoch 75. The grok's own lr/wd at init×1 gets the highest
# number, 0.5736, but not until epoch 950 — so "0.57" and "epoch ~125" come from DIFFERENT runs and
# must not be quoted as one result. Note that config peaks at 950 while memorising only at 3,950:
# its best point is four times BEFORE memorisation, so there is no post-memorisation rise to see.

def summarize(r):
    eps = np.array(r["epochs"]); va = np.array(r["val_accuracy"]); ta = np.array(r["train_accuracy"])
    bi = int(np.argmax(va))
    nearest = lambda e: int(np.argmin(np.abs(eps - e)))
    j50, j500 = nearest(50), nearest(500)
    return dict(best_val=float(va[bi]), ep_best=int(eps[bi]),
                v50=float(va[j50]), e50=int(eps[j50]),
                v500=float(va[j500]), e500=int(eps[j500]),
                final=float(va[-1]), final_ep=int(eps[-1]),
                overfit=float(va[bi] - va[-1]),
                mem_ep=(int(eps[int(np.argmax(ta >= 0.999))]) if np.any(ta >= 0.999) else None))

rows = []
for cfg in CONFIGS:
    logline("=" * 60); logline("CONFIG " + json.dumps(cfg))
    r = run_grok_pilot(label_noise=0.0, **cfg)
    s = summarize(r); rows.append((cfg, s))
    logline("SUMMARY " + json.dumps(s))

logline("\n===== BASELINE TABLE (normal init, same balanced 8000-row val as the grok) =====")
logline("chance=0.125   grok(init10x, 450k) = 0.583 final / 0.595 raw peak   "
        "['best' below is likewise a raw trajectory max, not an early-stopping result]")
for cfg, s in rows:
    logline("init1x lr={lr:g} wd={wd:g}: best={best:.4f}@ep{eb} | val@~50={v50:.4f} "
            "| val@~500={v500:.4f} | final={fin:.4f}@{fe} | overfit_drop={ov:+.4f} | mem@{mem}".format(
                lr=cfg["lr"], wd=cfg["wd"], best=s["best_val"], eb=s["ep_best"],
                v50=s["v50"], v500=s["v500"], fin=s["final"], fe=s["final_ep"],
                ov=s["overfit"], mem=s["mem_ep"]))
logline("DONE")
