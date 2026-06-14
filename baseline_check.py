"""Scratch baseline: what does a NORMAL-init net get on the exact P10b split?

Same pipeline / split / subsample seeds / 8000-sample val as the grok runs
(run_grok_pilot defaults: train_subset=100 seed=42, val_subset=8000, arch [200,100,70],
adamw, clean labels). Only init_scale=1.0 (no Omnigrok large init) + sensible fast,
early-stoppable regimes. Answers: best val a vanilla early-stopped net reaches, and how fast.
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
logline("CHANCE = 1/8 = 0.125 ; P10b grok endpoint = 0.62 @450k (init10x)")

CONFIGS = [
    {"init_scale": 1.0, "lr": 1e-3, "wd": 0.0,  "epochs": 3000, "log_every": 25},  # pure vanilla + early stop
    {"init_scale": 1.0, "lr": 1e-3, "wd": 1e-2, "epochs": 3000, "log_every": 25},  # light wd
    {"init_scale": 1.0, "lr": 3e-3, "wd": 1e-2, "epochs": 3000, "log_every": 25},  # faster lr
    {"init_scale": 1.0, "lr": 1e-4, "wd": 0.15, "epochs": 6000, "log_every": 50},  # P10b regime, normal init
]

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

logline("\n===== BASELINE TABLE (normal init, same split as P10b) =====")
logline("chance=0.125   P10b(grok,init10x)=0.62@450k")
for cfg, s in rows:
    logline("init1x lr={lr:g} wd={wd:g}: best={best:.4f}@ep{eb} | val@~50={v50:.4f} "
            "| val@~500={v500:.4f} | final={fin:.4f}@{fe} | overfit_drop={ov:+.4f} | mem@{mem}".format(
                lr=cfg["lr"], wd=cfg["wd"], best=s["best_val"], eb=s["ep_best"],
                v50=s["v50"], v500=s["v500"], fin=s["final"], fe=s["final_ep"],
                ov=s["overfit"], mem=s["mem_ep"]))
logline("DONE")
