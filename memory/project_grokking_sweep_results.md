---
name: grokking sweep shows delayed-generalization drift, not textbook grokking
description: The 15-run AdamW grokking sweep in myo-keras.ipynb shows late val-acc gains of +0.03 to +0.04 after a ~30k-epoch plateau, but the shape is slow drift (Run 1) or dip-and-rebound (Run 7), not the textbook flat-plateau→sharp-edge grokking signature
type: project
---

The stored sweep in `myo-keras.ipynb` (wd ∈ {0.09, 0.10, 0.11}, 5 seeds, arch=[200,100,70], lr=3e-4, 100k epochs, 100-sample train subset, 8k val subset, full-batch AdamW, log every 100) shows delayed generalization but **not** textbook grokking. No run has a flat plateau followed by a sharp vertical rise.

The 15 runs split into two distinct shapes, neither textbook grokking:

**Shape A — slow drift generalization** (Runs 1, 14, 11, 4, 8, 9). Flat plateau at 0.53–0.56 from ~5k to ~30k → gradual monotone climb to a higher band over 40–60k epochs. No dip, no sharp edge. Max rise in any 10k-epoch window is only +0.012 to +0.019.
- **Run 1** (wd=0.09, s=100) is the best example: plateau std 0.0015, no pre-rise dip (−0.002), shift +0.041 (plateau 0.555 → late band 0.596), highest late-band val in the sweep.
- **Run 14** (wd=0.11, s=420): plateau std 0.0015, no dip, shift +0.031, very slow monotone climb.

**Shape B — dip and rebound** (Runs 7, 10, 13). Plateau → real drawdown (val drops ~0.013 below plateau for 2k+ epochs) → sharper recovery + modest band elevation. The sharpness comes mostly from recovery, not from new generalization.
- **Run 7** (wd=0.1, s=123): plateau 0.5535 (epochs 10k–37k, std 0.0034) → real dip to ~0.540 around epochs 38–41k (not a noise spike — the single-epoch 0.518 at ep 38,900 is ~3σ Bernoulli noise, but the smoothed excursion to ~0.546 with +0.007 depth is real) → fast recovery to ~0.585 by ep 50k → sustained late band 0.588–0.605. Shift +0.042, and by far the **sharpest late 10k-window rise** (+0.042 over ep 41k–51k). But the pre-rise dip disqualifies it from "clean grokking shape".

**Run 6** (wd=0.1, s=100) is a separate anomaly: val collapses to 0.445 at ep 68k then recovers to ~0.56 — large non-monotone excursion, not a grokking shape at all.

Which run is "cleanest" depends on what you privilege:
- Grokking hallmark (flat plateau → sudden vertical rise): **no run qualifies**.
- Clean plateau + no dip + real delayed gain: **Run 1** is best.
- Biggest sharp late edge within any 10k window: **Run 7**, but contaminated by the dip.
- Tightest plateau (std only): **Run 11** (0.0008) or Run 5 (0.0011), but both have small shifts (+0.016, +0.010).

Weight norm: rises to ~38–40 by epoch ~15k, then oscillates with quasi-period ~25–30k epochs (3–4 peak/trough cycles per run). Correlation (smoothed wn vs smoothed val) is weakly negative (~−0.1 to −0.25) across most runs. Run 7's sharpest rise coincides with weight-norm compression from ~38 to ~32 around epochs 42k–47k. Wall time: 17h 25m total; ~1h 05m–1h 23m per run.

**Measurement pitfalls encountered during this analysis:**
1. Measuring net rise from `va@5k` understates it — epoch 5k is still in the memorization transient descent, not on the plateau.
2. Measuring rise from `min(va) in plateau window` **overstates** it — the raw min is usually a single-epoch Bernoulli noise spike (3σ ≈ ±0.028 on an 8k val set at val=0.55). Use plateau mean or a windowed median. Run 7's raw min 0.518 vs plateau mean 0.5535 is a 0.035 gap that is mostly noise.
3. Conflating "sharpest late rise" with "cleanest grokking shape" — they're different runs. A sharp late edge preceded by a drawdown is not the same as a flat plateau followed by a phase transition. Rank by shift, plateau std, AND dip depth separately.
4. When computing "max rise over window W", exclude the memorization transient (epoch < ~15k), or every run's winner will be the post-memorization decay, not a late-training event.

Multi-grok observation: several runs have a staircase structure — flat plateau, small step up to a mini-plateau, another step, etc. (Run 1 is the clearest example: ~0.555 flat 5k–35k → ~0.570 band 40k–65k → ~0.595 band 75k–100k.) These correspond loosely to weight-norm compression events. Each step is small (+0.01 to +0.02), so the "multiple grokkings" question is better framed as "delayed generalization arrives in stacked small increments driven by weight-decay-induced norm compression" rather than "multiple full grokking events".

Why the signal is weak: the task is noisy real-world EMG (not algorithmic), baseline statistical generalization is already partial immediately after memorization (~0.55, far above the 0.125 random baseline), so the memorize→generalize gap is narrow and the ceiling is low (~0.61). Only ~12 samples/class also caps reachable validation accuracy regardless of training length. The dataset doesn't have a clean "right circuit" hidden behind a memorization wall the way modular arithmetic does.

**Why:** user went through three rounds of pushback to get the analysis honest: (1) initial "no grokking" was wrong — it measured from va@5k inside the memorization transient; (2) corrected "Run 7 is cleanest +0.088" was also wrong — it used a noise-spike min as the baseline; (3) the +0.047 framing for Run 7 was *still* wrong — Run 7 has a real pre-rise dip and its sharpness is recovery, not phase-transition. The defensible answer is the two-shape taxonomy above.

**How to apply:** do not describe any run in this sweep as "textbook grokking". The most defensible characterizations are (a) Run 1 for delayed-generalization drift with a clean plateau and no dip, and (b) Run 7 for the sharpest late transition, with the caveat that its sharpness is partly rebound from a drawdown. Rank candidates by three separate metrics (plateau std, dip depth, sharpness) rather than a single score — they don't agree. If asked to strengthen the result, suggest smaller train subset (20–40), lower lr (1e-4), larger wd (0.2–0.5), label noise, or switching to an algorithmic toy task (modular arithmetic) where the memorize→generalize gap is actually sharp.
