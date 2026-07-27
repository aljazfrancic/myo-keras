# myo-keras

EMG gesture classification on Myo armband data — and an attempt to produce **grokking** on it.

Two things live here. The first is an ordinary supervised classifier: a dense network that reads
8-channel surface EMG and predicts which of 8 hand gestures is being held. The second is a
27-run investigation into whether *grokking* — delayed generalisation, normally demonstrated on
algorithmic toy tasks — can be produced on real physiological signal. It can, in shape if not in
payoff, and the write-up below is the honest account of how, including everything that failed.

Everything runs from [`myo-keras.ipynb`](myo-keras.ipynb), top to bottom.

---

## Results at a glance

| | protocol | accuracy |
|---|---|---|
| **Baseline classifier** (this notebook, full data) | within-subject cross-session | **0.744 test** / 0.827 val |
| **Ceiling** — good model, full data ([`ceiling_baseline.py`](ceiling_baseline.py)) | within-subject cross-session | **0.81 test / 0.85 val** |
| **Ceiling** — good model, full data | leave-one-subject-out, 5-fold | **0.65 ± 0.07** |
| **Grokking run** — 100 training samples, 450k epochs | within-subject cross-session, val | 0.619 peak / 0.600 final |
| Vanilla early-stopped net, *same* 100 samples | within-subject cross-session, val | 0.58 @ epoch ~150 |
| Chance | | 0.125 |

The grokking result is a **dynamics** result, not a performance result. Both facts are in the
write-up below, in that order.

---

## Setup

Expects the [myo-readings-dataset](https://github.com/aljazfrancic/myo-readings-dataset) checked out
alongside this repo:

```
.
├── myo-keras/              <- this repo
└── myo-readings-dataset/
    ├── _readings_right_hand/
    └── curated.txt
```

Then open [`myo-keras.ipynb`](myo-keras.ipynb) and run cells top to bottom. The first cell installs
[`requirements.txt`](requirements.txt).

**Runtime:** the committed run took **3 h 46 m** end to end on a 12-core CPU — 18 min for the sweep,
3 h 28 m for the 450k-epoch grokking run, and well under a minute for the baseline classifier. That
works out to ~28 ms per full-batch epoch; on the slower machine the earlier experiments used it was
~77 ms/epoch, so budget up to ~11 h if your hardware is closer to that. To shorten it, set
`GROKKING_PILOT_EPOCHS` down in [`myo_utils.py`](myo_utils.py) — memorisation and the whole rise are
over by 60k epochs.

**CPU is faster than GPU here.** The workloads are small dense models with full-batch updates and
periodic validation, not large batched matmuls, so the CPU TensorFlow stack in `requirements.txt` is
both the documented setup and the practical default.

### Project structure

| File | Description |
|---|---|
| [`myo-keras.ipynb`](myo-keras.ipynb) | Everything: data, baseline classifier, both grokking experiments |
| [`myo_utils.py`](myo_utils.py) | Constants and all hyperparameters, RMS, data loading, auto-curation |
| [`grokking.py`](grokking.py) | Model build, sweep/pilot runners, run summaries, plot helpers |
| [`ceiling_baseline.py`](ceiling_baseline.py) | What the task *actually* supports — within-subject and LOSO |
| [`baseline_check.py`](baseline_check.py) | What a vanilla net gets on the grok's exact 100-sample split |
| [`requirements.txt`](requirements.txt) | TensorFlow, NumPy, Matplotlib, scikit-learn |

### Data and splits

Features are a **causal** sliding-window RMS over each of the 8 EMG channels (window
`RMS_WINDOW_SIZE`, normalised by 128) — no future sample ever enters a feature.

Sessions split deterministically by directory suffix: `-1` train, `-2` validation, `-3` test. The
three splits are loaded by independent calls and **never concatenated then shuffled**, so the
window-overlap leak that would otherwise contaminate a sliding-window pipeline cannot happen. The
asterisk worth stating plainly: this is **within-subject cross-session** (same five participants,
different recording days), not leave-one-subject-out. `ceiling_baseline.py` reports both.

---

## The baseline classifier

`Dense(200) → Dense(100) → Dense(70) → softmax(8)` on the 8-dim RMS features, Adam at 1e-3, early
stopping on validation loss. Trained on the full curated data; stops after 6 epochs at
**0.744 test accuracy** (0.827 best val).

![baseline confusion matrix](pics/baseline_confusion_matrix.png)

Worth reading the per-class table next to that number. The class distribution is heavily skewed —
`hibernation` is 270k of the 480k test rows — and the model leans on it: recall is 0.96 for
hibernation but 0.26–0.66 for the eight active gestures, with `supination` worst. Macro-average F1
is 0.61 against an accuracy of 0.74. The aggregate number flatters the model.

For the strongest number this task supports, see `ceiling_baseline.py`, which adds z-scoring,
dropout, a wider net, and reports leave-one-subject-out as well: **0.81 test / 0.85 val**
within-subject, **0.65 ± 0.07** cross-subject.

---

# Grokking on EMG

## What grokking is, and what showing it requires

**Grokking** (Power et al., 2022) is delayed generalisation. A network memorises its training set —
train accuracy 1.0, validation near chance — sits on a flat low plateau for far longer than
memorisation took, and then, with train accuracy *already pinned at 1.0*, validation accuracy
suddenly climbs. Because train accuracy cannot go up, the improvement is not "more fitting"; the
network has reorganised into a genuinely different, simpler solution.

| Phase | Train accuracy | Val accuracy | Weight norm |
|---|---|---|---|
| 1 — Memorisation | rapid rise to ~100% | stays low | growing |
| 2 — Plateau | ~100% | unchanged | slowly declining |
| 3 — Grokking | ~100% | sudden jump upward | declining further |

It is nearly always shown on algorithmic tasks (modular arithmetic and friends). The goal here was
to produce it on this real 8-channel EMG dataset — and *only* on this dataset. Switching to a toy
task to "calibrate" was ruled out from the start; that forfeits the entire point.

Three conditions must hold **simultaneously**:

1. **Validation pinned low right after memorisation** — there has to be somewhere to jump *from*.
2. **A generalising solution behind an optimisation barrier** that memorisation does not cross.
3. **A weight-norm compression path** — the generalising solution lives at a *smaller* norm than
   the memorising one, and weight decay drives the network down through it.

## Part one — the natural-initialisation sweep (the negative result)

The textbook setup: subsample training data hard (100 samples, ~12 per class), full-batch AdamW with
weight decay, no early stopping, train orders of magnitude past convergence. 15 runs —
wd ∈ {0.09, 0.10, 0.11} × 5 seeds, arch [200,100,70], lr 3e-4, 100k epochs each.

**It does not grok.** What it produces is *delayed-generalisation drift*. From the committed
two-seed pass:

| run | memorised | plateau val | best 10k rise | corr(val, ‖w‖) | ‖w‖ start→end | peak val | final val |
|---|---|---|---|---|---|---|---|
| natural init, wd=0.09, seed 100 | ep 1,500 | 0.551 ± 0.002 | **+0.001** | −0.36 | 17 → 38 | 0.574 @ **ep 500** | 0.549 |
| natural init, wd=0.09, seed 123 | ep 1,400 | 0.577 ± 0.002 | **+0.000** | −0.93 | 17 → 37 | 0.594 @ **ep 500** | 0.570 |

- Validation is already ~0.55 the moment train accuracy hits 1.0. Condition (1) fails outright.
- The largest rise anywhere after memorisation is **+0.001**. That is not a slope, let alone an edge.
- **Peak validation occurs at epoch 500** — *before* memorisation completes — and everything after
  is a slow decay. Early stopping at partial memorisation beats the entire rest of the run. This
  held at 1.2M epochs too, and is a notable standalone finding about this dataset.
- The weight norm **grows** 17 → 38 and then oscillates with a ~25–30k quasi-period. It never
  compresses, so condition (3) fails and there is no mechanism for a late transition. (The −0.93
  correlation on seed 123 is not evidence of one: validation drifts gently down while the norm
  drifts up, which is anti-correlation without a transition — the reason `grok_summary` reports
  correlation *next to* rise rather than on its own.)

Over the original 100k-epoch runs the drift continues to ~0.596 — **+0.041 spread over 70,000
epochs**. One run there (wd=0.10, seed 123) showed a sharper late rise (+0.042 over 10k), but it was
a *rebound* from a pre-rise dip, not a phase transition. Ranking runs by sharpness alone finds that
artefact every time; see [measurement discipline](#measurement-discipline) below.

![sweep validation accuracy](pics/sweep_val_accuracy.png)

![sweep weight norm](pics/sweep_weight_norm.png)

## The diagnosis

Two separate causes, and the second is the one that mattered.

**Why validation never sits low:** EMG RMS features live on a **smooth manifold**. Nearby inputs
share labels, so low-complexity boundaries generalise from the first gradient step. The network
never enters a pure-memorisation regime, so validation never sits near chance.

**Why the norm never compressed:** every run started at the natural Glorot initialisation, weight
norm ~16 — *already at or below the generalising norm*. Weight decay had nothing to compress
**through**. This is the precise gap in the whole project, and it went unnoticed for 24 runs.

## Part two — Omnigrok large initialisation (the result)

The fix, from Liu, Michaud & Tegmark's *Omnigrok* (2022) — the canonical way to induce grokking on
non-algorithmic data: **multiply the initial Dense kernels by `init_scale` ≫ 1**. The network starts
at a large norm with a jagged initial function, so it memorises first with validation pinned low, and
weight decay must then compress the norm down through the generalising "Goldilocks" zone.
Validation rises as it crosses. This manufactures all three missing conditions at once.

`init_scale` and `wd` are **synergistic** and have to be tuned together:

| Run | Config | Outcome |
|---|---|---|
| **P9** | init×5, wd=0.3, 280k epochs (6h31m) | First real compression event: norm **78 → 28, monotone**. Validation starts *below chance* (~0.09) — large init killed the smooth-manifold shortcut. Broke the old 0.59 ceiling (band ~0.602, peak 0.6143). But **no edge**: validation peaks early then declines to 0.576 as the norm keeps falling. Post-memorisation corr(val, ‖w‖) = **+0.70** — validation is a *unimodal* function of the norm, optimal at ‖w‖ ≈ 45–60, and **wd=0.3 overshot the zone**. |
| **P10a** | init×5, wd=0.12, 120k epochs | Control. Validation climbs gently to a **flat hold at ~0.576**, norm equilibrates 44–48, corr(val, ‖w‖) ≈ 0. Stable, mid-band, no edge — too little decay parks the network above the transition. |
| **P10b** | init×10, wd=0.15, 220k epochs | **The grokking shape.** See below. |
| **P11** | init×10, wd=0.15, 450k epochs (10h53m) | Extends P10b to saturation: peak **0.6206 @ epoch 441k**, final 0.6125, norm equilibrated ~40–44. Shape and mechanism reproduce — corr(val, ‖w‖) = **−0.85**, rise **+0.187 over epochs 5k–55k**. This is the configuration the notebook runs. |

### The result

![grokking on EMG](pics/grokking_logx.png)

The figure above is the committed run — init×10, wd 0.15, seed 100, 450k epochs, 3 h 28 m. It shows
the textbook three-phase signature:

1. **Memorise** — train accuracy hits 1.0 at **epoch 2,100**. Validation starts at **0.134 — chance** —
   and rises only to ~0.34. Large init has killed the smooth-manifold shortcut entirely.
2. **Plateau** — train stays at 1.0, validation sits flat and low at **0.369 ± 0.015** for roughly
   ten times longer than memorisation took.
3. **Grok** — validation climbs **+0.090 in a single 10k-epoch window (ep 12,900 → 22,900)** and
   ~+0.25 in total, while train accuracy never changes, reaching **0.619 @ epoch 196k** and settling
   at 0.600.
4. **The mechanism** — the weight norm falls **156 → 45** in mirror image, with
   **corr(val, ‖w‖) = −0.87**. Generalisation is *driven by* norm compression.

Read on a **log-epoch axis** it is the canonical grokking curve, and qualitatively unlike the "peak
at epoch 500, then decay" of every natural-init run.

Set against the same-notebook natural-init runs, the contrast is the whole result:

| | plateau val | best 10k rise | corr(val, ‖w‖) | ‖w‖ start→end | peak val |
|---|---|---|---|---|---|
| natural init (×1), wd 0.09, seed 100 | 0.551 ± 0.002 | +0.001 | −0.36 | 17 → 38 (**grows**) | 0.574 @ ep 500 |
| natural init (×1), wd 0.09, seed 123 | 0.577 ± 0.002 | +0.000 | −0.93 | 17 → 37 (**grows**) | 0.594 @ ep 500 |
| **large init (×10), wd 0.15, seed 100** | **0.369 ± 0.015** | **+0.090** | **−0.87** | **156 → 45 (compresses)** | **0.619 @ ep 196,100** |

**Two honest caveats.** The plateau sits at ~0.37, not the 0.125 chance line — EMG keeps some
smooth-manifold generalisation even at high norm. And the rise spans ~20k epochs rather than a
vertical cliff. This is *grokking-shaped delayed generalisation on real data*, not the purest
algorithmic-task version.

## Everything we tried, and what each one disproved

27 training runs. The failures are the interesting part; each one closed a door.

| Run | What changed | Result | What it disproved |
|---|---|---|---|
| Sweep (15 runs) | wd ∈ {0.09,0.10,0.11} × 5 seeds, natural init, 100k epochs | Drift +0.041 over 70k epochs; norm oscillates 38–40 | Weight decay alone cannot produce an edge at natural init |
| **P1** | AdamW wd=0.5, n=40, 25% label noise | *Anti*-grokking: val peaks 0.40 then **decays** | High wd compresses incidental generalisation but cannot dislodge flipped-label basins |
| **P2** | AdamW wd=0.1, n=80, 30% noise | Flat plateau at 0.45, **zero liftoff in 149k epochs** | A clean Omnigrok-style plateau with no transition — the plateau is not sufficient |
| **P3** | SGD+Nesterov, batch 16, lr 0.01, wd 0.01, 25% noise | Basin wobble then re-collapse, ceiling 0.39 | SGD noise delays basin formation; the basin still forms |
| **P4** | SGD+Nesterov + **mixup α=1.0**, wd 0.01, 25% noise | Best absolute val **0.58**, sustained 0.55–0.58 for 90k epochs | Mixup is the only thing that *partially* breaks basins — still drift, no edge |
| **P5** | mixup α=4, **wd=0**, 35% noise | Regression to 0.40; weight norm **exploded 16 → 100** | wd is load-bearing with mixup — it caps the norm-inflation escape route |
| **P6** | mixup α=4, wd 0.01, 25% noise | Small regression (0.53) | α saturates at 1.0; Beta(4,4) over-smooths with only 80 samples |
| **P7** | **arch shrink to [64,32]** (12× fewer params) + mixup | *Identical* trajectory shape | **Over-parameterisation is not the bottleneck** — capacity is not the story |
| **P8** | Clean labels, wd 0.09, n=100, **1.2M full-batch steps (14h)** | Drift saturates ~500k–700k; peak 0.5901 @ 548k, final 0.5696 | **More epochs is not the answer.** 1.2M steps added +0.006 over the *pre-memorisation* peak of 0.5845 @ epoch **400** |
| **P9** | **init×5**, wd 0.3, 280k | Norm 78 → 28 monotone; ceiling broken (0.6143); no edge | The mechanism works — but wd=0.3 overshoots the Goldilocks zone |
| **P10a** | init×5, wd 0.12, 120k | Flat hold 0.576, corr ≈ 0 | Too little decay parks the net above the transition |
| **P10b** | **init×10, wd 0.15**, 220k | **Plateau 0.35 → +0.17 rise, corr −0.85, val 0.608 still rising** | — this is the result |
| **P11** | init×10, wd 0.15, **450k** | Saturates: peak 0.6206 @ 441k, corr −0.85 | The ~0.62 ceiling is the n=100 information limit, not an optimisation failure |

### Levers ruled out

- ❌ **Label noise of any kind** as the gap-maker. P2 (noise) vs sweep run 1 (clean) differ *only* in
  label noise, and **noise killed the drift that clean labels produced**. On a smooth-signal task,
  flipped labels create a permanent memorisation floor, not a memorise→generalise gap. Wrong
  framework for EMG.
- ❌ **wd = 0 with mixup** — weight-norm explosion (P5). Treat `wd ≥ 0.01` as non-negotiable.
- ❌ **mixup α ≥ 4** on an ~80-sample set — over-smooths (P6).
- ❌ **Shrinking the architecture** to force a plateau — no effect on shape (P7).
- ❌ **Just more epochs** at the existing config — saturates by ~500k (P8).
- ❌ **Chasing a higher validation number** — settled by the baselines below.

## The verdict — three things are true at once

**1. The split is clean.** Train/val/test are separate recording sessions, loaded independently,
never concatenated then shuffled. The temporal window-overlap leak cannot happen. The delayed rise
is real generalisation, not fit-to-leaked-neighbours. (Asterisk: within-subject, not LOSO.)

**2. The dynamics are real.** Large-init plateau → delayed rise → *monotone* norm compression,
corr(val, ‖w‖) = −0.85, visibly unlike a baseline that memorises and overfits in 150 epochs. The
Omnigrok mechanism genuinely fires on real physiological signal.

**3. The payoff is marginal, and this is the part that closes the project.** Two baselines settle it:

- **Matched baseline** ([`baseline_check.py`](baseline_check.py)) — a vanilla early-stopped net on
  the *exact same* 100-sample split, pipeline, and seeds reaches **best val ≈ 0.58 at epoch ~150**,
  then overfits back down. The grok's 0.600 final / 0.619 peak at 450k is **+0.02 to +0.04 for
  3000× the compute — inside the noise, n=1.**
- **Ceiling baseline** ([`ceiling_baseline.py`](ceiling_baseline.py)) — a good model on the **full**
  data reaches **0.81 test / 0.85 val** on the grok's own within-subject protocol, and
  **0.65 ± 0.07** leave-one-subject-out. So the grok's ~0.61 is the **n=100 *sample* ceiling, not
  the *task* ceiling**. The network is not wandering — it is data-starved by design — but it has no
  performance story at any budget.

**The reframe that shrinks the claim:** this is grokking-**the-dynamics**, not
grokking-**the-outcome**. In canonical grokking the train/val gap *closes* — validation climbs to
meet train. Here train is 1.0, validation saturates at ~0.61, and the gap narrows from 0.63 at the
plateau to 0.40 and then **stops**. It never closes.

The defensible sentence is: *grokking-flavoured delayed generalisation shows up cleanly on a non-toy
EMG dataset.* Not: *grokking solved EMG decoding.*

### Reproducibility caveat

Three runs of nominally the same configuration at seed 100 landed at peak val 0.6084 (P10b, 220k),
0.6206 (P11, 450k) and 0.619 (the committed run, 450k) — but by visibly different paths, and P11
disagreed with P10b by 0.04 *at the same epoch*. Cause: RNG context (P10a ran before P10b
originally) plus floating-point non-determinism compounding over 10⁵ full-batch steps in the drift
regime.

What is **robust**: the three-phase shape, the norm compression, corr(val, ‖w‖) ≈ −0.85 to −0.87,
and saturation at ~0.60–0.62. What is **not reproducible**: validation-at-a-given-epoch, to about
±0.03, and the epoch at which the peak happens (196k here, 441k in P11). Expect your run to reach
the same place by a slightly different path — and do not read anything into a 0.02 difference.

### Measurement discipline

Five pitfalls that cost three rounds of wrong conclusions on the sweep. `grok_summary()` in
[`grokking.py`](grokking.py) encodes all of them, and the notebook prints it for every run.

1. **Do not measure net rise from val@5k** — epoch 5k is still inside the memorisation transient (a
   *descent*), which understates the gain. Measure from the plateau.
2. **Do not measure from `min(val)` in the plateau** — a raw minimum is almost always a single-tick
   Bernoulli spike, which *overstates* the rise. Use the plateau **mean** or a windowed median.
3. **"Sharpest late rise" ≠ "cleanest grokking shape"** — they are different runs. A sharp edge
   preceded by a drawdown is a rebound, not a phase transition. Rank by plateau-std, dip-depth, and
   sharpness *separately*; they disagree.
4. **Exclude the memorisation transient** when computing "max rise over window W", or every run's
   winner is just its post-memorisation decay.
5. **Know the noise floor.** On an 8k validation set at p ≈ 0.55, 1σ ≈ 0.006 and single-tick spikes
   over a long run reach ~3σ (±0.017). Worse, causal RMS emits one row per sample, so adjacent
   validation rows overlap by n−1 and the *effective* N is far below 8000. That ±0.006 is a floor —
   trust small edges **less**, not more.

### What we did not do

- **Multi-seed error bands** (4 more seeds, ~4.5 h). Wired as `GROKKING_PILOT_MULTISEED` and never
  run. Once the ceiling closed the performance story, error bars on a no-performance curve stopped
  being decision-relevant — it would only make a non-result robust.
- **The causal init ablation** (init×1 at the full 450k regime, ~10 h) — the airtight version of "is
  large init load-bearing or just an accelerant?". The tuning pass already hints at the answer:
  init×1 in the same regime reached 0.58 with **no plateau and no delayed rise**, so init looks
  load-bearing for the *shape*.
- **Rigid features** — raw `W×8` signal windows instead of RMS, or a much smaller RMS window, to
  remove the smooth shortcut and push the plateau toward chance. The biggest untried structural
  lever, and the one most likely to sharpen the edge.
- **Landscape knobs** — MSE loss, `use_bias=False`, GELU/SiLU, a single wide `[512]` layer.
- **Instrumentation** — per-layer norms and effective rank (grokking as rank collapse, Nanda et al.),
  per-class validation accuracy (grokking often shows one class snapping into place first).

## Reproducing a specific pass

All hyperparameters live in [`myo_utils.py`](myo_utils.py). `GROKKING_PILOT_CONFIGS` selects which
pilot(s) `run_grok_pilots()` executes:

Costs below are measured at ~28 ms/epoch (12-core CPU); multiply by ~2.8 for a ~77 ms/epoch machine.

| Set it to | What runs | Cost |
|---|---|---|
| `GROKKING_PILOT_HEADLINE` *(default)* | init×10, wd 0.15, 450k — the result | ~3.5 h |
| `GROKKING_PILOT_INIT_SWEEP` | the tuning pass: init×5/wd 0.12 control, then init×10/wd 0.15 | ~2.5 h |
| `GROKKING_PILOT_MULTISEED` | 4 extra seeds at 150k for an error band | ~4.5 h |

The natural-init sweep is configured separately by `GROKKING_ARCHITECTURES` × `GROKKING_LRS` ×
`GROKKING_WEIGHT_DECAYS` × `GROKKING_SEEDS` (any list may have length 1). The notebook ships a
trimmed two-seed, 20k-epoch pass (18 min measured); the original was wd ∈ {0.09, 0.10, 0.11} ×
5 seeds × 100k epochs and reached the same conclusion.

`run_grok_sweep` calls `tf.random.set_seed(seed)` before each build and prints per-run wall time.
For reproducibility across machines, confirm `curated.txt` in the dataset repo has not changed since
your reference run — regenerating it with `generate_curated()` changes which sessions enter the split.

### Key parameters

| What | Constant | Role |
|---|---|---|
| Initial weight scale | `GROKKING_PILOT_INIT_SCALE` | **The lever.** Multiplies Dense kernels (not biases) after build. ×10 → starting norm ~157 |
| Weight decay | `GROKKING_PILOT_WD` | Must be tuned *with* init_scale — sets where the norm equilibrates relative to the Goldilocks zone |
| Learning rate | `GROKKING_PILOT_LR` | 1e-4; lowered from 3e-4 for stability at large init |
| Epochs | `GROKKING_PILOT_EPOCHS` | No early stopping. Full-batch, so one optimizer step per epoch |
| Train subset | `GROKKING_TRAIN_SUBSET` | Stratified subsample, n=100 (~12 per class) |
| Val subset | `GROKKING_VAL_SUBSET` | Stratified, 8000 — used for all logged validation metrics |
| Log / eval cadence | `GROKKING_LOG_EVERY` | Training runs every epoch; metrics are *recorded* at multiples of this and the final epoch |
| RMS window (grok reload) | `GROKKING_RMS_WINDOW` | 30, for the grokking data load only |
| Label noise, mixup | `GROKKING_PILOT_LABEL_NOISE`, `..._MIXUP_ALPHA` | Dead ends here; wired for reproduction of P1–P6 |

### Plots

`plot_grok_logx` is the canonical view (log-epoch, accuracy + weight norm, memorisation marked).
`plot_grok_pilot` is the linear-x diagnostic set. `plot_grok_seed_overlay`,
`plot_grok_run_loss_accuracy` and `plot_grok_run_valacc_weight_norm` cover the sweep;
`plot_grok_logx_overlay` overlays several runs with their mean. All accept `savepath`; the notebook
writes every figure in this README into `pics/`.

## References

- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization
  Beyond Overfitting on Small Algorithmic Datasets*. [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- Liu, Z., Michaud, E. J., & Tegmark, M. (2022). *Omnigrok: Grokking Beyond Algorithmic Data*.
  [arXiv:2210.01117](https://arxiv.org/abs/2210.01117)
- Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). *Progress Measures for
  Grokking via Mechanistic Interpretability*. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019.

---

## Appendix

### Auto-curation

`curated.txt` lists the participants whose per-participant test accuracy clears
`CURATION_ACCURACY_THRESHOLD` (default 0.7). To regenerate it — note this **writes into the dataset
repo** and can change which sessions enter the splits:

```python
generate_curated()  # uses READINGS_DIR, CURATED_FILE, CURATION_ACCURACY_THRESHOLD
```

Optional arguments: `generate_curated(readings_dir=..., output_file=..., accuracy_threshold=...)`.

### Gesture labels

| Index | Gesture | | Index | Gesture |
|---|---|---|---|---|
| 0 | hibernation | | 4 | ulnar deviation |
| 1 | flexion | | 5 | pronation |
| 2 | extension | | 6 | supination |
| 3 | radial deviation | | 7 | fist |
