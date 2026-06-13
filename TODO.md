# TODO — Prove Grokking on the EMG (Myo) Dataset

**Goal:** demonstrate *textbook* grokking on this real 8-channel EMG gesture dataset —
a **flat validation plateau followed by a sharp, late phase-transition upward**,
ideally coincident with a weight-norm compression event.
Not "delayed-generalization drift" (which we already have), and **never** a toy /
algorithmic task — the EMG dataset is the experiment, not a vehicle for it
(see *Disproven / forbidden* below).

> Branch: **pilot**. This file is the single source of truth for the grokking effort —
> roadmap, the 26 runs done, banked findings, and analysis discipline in one place.
> Per-run detail otherwise lives in `git log` (master = the 15-run sweep, pilot = pilots P1–P10).
>
> **STATUS — grokking demonstrated & confirmed (P10b → P11):** flat low-val plateau (≈0.37) → delayed,
> weight-norm-compression-driven rise (corr −0.85) → **saturates at val ~0.61–0.62** (P11 extend, 450k).
> Figure: [`pics/p10b_grokking_logx.png`](pics/p10b_grokking_logx.png). See **Tier 0 → P10/P11 RESULTS**.
> Next overnight run is **wired & active**: multi-seed robustness (P12, `GROKKING_PILOT_MULTISEED`).

---

## Definition of done — what "showing grokking" actually requires

All three must hold simultaneously. Every run so far has failed condition (1), which is why
nothing has produced a sharp edge:

1. **Val pinned LOW right after memorization** (near chance, far below the ceiling) — there must
   be somewhere to jump *from*. On EMG-with-RMS this fails: val is already ~0.55 the moment train
   hits 100%, because RMS features make the task a smooth manifold (nearest-neighbour / linear
   boundaries generalize from gradient step 1).
2. **A generalizing solution hidden behind an optimization barrier** that memorization does not
   cross — on algorithmic tasks this is the algebraic structure; on EMG-with-RMS there is no such
   barrier (the easy memorizing solution *already* generalizes partially).
3. **A weight-norm compression path**: the generalizing solution lives at a *smaller* weight norm
   than the memorizing one, and weight decay drives the network down through it. **We have never
   observed compression** — see diagnosis below.

The plan is organized around *manufacturing* conditions (1)–(3), which no knob we've turned so far
does.

---

## ✅ Already tried (26 training runs — **P10b produced the grokking shape**; P9 broke the val ceiling)

### Master branch — the 15-run AdamW sweep
- [x] **wd ∈ {0.09, 0.10, 0.11} × 5 seeds**, arch `[200,100,70]`, lr 3e-4, 100k full-batch epochs,
  n=100 train / 8k val, RMS window 30. → **Delayed-generalization *drift*, not grokking.**
  - Shape A (slow drift): **Run 1** (wd=0.09, s=100) cleanest — plateau 0.555 (5k–30k), climbs to
    0.596 by 100k, +0.041, no dip. Best "clean shape."
  - Shape B (dip-rebound): **Run 7** (wd=0.10, s=123) — sharpest late edge (+0.042 over 10k) but
    *contaminated by a real pre-rise dip* (sharpness is rebound, not phase transition).
  - Weight norm rises to ~38–40 by 15k then **oscillates** (no clean compression). Ceiling ~0.61.
  - Measurement lessons from analyzing this sweep are banked in *Analysis discipline* below.

### Pilot branch — 11 pilots (P1–P9, P10a, P10b)
- [x] **P1** — AdamW wd=0.5, n=40, 25% noise → *anti-grokking decay* (val peaks 0.40 then decays).
  wd compresses incidental generalization but can't dislodge flipped-label basins.
- [x] **P2** — AdamW wd=0.1, n=80, 30% noise → *flat plateau val ~0.45, ZERO liftoff in 149k epochs.*
  A clean Omnigrok-style plateau with no transition.
- [x] **P3** — SGD+Nesterov, batch=16, lr=0.01, wd=0.01, n=80, 25% noise → *basin wobble then
  re-collapse*, val ceiling 0.39. SGD noise delays basin formation but the basin still forms.
- [x] **P4** — SGD+Nesterov + **mixup α=1.0** + wd=0.01, n=80, 25% noise → **best absolute val 0.58**,
  partial basin escape, sustained 0.55–0.58 band for 90k epochs. **Still drift, not a sharp edge.**
  Mixup (between-sample perturbation) is the only thing that partially breaks basins.
- [x] **P5** — α=4 + **wd=0** + noise=0.35 → regression to 0.40; weight norm **exploded 16→100**.
  Diagnostic: **wd is load-bearing with mixup** (caps the norm-inflation escape route).
- [x] **P6** — α=4, wd=0.01, noise=0.25 → small regression (0.53). **α saturated at 1.0**
  (Beta(4,4) over-smooths with only 80 samples).
- [x] **P7** — **arch shrink `[64,32]`** (12× fewer params) + mixup → *identical trajectory shape*.
  **Disproves the over-parameterization diagnosis** — capacity is not the bottleneck.
- [x] **P8** — **clean-label** wd=0.09, n=100, **1.2M full-batch steps (14h)** → drift extends past
  100k but **saturates ~500k–700k**; all-time peak val **0.5901 @ ep 548k**, final 0.5696.
  The entire 1.2M only added **+0.006** over the *pre-memorization* peak (0.5845 @ ep 400).
  No compression event. (See *Banked findings* under the diagnosis for the pre-memorization-peak result.)
- [x] **P9** — **Omnigrok large-init** (init×5, wd=0.3, lr=1e-4, clean, n=100, 280k full-batch, 6h 31m) →
  **broke the 0.59 ceiling** (band ~0.602, peak 0.6143) with the **first real weight-norm compression
  event** (78→28), but **no sharp edge** — val peaks early then declines as wd overshoots the Goldilocks
  zone (wn≈45–60). *(Superseded by P10b.)*
- [x] **P10a / P10b** — **Goldilocks-tuned large-init** (init×5 wd=0.12 / init×10 wd=0.15, 120k+220k, 8h) →
  **P10b is the grokking shape**: flat low plateau (val ≈0.35, ep 2–15k) → **delayed rise +0.17 over ep 11–31k**
  driven by norm compression (corr(val,wn) = −0.85) → **val 0.608 @ ep 220k, still rising**. P10a holds a flat
  0.576 (control). Full result + next steps in **Tier 0 → P10 RESULTS** below. ***Best result to date.***

### Meta-levers ruled out by the above
- [x] **Longer timescale** (1.2M steps) — saturates; not the answer.
- [x] **Arch capacity** (30k vs 2.5k params) — not the bottleneck (P7).
- [x] **Label noise / Omnigrok-*noise* recipe** — *wrong framework for EMG*. P2 (noise) vs sweep
  Run 1 (clean) differ only in label noise: **noise killed the drift that clean labels produced.**
  On a smooth-signal task, flipped labels create a permanent memorization floor, not a
  memorize→generalize gap.

---

## 🔬 Current diagnosis (why it's been hard)

EMG RMS features live on a **smooth manifold**: nearby inputs share labels, so low-complexity
boundaries generalize from the first gradient step. That *defeats condition (1)* — the network never
enters a pure-memorization regime, so val never sits at chance, so there's no plateau to grok out of.
And because every run started at the **natural Glorot init norm (~16)**, which is already at/below the
generalizing norm, weight decay had **nothing to compress through** → *condition (3)* never fired →
no sharp edge, ever.

**The two untried families below attack the root cause directly:** (A) force memorization-first by
*starting at a large weight norm* (Omnigrok), and (B) *remove the smooth shortcut* by feeding rawer
features. Everything previously tried turned knobs that leave the smooth-manifold property intact.

**Banked findings (keep in mind when reading new runs):**
- **The ceiling is real — ~0.58–0.59 val at n=100.** Every config converges to this band (P4 0.58,
  sweep Run 1 0.596, P8 0.59). It's the information content of 100 samples / 8 classes, not an
  optimization failure. → **We chase the *shape* (low plateau → sharp jump), not a higher number.**
  A sharp jump *to* ~0.58 from a depressed plateau still counts as grokking.
- **Best generalization happens BEFORE memorization completes.** P8's val peaked 0.5845 @ ep 400
  (train only 0.81) ≈ its post-mem peak 0.5901 @ ep 548k. Early-stopping at *partial* memorization
  matched 14h of training — the opposite of textbook grokking, and a notable standalone EMG finding
  worth reporting on its own.
- **Weight-norm "breathing" ≠ signal.** Norm oscillates with quasi-period ~25–30k epochs, weakly
  anti-correlated with val (~−0.1 to −0.25). Don't misread oscillation as a transition — a real grok
  needs *monotone* norm compression through the Goldilocks zone.

---

## 🚫 Disproven / forbidden — do **not** repeat

- ❌ **Label noise of any kind** as the gap-maker (P1–P3, P2) — wrong framework for smooth EMG.
- ❌ **wd = 0 with mixup** — weight-norm explosion (P5). Treat `wd ≥ 0.01` as non-negotiable.
- ❌ **mixup α ≥ 4** on an ~80-sample set — over-smooths, small regression (P6).
- ❌ **Shrinking arch alone** to force a plateau — no effect on shape (P7).
- ❌ **Just more epochs** at the existing config — saturates by ~500k (P8).
- ❌ **Switching to a toy/algorithmic task** — ever, even as a "sanity check" or "calibration run."
  Verbatim directive: *"dont ever suggest dropping this dataset for a algoritmic toy task; that
  foregoes the whole point of the experiment."* Allowed lever space is restricted to changes that
  keep the EMG dataset, curated sessions, and gesture task: optimizer, regularization, lr schedule,
  architecture/activations, data preprocessing (rms_window, raw windows, normalization), and training
  regime (batch size, full vs mini-batch, train_subset, init scale). If a config genuinely can't grok
  on EMG, say so honestly and stop — do **not** redirect to a different problem.

---

## 🎯 To try next — prioritized by expected leverage

### Tier 0 — Omnigrok large-norm initialization  ⭐ HEADLINE, do this first
- [x] **Scale the initial weights by α ≫ 1 and let weight decay compress the norm down through the
  "Goldilocks zone."** *(P9 done — see RESULT below; follow-ups still open.)* This is *the* canonical mechanism for inducing grokking on **non-algorithmic
  real data** (Liu, Michaud & Tegmark, *Omnigrok*, 2022 — already cited in our README/memory; they
  grok MNIST, IMDb, and molecules this way). It manufactures **all three** missing conditions at once:
  large init → jagged initial function → **memorizes first with val pinned low** (cond. 1) → wd must
  **compress the norm** (cond. 3) → when the norm enters the Goldilocks zone, **val jumps** (cond. 2).
  - **Why we never saw it before:** every prior run started at natural norm ~16 (already inside/below
    the zone) so there was nothing to compress *from*. This is the precise gap in the whole project.
  - **Sweep:** `init_scale α ∈ {2, 3, 5, 8}` × `wd ∈ {0.1, 0.3, 1.0}`, clean labels, n=100,
    arch `[200,100,70]`, **lr 1e-4** (drop from 3e-4 for stability at large init), full-batch,
    ~300k–500k epochs, seed 100 first. α and wd are **synergistic** — sweep them together.
  - **Code (minimal):** add `GROKKING_PILOT_INIT_SCALE` to `myo_utils.py`; in
    `build_grok_model` accept `init_scale=1.0` and, after building, multiply each Dense kernel:
    `for lyr in model.layers: w = lyr.get_weights(); lyr.set_weights([w[0]*init_scale, *w[1:]])`
    (scale kernels only; leave biases). Thread `init_scale` through `run_grok_pilot`.
  - **Success criterion:** post-memorization val sits **below ~0.45** for ≥10k epochs (a real low
    plateau, *lower* than the usual 0.55 — large init should depress it), then **rises ≥ +0.10 within
    a ≤10k-epoch window**, coincident with monotone weight-norm decline. That is the sharp edge.
  - **If α too large diverges / never memorizes:** back off α; if val never leaves the floor, lower wd
    or raise lr slightly. Grade α upward until you find the memorize-then-grok window.
  - Cost (measured, this session): **76.6 ms/epoch at log_every=100** (train-bound — evals are ~11%),
    so **~6h ≈ 280k epochs**; at the fine log_every=20 it's ~110 ms/epoch (eval-heavy) → 6h ≈ 195k.
    The 6h epoch ceiling is ~315k even with no evals (the 100-sample full-batch step alone is ~68 ms).
  - **RESULT (P9 complete — 280k epochs, 6h 31m): partial success. Ceiling broken; compression event
    achieved; no textbook edge yet.** First config on this dataset to do all of: **(a) val starts *below
    chance* (~0.09)** — large init killed the smooth-manifold shortcut; **(b) a real, dramatic weight-norm
    compression event** — 78 → 28 monotonic (every prior run oscillated 33–40 and never compressed);
    **(c) broke the 0.59 ceiling** — sustained val band **~0.602 (ep 20–60k)**, single-point peak **0.6143
    @ ep 12.4k** (vs P8 0.5901 / P4 0.58). **New project best.** BUT **no sharp late edge** (max smoothed
    post-transient rise +0.0017): val rises *with* memorization (ep 1.5k) to its peak by ep ~12–40k, then
    **slowly declines to 0.576** as the norm keeps compressing. Post-mem **corr(val, wn) = +0.70** — val is
    a **unimodal function of weight norm, optimal at wn ≈ 45–60** (the Goldilocks zone), and **wd=0.3
    overshot it**, compressing to 28 and dragging val back down. The Omnigrok mechanism works on EMG; the
    knob is just mistuned (too much wd → past the zone).
  - **P10 RESULTS (complete — 8h, both configs): 🎉 GROKKING SHAPE ACHIEVED in P10b.**
    - [x] **P10a** (init×5, wd=0.12, 120k): val climbs gently 0.565 → **holds ~0.576** (final 0.5764), norm
      equilibrated ~44–48. No edge (max smoothed rise +0.012), corr(val,wn)≈0. Clean stable mid-band but
      *below* P9's peak — wd=0.12 at init×5 just gives a flat plateau. Useful control, not the prize.
    - [x] **P10b** (init×10, wd=0.15, 220k): **the textbook three-phase signature, first time on this data.**
      (1) memorize at ep 1.1k; (2) **flat LOW plateau val ≈0.35** (ep 2–15k, std 0.013) while norm is high
      (~135); (3) **delayed sharp rise** — val **+0.17 over ep 11–31k** (+0.105 in a single 10k window) as the
      norm compresses. **corr(val, wn) = −0.852** — generalization is *driven by* norm compression, the grokking
      mechanism. Val reaches **0.6084 @ ep 220k and is STILL RISING** (norm equilibrated ~44, in the Goldilocks
      zone — wd=0.15 landed it right). New project best on *both* shape and final val.
  - **This is grokking on the EMG dataset.** Flat low plateau → delayed generalization driven by weight-norm
    compression. Honest caveats: the plateau sits at ~0.35, not chance 0.125 (EMG keeps some smooth-manifold
    generalization even at high norm), and the rise spans ~20k epochs rather than a vertical cliff — but on a
    **log-epoch axis** it is the canonical grokking curve, and qualitatively unlike the "smooth drift from
    step 1" of all 24 prior runs.
  - **Next, in priority order** (wired in `myo_utils.py`; `GROKKING_PILOT_CONFIGS` selects the active run):
    - [x] **Extend P10b to 450k (P11, done, 10h53m): saturates at val ~0.61–0.62.** Peak **0.6206 @ ep 441k**,
      final 0.6125, **norm equilibrated ~40–44**; shape + mechanism reproduce (corr(val,wn) **−0.85**, rise
      **+0.187 over ep 5–55k**). **Caveat found:** this standalone run *diverged* from the original P10b at the
      *same* seed (val 0.567 vs 0.608 @ ep 220k) — RNG-context (P10a ran before P10b originally) + FP
      non-determinism over 10⁵ steps in the drift regime. **Robust** = the shape + saturation (~0.61); **not
      reproducible** = exact val-at-epoch (±~0.03). Ceiling ~0.62 is the n=100 information limit, not optimization.
    - [ ] **Multi-seed P10b (P12)** (seeds 123, 256, 420, 789) — prove it's robust, not seed/run-luck; now
      *doubly* motivated by the ±0.03 divergence above. Enables a mean±std error-band figure. **WIRED & ACTIVE**
      (`GROKKING_PILOT_MULTISEED`, ~3.6h/seed at 150k → ~14h). Each seed shows plateau→rise; saturation level
      is already established by P11.
    - [x] **Publication figure** — log-x val + weight-norm: [`pics/p10b_grokking_logx.png`](pics/p10b_grokking_logx.png).
      Train→1.0 at ep 1.1k, flat val plateau ~0.33 to ep ~10k, then the rise tracks the norm compression
      (157→44). Reusable via `plot_grok_logx()`; the notebook plot cell now emits it for every run.
    - [ ] **Sharpen the edge** (optional): even bigger init (×15–20) → longer flatter plateau + more abrupt entry
      into the zone; and/or Tier 1 rigid features (raw windows / tiny RMS) to push the plateau toward chance.

### Tier 1 — Make the EMG task rigid (remove the smooth shortcut)
Attacks condition (1) structurally. Natural follow-up if Tier 0 alone isn't enough — and **combines**
with Tier 0 (rigid features + large init is the strongest single bet).
- [ ] **Raw EMG windows instead of RMS features.** Feed flattened raw `W×8` signal windows so the
  network must *discover* the RMS-like power computation itself — the discriminative feature is now
  hidden behind a nonlinearity it has to learn (a real barrier → potential memorize-then-generalize
  gap). Flagged in `grokking.txt`, never tried. Needs a `get_raw_windows()` loader alongside
  `get_rms` and a wider input layer (`W*8`). **Biggest structural lever.**
- [ ] **Tiny RMS window** (`GROKKING_RMS_WINDOW ∈ {1, 3, 5, 10}` vs 30). Less temporal smoothing →
  noisier, harder, less pre-generalized features → wider gap. One-line change + clean-label run.
  Cheapest way to probe the same axis before committing to raw windows.

### Tier 2 — Landscape / objective knobs (fold in as extra sweep dims once Tier 0/1 shows life)
All from `grokking.txt`; cheap, each is a small `build_grok_model` change.
- [ ] **Clean-label HIGH weight decay** `wd ∈ {0.2, 0.3, 0.5, 1.0}`. We only ever tried wd ≥ 0.3
  *with label noise* (P1). The Omnigrok "grok window" for dense nets this size is 0.3–1.0 and the
  **clean-label** high-wd regime is **completely untried**. (Pairs naturally with Tier 0.)
- [ ] **MSE loss** instead of cross-entropy (one-hot targets). Changes the loss landscape; several
  grokking results are loss-function-sensitive. `build_grok_model` already takes a `loss` kwarg.
- [ ] **`use_bias=False`** in all Dense layers. Removes a degree of freedom and ties capacity more
  tightly to weight norm (cleaner Omnigrok-style norm dynamics).
- [ ] **Smoother activations: GELU / SiLU(Swish)** vs ReLU. Affects when/how sharply the transition
  fires in some grokking studies.
- [ ] **Single wide layer `[512]`** (vs deep `[200,100,70]`). Different inductive bias; shallow-wide
  nets show cleaner norm→generalization coupling in Omnigrok.

### Tier 3 — Instrumentation (so we can SEE a transition we'd otherwise miss)
Do this *before/alongside* the first Tier-0 run — these reveal grokking onset earlier than raw val acc
and stop us from smearing a sharp edge.
- [ ] **Per-layer weight norm + effective rank / top singular values of layer 1.** Nanda et al. show
  grokking ≈ a rank-collapse / structural simplification. Cheap to compute once per log tick in
  `GrokLoggingCallback`; may fire *before* val moves.
- [ ] **Per-class val accuracy.** Grokking often shows up as one class suddenly snapping into place
  while others lag — invisible in the aggregate.
- [ ] **Tighter logging (`log_every = 20`)** inside the expected transition window — 100-epoch logging
  smears sharp edges and inflates Bernoulli noise spikes (this already bit us on Run 7's 0.518).
- [ ] **Multi-seed overlay/averaging** for the final figure — 8k-val Bernoulli noise is ~±0.005 (1σ);
  averaging 3–5 seeds gives a publication-clean curve and separates real edges from noise.

### Tier 4 — Combine the winners
- [ ] Take the best (init_scale, wd, features, activation, loss) from Tiers 0–2, run 3–5 seeds at
  300k–500k epochs with full Tier-3 instrumentation, and produce the final grokking figure.

---

## ▶️ Recommended immediate next run

**Omnigrok large-init pilot, single seed, exact config:**

```
optimizer   = adamw
init_scale  = 5.0          # NEW — scale Glorot kernels ×5 (start here; grade 3↔8 as needed)
wd          = 0.3          # NEW — clean-label high wd (Omnigrok grok window)
lr          = 1e-4         # lowered from 3e-4 for stability at large init
arch        = [200,100,70]
batch_size  = None         # full-batch
train_subset= 100
label_noise = 0.0          # clean — noise is a proven dead end on EMG
mixup_alpha = 0
epochs      = 280_000      # ~5.95h at measured 76.6 ms/epoch; 6h epoch-ceiling this machine ~315k
seed        = 100
subsample_seed = 42
log_every   = 100          # train-bound here; ~2800 pts resolves the curve (every-20 is eval-heavy, ~halves epochs in 6h)
```

Add `GROKKING_PILOT_INIT_SCALE = 5.0` to `myo_utils.py`, wire `init_scale` through
`build_grok_model` → `run_grok_pilot`, then run `run_grok_pilot()` and `plot_grok_pilot(result)` in
the notebook. **Watch for (updated from calibration):** val starts *below chance* (~0.09) and climbs
*with* memorization to ~0.56 by ep ~2k — there is **no flat low plateau**, the climb is gradual — while
the weight norm descends monotonically from ~78. The open question the full run answers: does the
**deep** norm compression past ep ~5k (toward the wd-equilibrium) trigger a **late sharp val jump**
(textbook grokking) or a P8-style saturation near the ~0.59 ceiling? So far the *shape* is smooth drift,
but the below-chance start + clean monotone compression are both unprecedented on this data.
If it diverges, α→3; if val saturates early, try wd→0.5–1.0 (stronger compression). **Expected ~5.95h**
(measured 76.6 ms/epoch this session). For a faster first look, set `GROKKING_PILOT_EPOCHS = 50_000`
(~1.1h) — memorization (ep ~2k) and the norm-compression trend are fully visible by then.

All constants in `myo_utils.py` are already set to this config and the run is smoke-tested
(see Tier 0 STATUS). To launch: restart the notebook kernel (to re-import the constants), then run the
`run_grok_pilot()` cell followed by `plot_grok_pilot(pilot_result)`.

---

## 📏 Analysis discipline — measurement pitfalls (banked; don't re-make these)

These cost us three rounds of wrong conclusions on the original sweep. Apply them to every new run:

1. **Don't measure net rise from val@5k** — ep 5k is still in the memorization transient (a
   *descent*), so it understates the gain. Measure from the plateau, not the transient.
2. **Don't measure from `min(val)` in the plateau** — the raw min is almost always a single-epoch
   Bernoulli spike, which *overstates* the rise. Use the plateau **mean** or a **windowed median**.
3. **Don't conflate "sharpest late rise" with "cleanest grokking shape"** — they're different runs.
   A sharp edge preceded by a drawdown (rebound) is not a phase transition. Rank candidates by
   **plateau-std, dip-depth, and sharpness *separately*** — they disagree.
4. **When computing "max rise over window W", exclude the memorization transient** (ep < ~15k), or
   every run's "winner" is just the post-memorization decay, not a late-training event.
5. **Noise floor:** on an 8k val set at p≈0.55, **1σ ≈ 0.006**; single-epoch spikes over a long run
   reach ~3σ (±0.017). Smooth (or use a wider val subset / multi-seed) before declaring an edge.

## How to run / where things live

| What | Where |
|---|---|
| All hyperparameters (`GROKKING_*`, `GROKKING_PILOT_*`) | [`myo_utils.py`](myo_utils.py) |
| Model build, sweep/pilot runners, mixup, plots | [`grokking.py`](grokking.py) |
| Invocation | `run_grok_pilot()` then `plot_grok_pilot(result)` (notebook); sweep via `run_grok_sweep(...)` |
| Default model today | `Dense` ReLU, glorot init, cross-entropy, `use_bias=True`, AdamW |
| Per-run history | `git log` — master = 15-run sweep, pilot = pilots P1–P8 |

*Record each experiment's outcome inline in this file as it completes, and tick the item.*
