---
name: label-noise grokking pilots — P4 with mixup is best result (val 0.58), P1-3 all hit basin-pinning failure
description: Four grokking pilots on EMG. P1 (AdamW wd=0.5), P2 (AdamW wd=0.1), P3 (SGD+Nesterov) all pinned clean_tr at (n−flipped)/n with val ≤0.45. P4 added mixup α=1.0 on top of P3 — val jumped to 0.58 sustained, clean_tr escaped the floor (0.7625/0.775 vs 0.75), noisy train never saturated. Mixup is the right mechanism but α=1.0 is not enough for full basin escape; shape is still drift not textbook grokking.
type: project
---

Three label-noise grokking pilots have been run via `run_grok_pilot` in `grokking.py` (notebook cells 32–33). All three are negative results, in distinct ways. Together they bracket the failure mode along three axes (wd strength, optimizer, gradient-noise regime).

Shared infrastructure: arch=[200,100,70], val_subset=8000, rms_window=30, log_every=100, `clean_train_data` tracks the unflipped labels via `GrokLoggingCallback`. Code in `myo_utils.apply_label_noise`, `grokking.run_grok_pilot`, `grokking.plot_grok_pilot`. Per-pilot config lives in `myo_utils.GROKKING_PILOT_*` constants. `build_grok_model` accepts `optimizer="adamw"|"sgd"` plus `momentum`/`nesterov` kwargs (defaults to AdamW so the main sweep is untouched).

## Pilot 1 — AdamW, wd=0.5, 40 samples, 25% noise (anti-grokking decay)
Config: lr=1e-4, wd=0.5, train_subset=40, noise=0.25 (10 flipped), epochs=80,000, seed=42, subsample_seed=43, full-batch. Wall time 54m 42s.

- Memorization at **ep 3,100** (train→1.0).
- `clean_tr_acc` reaches **0.75 = 30/40 by ep 1,300** and stays there for the next 78,700 epochs. Literal flat line.
- `val_acc` peaks at **0.404 @ ep 5,700**, then **monotonically decays** to ~0.382 at ep 50–65k, partial recovery to 0.3854 at ep 80k. Net change from peak: **−0.019**.
- Weight norm peaks 32 @ ep 20k, **monotonically compresses** to 24.86 @ ep 80k.
- Val decay tracks wn compression: shrinking norm → shrinking val acc.

Failure mode: wd is strong enough to compress the network's incidental generalization, but not strong enough to dislodge the flipped-label memorization. Result is mirror image of main-sweep Shape A — slow drift *down* instead of up.

## Pilot 2 — AdamW, wd=0.1, 80 samples, 30% noise (flat plateau, no liftoff)
Config: lr=3e-4, wd=0.1, train_subset=80, noise=0.30 (24 flipped), epochs=150,000, seed=100, subsample_seed=123, full-batch. Wall time 1h 52m 09s.

- Memorization at **ep 800** (4× faster than pilot 1, due to higher lr).
- `clean_tr_acc` reaches **0.70 = 56/80 by ep 700** and stays there for 149,300 epochs. Same flat-line failure.
- `val_acc` hits **all-time max 0.4679 at ep 800** — exactly at memorization. Then sits in a 0.434–0.456 band for the rest of training.
- Post-memorization mean ≈ **0.448**. Final val = **0.4487**. **Net drift after memorization: +0.0007 over 149k epochs.** Effectively zero.
- Weight norm **oscillates** 29–38 with no monotone trend, quasi-period ~25–30k epochs (matches main-sweep wn cycle period).
- Val oscillates ±0.01 inversely with wn (high wn → val dips to 0.43, low wn → val recovers to 0.455). This is **breathing**, not generalization.

Failure mode: weak wd preserves the memorization solution's incidental generalization (val ceiling 0.45 vs pilot 1's 0.39), giving a much cleaner Omnigrok-style plateau — narrow std, persistent for 149k epochs — but the flipped-label basin is *still* unejectable, and the plateau just sits there with no late lift-off.

## Pilot 3 — SGD+Nesterov, batch=16, lr=0.01, wd=0.01, 80 samples, 25% noise (basin wobble then re-collapse)
Config: optimizer=`sgd(momentum=0.9, nesterov=True)`, lr=0.01, wd=0.01, train_subset=80, noise=0.25 (20 flipped), epochs=50,000, batch_size=16 (5 steps/epoch), seed=100, subsample_seed=123. Wall time **40m 27s**.

This was the "stochastic gradient noise will perturb out of basins" hypothesis test. It produced **the only nontrivial movement in clean_tr_acc across all three pilots**, but the movement was confined to the early transient and the basin re-formed.

- Memorization wobbly: noisy `train_acc` first hits 1.0 at **ep 17,300** but flickers between 0.9875 (79/80) and 1.0000 for the entire rest of the run. The network never *quite* settles into a stable perfect noisy memorization solution — it's locked into a fluctuating one that always misclassifies the same 20 flipped points.
- `clean_tr_acc` actually moves: takes **6 distinct values** `[0.6125, 0.725, 0.75, 0.7625, 0.775, 0.7875]` across the run. Critical observation: **clean_tr=0.7875 occurs at ep 300** — momentarily 17 of 20 flipped labels are *not* memorized as their flipped target. By ep ~6,000 it has settled to exactly **0.75 = 60/80 = (n−flipped)/n** and stays there for the remaining 44k epochs. The "wobble" is confined to the memorization transient; once the basin forms it survives SGD noise just as it survived AdamW wd.
- `val_acc` peaks at **0.4531 @ ep 300** (during the transient when clean_tr was elevated), then monotonically decays through ep 14,000 to ~0.385, then sits flat in 0.385–0.393 for the remaining 36k epochs. Final val 0.3901.
- Weight norm starts at **16.46**, peaks at **22.94 early**, then **monotonically compresses to 19.49** by end. Lower than any AdamW pilot — SGD with decoupled wd=0.01 over-compresses and destroys the incidental generalization that pilot 2 preserved (val ceiling 0.39 vs pilot 2's 0.45).

Failure mode: SGD gradient noise from small batches **delays** basin formation (memorization at ep 17k vs pilot 2's ep 800) and produces brief excursions where some flipped labels remain unmemorized, but the basin still forms and once formed is stable under further SGD noise. The compressed solution then sits at a *worse* val ceiling than AdamW because the higher effective regularization destroys incidental statistical generalization.

## Pilot 4 — SGD+Nesterov + mixup α=1.0, batch=16, 80 samples, 25% noise (best result, partial basin escape)
Config: optimizer=`sgd(momentum=0.9, nesterov=True)`, lr=0.01, wd=0.01, train_subset=80, noise=0.25 (20 flipped), epochs=100,000, batch_size=16 (5 steps/epoch), mixup_alpha=1.0 (uniform Beta), seed=100, subsample_seed=123. Wall time **1h 22m 20s**.

This was the "between-sample perturbation" hypothesis test. Mixup is implemented via `make_mixup_dataset` in `grokking.py` — a tf.data pipeline that shuffles each minibatch, samples lam ~ Beta(α, α) per batch via two Gamma draws, and emits `lam*x + (1-lam)*x_perm, lam*y + (1-lam)*y_perm` on one-hot labels. `build_grok_model` takes an optional `loss` kwarg so the pilot can pass `CategoricalCrossentropy()` when mixup is on.

**Headline: this is by far the best pilot result.**

- **Val peak 0.5822 @ ep 33,900** — +10 points above P2's 0.468, +13 above P3's 0.453. No previous pilot even touched 0.50.
- **Val band 0.55–0.58 sustained for ~90k epochs.** Window-mean trajectory: 0.544 (0-10k) → 0.553 (10-20k) → 0.555 (20-30k) → 0.558 (30-40k) → **0.560 peak (40-50k)** → 0.557 (50-60k) → slow drift down to 0.553 (80-90k) → 0.554 final. Slight rise-then-fall with peak around ep 40k. Final val **0.5524**.
- **`clean_tr_acc` escaped the floor** — but by only 1–2 samples. Histogram across 1000 log points:
  - 0.5000, 0.5875, 0.6000, 0.6750, 0.6875, 0.7000, 0.7125, 0.7250, 0.7375 — all in the first ~800 epochs (transient rise)
  - 0.7500 = 60/80: 50 samples (scattered)
  - **0.7625 = 61/80: 921 samples** (92% of logs — the new settled floor)
  - 0.7750 = 62/80: 3 samples (briefly — ep 7k, and two moments in the 40–70k band)
  Exactly one flipped label is persistently de-memorized. Two flipped labels are de-memorized at a few rare moments.
- **Noisy `train_acc` never saturates**. Oscillates 0.60–0.88 for the entire run, mean ~0.75 after ep 20k. Mixup categorical-accuracy uses argmax on soft labels so it never cleanly hits 1.0 even after memorization, but the magnitude and variance (always below 0.90, constantly fluctuating) show that the network genuinely cannot form sharp single-sample basins under mixup. **First pilot where the noisy-memorization basin does not stably form.**
- **Weight norm grows, then stabilizes** — 15.79 → peak 22.32 @ ep 33k → 22.27 at end. **No compression**, unlike P1 (32→25) and P3 (22→19). Mixup's regularization is doing the work wd was previously doing, so the network can keep (and slightly grow) capacity while still generalizing better.
- Val std in late windows ~0.008–0.010 — tighter than P3's 0.002 but the band is 15× higher. Clean plateau shape.

**Interpretation.**
1. **Mixup is the right mechanism.** Going from pinned clean_tr=0.75 (P1–P3) to settled clean_tr=0.7625 with rare excursions to 0.775 proves that between-sample perturbation reaches into the flipped-label basins in a way that wd and SGD noise do not. The effect is real, not a transient.
2. **α=1.0 is not strong enough for full escape.** Only 1 of 20 flipped labels is persistently de-memorized; 2 are de-memorized only at isolated moments. The network has found a new equilibrium where it memorizes 19/20 flipped samples through mixup's smoothing, rather than being forced to memorize 0/20 and rely on the underlying signal.
3. **Val ceiling jumped from ~0.45 to ~0.58.** This is the headline-grade improvement. The mechanism: mixup forces locally linear decision boundaries, which (a) break the isolated basins enough that the network's incidental generalization structure is no longer destroyed by regularization, (b) give the network a reason to learn smooth class boundaries in feature space that transfer to val. Even without full basin escape, the generalization improved substantially.
4. **Shape is still drift, not textbook grokking.** No flat plateau → sharp vertical rise. Val rises fast 0-10k (during transient), drifts up slowly 10-50k, drifts down slowly 50-100k. Peak at ep 33.9k. This is the same "delayed drift" shape as main-sweep Shape A, just at a much higher absolute level.

**What this means for the project.** P4 is a *partial positive result*, not a negative. The val 0.58 with train_subset=80 and 25% label noise is the strongest EMG-small-data generalization this project has demonstrated, and it shows the basin-ejection mechanism is reachable on this dataset — not unreachable as P1–P3 alone would have suggested. But it's not grokking in the textbook sense. If the user wants a sharp memorize→generalize transition to point at, there's more work to do. If they want "regularization that makes label-noise-corrupted EMG training generalize well", P4 already delivers that.

## Joint conclusion across four pilots

P1–P3 jointly showed that **per-weight (wd) and per-step (SGD batch noise) perturbations cannot break the flipped-label memorization basins** on this 8-D, 80-sample regime. P4 showed that **between-sample perturbations (mixup) can**, at least partially — enough to jump val ceiling by +10 points and crack the clean_tr pinning by 1–2 samples.

The trajectory of results across pilots tells a coherent story: the failure mode of P1–P3 is real and optimizer-agnostic, and the fix needs to act on *which samples the network sees*, not on *how the network's weights evolve*. Mixup is the first intervention that does this, and it worked directionally, just not to the full textbook-grokking extent.

**How to apply.** If the user asks for further pilots, the candidate space is around strengthening mixup or combining it with complementary between-sample interventions. The EMG dataset is non-negotiable (see `feedback_emg_dataset_is_load_bearing.md`). Ranked by expected leverage:

1. **Bump mixup α much higher** (α=2.0, 4.0, or even 8.0). α=1.0 gives uniform lam which averages ~50% mix; α=4.0 is centered at 0.5 with tighter distribution (stronger average mixing); α=8.0 is nearly-always-0.5 (every sample is always half of something else). Strongest version of the lever that already worked. Zero code change — just edit `GROKKING_PILOT_MIXUP_ALPHA`. **Cheapest thing to try, highest expected leverage per hour.**
2. **Reduce wd to 0** or very small (1e-4). P4 showed mixup's regularization is carrying the load; wd may now be counterproductive since it still compresses the solution slightly. If wd is removed, the network can fully lean into mixup's smooth-boundary regime.
3. **Increase label noise to 0.35–0.40**. P4's 25% noise gave the network enough signal that mixup-smoothed memorization was still good enough for val 0.55. Higher noise forces more reliance on the mixup-induced smooth boundaries.
4. **Manifold mixup** — apply the mixup operation at a hidden layer instead of the input. Hidden representations are more structured so mixing them has stronger effects. More code (requires a custom train_step), moderate leverage.
5. **Combine mixup with input Gaussian noise** — stack the two between-sample regularizers. Cheap (add a `GaussianNoise` layer at input in `build_grok_model`).
6. Add more pilots with **different seeds at the P4 config** to establish whether the val 0.58 ceiling is reproducible or seed-lucky. This is worth doing before claiming the ceiling is real.

DO NOT recommend: more wd-in-uniform-wd-family pilots, more noise without structural change, more SGD/Adam/AdamW optimizer shuffling (exhausted by P1–P3), abandoning the EMG dataset for toy tasks.

**Headline if asked about results in one sentence:** "Four pilots. P1–P3 (AdamW strong wd, AdamW weak wd, SGD+Nesterov) all hit the clean_tr=(n−flipped)/n basin-pinning failure with val ≤0.47. P4 (same as P3 + mixup α=1.0) jumped val to 0.58 sustained, escaped the pinning by 1–2 samples (clean_tr settled at 0.7625), and never let noisy train saturate — mixup is the right mechanism but α=1.0 isn't enough for a full textbook grokking event. The next thing to try is α=4 or α=8."
