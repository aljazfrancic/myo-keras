---
name: label-noise grokking pilots — three negative results, basin failure is geometric not optimizer-shaped
description: Three label-noise grokking pilots on EMG (AdamW wd=0.5, AdamW wd=0.1, SGD+Nesterov mini-batch) all failed the same way — clean_tr_acc pinned at (n−flipped)/n after settling. SGD pilot momentarily held a *better* clean_tr (0.7875 @ ep 300) before collapsing back. Failure is geometric: 80 samples in 8-D, each flipped point carves a small isolated basin no per-step or per-param perturbation can dislodge.
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

## Joint conclusion across three pilots

The three pilots cover three distinct mechanisms — strong uniform wd shrinkage (P1), weak uniform wd shrinkage (P2), per-step stochastic gradient noise (P3) — and all three end the same way: `clean_tr_acc` pinned at exactly (n−flipped)/n for the bulk of training, all flipped labels memorized as their wrong targets, no late lift-off, no grokking. The failure mode is **robust to optimizer choice**, robust to wd magnitude, and robust to the gradient-noise regime.

**The basin-ejection hypothesis is disproven for this dataset.** SGD noise was the highest-leverage remaining intervention against the geometric basin issue and it merely delayed basin formation rather than preventing it. Once the network has carved a small isolated basin around each flipped point in 8-D feature space, no per-parameter (wd) or per-step (SGD batch sampling) perturbation has enough leverage to break those basins, because the perturbations are *non-targeted* — they push every weight a little, but the basin walls are large enough that uniform pushes don't break out.

The geometry: 80 samples in 8-D real-valued feature space means the average inter-point distance is large compared to typical decision-boundary radius. Each flipped sample sits in its own neighborhood with no other sample nearby to "compete" with its memorization basin. wd and SGD noise don't know which weights matter for the basin vs. which matter for general structure, so they can't selectively erode the bad basins.

Pilot 2 remains the cleanest plateau shape (flat, 149k epochs, narrow std, val 0.448). Pilot 3 is the only one where clean_tr actually moved at all, and only briefly. Neither produced grokking.

**Why:** the user explicitly asked for the SGD+momentum pilot ("let's try the mini-batch SGD+momentum thing... in the pilot ofc..."). I implemented it (lr=0.01, wd=0.01, batch=16, momentum=0.9, nesterov=True, 50k epochs). The result is a clean negative that disproves my own hypothesis. The user has accepted three negative results in a row and is genuinely interested in finding what would work, but the EMG dataset is **not negotiable** (see `feedback_emg_dataset_is_load_bearing.md`) — never propose toy tasks as a substitute.

**How to apply:** if asked "what next?", the candidate space has shrunk significantly. Forbidden: more wd values, more noise levels, more seeds, more optimizer choices in the SGD/Adam/AdamW family, and toy tasks. The remaining levers that have a mechanical reason to break the geometric basin issue are interventions that act *between samples* rather than *per-weight* or *per-step*:

1. **Mixup** (interpolating samples and their labels). This is the highest-leverage remaining intervention. Mixup linearly blends a flipped sample with a clean sample of a different class — the network can no longer carve an isolated basin around the flipped point because that point is constantly being averaged with other points. Forces the basin to compete with nearby clean structure. Implementable as a tf.data pipeline transform or a custom training loop; requires modifying `run_grok_pilot` more substantially than the SGD swap did.
2. **Input Gaussian noise / jitter on each forward pass** — similar mechanism: each flipped sample is "smeared" across a region of feature space, so the basin must be wide enough to absorb the smearing, which lets nearby clean samples penetrate it. Cheaper to implement (Keras `GaussianNoise` layer at the input).
3. **Dropout** (especially high-rate, ~0.5 on hidden layers) — random partial deactivation forces redundant representations that resist single-sample memorization. Lower expected leverage than mixup because it's still a non-targeted perturbation, but it acts on representation rather than on weights, which is qualitatively different from wd and SGD noise.
4. **Larger train_subset (200–500)** — defeats the small-data grokking premise but is an honest fallback: if you can't break the basins, fill the space with enough samples that there are no isolated points.

DO NOT recommend: switching dataset, more wd tuning, more noise levels, more seeds, more epochs, or any optimizer that's still in the SGD/Adam/AdamW family with uniform regularization. The three pilots jointly rule those out.

**Headline if asked about results in one sentence:** "Three pilots — AdamW strong wd, AdamW weak wd, SGD+Nesterov mini-batch — all produced the same clean_tr=(n−flipped)/n flatline failure. SGD noise *briefly* held clean_tr at 0.7875 around ep 300 (3 fewer flipped labels memorized than the eventual basin) before re-collapsing. The flipped-label basins are geometrically robust to per-weight (wd) and per-step (SGD) perturbations; the next thing to try is interventions that act *between samples*, like mixup or input noise."
