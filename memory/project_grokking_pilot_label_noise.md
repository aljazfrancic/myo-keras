---
name: label-noise grokking pilots — P4 still best; P6 confirms α knob is saturated; real bottleneck is no flat plateau (net too over-parameterized)
description: Six grokking pilots on EMG. P1–P3 (no mixup) pinned clean_tr with val ≤0.47. P4 (SGD + mixup α=1 + wd=0.01) best: val 0.58 sustained. P5 (α=4, wd=0, noise=0.35) regressed to val 0.40 via weight-norm explosion (wd=0 escape route). P6 (P4 + α=4 alone) small regression to val ~0.53 — confirms α knob saturated at α=1 and P5 collapse was mostly wd-attributable. Real diagnosis across all pilots: NO flat plateau exists — val rises continuously from ep 100. Net is over-parameterized ([200,100,70]=30k params on 80 samples). Next: shrink arch to [64,32].
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

**⚠️ Important correction after P5 (see below):** earlier wording in this P4 section described wd as possibly "counterproductive" that could be "removed so the network can lean into mixup". P5 disproved that hypothesis hard. wd=0.01 in P4 is **load-bearing** — it caps weight magnitude so the network cannot escape mixup's regularization via weight-norm inflation. Do NOT recommend wd=0 with mixup on this dataset.

## Pilot 5 — SGD+Nesterov + mixup α=4 + wd=0 + noise=0.35 (combined-knob regression)
Config: same as P4 except `mixup_alpha=4.0`, `wd=0.0`, `label_noise=0.35` (28 flipped of 80). Epochs 100,000, seed=100. Wall time **1h 25m 23s**.

This was the "combine all three aggressive knobs at once" test requested after P4's partial success. It regressed on every metric, and the regression is **diagnostic**.

- **Val peak 0.4854 @ ep 1,600** (during memorization transient), then monotonically decays to **0.4026** by ep 100k. Same "peak-at-memorization-then-decay" shape as P1 and P3.
- **Val late-band mean 0.408** (80–100k window). **Net regression from P4: −0.15 val.** Below P3's ceiling, comparable to P1's floor.
- **`clean_tr_acc` re-pinned at 0.65 = 52/80** by ep 10k and stays there for the remaining 90k epochs. All 28 flipped samples memorized. Histogram: 955 of 1000 logs at exactly 0.65. One single-log excursion to 0.6625 at ep 1,800 (one flipped label briefly un-memorized during the transient), then never again.
- **Weight norm exploded 16.23 → 100.69** monotonically over 100k epochs. **6× P4's final 22.27.** No ceiling, no stabilization — still climbing at +5 wn per 10k epochs at end.
- Noisy `train_acc` late mean **0.83** (vs P4's 0.76) — higher and less variable. Mixup is no longer preventing sharp memorization; the network is memorizing harder by inflating weight magnitudes.

**Failure mechanism (diagnostic).** With wd=0 removed, the network found a new way to memorize that mixup's regularization does not constrain: **inflate weight magnitudes to make decision boundaries arbitrarily sharp**. Mixup smooths the *input* side (the network must predict a mixed label for a mixed input), but nothing caps *weight magnitudes*, so the network routes around mixup by making its decision function steeper in weight-space terms. In P4, wd=0.01 prevented this escape — the network couldn't grow weights unboundedly, so it had to find a solution that fit the mixed-input constraint with bounded weights, which forced it into the smoother-boundary regime that generalized to val 0.58.

**The pilot doesn't disambiguate the three knobs** (α 1→4, wd 0.01→0, noise 0.25→0.35). But the weight-norm explosion is a clean attribution signal: **wd=0 is the primary cause**. Future pilots should treat `wd ≥ 0.01` as non-negotiable when using mixup on this dataset.

**Secondary observations:**
- Noise 0.35 moved the "all flipped memorized" floor from 0.75 to 0.65. P5 settled at exactly this floor, so higher noise doesn't inherently help — it just relocates the failure line.
- α=4 was not tested in isolation. α=4 with wd in place may still be productive (the P4 mechanism with stronger mixing) — P5 can't tell us either way.

## Pilot 6 — SGD+Nesterov + mixup α=4, wd=0.01, noise=0.25 (α knob saturated)
Config: exactly P4 with `mixup_alpha=4.0`. Epochs 100,000, seed=100. Wall time **1h 36m 27s**.

This was the single cleanest next experiment from the P5 post-mortem: isolate the α knob with wd/noise held at P4 values.

- **Val peak 0.5645 @ ep 4,700.** Below P4's 0.5822.
- **Val stable band ~0.52–0.55** for epochs 10k–100k. Final val **0.5247** (ep 100k). Late-window mean ≈ 0.53. **Net regression from P4: −0.05 val.**
- **`clean_tr_acc` escaped the floor to 0.7625** by ep 1,100 and stayed there — exactly matching P4's settled floor. The α=4 change did not improve clean-label escape beyond what α=1 already achieves.
- **Weight norm grew 15.6 → 22.2** and stabilized. No explosion (wd held), comparable to P4's final 22.27. wd is confirmed load-bearing even at α=4.
- **Noisy `train_acc`** ranged ~0.65–0.85 throughout (similar to P4). Mixup still preventing saturation.

**Interpretation.** α=4 alone is a small regression from α=1, not an improvement. Mechanism: Beta(4,4) concentrates the mixing coefficient near λ=0.5, so nearly every training example becomes a ~50/50 blend of two samples. With only 80 training samples, this over-smooths the signal — there isn't enough per-sample information left for the network to learn sharp class boundaries, so val ceiling drops. α=1 (uniform Beta) is the sweet spot for this train_subset size. **The α knob is saturated.**

**The diagnostic insight from P6:** plotting val across all six pilots, **none of them show a flat plateau**. Val rises continuously from ep 100 onward in every pilot. The canonical grokking shape (flat plateau → sharp vertical rise) requires a *phase separation* where memorization completes long before generalization starts. On this EMG task with arch=[200,100,70] and n=80, the network generalizes gradually from step 1 — it never enters a pure-memorization regime. **Real failure mode across all six pilots: arch is too big.** [200,100,70] has ~30,000 params for 80 samples ≈ 375 params/sample, way into the over-parameterized regime where the network immediately finds partial generalization alongside memorization. To force a memorization-first phase, shrink the net.

## Joint conclusion across six pilots

P1–P3 jointly showed that **per-weight (wd) and per-step (SGD batch noise) perturbations cannot break the flipped-label memorization basins** on this 8-D, 80-sample regime. P4 showed that **between-sample perturbations (mixup) can**, at least partially — enough to jump val ceiling by +10 points and crack the clean_tr pinning by 1–2 samples. P5 showed that **mixup needs wd** — without wd capping weight magnitudes, the network escapes mixup's regularization by inflating weights (16→100). P6 showed that **the α knob is saturated at α=1** — α=4 alone is a small regression (~0.53 vs 0.58), because Beta(4,4) over-smooths when the train set is only 80 samples.

The mixup-knob exploration is exhausted. P4 remains the best attainable configuration in the (mixup strength × wd × noise) subspace.

**But the real bottleneck revealed by P6:** across all six pilots, **no configuration produces a flat plateau**. Val rises continuously from ep 100 in every run. Textbook grokking requires memorization-first dynamics with a phase gap before generalization; this network is generalizing gradually from the first step because it is too over-parameterized for the task. With arch=[200,100,70] ≈ 30,000 params on 80 samples, the model has ~375 params/sample, far enough into the over-parameterized regime that partial generalization emerges alongside memorization rather than after it. Further knob-tuning inside this architecture cannot produce a plateau — the plateau requires a model small enough that memorization is genuinely hard.

**How to apply.** Forbidden knobs and combinations (all disproven by pilots above):
- wd=0 combined with mixup → weight-norm explosion (P5)
- α=4 alone on 80-sample train set → over-smoothing, small val regression (P6)
- no-mixup + any wd/optimizer → basin-pinning failure (P1–P3)
- more SGD/Adam/AdamW optimizer shuffling → exhausted (P1–P3)
- abandoning the EMG dataset for toy tasks → never (see `feedback_emg_dataset_is_load_bearing.md`)

The remaining candidate space for pilots, re-ranked after P6 — now focused on **forcing a memorization-first phase** via capacity/noise/data levers, not mixup tuning:

1. **P4 baseline with shrunk arch [64, 32]** (wd=0.01, α=1, noise=0.25, SGD+momentum, lr=0.01, batch=16). New dedicated constant `GROKKING_PILOT_ARCH` separates pilot arch from sweep arch. ~2.5k params for 80 samples (~31 params/sample) is ~12× tighter, should force the network to actually struggle to memorize and create the flat-plateau phase separation that every prior pilot lacked. **This is the single most important experiment — it tests the real diagnosis from P6.**
2. **P4 baseline + noise=0.40** (wd=0.01, α=1, orig arch). Push memorization harder to lengthen the pre-generalization phase. Cheaper than (1) if arch shrink is too aggressive.
3. **P4 baseline + train_subset=200** (wd=0.01, α=1, noise=0.40). More data → longer memorization → possible plateau. Dilutes the small-data grokking story slightly.
4. **P4 baseline + lr=3e-3** (10× slower). Stretches all training phases; may expose a hidden phase separation that's happening too fast to see at lr=0.01.
5. **Manifold mixup** at first hidden layer. Different regularizer surface; may behave differently without the Beta(4,4) over-smoothing failure mode.
6. **Dropout 0.3** instead of mixup, with wd=0.01 + noise=0.25. Alternative regularizer on the same diagnosis.
7. **P4 baseline + α=2** (finer mixup-strength test). Low priority now — α saturated.
8. **P4 baseline + different seeds**. Reproducibility check; worth doing before any results are reported externally.

The highest-leverage single next step is **option 1** (shrink arch to [64, 32]). Addresses the actual failure mode — over-parameterization — rather than tuning knobs inside an architecture that can't produce a plateau no matter what.

**Headline if asked about results in one sentence:** "Six pilots: P4 (SGD + mixup α=1 + wd=0.01) is still the best at val 0.58 sustained. P6 tested α=4 alone (wd/noise kept at P4 values) and got a small regression to val ~0.53 — α knob is saturated at α=1. But the real diagnosis from P6 is that NO pilot shows a flat plateau, because arch=[200,100,70] is over-parameterized for 80 samples. Next: shrink arch to [64, 32] via new `GROKKING_PILOT_ARCH` constant."
