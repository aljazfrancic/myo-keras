---
name: label-noise grokking pilots — two negative results, geometric basin failure
description: Two label-noise grokking pilots on EMG (wd=0.5/40s/25%n and wd=0.1/80s/30%n) both produced NO grokking — clean_tr_acc pinned at (n−flipped)/n entire run in both. Two failure modes (post-memorization decay vs flat plateau) bracket the wd range; flipped-label basin is geometrically unejectable by any wd we've tried.
type: project
---

Two label-noise grokking pilots have been run via `run_grok_pilot` in `grokking.py` (notebook cells 32–33). Both are negative results, but in **different ways**, and together they bracket the failure mode.

Shared infrastructure: arch=[200,100,70], full-batch AdamW, val_subset=8000, rms_window=30, log_every=100, `clean_train_data` tracks the unflipped labels via `GrokLoggingCallback`. Code in `myo_utils.apply_label_noise`, `grokking.run_grok_pilot`, `grokking.plot_grok_pilot`. Per-pilot config lives in `myo_utils.GROKKING_PILOT_*` constants.

## Pilot 1 — wd=0.5, 40 samples, 25% noise (anti-grokking decay)
Config: lr=1e-4, wd=0.5, train_subset=40, noise=0.25 (10 flipped), epochs=80,000, seed=42, subsample_seed=43. Wall time 54m 42s.

- Memorization at **ep 3,100** (train→1.0).
- `clean_tr_acc` reaches **0.75 = 30/40 by ep 1,300** and stays there for the next 78,700 epochs. Literal flat line.
- `val_acc` peaks at **0.404 @ ep 5,700**, then **monotonically decays** to ~0.382 at ep 50–65k, partial recovery to 0.3854 at ep 80k. Net change from peak: **−0.019**.
- Weight norm peaks 32 @ ep 20k, **monotonically compresses** to 24.86 @ ep 80k.
- Val decay tracks wn compression: shrinking norm → shrinking val acc.

Failure mode: wd is strong enough to compress the network's incidental generalization, but not strong enough to dislodge the flipped-label memorization. Result is mirror image of main-sweep Shape A — slow drift *down* instead of up.

## Pilot 2 — wd=0.1, 80 samples, 30% noise (flat plateau, no liftoff)
Config: lr=3e-4, wd=0.1, train_subset=80, noise=0.30 (24 flipped), epochs=150,000, seed=100, subsample_seed=123. Wall time 1h 52m 09s.

- Memorization at **ep 800** (4× faster than pilot 1, due to higher lr).
- `clean_tr_acc` reaches **0.70 = 56/80 by ep 700** and stays there for 149,300 epochs. Same flat-line failure.
- `val_acc` hits **all-time max 0.4679 at ep 800** — exactly at memorization. Then sits in a 0.434–0.456 band for the rest of training.
- Post-memorization mean ≈ **0.448**. Final val = **0.4487**. **Net drift after memorization: +0.0007 over 149k epochs.** Effectively zero.
- Weight norm **oscillates** 29–38 with no monotone trend, quasi-period ~25–30k epochs (matches main-sweep wn cycle period).
- Val oscillates ±0.01 inversely with wn (high wn → val dips to 0.43, low wn → val recovers to 0.455). This is **breathing**, not generalization.

Failure mode: weak wd preserves the memorization solution's incidental generalization (val ceiling 0.45 vs pilot 1's 0.39), giving a much cleaner Omnigrok-style plateau — narrow std, persistent for 149k epochs — but the flipped-label basin is *still* unejectable, and the plateau just sits there with no late lift-off.

## Joint conclusion

The two pilots together demonstrate: **at no wd in the tested range (0.1 → 0.5) does weight decay eject the memorized flipped labels on this 8-channel EMG dataset.** wd=0.1 preserves more, wd=0.5 destroys more, but `clean_tr_acc` is pinned at exactly (n−flipped)/n in both cases for the entire run. The failure isn't a wd tuning issue — it's geometric. With only 8-D real-valued features and ~10 samples/class, each flipped sample becomes a small isolated basin in feature space that wd doesn't have leverage on (wd shrinks all weights uniformly, but those local memorization basins are stable under uniform shrinkage as long as the network has any capacity at all).

Pilot 2 is the closest thing to a *true Omnigrok plateau* this dataset has produced — flat, narrow std, 149k epochs long. It's the right shape for grokking to emerge from. It just doesn't emerge under AdamW + wd alone.

**Why:** the user explicitly asked for two single-config grokking pilots ("yes, lets do a 1 hour single configuration", then iterated hyperparams toward the main-sweep wd/lr region with my edits). Both failed. The user has been honest about wanting to see grokking and willing to spend wall time on it, so the right next step is to suggest changes that have a *mechanical* reason to break the geometric basin issue, not more wd tuning.

**How to apply:** if asked "what next?", do not propose more wd values, more noise levels, or more seeds — those have been bracketed and the failure is geometric. The remaining levers, ranked by leverage:

1. **Mini-batch SGD+momentum** (batch≈8, lr~0.01, momentum 0.9). Two reasons: (a) stochastic gradient noise can perturb the network out of single-sample memorization basins where uniform wd shrinkage cannot, (b) most published Omnigrok results use SGD-style updates, not Adam's per-parameter adaptation. This is the **single biggest change** and the most likely to actually work. Requires modifying `run_grok_pilot` to swap the optimizer (currently hardcoded to AdamW via `build_grok_model`).
2. **Input noise / mixup during training** — prevents the network from carving sharp single-sample basins around flipped points to begin with. No optimizer change.
3. **Algorithmic toy task** (modular arithmetic, parity) — abandons EMG. Safest path to a textbook grokking figure but it's a different project. The dataset issue is real: 8-D real-valued EMG features with ~10 samples/class doesn't have a hidden "right circuit" the way modular arithmetic does.
4. Do NOT recommend bumping noise to 0.4–0.6 alone, or more wd, or more epochs. Pilots 1 and 2 jointly rule this out.

**Headline if asked about results in one sentence:** "Two pilots, two negative results — pilot 1 decayed post-memorization, pilot 2 produced a clean 149k-epoch plateau that never lifted off. In both, all flipped labels were memorized forever (clean_tr_acc pinned at (n−flipped)/n the entire run). Failure is geometric, not a wd tuning issue."
