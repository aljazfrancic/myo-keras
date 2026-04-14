---
name: label-noise grokking pilot — negative result (anti-grokking decay)
description: 1-hour single-config grokking pilot on EMG with 25% label noise produced NO grokking — clean-label train acc stuck at 0.75 entire run, val acc peaks at 0.404 right after memorization and monotonically decays to 0.385 over 80k epochs
type: project
---

Pilot config (stored in `myo_utils.GROKKING_PILOT_*`, executed via `run_grok_pilot` in `grokking.py`, notebook cells 32–33):
arch=[200,100,70], lr=1e-4, wd=0.5, epochs=80,000, train_subset=40, label_noise=0.25 (10/40 labels flipped), val_subset=8000, seed=42, subsample_seed=43, rms_window=30, full-batch AdamW, `clean_train_data` tracked against original (unflipped) labels. Wall time 54m 42s.

**Result: no grokking, the opposite of grokking.**

Trajectory (from the 800 logged epochs in the notebook output):
- Memorization fast: `train_acc` (on noisy labels) hits 1.0 at **epoch 3100** and stays there.
- `clean_tr_acc` climbs to **0.75 = 30/40 by epoch 1300** and is **pinned there for the entire remaining 78,700 epochs**. The 10 flipped labels are memorized as their flipped targets; the network never "un-memorizes" them. There is no late lift-off, no grok moment, not even a wiggle — `cl` is literally constant at 0.75 for every single logged epoch from ~1.3k onward.
- `val_acc` **peaks at 0.404 at epoch 5700**, immediately after memorization, then **monotonically decays** through the rest of training: 0.400 @ 5k → 0.399 @ 10k → 0.390 @ 20k → 0.385 @ 30k → 0.384 @ 40k → 0.383 @ 50k → 0.382 @ 60k-65k (minimum 0.3815) → slight rebound to **0.3854 @ 80k**. Net change from peak: **−0.019**. Plateau std after epoch 20k is ~0.0003–0.001 — extremely tight, just slowly drifting down.
- Weight norm rises from ~24 to a peak of **32.03 around epoch 20k**, then wd-driven compression pulls it steadily down to **24.86 by ep 80k**. The val decay tracks the weight-norm compression: both fall monotonically from ep 10k to ~65k. (This is the inverse of the main sweep's weak negative correlation — here it's a clean anti-correlation in the late phase: shrinking norm → shrinking val acc.)

**Interpretation.** The pilot was designed on Omnigrok logic: add label noise to force a real memorize→generalize gap that weight decay can then close via a phase transition. It failed for a specific, informative reason:

1. The flipped-label basin is **stable under this wd**. Once the network memorizes the 10 flipped labels, weight decay is not strong enough to eject it from that basin — `clean_tr_acc` stays exactly at 0.75 for 77k epochs. For grokking to occur you need wd to eventually make the memorization solution inaccessible; here the memorization solution is robust to the chosen wd.
2. What little generalization exists is a **statistical shortcut** accumulated during the memorization transient (val 0→0.40 over the first 5k epochs tracks the train curve closely). There is no hidden "correct circuit" to grok into — val peaks at 0.404 because that's what the initial memorization-dominated solution gets on 8k held-out samples, and the later wd-driven compression *destroys* rather than reveals structure.
3. The result is the **mirror image** of the main sweep's Shape A (slow drift up): here it's slow drift **down**, because the starting point after memorization is above, not below, the wd-compressed equilibrium.

**What to tell the user if they ask about this pilot.** It is a clean negative result. The headline is: "clean_tr_acc pinned at 0.75 the entire run — the flipped labels are memorized forever — and val peaks at 0.404 right after memorization then decays to 0.385, the opposite of grokking." Do not dress it up as partial grokking; the `cl=0.75` flatline is unambiguous.

**Why:** this pilot was run in response to "if I wish to show grokking, what next steps would you suggest?" The user agreed to a 1-hour single-config label-noise attempt. The config I picked (wd=0.5, lr=1e-4, 40 samples, 25% noise) did not produce grokking. The negative result is publishable as-is in the notebook, but if the user wants to actually show grokking on this dataset, the next iteration needs different knobs.

**How to apply:** if the user wants to retry:
- The first thing to test is **higher label noise (0.4–0.6)** — with 25% noise, the memorization solution still has enough signal overlap with the true labels to give val 0.40 immediately, so there's no plateau to escape. Higher noise widens the gap.
- Second lever: **much stronger wd (1.0–2.0)** or **larger model** — wd=0.5 was not enough to eject from the flipped-label basin. Either make the basin smaller (more wd) or make the generalization basin more attractive (more params).
- Third lever: **lr decay / cosine schedule**, or **AdamW → SGD+momentum** — Omnigrok grokking often needs SGD-style dynamics, not Adam's per-parameter adaptation, to get the sharp phase transition.
- Do NOT interpret "val peaks early then decays" as "partial grokking running backward" — it's just the wd compression schedule revealing the memorization solution is fragile under decay. Report as a clean negative.
- The pilot helper infrastructure (`apply_label_noise`, `GrokLoggingCallback.clean_train_data`, `run_grok_pilot`, `plot_grok_pilot`) is reusable — only the `GROKKING_PILOT_*` constants need to change for the next iteration. Encourage parameter changes via those constants rather than code edits.
