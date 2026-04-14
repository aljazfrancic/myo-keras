---
name: never suggest abandoning the EMG dataset for an algorithmic toy task
description: The whole point of this grokking project is to demonstrate the phenomenon on the user's real EMG gesture dataset. Suggesting modular arithmetic, parity, or any toy task as an "easier path" defeats the experiment.
type: feedback
---

When grokking experiments on the EMG dataset fail or stall, **do not** suggest switching to an algorithmic toy task (modular arithmetic, parity, MNIST-style, etc.) as a "safer path to a textbook grokking figure". The point of the experiment is to investigate grokking *on this specific 8-channel real-world EMG gesture dataset*. A toy-task result is irrelevant to that goal.

**Why:** the user pushed back hard after I included "drop to algorithmic toy task" as option 3 in a ranked list of next steps. Their words: "dont ever suggest dropping this dataset for a algoritmic toy task; that foregoes the whole point of the exepriment". This is a foundational scope decision, not a momentary preference — the EMG dataset is the experiment, not a vehicle for it.

**How to apply:** when proposing next steps for grokking work in this repo, the candidate space is restricted to changes that keep the EMG dataset, the curated sessions, and the gesture classification task. Allowed levers: optimizer (SGD/momentum/Lion/Lamb), regularization (dropout, mixup, input noise, gradient noise), schedule (lr decay, cosine, warmup), architecture (depth/width/activations), data preprocessing (rms_window, normalization), training regime (batch size, full-batch vs mini-batch, train_subset size, label noise level). Forbidden: switching to any task other than the EMG gesture classification, even briefly, even as a "sanity check" or "calibration run". If a config genuinely cannot produce grokking on this dataset, say so honestly and stop — do not redirect to a different problem.
