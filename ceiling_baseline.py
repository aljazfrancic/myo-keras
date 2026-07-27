"""Ceiling baseline — what val/test accuracy is *actually achievable* on this 8-class EMG task
with a properly-trained model? This gates the grokking interpretation: if a good model on the full
data lands near the grok's number, the grok is near the task ceiling; if it lands far above, the
grok's ~0.60 is a small-sample ceiling rather than a task ceiling.

Read the two metrics side by side, because the class distribution decides which one means what:
the full splits are ~56% `hibernation`, so plain accuracy there has a majority-class floor of 0.56,
not 0.125. The grok is scored on a class-BALANCED 8000-row val subsample (1000/class, floor 0.125),
so plain accuracy on the full splits is NOT comparable to it. Two like-for-like numbers are
reported alongside it:
  * balanced accuracy (macro recall) on the full splits, and
  * accuracy on the grok's own balanced 8000-row val subsample. Note this subsample is drawn from
    the SAME -2 split protocol (A) early-stops on, so it is mildly optimistic (~1e-3: a max over
    <=300 epochs on 479,667 rows). The strictly held-out ceiling is the balanced TEST number.

Two protocols, curated 5 participants, RMS window 30 (same features the grok saw; input is 8-dim):
  (A) within-subject cross-session: train -1, early-stop on -2 (val), report best-val(-2) and held-out test(-3)
  (B) leave-one-subject-out (5-fold): test one held-out participant, early-stop on a second, train
      on the remaining three; mean +/- std  (the strongest protocol)
Protocol (B) holds a whole *participant* out of the training pool for early stopping rather than a
random row split: adjacent RMS rows overlap by window-1, so a shuffled row split would put
near-duplicate rows on both sides of the early-stopping criterion. Two consequences worth knowing:
each fold trains on 3 participants rather than 4, and the deterministic `pool[-1]` choice makes
78945 the val subject in 4 of the 5 folds. Measured cost of both: none visible — plain LOSO test
came out 0.6502 +/- 0.0768 against 0.65 +/- 0.07 under the old shuffled split.

The LOSO result only makes sense in the balanced column: 0.650 plain sits barely above the 0.564
majority-class floor, while balanced accuracy is 0.358 +/- 0.173 and the hold-12345 fold scores
0.1224 -- the chance line -- while still reporting 0.5455 plain. Cross-subject transfer here is weak
and highly subject-dependent, and the plain number hides that completely.
Model: MLP [256,128,64] + dropout, z-scored inputs, Adam, early stopping (a genuinely good supervised run).
"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import recall_score
from myo_utils import (load_data_curated, get_values, split_features_labels, subsample_data,
                       NUM_GESTURES, CURATED_FILE, READINGS_DIR, GROKKING_VAL_SUBSET)

RMS_W = 30
LOG = "ceiling_out.log"
# Protocol (B) budget. Capped well below (A)'s 300/25 because the participant-disjoint val split
# makes cross-subject val accuracy noisy enough that patience rarely fires — see train_eval. (A)
# converges by epoch ~59 on an easier val split, so 120 leaves ample headroom at ~1/3 the wall time.
LOSO_EPOCHS = 120
LOSO_PATIENCE = 15

def logline(s):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")

open(LOG, "w", encoding="utf-8").close()

def zfit(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return mu, sd

def make_model(in_dim, hidden=(256, 128, 64), dropout=0.3, lr=1e-3):
    m = keras.Sequential([keras.layers.Input((in_dim,))])
    for h in hidden:
        m.add(keras.layers.Dense(h, activation="relu"))
        m.add(keras.layers.Dropout(dropout))
    m.add(keras.layers.Dense(NUM_GESTURES, activation="softmax"))
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def train_eval(Xtr, ytr, Xva, yva, Xte, yte, tag, epochs=300, patience=25, seed=0, extra=None):
    """Fit with early stopping; report plain and balanced accuracy on the held-out test set.

    *extra* is an optional list of (name, X, y) evaluated with the same fitted model — used to
    score protocol (A) on the grok's own balanced val subsample, the only like-for-like comparison.

    Protocol (B) passes a tighter budget than (A) because its epoch cap is load-bearing. With a
    participant-disjoint validation split, cross-subject val accuracy is noisy, and at patience=25
    the folds ran past 22 min apiece without stopping (~100 min each at the 300 cap, ~8 h for five).
    At 120/15 they stop at epochs 16-34 and the whole script finishes in well under an hour, with
    the same plain-accuracy mean. (A), whose val split is same-subject and therefore easy,
    converges by epoch ~59.
    """
    keras.utils.set_random_seed(seed)
    mu, sd = zfit(Xtr)
    Xtr_, Xva_, Xte_ = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    m = make_model(Xtr.shape[1])
    es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience,
                                       restore_best_weights=True)

    class Progress(keras.callbacks.Callback):
        """These fits take tens of minutes; verbose=0 with no ticks is unreadable from a log."""
        def __init__(self):
            super().__init__()
            self.t0 = time.time()
        def on_epoch_end(self, ep, logs=None):
            if (ep + 1) % 10 == 0:
                logline(f"      [{tag}] ep {ep + 1}/{epochs}  val_acc {logs['val_accuracy']:.4f}"
                        f"  ({(time.time() - self.t0) / 60:.1f} min)")

    h = m.fit(Xtr_, ytr.astype(int), validation_data=(Xva_, yva.astype(int)),
              epochs=epochs, batch_size=128, callbacks=[es, Progress()], verbose=0)
    best_val = float(np.max(h.history["val_accuracy"]))
    te_acc = float(m.evaluate(Xte_, yte.astype(int), verbose=0)[1])
    te_bal = float(recall_score(yte.astype(int), np.argmax(m.predict(Xte_, verbose=0), axis=1),
                                average="macro"))
    logline(f"  [{tag}] best_val={best_val:.4f}  test={te_acc:.4f}  test_balanced={te_bal:.4f}"
            f"  (stopped ep {len(h.history['val_accuracy'])})")
    extras = {}
    for name, Xe, ye in (extra or []):
        acc = float(m.evaluate((Xe - mu) / sd, ye.astype(int), verbose=0)[1])
        extras[name] = acc
        logline(f"      {name}: acc={acc:.4f}")
    return best_val, te_acc, te_bal, extras

logline("CEILING BASELINE | RMS window=30 | input=8-dim")
logline("chance=0.125 | grok val (balanced 8000, floor 0.125) = 0.600 final / 0.619 raw peak"
        " | vanilla net on the same balanced subset = 0.574 (its own trajectory max)")

# ---- (A) within-subject cross-session ----
logline("\n[A] WITHIN-SUBJECT CROSS-SESSION  (train -1, val -2, test -3) -----------------------")
tr_x, tr_y, va_x, va_y, te_x, te_y = load_data_curated(rms_window=RMS_W)
maj_va = float(np.bincount(va_y.astype(int)).max() / len(va_y))
maj_te = float(np.bincount(te_y.astype(int)).max() / len(te_y))
logline(f"  sizes: train={len(tr_x)} val={len(va_x)} test={len(te_x)} feat_dim={tr_x.shape[1]}")
logline(f"  majority-class floor on the FULL splits: val={maj_va:.4f} test={maj_te:.4f}"
        f"  (vs 0.1250 on the grok's balanced subsample)")

# the grok's own validation set: subsample_data(val, 8000, seed=42) -> exactly 1000 rows/class
gv_x, gv_y = subsample_data(va_x, va_y, GROKKING_VAL_SUBSET, seed=42)
A_best_val, A_test, A_test_bal, A_extra = train_eval(
    tr_x, tr_y, va_x, va_y, te_x, te_y, "within-subj", seed=0,
    extra=[("grok's balanced val subsample (n=8000, 1000/class)", gv_x, gv_y)],
)
A_grok_val = A_extra["grok's balanced val subsample (n=8000, 1000/class)"]

# ---- (B) leave-one-subject-out ----
logline("\n[B] LEAVE-ONE-SUBJECT-OUT (5-fold cross-subject) -----------------------------------")
with open(CURATED_FILE) as f:
    names = [l.strip() for l in f if l.strip()]
pids = sorted({n.rsplit("-", 1)[0] for n in names})
logline(f"  participants: {pids}")
logline("  early-stopping val = one held-out *participant* from the training pool"
        " (not a shuffled row split: adjacent RMS rows overlap by window-1)")

def pid_dirs(pid):
    return [os.path.join(READINGS_DIR, f"{pid}-{i}") for i in (1, 2, 3)]

def load_dirs(dirs):
    return split_features_labels(get_values(dirs, verbose=False, rms_window=RMS_W))

loso, loso_bal = [], []
for held in pids:
    pool = [p for p in pids if p != held]
    val_pid, train_pids = pool[-1], pool[:-1]   # deterministic: last of the remaining four
    Xtr, ytr = load_dirs([d for p in train_pids for d in pid_dirs(p)])
    Xva, yva = load_dirs(pid_dirs(val_pid))
    Xte, yte = load_dirs(pid_dirs(held))
    _, ta, tb, _ = train_eval(Xtr, ytr, Xva, yva, Xte, yte,
                              f"LOSO/hold-{held} (val={val_pid})", seed=0,
                              epochs=LOSO_EPOCHS, patience=LOSO_PATIENCE)
    loso.append(ta)
    loso_bal.append(tb)
loso, loso_bal = np.array(loso), np.array(loso_bal)
logline(f"\n  LOSO test mean={loso.mean():.4f} std={loso.std():.4f} folds={[round(float(x),4) for x in loso]}")
logline(f"  LOSO balanced  mean={loso_bal.mean():.4f} std={loso_bal.std():.4f} "
        f"folds={[round(float(x),4) for x in loso_bal]}")

logline("\n===== CEILING SUMMARY =====")
logline(f"[A] within-subject cross-session (FULL splits, majority floor {maj_te:.2f}): "
        f"best_val(-2)={A_best_val:.4f}  test(-3)={A_test:.4f}")
logline(f"[A] balanced (floor 0.125): test balanced accuracy={A_test_bal:.4f}  |  "
        f"same model on the grok's balanced 8000-row val subsample={A_grok_val:.4f}")
logline(f"[B] LOSO cross-subject: test={loso.mean():.4f} +/- {loso.std():.4f}  "
        f"balanced={loso_bal.mean():.4f} +/- {loso_bal.std():.4f}")
logline("Compare the grok's 0.60/0.62 against the BALANCED numbers, not the full-split accuracy.")
logline("DONE")
