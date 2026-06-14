"""Ceiling baseline — what val/test accuracy is *actually achievable* on this 8-class EMG task
with a properly-trained model? This gates the grokking interpretation:
  ceiling ~0.62-0.65  -> task is hard, grok's 0.62 is near-Bayes (phenomenology claim live)
  ceiling ~0.85       -> grok is wandering, not converging (shelve the write-up)

Two protocols, curated 5 participants, RMS window 30 (same features the grok saw; input is 8-dim):
  (A) within-subject cross-session: train -1, early-stop on -2 (val), report best-val(-2) and held-out test(-3)
  (B) leave-one-subject-out (5-fold): train 4 participants, test the 5th; mean +/- std  (the strongest protocol)
Model: MLP [256,128,64] + dropout, z-scored inputs, Adam, early stopping (a genuinely good supervised run).
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from myo_utils import (load_data_curated, get_values, split_features_labels,
                       NUM_GESTURES, CURATED_FILE, READINGS_DIR)

RMS_W = 30
LOG = "ceiling_out.log"

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

def train_eval(Xtr, ytr, Xva, yva, Xte, yte, tag, epochs=300, patience=25, seed=0):
    keras.utils.set_random_seed(seed)
    mu, sd = zfit(Xtr)
    Xtr_, Xva_, Xte_ = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    m = make_model(Xtr.shape[1])
    es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience,
                                       restore_best_weights=True)
    h = m.fit(Xtr_, ytr.astype(int), validation_data=(Xva_, yva.astype(int)),
              epochs=epochs, batch_size=128, callbacks=[es], verbose=0)
    best_val = float(np.max(h.history["val_accuracy"]))
    te_acc = float(m.evaluate(Xte_, yte.astype(int), verbose=0)[1])
    logline(f"  [{tag}] best_val={best_val:.4f}  test={te_acc:.4f}  (stopped ep {len(h.history['val_accuracy'])})")
    return best_val, te_acc

logline("CEILING BASELINE | chance=0.125 | grok=0.62 | vanilla-early-stop~0.58 | RMS window=30 | input=8-dim")

# ---- (A) within-subject cross-session ----
logline("\n[A] WITHIN-SUBJECT CROSS-SESSION  (train -1, val -2, test -3) -----------------------")
tr_x, tr_y, va_x, va_y, te_x, te_y = load_data_curated(rms_window=RMS_W)
logline(f"  sizes: train={len(tr_x)} val={len(va_x)} test={len(te_x)} feat_dim={tr_x.shape[1]}")
A_best_val, A_test = train_eval(tr_x, tr_y, va_x, va_y, te_x, te_y, "within-subj", seed=0)

# ---- (B) leave-one-subject-out ----
logline("\n[B] LEAVE-ONE-SUBJECT-OUT (5-fold cross-subject) -----------------------------------")
with open(CURATED_FILE) as f:
    names = [l.strip() for l in f if l.strip()]
pids = sorted({n.rsplit("-", 1)[0] for n in names})
logline(f"  participants: {pids}")

def pid_dirs(pid):
    return [os.path.join(READINGS_DIR, f"{pid}-{i}") for i in (1, 2, 3)]

def load_dirs(dirs):
    return split_features_labels(get_values(dirs, verbose=False, rms_window=RMS_W))

loso = []
rng = np.random.default_rng(0)
for held in pids:
    tr_dirs = [d for p in pids if p != held for d in pid_dirs(p)]
    Xtr_all, ytr_all = load_dirs(tr_dirs)
    Xte, yte = load_dirs(pid_dirs(held))
    idx = rng.permutation(len(Xtr_all)); cut = int(0.85 * len(idx))
    bv, ta = train_eval(Xtr_all[idx[:cut]], ytr_all[idx[:cut]],
                        Xtr_all[idx[cut:]], ytr_all[idx[cut:]],
                        Xte, yte, f"LOSO/hold-{held}", seed=0)
    loso.append(ta)
loso = np.array(loso)
logline(f"\n  LOSO test mean={loso.mean():.4f} std={loso.std():.4f} folds={[round(float(x),4) for x in loso]}")

logline("\n===== CEILING SUMMARY =====")
logline("chance=0.125 | grok=0.62 | vanilla-early-stop~0.58")
logline(f"[A] within-subject cross-session: best_val(-2)={A_best_val:.4f}  test(-3)={A_test:.4f}")
logline(f"[B] LOSO cross-subject: test={loso.mean():.4f} +/- {loso.std():.4f}")
logline("DONE")
