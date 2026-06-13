"""Grokking sweep: callback, model build, sweep runner, and matplotlib plot helpers."""

from __future__ import annotations

import time
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from myo_utils import (
    FIGURE_DPI,
    GROKKING_ARCHITECTURES,
    GROKKING_EPOCHS,
    GROKKING_LOG_EVERY,
    GROKKING_LRS,
    GROKKING_PILOT_ARCH,
    GROKKING_PILOT_BATCH_SIZE,
    GROKKING_PILOT_CONFIGS,
    GROKKING_PILOT_EPOCHS,
    GROKKING_PILOT_INIT_SCALE,
    GROKKING_PILOT_LABEL_NOISE,
    GROKKING_PILOT_LOG_EVERY,
    GROKKING_PILOT_LR,
    GROKKING_PILOT_MIXUP_ALPHA,
    GROKKING_PILOT_MOMENTUM,
    GROKKING_PILOT_NESTEROV,
    GROKKING_PILOT_OPTIMIZER,
    GROKKING_PILOT_SEED,
    GROKKING_PILOT_SUBSAMPLE_SEED,
    GROKKING_PILOT_TRAIN_SUBSET,
    GROKKING_PILOT_WD,
    GROKKING_RMS_WINDOW,
    GROKKING_SEEDS,
    GROKKING_VAL_SUBSET,
    GROKKING_WEIGHT_DECAYS,
    NUM_EMG_CHANNELS,
    NUM_GESTURES,
    apply_label_noise,
    load_data_curated,
    subsample_data,
)


def format_elapsed(seconds: float) -> str:
    seconds = int(round(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def sweep_key(arch: Union[List[int], Tuple[int, ...]], lr: float, wd: float, seed: int):
    """Canonical dict key for one sweep run (matches training and plotting)."""
    return (tuple(arch), lr, wd, seed)


def load_curated_for_grok(rms_window: Optional[int] = None):
    """Load curated splits for grokking; returns train/val features and labels only."""
    w = GROKKING_RMS_WINDOW if rms_window is None else rms_window
    train_x, train_y, val_x, val_y, _test_x, _test_y = load_data_curated(rms_window=w)
    return train_x, train_y, val_x, val_y


class GrokLoggingCallback(keras.callbacks.Callback):
    """Logs train metrics + periodic validation metrics + weight norm.

    If clean_train_data is provided, also tracks train loss/accuracy against the
    CLEAN (unflipped) labels on each logging tick. This is the key signal for
    label-noise grokking: clean-label loss drops *after* noisy-label loss
    saturates, and its onset is the moment the network stops fitting the noise.
    """

    def __init__(
        self,
        val_data,
        log_every: int = GROKKING_LOG_EVERY,
        clean_train_data=None,
    ):
        super().__init__()
        self.val_features, self.val_labels = val_data
        self.log_every = log_every
        self.clean_train_data = clean_train_data
        self.epochs: List[int] = []
        self.train_loss: List[float] = []
        self.train_accuracy: List[float] = []
        self.val_loss: List[float] = []
        self.val_accuracy: List[float] = []
        self.weight_norms: List[float] = []
        self.clean_train_loss: List[float] = []
        self.clean_train_accuracy: List[float] = []

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        if ep % self.log_every != 0 and ep != self.params["epochs"]:
            return

        self.epochs.append(ep)
        self.train_loss.append(float(logs["loss"]))
        self.train_accuracy.append(float(logs["accuracy"]))

        val_loss, val_acc = self.model.evaluate(self.val_features, self.val_labels, verbose=0)
        self.val_loss.append(float(val_loss))
        self.val_accuracy.append(float(val_acc))

        weight_norm = np.sqrt(
            sum(float(np.sum(w.numpy() ** 2)) for w in self.model.trainable_weights)
        )
        self.weight_norms.append(float(weight_norm))

        extra = ""
        if self.clean_train_data is not None:
            cf, cl = self.clean_train_data
            cl_loss, cl_acc = self.model.evaluate(cf, cl, verbose=0)
            self.clean_train_loss.append(float(cl_loss))
            self.clean_train_accuracy.append(float(cl_acc))
            extra = f"  clean_tr_acc={cl_acc:.4f}"

        print(
            f"Epoch {ep:>6d}  "
            f"train_acc={logs['accuracy']:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"wnorm={weight_norm:.2f}{extra}"
        )


def build_grok_model(
    arch,
    lr: float,
    wd: float,
    *,
    optimizer: str = "adamw",
    momentum: float = 0.9,
    nesterov: bool = True,
    loss: Optional[Any] = None,
    init_scale: float = 1.0,
):
    layers = [keras.layers.Input(shape=(NUM_EMG_CHANNELS,))]
    for units in arch:
        layers.append(keras.layers.Dense(units, activation="relu"))
    layers.append(keras.layers.Dense(NUM_GESTURES, activation="softmax"))
    model = keras.Sequential(layers)
    if init_scale != 1.0:
        # Omnigrok large-norm init: scale each Dense kernel (not the zero-init bias) so the
        # network starts far OUTSIDE the generalizing weight-norm "Goldilocks zone" and must
        # compress into it under weight decay — the mechanism that produces a sharp memorize->
        # generalize transition on non-algorithmic data. Glorot init is seeded upstream
        # (tf.random.set_seed before this call), so this scaling is deterministic.
        for lyr in model.layers:
            w = lyr.get_weights()
            if w:  # Dense -> [kernel, bias]; Input has no weights -> []
                w[0] = w[0] * init_scale
                lyr.set_weights(w)
    if optimizer == "adamw":
        opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    elif optimizer == "sgd":
        opt = keras.optimizers.SGD(
            learning_rate=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=wd,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r} (expected 'adamw' or 'sgd')")
    if loss is None:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def make_mixup_dataset(x, y_onehot, batch_size: int, alpha: float, seed: int):
    """tf.data pipeline that yields mixup-augmented mini-batches.

    Within each batch, sample lam ~ Beta(alpha, alpha), shuffle the batch to
    pair each sample with another, and emit (lam*x + (1-lam)*x_perm,
    lam*y + (1-lam)*y_perm). One lam per batch (standard mixup recipe).
    Reshuffles between epochs so different pairings are seen each pass.
    """
    n = len(x)
    ds = tf.data.Dataset.from_tensor_slices((x.astype(np.float32), y_onehot.astype(np.float32)))
    ds = ds.shuffle(n, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)

    alpha_t = tf.constant(alpha, dtype=tf.float32)

    def _mixup(xb, yb):
        bs = tf.shape(xb)[0]
        # Sample lam ~ Beta(alpha, alpha) via two Gammas (no tfp dependency).
        g1 = tf.random.gamma([], alpha=alpha_t)
        g2 = tf.random.gamma([], alpha=alpha_t)
        lam = g1 / (g1 + g2)
        idx = tf.random.shuffle(tf.range(bs))
        xb_perm = tf.gather(xb, idx)
        yb_perm = tf.gather(yb, idx)
        xb_mix = lam * xb + (1.0 - lam) * xb_perm
        yb_mix = lam * yb + (1.0 - lam) * yb_perm
        return xb_mix, yb_mix

    ds = ds.map(_mixup, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def run_grok_sweep(
    grok_train,
    grok_train_labels,
    grok_valid,
    grok_valid_labels,
    *,
    architectures=None,
    lrs=None,
    weight_decays=None,
    seeds=None,
    epochs=None,
    log_every=None,
) -> Tuple[Dict[Tuple, Dict[str, Any]], List[Tuple[Tuple, float]], float]:
    architectures = architectures if architectures is not None else GROKKING_ARCHITECTURES
    lrs = lrs if lrs is not None else GROKKING_LRS
    weight_decays = weight_decays if weight_decays is not None else GROKKING_WEIGHT_DECAYS
    seeds = seeds if seeds is not None else GROKKING_SEEDS
    epochs = epochs if epochs is not None else GROKKING_EPOCHS
    log_every = log_every if log_every is not None else GROKKING_LOG_EVERY

    sweep_results: Dict[Tuple, Dict[str, Any]] = {}
    run_specs = list(product(architectures, lrs, weight_decays, seeds))
    n_total = len(run_specs)
    run_timings: List[Tuple[Tuple, float]] = []

    t_sweep0 = time.perf_counter()
    for run_idx, (arch, lr, wd, seed) in enumerate(run_specs, start=1):
        arch_t = tuple(arch)
        key = sweep_key(arch, lr, wd, seed)
        print(f"\n{'='*60}")
        print(f"  Run {run_idx}/{n_total}  arch={list(arch_t)}  lr={lr:g}  wd={wd}  seed={seed}")
        print(f"{'='*60}")

        tf.random.set_seed(seed)
        grok_model = build_grok_model(arch, lr, wd)
        grok_cb = GrokLoggingCallback(val_data=(grok_valid, grok_valid_labels), log_every=log_every)
        t0 = time.perf_counter()
        grok_model.fit(
            grok_train,
            grok_train_labels,
            epochs=epochs,
            batch_size=len(grok_train),
            callbacks=[grok_cb],
            verbose=0,
        )
        elapsed = time.perf_counter() - t0
        run_timings.append((key, elapsed))
        print(f"  >>> Run wall time: {format_elapsed(elapsed)}")

        sweep_results[key] = {
            "arch": arch_t,
            "lr": lr,
            "wd": wd,
            "seed": seed,
            "epochs": grok_cb.epochs,
            "train_loss": grok_cb.train_loss,
            "train_accuracy": grok_cb.train_accuracy,
            "val_loss": grok_cb.val_loss,
            "val_accuracy": grok_cb.val_accuracy,
            "weight_norms": grok_cb.weight_norms,
            "model": grok_model,
        }

    sweep_total = time.perf_counter() - t_sweep0
    print(f"\n{'='*60}")
    print(f"  Full sweep wall time: {format_elapsed(sweep_total)}")
    print(f"{'='*60}\n")
    print("Per-run timings:")
    for key, sec in run_timings:
        arch_t, lr, wd, seed = key
        print(
            f"  arch={list(arch_t)}  lr={lr:g}  wd={wd}  seed={seed}  ->  {format_elapsed(sec)}"
        )
    return sweep_results, run_timings, sweep_total


def iter_grok_overlay_specs(
    architectures=None,
    lrs=None,
    weight_decays=None,
) -> Iterable[Tuple[Any, float, float]]:
    architectures = architectures if architectures is not None else GROKKING_ARCHITECTURES
    lrs = lrs if lrs is not None else GROKKING_LRS
    weight_decays = weight_decays if weight_decays is not None else GROKKING_WEIGHT_DECAYS
    return product(architectures, lrs, weight_decays)


def iter_grok_run_specs(
    architectures=None,
    lrs=None,
    weight_decays=None,
    seeds=None,
) -> Iterable[Tuple[Any, float, float, int]]:
    architectures = architectures if architectures is not None else GROKKING_ARCHITECTURES
    lrs = lrs if lrs is not None else GROKKING_LRS
    weight_decays = weight_decays if weight_decays is not None else GROKKING_WEIGHT_DECAYS
    seeds = seeds if seeds is not None else GROKKING_SEEDS
    return product(architectures, lrs, weight_decays, seeds)


def default_overlay_title(arch, lr: float, wd: float) -> str:
    return f"Grok sweep: arch={list(arch)}  lr={lr:g}  wd={wd}"


def plot_grok_seed_overlay(
    sweep_results,
    arch,
    lr: float,
    wd: float,
    *,
    y_metric_key: str,
    ylabel: str,
    title_fn: Optional[Callable[[Any, float, float], str]] = None,
    seeds=None,
    figsize=(10, 6),
):
    """One figure: overlay one curve per seed for fixed (arch, lr, wd)."""
    import matplotlib.pyplot as plt

    seeds = seeds if seeds is not None else GROKKING_SEEDS
    title_fn = title_fn or default_overlay_title
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    for seed in seeds:
        key = sweep_key(arch, lr, wd, seed)
        res = sweep_results[key]
        ax.plot(res["epochs"], res[y_metric_key], label=f"seed={seed}", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title_fn(arch, lr, wd))
    ax.legend(fontsize=8, title="seed")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_grok_run_loss_accuracy(sweep_results, arch, lr: float, wd: float, seed: int):
    """Two subplots: train vs val loss and accuracy for one run."""
    import matplotlib.pyplot as plt

    res = sweep_results[sweep_key(arch, lr, wd, seed)]
    x = res["epochs"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4.5), dpi=FIGURE_DPI)
    ax0.plot(x, res["train_loss"], label="train loss", alpha=0.9)
    ax0.plot(x, res["val_loss"], label="val loss", alpha=0.9)
    ax1.plot(x, res["train_accuracy"], label="train accuracy", alpha=0.9)
    ax1.plot(x, res["val_accuracy"], label="val accuracy", alpha=0.9)
    ax0.set_title("Loss")
    ax1.set_title("Accuracy")
    ax0.set_xlabel("Epoch")
    ax1.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax1.set_ylabel("Accuracy")
    ax0.legend(loc="best")
    ax1.legend(loc="best")
    ax0.grid(True, alpha=0.3)
    ax1.grid(True, alpha=0.3)
    fig.suptitle(
        f"arch={list(arch)}  lr={lr:g}  wd={wd}  seed={seed}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    return fig, (ax0, ax1)


def plot_grok_run_valacc_weight_norm(sweep_results, arch, lr: float, wd: float, seed: int):
    """Dual-axis: validation accuracy and L2 weight norm vs epoch."""
    import matplotlib.pyplot as plt

    res = sweep_results[sweep_key(arch, lr, wd, seed)]
    x = res["epochs"]
    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=FIGURE_DPI)
    ax2 = ax1.twinx()
    ax1.plot(
        x,
        res["val_accuracy"],
        color="C0",
        linestyle="-",
        linewidth=1.4,
        label="val accuracy",
        alpha=0.9,
    )
    ax2.plot(
        x,
        res["weight_norms"],
        color="orange",
        linestyle="--",
        linewidth=1.4,
        label="L2 weight norm",
        alpha=0.9,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation accuracy")
    ax2.set_ylabel("L2 weight norm")
    ax1.set_title(f"arch={list(arch)}  lr={lr:g}  wd={wd}  seed={seed}")
    ax1.grid(True, alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="best")
    plt.tight_layout()
    return fig, (ax1, ax2)


def run_grok_pilot(
    *,
    arch: Optional[List[int]] = None,
    lr: Optional[float] = None,
    wd: Optional[float] = None,
    seed: Optional[int] = None,
    epochs: Optional[int] = None,
    log_every: Optional[int] = None,
    train_subset: Optional[int] = None,
    val_subset: Optional[int] = None,
    label_noise: Optional[float] = None,
    subsample_seed: Optional[int] = None,
    rms_window: Optional[int] = None,
    optimizer: Optional[str] = None,
    momentum: Optional[float] = None,
    nesterov: Optional[bool] = None,
    batch_size: Optional[int] = None,
    mixup_alpha: Optional[float] = None,
    init_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """Single-run grokking pilot with label noise.

    Widens the memorize->generalize gap by flipping a fraction of training
    labels: the network must memorize noise first, which blocks statistical
    shortcuts and creates a real plateau to grok out of. Tracks both
    noisy-label (optimizer target) and clean-label (ground-truth) metrics so
    the grokking onset is visible as clean-label accuracy lifting off while
    noisy-label accuracy stays pinned at 1.0.
    """
    arch = arch if arch is not None else GROKKING_PILOT_ARCH
    lr = lr if lr is not None else GROKKING_PILOT_LR
    wd = wd if wd is not None else GROKKING_PILOT_WD
    seed = seed if seed is not None else GROKKING_PILOT_SEED
    epochs = epochs if epochs is not None else GROKKING_PILOT_EPOCHS
    log_every = log_every if log_every is not None else GROKKING_PILOT_LOG_EVERY
    train_subset = train_subset if train_subset is not None else GROKKING_PILOT_TRAIN_SUBSET
    val_subset = val_subset if val_subset is not None else GROKKING_VAL_SUBSET
    label_noise = label_noise if label_noise is not None else GROKKING_PILOT_LABEL_NOISE
    subsample_seed = subsample_seed if subsample_seed is not None else GROKKING_PILOT_SUBSAMPLE_SEED
    rms_window = rms_window if rms_window is not None else GROKKING_RMS_WINDOW
    optimizer = optimizer if optimizer is not None else GROKKING_PILOT_OPTIMIZER
    momentum = momentum if momentum is not None else GROKKING_PILOT_MOMENTUM
    nesterov = nesterov if nesterov is not None else GROKKING_PILOT_NESTEROV
    batch_size = batch_size if batch_size is not None else GROKKING_PILOT_BATCH_SIZE
    mixup_alpha = mixup_alpha if mixup_alpha is not None else GROKKING_PILOT_MIXUP_ALPHA
    init_scale = init_scale if init_scale is not None else GROKKING_PILOT_INIT_SCALE

    print(f"Loading curated data (rms_window={rms_window})...")
    train_x_full, train_y_full, val_x_full, val_y_full = load_curated_for_grok(rms_window=rms_window)

    clean_tr_x, clean_tr_y = subsample_data(
        train_x_full, train_y_full, train_subset, seed=subsample_seed
    )
    noisy_tr_y = apply_label_noise(clean_tr_y, label_noise, seed=seed)
    n_flipped = int(np.sum(noisy_tr_y != clean_tr_y))
    v_x, v_y = subsample_data(val_x_full, val_y_full, val_subset, seed=42)

    effective_batch_size = batch_size if batch_size is not None else len(clean_tr_x)
    steps_per_epoch = max(1, int(np.ceil(len(clean_tr_x) / effective_batch_size)))
    use_mixup = mixup_alpha is not None and mixup_alpha > 0
    if optimizer == "sgd":
        opt_str = f"sgd(momentum={momentum}, nesterov={nesterov})"
    else:
        opt_str = optimizer
    mixup_str = f"mixup(alpha={mixup_alpha})" if use_mixup else "no-mixup"

    print(f"Train: {len(clean_tr_x)} samples, {n_flipped} labels flipped ({label_noise*100:.0f}% noise)")
    print(f"Val:   {len(v_x)} samples")
    init_str = f"init={init_scale:g}x" if init_scale != 1.0 else "init=1x"
    print(f"Config: arch={list(arch)}  opt={opt_str}  lr={lr:g}  wd={wd}  {init_str}  "
          f"batch={effective_batch_size} ({steps_per_epoch} step{'s' if steps_per_epoch != 1 else ''}/epoch)  "
          f"{mixup_str}  seed={seed}  epochs={epochs}")
    if init_scale != 1.0:
        print(f"Expected shape (Omnigrok large-init): train memorizes fast; val sits at a LOW plateau")
        print(f"while the weight norm (starts ~{init_scale:g}x natural) compresses under weight decay;")
        print(f"val should jump sharply as the norm enters the generalizing zone.\n")
    elif label_noise > 0:
        print(f"Expected shape: noisy-label train_acc saturates early at ~{1.0-label_noise:.2f}+")
        print(f"then climbs to 1.00 as the network memorizes noise; val_acc sits flat;")
        print(f"clean-label train_acc lifts off late as the network groks the true signal.\n")
    else:
        print(f"Expected shape: clean-label delayed-generalization drift (no sharp edge expected).\n")

    tf.random.set_seed(seed)
    if use_mixup:
        # one-hot path: categorical loss, mixup tf.data pipeline. Convert val and clean_tr
        # labels to one-hot so the callback's model.evaluate calls match the loss format.
        noisy_tr_y_oh = np.eye(NUM_GESTURES, dtype=np.float32)[noisy_tr_y.astype(np.int64)]
        clean_tr_y_oh = np.eye(NUM_GESTURES, dtype=np.float32)[clean_tr_y.astype(np.int64)]
        v_y_oh = np.eye(NUM_GESTURES, dtype=np.float32)[v_y.astype(np.int64)]
        train_ds = make_mixup_dataset(
            clean_tr_x, noisy_tr_y_oh, effective_batch_size, mixup_alpha, seed
        )
        model = build_grok_model(
            arch, lr, wd,
            optimizer=optimizer, momentum=momentum, nesterov=nesterov,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            init_scale=init_scale,
        )
        cb = GrokLoggingCallback(
            val_data=(v_x, v_y_oh),
            log_every=log_every,
            clean_train_data=(clean_tr_x, clean_tr_y_oh),
        )
        t0 = time.perf_counter()
        model.fit(
            train_ds,
            epochs=epochs,
            callbacks=[cb],
            verbose=0,
        )
    else:
        model = build_grok_model(
            arch, lr, wd, optimizer=optimizer, momentum=momentum, nesterov=nesterov,
            init_scale=init_scale,
        )
        cb = GrokLoggingCallback(
            val_data=(v_x, v_y),
            log_every=log_every,
            clean_train_data=(clean_tr_x, clean_tr_y),
        )
        t0 = time.perf_counter()
        model.fit(
            clean_tr_x,
            noisy_tr_y,
            epochs=epochs,
            batch_size=effective_batch_size,
            callbacks=[cb],
            verbose=0,
            shuffle=True,
        )
    elapsed = time.perf_counter() - t0
    print(f"\nPilot wall time: {format_elapsed(elapsed)}")

    return {
        "arch": tuple(arch),
        "lr": lr,
        "wd": wd,
        "seed": seed,
        "optimizer": optimizer,
        "momentum": momentum,
        "nesterov": nesterov,
        "batch_size": effective_batch_size,
        "mixup_alpha": mixup_alpha if use_mixup else 0.0,
        "init_scale": init_scale,
        "label_noise": label_noise,
        "train_subset": train_subset,
        "n_flipped_labels": n_flipped,
        "epochs": cb.epochs,
        "train_loss": cb.train_loss,
        "train_accuracy": cb.train_accuracy,
        "val_loss": cb.val_loss,
        "val_accuracy": cb.val_accuracy,
        "weight_norms": cb.weight_norms,
        "clean_train_loss": cb.clean_train_loss,
        "clean_train_accuracy": cb.clean_train_accuracy,
        "elapsed": elapsed,
        "model": model,
    }


def run_grok_pilots(configs=None) -> List[Dict[str, Any]]:
    """Run several grokking pilots sequentially — one full training job per config dict.

    Each entry in *configs* is a dict of keyword overrides forwarded to ``run_grok_pilot``
    (e.g. ``{"init_scale": 10.0, "wd": 0.15, "epochs": 220_000}``); any key not given falls
    back to the ``GROKKING_PILOT_*`` defaults. Returns the list of per-run result dicts, in
    order. Each result already carries its own ``init_scale``/``wd``/``epochs`` so
    ``plot_grok_pilot`` labels every figure with the config that produced it.
    """
    configs = configs if configs is not None else GROKKING_PILOT_CONFIGS
    n = len(configs)
    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, cfg in enumerate(configs, start=1):
        print(f"\n{'#'*70}")
        print(f"# PILOT {i}/{n}: {cfg}")
        print(f"{'#'*70}")
        results.append(run_grok_pilot(**cfg))
    print(f"\nAll {n} pilots done. Total wall time: {format_elapsed(time.perf_counter() - t0)}")
    return results


def plot_grok_pilot(pilot_result: Dict[str, Any]):
    """Four-panel summary of a label-noise grokking pilot.

    Panels:
      (0) Accuracy — noisy-label train (optimizer target), clean-label train
          (ground-truth fit), val. The grokking gap is between noisy-label
          train saturation and clean-label train / val lift-off.
      (1) Loss — same three series in loss space; clean-label loss is the
          sharpest early-warning of grokking onset.
      (2) Val accuracy vs L2 weight norm (dual axis).
      (3) Clean-minus-noisy train accuracy gap — negative during memorization,
          climbs toward zero as the network stops fitting the noise. Crossing
          zero is the unambiguous grokking moment.
    """
    import matplotlib.pyplot as plt

    r = pilot_result
    x = r["epochs"]
    has_clean = len(r["clean_train_accuracy"]) > 0

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), dpi=FIGURE_DPI)
    ax0, ax1, ax2, ax3 = axes

    ax0.plot(x, r["train_accuracy"], label="train acc (noisy labels)", color="C0", alpha=0.9)
    if has_clean:
        ax0.plot(x, r["clean_train_accuracy"], label="train acc (clean labels)", color="C2", alpha=0.9)
    ax0.plot(x, r["val_accuracy"], label="val acc", color="C3", alpha=0.9)
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Accuracy")
    ax0.set_title("Accuracy")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8, loc="best")

    ax1.plot(x, r["train_loss"], label="train loss (noisy)", color="C0", alpha=0.9)
    if has_clean:
        ax1.plot(x, r["clean_train_loss"], label="train loss (clean)", color="C2", alpha=0.9)
    ax1.plot(x, r["val_loss"], label="val loss", color="C3", alpha=0.9)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="best")

    ax2b = ax2.twinx()
    ax2.plot(x, r["val_accuracy"], color="C0", linewidth=1.4, label="val acc", alpha=0.9)
    ax2b.plot(x, r["weight_norms"], color="orange", linestyle="--", linewidth=1.4, label="L2 weight norm", alpha=0.9)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val accuracy")
    ax2b.set_ylabel("L2 weight norm")
    ax2.set_title("Val acc vs weight norm")
    ax2.grid(True, alpha=0.3)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")

    if has_clean:
        gap = np.array(r["clean_train_accuracy"]) - np.array(r["train_accuracy"])
        ax3.plot(x, gap, color="C4", alpha=0.9)
        ax3.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("clean_train_acc − noisy_train_acc")
        ax3.set_title("Grokking gap (→ 0 when groked)")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.axis("off")

    opt_label = r.get("optimizer", "adamw")
    if opt_label == "sgd":
        opt_label = f"sgd(m={r.get('momentum', 0.9)},nesterov={r.get('nesterov', True)})"
    bs_label = r.get("batch_size", r["train_subset"])
    mixup_a = r.get("mixup_alpha", 0.0)
    mixup_label = f"  mixup={mixup_a}" if mixup_a else ""
    init_s = r.get("init_scale", 1.0)
    init_label = f"  init×{init_s:g}" if init_s != 1.0 else ""
    fig.suptitle(
        f"Grokking pilot: arch={list(r['arch'])}  opt={opt_label}  "
        f"lr={r['lr']:g}  wd={r['wd']}{init_label}  batch={bs_label}{mixup_label}  "
        f"seed={r['seed']}  noise={r['label_noise']:.2f}  train_n={r['train_subset']}",
        fontsize=12,
        y=1.03,
    )
    plt.tight_layout()
    return fig, axes


def plot_grok_logx(result: Dict[str, Any], *, savepath: Optional[str] = None, title: Optional[str] = None):
    """Canonical grokking view: dual-axis, LOG-x — train+val accuracy and L2 weight norm vs epoch.

    On a log-epoch axis the flat low-val plateau and the delayed generalization rise become
    visible, and the weight-norm curve (right axis) shows the compression that drives it. A dotted
    marker flags where train accuracy first hits 1.0 (memorization), the start of the grok window.
    """
    import matplotlib.pyplot as plt

    r = result
    x = np.asarray(r["epochs"], dtype=float)
    tr = np.asarray(r["train_accuracy"])
    va = np.asarray(r["val_accuracy"])
    wn = np.asarray(r["weight_norms"])

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=FIGURE_DPI)
    ax2 = ax1.twinx()
    ax1.plot(x, tr, color="C7", lw=1.0, alpha=0.55, label="train acc")
    ax1.plot(x, va, color="C0", lw=2.0, label="val acc")
    ax2.plot(x, wn, color="orange", ls="--", lw=1.6, alpha=0.9, label="L2 weight norm")

    mem_idx = int(np.argmax(tr >= 1.0)) if np.any(tr >= 1.0) else None
    if mem_idx is not None and x[mem_idx] > 0:
        ax1.axvline(x[mem_idx], color="C3", ls=":", lw=1.0, alpha=0.7)
        ax1.annotate(
            f"train→1.0 (ep {int(x[mem_idx]):,})",
            xy=(x[mem_idx], 0.03), xycoords=("data", "axes fraction"),
            fontsize=8, color="C3", ha="left", va="bottom",
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Epoch (log scale)")
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("L2 weight norm")
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True, which="both", alpha=0.25)
    is_ = r.get("init_scale", 1.0)
    ttl = title or (
        f"Grokking on EMG — init×{is_:g}, wd={r.get('wd')}, "
        f"n={r.get('train_subset')}, seed={r.get('seed')}   (val {va.min():.3f}→{va.max():.3f})"
    )
    ax1.set_title(ttl, fontsize=11)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig
