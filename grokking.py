"""Grokking on EMG: model build, sweep/pilot runners, run summaries, and plot helpers.

Two experiments live here. ``run_grok_sweep`` trains a grid of natural-init models (the negative
result: delayed-generalization *drift*, no plateau to grok out of). ``run_grok_pilot`` /
``run_grok_pilots`` train Omnigrok large-init models, which do produce the three-phase grokking
signature. ``grok_summary`` reduces either to the numbers that decide which of the two happened.
See README.md for the full write-up.
"""

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


def _smooth(y: np.ndarray, w: int = 5) -> np.ndarray:
    """Centered moving average — single-tick Bernoulli spikes are not edges.

    Pads by edge replication: np.convolve's "same" mode zero-pads instead, which drags the first
    and last w//2 points toward zero and fabricates a rise at the start of every run.
    """
    if len(y) < w:
        return y
    pad = w // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(w) / w, mode="valid")


def grok_summary(result: Dict[str, Any], *, rise_window: int = 10_000) -> Dict[str, Any]:
    """Reduce one run to the numbers that decide whether it groked.

    Follows the measurement discipline the sweep taught us: measure the plateau by its *mean*
    (not its min, which is a noise spike), exclude the memorization transient from the "biggest
    rise" search (or the winner is just post-memorization decay), and smooth before declaring an
    edge. Grokking = low flat plateau, a large delayed rise, and val anti-correlated with ‖w‖.
    """
    eps = np.asarray(result["epochs"], dtype=float)
    tr = np.asarray(result["train_accuracy"], dtype=float)
    va = np.asarray(result["val_accuracy"], dtype=float)
    wn = np.asarray(result["weight_norms"], dtype=float)
    va_s = _smooth(va)

    mem_i = int(np.argmax(tr >= 0.999)) if np.any(tr >= 0.999) else None
    mem_ep = int(eps[mem_i]) if mem_i is not None else None

    plateau_mean = plateau_std = None
    if mem_ep:  # the decade of training right after memorization
        m = (eps >= mem_ep) & (eps <= 5 * mem_ep)
        if m.sum() >= 3:
            plateau_mean, plateau_std = float(va[m].mean()), float(va[m].std())

    # Biggest val gain over any rise_window-wide window, past the memorization transient.
    start = 2 * mem_ep if mem_ep else 0
    idx = np.flatnonzero(eps >= start)
    best_rise, rise_from, rise_to = 0.0, None, None
    if len(idx) >= 2:
        js = np.clip(np.searchsorted(eps, eps[idx] + rise_window, side="right") - 1, 0, len(eps) - 1)
        gains = va_s[js] - va_s[idx]
        k = int(np.argmax(gains))
        best_rise = float(gains[k])
        rise_from, rise_to = int(eps[idx[k]]), int(eps[js[k]])

    post = eps >= (mem_ep or 0)
    corr = float(np.corrcoef(va[post], wn[post])[0, 1]) if post.sum() > 2 else float("nan")
    peak_i = int(np.argmax(va))

    return {
        "mem_epoch": mem_ep,
        "plateau_val": plateau_mean,
        "plateau_std": plateau_std,
        "best_rise": best_rise,
        "rise_from_epoch": rise_from,
        "rise_to_epoch": rise_to,
        "corr_val_wnorm": corr,
        "peak_val": float(va[peak_i]),
        "peak_epoch": int(eps[peak_i]),
        "final_val": float(va[-1]),
        "wnorm_start": float(wn[0]),
        "wnorm_final": float(wn[-1]),
        "epochs_run": int(eps[-1]),
    }


def print_grok_summary(result: Dict[str, Any], *, rise_window: int = 10_000) -> Dict[str, Any]:
    """Print grok_summary as a handful of aligned lines (see also `grok_summary_table`)."""
    s = grok_summary(result, rise_window=rise_window)
    fmt = lambda v, p=3: "n/a" if v is None else f"{v:.{p}f}"
    print(
        f"  memorized (train→1.0) at epoch {s['mem_epoch']:,}" if s["mem_epoch"]
        else "  never fully memorized"
    )
    print(f"  plateau val        {fmt(s['plateau_val'])} ± {fmt(s['plateau_std'])}")
    print(f"  largest {rise_window//1000}k-epoch rise  +{s['best_rise']:.3f}"
          f"  (ep {s['rise_from_epoch']:,} → {s['rise_to_epoch']:,})"
          if s["rise_from_epoch"] else "  largest rise      n/a")
    print(f"  corr(val, ‖w‖)     {fmt(s['corr_val_wnorm'])}   "
          f"‖w‖ {s['wnorm_start']:.0f} → {s['wnorm_final']:.0f}")
    print(f"  val peak {s['peak_val']:.3f} @ ep {s['peak_epoch']:,}   "
          f"final {s['final_val']:.3f} @ ep {s['epochs_run']:,}")
    return s


def grok_summary_table(results: List[Dict[str, Any]], labels: Optional[List[str]] = None) -> str:
    """Markdown table of grok_summary across several runs — the notebook's results block."""
    labels = labels or [
        f"init×{r.get('init_scale', 1.0):g} wd={r.get('wd')} seed={r.get('seed')}" for r in results
    ]
    head = ("| run | memorized | plateau val | best 10k rise | corr(val,‖w‖) | ‖w‖ start→end | "
            "peak val | final val |\n|---|---|---|---|---|---|---|---|")
    rows = []
    for lbl, r in zip(labels, results):
        s = grok_summary(r)
        plateau = "n/a" if s["plateau_val"] is None else f"{s['plateau_val']:.3f} ± {s['plateau_std']:.3f}"
        mem = "never" if s["mem_epoch"] is None else f"ep {s['mem_epoch']:,}"
        rows.append(
            f"| {lbl} | {mem} | {plateau} | +{s['best_rise']:.3f} | "
            f"{s['corr_val_wnorm']:+.2f} | {s['wnorm_start']:.0f} → {s['wnorm_final']:.0f} | "
            f"{s['peak_val']:.3f} @ {s['peak_epoch']:,} | {s['final_val']:.3f} |"
        )
    return "\n".join([head] + rows)


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
        print_every: int = 0,
    ):
        super().__init__()
        self.val_features, self.val_labels = val_data
        self.log_every = log_every
        self.clean_train_data = clean_train_data
        # Metrics are *recorded* every log_every epochs but only *printed* every print_every
        # (0 = never). A 450k-epoch run logs ~4500 points; printing them all buries the notebook.
        self.print_every = print_every
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

        if self.clean_train_data is not None:
            cf, cl = self.clean_train_data
            cl_loss, cl_acc = self.model.evaluate(cf, cl, verbose=0)
            self.clean_train_loss.append(float(cl_loss))
            self.clean_train_accuracy.append(float(cl_acc))

        if self.print_every and (ep % self.print_every == 0 or ep == self.params["epochs"]):
            print(
                f"    ep {ep:>7,d}  train {logs['accuracy']:.3f}  "
                f"val {val_acc:.3f}  ‖w‖ {weight_norm:6.1f}",
                flush=True,
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
        print(f"Run {run_idx}/{n_total}  arch={list(arch_t)} lr={lr:g} wd={wd} seed={seed} "
              f"epochs={epochs:,}", flush=True)

        tf.random.set_seed(seed)
        grok_model = build_grok_model(arch, lr, wd)
        grok_cb = GrokLoggingCallback(
            val_data=(grok_valid, grok_valid_labels),
            log_every=log_every,
            print_every=_progress_every(epochs, log_every),
        )
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
        print(f"  done in {format_elapsed(elapsed)}  "
              f"val {grok_cb.val_accuracy[-1]:.3f}  ‖w‖ {grok_cb.weight_norms[-1]:.1f}", flush=True)

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
    print(f"\nSweep complete: {n_total} run(s) in {format_elapsed(sweep_total)}")
    return sweep_results, run_timings, sweep_total


def _progress_every(epochs: int, log_every: int) -> int:
    """Print roughly ten progress lines per run, snapped to the logging cadence."""
    return max(log_every, (epochs // 10 // log_every) * log_every)


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
    savepath: Optional[str] = None,
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
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
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
    """Single grokking training run on a small stratified subsample of the curated data.

    The lever that works on EMG is ``init_scale`` (Omnigrok large-norm init): starting far above
    the generalizing weight norm forces memorization first with val pinned low, and weight decay
    then has something to compress *through*. ``label_noise`` and ``mixup_alpha`` are the other
    levers we tried; both are dead ends here (see README) but stay wired for reproduction. When
    label noise is on, clean-label metrics are tracked alongside the noisy ones the optimizer sees.
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

    noise_str = f"  noise={label_noise:.2f} ({n_flipped} flipped)" if label_noise else ""
    print(f"arch={list(arch)}  opt={opt_str}  lr={lr:g}  wd={wd}  init×{init_scale:g}  "
          f"batch={effective_batch_size} ({steps_per_epoch} step{'s' if steps_per_epoch != 1 else ''}/ep)  "
          f"{mixup_str}  seed={seed}  n={len(clean_tr_x)}/{len(v_x)}  epochs={epochs:,}{noise_str}",
          flush=True)

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
            print_every=_progress_every(epochs, log_every),
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
            print_every=_progress_every(epochs, log_every),
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
    print(f"  wall time {format_elapsed(elapsed)}", flush=True)

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
        print(f"\nPilot {i}/{n}", flush=True)
        r = run_grok_pilot(**cfg)
        print_grok_summary(r)
        results.append(r)
    if n > 1:
        print(f"\nAll {n} pilots done in {format_elapsed(time.perf_counter() - t0)}")
    return results


def plot_grok_pilot(pilot_result: Dict[str, Any], *, savepath: Optional[str] = None):
    """Diagnostic panel set for one grokking pilot (linear x-axis).

    Panels:
      (0) Accuracy — train and val.
      (1) Loss — same series in loss space (log y).
      (2) Val accuracy vs L2 weight norm (dual axis) — the grokking mechanism.
      (3) Only for label-noise runs: clean-minus-noisy train accuracy. It sits negative while the
          network memorizes flipped labels and climbs toward zero as it stops fitting the noise.
          Omitted for clean-label runs, where the clean series is identical to the train series.
    """
    import matplotlib.pyplot as plt

    r = pilot_result
    x = r["epochs"]
    has_clean = r.get("label_noise", 0.0) > 0 and len(r["clean_train_accuracy"]) > 0
    n_panels = 4 if has_clean else 3

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5), dpi=FIGURE_DPI)
    ax0, ax1, ax2 = axes[0], axes[1], axes[2]
    ax3 = axes[3] if has_clean else None

    ax0.plot(x, r["train_accuracy"],
             label="train acc (noisy labels)" if has_clean else "train acc", color="C0", alpha=0.9)
    if has_clean:
        ax0.plot(x, r["clean_train_accuracy"], label="train acc (clean labels)", color="C2", alpha=0.9)
    ax0.plot(x, r["val_accuracy"], label="val acc", color="C3", alpha=0.9)
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Accuracy")
    ax0.set_title("Accuracy")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8, loc="best")

    ax1.plot(x, r["train_loss"],
             label="train loss (noisy)" if has_clean else "train loss", color="C0", alpha=0.9)
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

    if ax3 is not None:
        gap = np.array(r["clean_train_accuracy"]) - np.array(r["train_accuracy"])
        ax3.plot(x, gap, color="C4", alpha=0.9)
        ax3.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("clean_train_acc − noisy_train_acc")
        ax3.set_title("Grokking gap (→ 0 when groked)")
        ax3.grid(True, alpha=0.3)

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
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
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


def plot_grok_logx_overlay(results, *, savepath: Optional[str] = None, title: Optional[str] = None):
    """Overlay val accuracy vs epoch (log-x) across several runs (e.g. a multi-seed pass).

    Each run is a thin line labelled by seed; the mean over the shared epoch grid is bold.
    Shows whether the grok (flat plateau → delayed rise) is robust across seeds/runs, and
    how wide the run-to-run spread is (see the reproducibility caveat in README.md).
    """
    import matplotlib.pyplot as plt

    minlen = min(len(r["epochs"]) for r in results)
    x = np.asarray(results[0]["epochs"][:minlen], dtype=float)
    vals = np.array([np.asarray(r["val_accuracy"][:minlen]) for r in results])
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=FIGURE_DPI)
    for r, v in zip(results, vals):
        ax.plot(x, v, lw=1.0, alpha=0.5, label=f"seed={r.get('seed')}")
    ax.plot(x, vals.mean(axis=0), color="k", lw=2.4, label="mean")
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.0, max(0.7, float(vals.max()) + 0.05))
    ax.grid(True, which="both", alpha=0.25)
    is_ = results[0].get("init_scale", 1.0)
    ax.set_title(
        title or f"Grokking robustness — init×{is_:g}, wd={results[0].get('wd')}, "
        f"n={results[0].get('train_subset')}, {len(results)} seeds",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig
