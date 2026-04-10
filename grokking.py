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
    GROKKING_RMS_WINDOW,
    GROKKING_SEEDS,
    GROKKING_WEIGHT_DECAYS,
    NUM_EMG_CHANNELS,
    NUM_GESTURES,
    load_data_curated,
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
    """Logs train metrics + periodic validation metrics + weight norm."""

    def __init__(self, val_data, log_every: int = GROKKING_LOG_EVERY):
        super().__init__()
        self.val_features, self.val_labels = val_data
        self.log_every = log_every
        self.epochs: List[int] = []
        self.train_loss: List[float] = []
        self.train_accuracy: List[float] = []
        self.val_loss: List[float] = []
        self.val_accuracy: List[float] = []
        self.weight_norms: List[float] = []

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

        print(
            f"Epoch {ep:>6d}  "
            f"train_acc={logs['accuracy']:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"wnorm={weight_norm:.2f}"
        )


def build_grok_model(arch, lr: float, wd: float):
    layers = [keras.layers.Input(shape=(NUM_EMG_CHANNELS,))]
    for units in arch:
        layers.append(keras.layers.Dense(units, activation="relu"))
    layers.append(keras.layers.Dense(NUM_GESTURES, activation="softmax"))
    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


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
