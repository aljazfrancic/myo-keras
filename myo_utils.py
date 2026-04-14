import os
import numpy as np

# Constants (all magic numbers in one place)
READINGS_DIR = "../myo-readings-dataset/_readings_right_hand/"
CURATED_FILE = "../myo-readings-dataset/curated.txt"
NUM_EMG_CHANNELS = 8
NUM_GESTURES = 8
NUM_COLUMNS = 9  # 8 channels + 1 label
RMS_WINDOW_SIZE = 80
RMS_NORMALIZATION = 128.0
WEIGHTS_FILE = "weights.weights.h5"
LEARNING_RATE = 0.001
EPOCHS = 20
PATIENCE = 5
LAYER_SIZES = [200, 100, 70]
# Grokking sweep: tune scalars and lists below. The notebook runs one training job per tuple in
# GROKKING_ARCHITECTURES × GROKKING_LRS × GROKKING_WEIGHT_DECAYS × GROKKING_SEEDS (any list may have length 1).
# Stratified train/val subsamples use subsample_data(..., seed=42) unless you change the notebook call site.
GROKKING_RMS_WINDOW = 30
GROKKING_EPOCHS = 100_000
GROKKING_TRAIN_SUBSET = 100
GROKKING_ARCHITECTURES = [[200, 100, 70]]
GROKKING_LRS = [3e-4]
GROKKING_WEIGHT_DECAYS = [0.09, 0.1, 0.11]
GROKKING_SEEDS = [100, 123, 256, 420, 789]
GROKKING_VAL_SUBSET = 8_000
GROKKING_LOG_EVERY = 100
# Pilot: single-run grokking experiment with label noise to widen the memorize->generalize gap.
# Rationale: pilots 1-3 (AdamW strong wd, AdamW weak wd, SGD+Nesterov mini-batch) all failed
# the same way — clean_tr_acc pinned at (n-flipped)/n once the network settled into a basin
# that memorized every flipped sample. Per-weight (wd) and per-step (SGD batch) perturbations
# don't have the targeted leverage to break those basins because they push every weight a
# little, but the basin walls are large enough that uniform pushes don't break out. This pilot
# adds *between-sample* perturbation via mixup: each training example is linearly blended with
# another sample in the same minibatch, so the basin around a flipped point must compete with
# the nearby clean samples that are now constantly being averaged into it.
GROKKING_PILOT_OPTIMIZER = "sgd"  # "adamw" or "sgd"
GROKKING_PILOT_MOMENTUM = 0.9
GROKKING_PILOT_NESTEROV = True
GROKKING_PILOT_BATCH_SIZE = 16  # set to None for full-batch
GROKKING_PILOT_MIXUP_ALPHA = 1.0  # Beta(alpha, alpha) mixing strength; 0 disables mixup
GROKKING_PILOT_TRAIN_SUBSET = 80
GROKKING_PILOT_LABEL_NOISE = 0.25
GROKKING_PILOT_WD = 0.01
GROKKING_PILOT_LR = 0.01
GROKKING_PILOT_EPOCHS = 100_000
GROKKING_PILOT_SEED = 100
GROKKING_PILOT_LOG_EVERY = 100
GROKKING_PILOT_SUBSAMPLE_SEED = 123
CURATION_ACCURACY_THRESHOLD = 0.7
FIGURE_SIZE = (20, 5)
FIGURE_DPI = 200  # 2× Matplotlib default (100) for sharper display and exports
GESTURE_LABELS = [
    "hibernation",
    "flexion",
    "extension",
    "radial deviation",
    "ulnar deviation",
    "pronation",
    "supination",
    "fist",
]


# Signal processing
def get_rms(data, n=RMS_WINDOW_SIZE):
    """Compute causal RMS over a sliding window for each EMG channel."""
    rows, cols = data.shape
    new = np.zeros((rows, cols), dtype=np.float64)
    new[:, NUM_EMG_CHANNELS] = data[:, NUM_EMG_CHANNELS]
    sq = np.asarray(data[:, :NUM_EMG_CHANNELS], dtype=np.float64) ** 2
    cumsum_sq = np.cumsum(sq, axis=0)
    roll_sum_sq = np.empty_like(sq)
    roll_sum_sq[:n] = cumsum_sq[:n]
    roll_sum_sq[n:] = cumsum_sq[n:] - cumsum_sq[:-n]
    count = np.minimum(np.arange(rows, dtype=np.float64) + 1, n)
    np.maximum(count, 1, out=count)
    new[:, :NUM_EMG_CHANNELS] = (
        np.sqrt(roll_sum_sq / count[:, np.newaxis]) / RMS_NORMALIZATION
    )
    return new


# Feature / label helpers
def split_features_labels(data):
    """Split a data matrix into (features, labels)."""
    return data[:, :NUM_EMG_CHANNELS], data[:, NUM_EMG_CHANNELS]


# Data loading
def get_sessions(readings_dir=READINGS_DIR):
    """Return a sorted list of session directory paths."""
    return sorted(
        [x[0] for x in os.walk(readings_dir) if x[0] != readings_dir]
    )


def get_values(seshes, verbose=True, rms_window=RMS_WINDOW_SIZE):
    """Load gesture files from *seshes* directories and return RMS matrix."""
    parts = []
    for sesh in seshes:
        for gesture in range(NUM_GESTURES):
            path = os.path.join(sesh, f"{gesture}.txt")
            if verbose:
                print(path)
            matrix = np.genfromtxt(path, delimiter=",")
            parts.append(get_rms(matrix, n=rms_window))
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, NUM_COLUMNS))


def _split_and_load(session_dirs, rms_window=RMS_WINDOW_SIZE):
    """Split session dirs by suffix (-1 train, -2 valid, -3 test), load each."""
    train_dirs = sorted([d for d in session_dirs if os.path.basename(d).endswith("-1")])
    valid_dirs = sorted([d for d in session_dirs if os.path.basename(d).endswith("-2")])
    test_dirs = sorted([d for d in session_dirs if os.path.basename(d).endswith("-3")])

    train_set = get_values(train_dirs, rms_window=rms_window)
    valid_set = get_values(valid_dirs, rms_window=rms_window)
    test_set = get_values(test_dirs, rms_window=rms_window)

    train, train_labels = split_features_labels(train_set)
    valid, valid_labels = split_features_labels(valid_set)
    test, test_labels = split_features_labels(test_set)

    return train, train_labels, valid, valid_labels, test, test_labels


def load_data_curated(
    curated_file=CURATED_FILE, readings_dir=READINGS_DIR, rms_window=RMS_WINDOW_SIZE
):
    """Load only curated sessions listed in *curated_file*."""
    with open(curated_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    session_dirs = [os.path.join(readings_dir, name) for name in names]
    return _split_and_load(session_dirs, rms_window=rms_window)


def load_data_all(readings_dir=READINGS_DIR, rms_window=RMS_WINDOW_SIZE):
    """Load all sessions from *readings_dir*."""
    session_dirs = get_sessions(readings_dir)
    return _split_and_load(session_dirs, rms_window=rms_window)


def apply_label_noise(labels, noise_fraction, num_classes=NUM_GESTURES, seed=42):
    """Flip a fraction of labels uniformly to a different class. Returns a new array."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).copy()
    n = len(labels)
    n_flip = int(round(n * noise_fraction))
    if n_flip == 0:
        return labels
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    for i in flip_idx:
        original = int(labels[i])
        wrong = int(rng.integers(0, num_classes - 1))
        if wrong >= original:
            wrong += 1
        labels[i] = wrong
    return labels


# Data subsampling
def subsample_data(features, labels, n, seed=42):
    """Return a stratified random subset of *n* samples."""
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    per_class = n // len(unique_labels)
    remainder = n - per_class * len(unique_labels)

    chosen = []
    for i, lbl in enumerate(unique_labels):
        idx = np.where(labels == lbl)[0]
        count = per_class + (1 if i < remainder else 0)
        count = min(count, len(idx))
        chosen.append(rng.choice(idx, size=count, replace=False))

    chosen = np.concatenate(chosen)
    rng.shuffle(chosen)
    return features[chosen], labels[chosen]


# Auto-curation
def _get_participant_ids(readings_dir=READINGS_DIR):
    """Return participant IDs that have all three sessions (-1, -2, -3)."""
    dirs = set(os.path.basename(d) for d in get_sessions(readings_dir))
    prefixes = sorted({d.rsplit("-", 1)[0] for d in dirs if "-" in d})
    return [pid for pid in prefixes
            if {f"{pid}-1", f"{pid}-2", f"{pid}-3"} <= dirs]


def generate_curated(readings_dir=READINGS_DIR,
                     output_file=CURATED_FILE,
                     accuracy_threshold=CURATION_ACCURACY_THRESHOLD):
    """Train a small model per participant; write curated.txt for those above threshold."""
    import tensorflow as tf
    from tensorflow import keras

    participant_ids = _get_participant_ids(readings_dir)
    curated = []

    for pid in participant_ids:
        print(f"--- evaluating participant {pid} ---")
        session_dirs = [os.path.join(readings_dir, f"{pid}-{i}") for i in range(1, 4)]
        train_set = get_values([session_dirs[0]], verbose=False)
        valid_set = get_values([session_dirs[1]], verbose=False)
        test_set = get_values([session_dirs[2]], verbose=False)
        tr, tr_l = split_features_labels(train_set)
        va, va_l = split_features_labels(valid_set)
        te, te_l = split_features_labels(test_set)

        model = keras.Sequential([
            keras.layers.Input(shape=(NUM_EMG_CHANNELS,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(NUM_GESTURES, activation="softmax"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        model.fit(tr, tr_l, validation_data=(va, va_l), epochs=10, verbose=0)
        _, acc = model.evaluate(te, te_l, verbose=0)
        print(f"  {pid}: test accuracy = {acc:.3f}")

        if acc >= accuracy_threshold:
            curated.append(pid)

    lines = [f"{pid}-{i}" for pid in curated for i in range(1, 4)]
    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(curated)} participants ({len(lines)} sessions) to {output_file}")
    return curated
