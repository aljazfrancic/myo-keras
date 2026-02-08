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
GROKKING_EPOCHS = 1000
CURATION_ACCURACY_THRESHOLD = 0.7
FIGURE_SIZE = (20, 5)
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


def get_values(seshes, verbose=True):
    """Load gesture files from *seshes* directories and return RMS matrix."""
    parts = []
    for sesh in seshes:
        for gesture in range(NUM_GESTURES):
            path = os.path.join(sesh, f"{gesture}.txt")
            if verbose:
                print(path)
            matrix = np.genfromtxt(path, delimiter=",")
            parts.append(get_rms(matrix))
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, NUM_COLUMNS))


def _split_and_load(session_dirs):
    """Split session dirs by suffix (-1 train, -2 valid, -3 test), load each."""
    train_dirs = sorted([d for d in session_dirs if os.path.basename(d).endswith("-1")])
    valid_dirs = sorted([d for d in session_dirs if os.path.basename(d).endswith("-2")])
    test_dirs = sorted([d for d in session_dirs if os.path.basename(d).endswith("-3")])

    train_set = get_values(train_dirs)
    valid_set = get_values(valid_dirs)
    test_set = get_values(test_dirs)

    train, train_labels = split_features_labels(train_set)
    valid, valid_labels = split_features_labels(valid_set)
    test, test_labels = split_features_labels(test_set)

    return train, train_labels, valid, valid_labels, test, test_labels


def load_data_curated(curated_file=CURATED_FILE, readings_dir=READINGS_DIR):
    """Load only curated sessions listed in *curated_file*."""
    with open(curated_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    session_dirs = [os.path.join(readings_dir, name) for name in names]
    return _split_and_load(session_dirs)


def load_data_all(readings_dir=READINGS_DIR):
    """Load all sessions from *readings_dir*."""
    session_dirs = get_sessions(readings_dir)
    return _split_and_load(session_dirs)


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
            keras.layers.Dense(NUM_GESTURES, activation="sigmoid"),
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
