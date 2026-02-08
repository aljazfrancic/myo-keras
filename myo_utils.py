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


def get_values(seshes):
    """Load gesture files from *seshes* directories and return RMS matrix."""
    big_matrix = np.zeros((0, NUM_COLUMNS))
    for sesh in seshes:
        for gesture in range(NUM_GESTURES):
            path = sesh + "/" + str(gesture) + ".txt"
            print(path)
            matrix = np.genfromtxt(path, delimiter=",")
            rms = get_rms(matrix, RMS_WINDOW_SIZE)
            big_matrix = np.concatenate((big_matrix, rms), axis=0)
    return big_matrix


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
