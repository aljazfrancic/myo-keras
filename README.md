# myo-keras

EMG gesture classification using Keras on Myo armband data.

## What it does

Trains a dense neural network to classify 8 hand gestures from 8-channel EMG signals recorded with a Myo armband. Features are extracted via a causal sliding-window RMS. The notebook supports two data loading modes (curated subset or all sessions) and includes a grokking experiment.

## Dataset

Expects the [myo-readings-dataset](https://github.com/aljazfrancic/myo-readings-dataset) to be located alongside this repo. Specifically, readings are loaded from `_readings_right_hand/` and curated session names from `curated.txt`.

## Setup

```bash
pip install -r requirements.txt
```

## Project structure

| File | Description |
|---|---|
| `myo_utils.py` | Constants and utility functions (RMS, data loading, auto-curation) |
| `myo-keras.ipynb` | Training, evaluation, and grokking experiment |
| `requirements.txt` | Python dependencies |

## Usage

Open `myo-keras.ipynb` and run cells top to bottom.

- **Curated sessions** (default): loads only participants listed in `curated.txt`.
- **All sessions**: uncomment `load_data_all()` and comment out `load_data_curated()`.

Sessions are split deterministically: suffix `-1` for training, `-2` for validation, `-3` for testing.

## Grokking experiment

The notebook includes a grokking section that trains the same architecture for many epochs (default 1000) **without** early stopping. The goal is to observe whether the model exhibits delayed generalization -- a sudden jump in validation accuracy well after the training loss has converged.

## Auto-curation

To regenerate `curated.txt` based on per-participant test accuracy (run after the first notebook cell so `myo_utils` is in scope):

```python
generate_curated()  # no args: uses READINGS_DIR, CURATED_FILE, CURATION_ACCURACY_THRESHOLD from myo_utils
```

Optional args: `generate_curated(readings_dir=..., output_file=..., accuracy_threshold=...)`.

The default accuracy threshold is 0.7 (configurable via `CURATION_ACCURACY_THRESHOLD`).

## Gesture labels

| Index | Gesture |
|---|---|
| 0 | hibernation |
| 1 | flexion |
| 2 | extension |
| 3 | radial deviation |
| 4 | ulnar deviation |
| 5 | pronation |
| 6 | supination |
| 7 | fist |
