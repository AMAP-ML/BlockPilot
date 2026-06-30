# Quickstart


## 1) Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Script Overview

### `run_benchmark_acceptance_lengths.sh`

Runs multiple datasets across multiple block sizes and writes acceptance-length JSON files.

- Output: `utils/{dataset}_0T0.0/{block_size}.json`
- Hidden states are disabled in this script via `--no-save-hidden-states`

Run:

```bash
bash run_benchmark_acceptance_lengths.sh
```

### `run_dataset_txt.sh`

Wrapper script for `dataset.py` using a TXT file (one prompt per line).

Run:

```bash
bash run_dataset_txt.sh --input-file TXT/copa.txt
```

Example with common options:

```bash
bash run_dataset_txt.sh \
  --input-file TXT/copa.txt \
  --block-sizes "1 2 4 8 14 15 16 17 18" \
  --max-samples 32 \
  --seed 0 \
  --no-save-hidden-states
```

### `utils/predict.py`

Loads:

- `utils/dataset.pt`
- `utils/model.pt`

Predicts block ids (`pred + 14`) and writes:

- `utils/{task}_0T0.0/1.json`

Run:

```bash
python3 utils/predict.py
```

### `map_selected_block_values.py`

Maps selected block ids in `1.json` to actual values from corresponding block JSON files.

Example logic:

- If `1.json["0"] = 18`, output value becomes `18.json["0"]`.

Run:

```bash
python3 map_selected_block_values.py utils/gsm8k_0T0.0
```

Custom output filename:

```bash
python3 map_selected_block_values.py \
  utils/gsm8k_0T0.0 \
  --output-file mapped_from_1.json
```

