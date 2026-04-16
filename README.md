# GTM Respiratory Disease Detection

Code accompanying the paper **Improving Respiratory Disease Detection Through SSL-Enhanced Acoustic Analysis and Exercise-Rest Measurements**.

This repository provides a pipeline to:
- extract **SSL embeddings** and **acoustic features** from audio recordings,
- configure experiments via a YAML file,
- train and evaluate **classification models**.

It is intended for users who want to reproduce or adapt the methodology described in the paper.

---

# Repository Workflow

These are the **core scripts required to run experiments**:

- `extract_embeddings.py` → Extract SSL embeddings from audio (Acoustic features are computed during classification using the `AudioProcessor` module)  
- `multiclass_classification_conf.yaml` → Configure experiment  
- `multiclass_classification.py` → Train & evaluate models  

---

# Installation

```bash
pip install -r requirements.txt
```

⚠️ If using GPU, ensure compatibility between:
- PyTorch
- CUDA
- NVIDIA drivers

---

# Expected Data

You need:
- A folder with `.wav` files  
- Two CSV files:
  - a condition CSV (labels)
  - a global CSV (metadata)

## Example structure

```
project_root/
├── data/
│   ├── wavs/
│   │   ├── 1848560680d_c6b9296a_before_a_01.wav
│   │   ├── 1848560680d_c6b9296a_after_cough_02.wav
│   │   └── ...
│   ├── labels.csv
│   └── metadata.csv
├── extracted_features/
├── src/
├── extract_embeddings.py
├── multiclass_classification.py
├── multiclass_classification_conf.yaml
```

---

# Condition CSV (labels)

Patient-level labels:

```
participant_id,label
1848560680d,1
983475920ab,0
```

Configured in `multiclass_classification_conf.yaml` via:

```yaml
condition_path: path/to/labels.csv
condition_id_column: <column_index>
condition_label_column: <column_index>
```

---

# Global CSV (metadata)

Audio-level metadata:

```
filename,participant_id,task,moment
1848560680d_c6b9296a_before_a_01.wav,1848560680d,a,before
1848560680d_c6b9296a_after_cough_02.wav,1848560680d,cough,after
983475920ab_f83aa12b_before_a_01.wav,983475920ab,a,before
```

Configured via:

```yaml
global_file_path: path/to/metadata.csv
```

---

# Linking audio to labels

Audio files are linked to labels using a simple convention: the patient ID is the prefix of the filename.

Example:

```
1848560680d_c6b9296a_before_a_01.wav → 1848560680d
```

This ID is matched with the condition CSV, so multiple recordings can share the same label. The global CSV is used to filter valid files and provide metadata (e.g., task and moment).

---

# Execution Workflow

## 1. Extract SSL embeddings

```bash
python extract_embeddings.py
```

This step extracts embeddings using:
- WavLM
- HuBERT
- wav2vec 2.0

You may need to adjust paths inside the script depending on your setup.

---

## 2. Configure experiment

Edit:

```
multiclass_classification_conf.yaml
```

### Key parameters

#### Paths

```yaml
condition_path: path/to/labels.csv
global_transcript_file: path/to/metadata.csv
wav_folder: path/to/wavs/
path_extracted_features: path/to/features/
model_folder: path/to/save/models/
save_metrics_path: path/to/save/metrics/
```

#### Execution mode

```yaml
run:
  extract_train_test: True
  evaluate_model: False
```

#### Feature usage

```yaml
use_acoustic_feat: True
use_ssl_wavlm: True
use_ssl_hubert: False
use_ssl_wav2vec: True
```

#### Feature loading

```yaml
load_extracted_features: True
```

Note:
- First run → set `load_extracted_features: False`
- Later → set `True` to reuse features and speed up experiments

---

## 3. Run classification

```bash
python multiclass_classification.py
```

This will:
- load configuration
- load or compute features
- train models
- evaluate performance
- save results

---

# Typical Usage Modes

## A. Using precomputed embeddings

```yaml
load_extracted_features: True
```

## B. Full pipeline from raw audio

1. Place `.wav` files in `wav_folder`  
2. Prepare labels and metadata CSVs  

Run:

```bash
python extract_embeddings.py
```

Then:

```bash
python multiclass_classification.py
```

---

# Post-classification Utilities

## 1. Combination-based aggregation (`process_combinations.py`)

This script implements late fusion by combining predictions from multiple experiments.

It:
- loads prediction CSVs
- aggregates predictions at subject level
- applies:
  - majority voting
  - threshold-based voting
  - average probability fusion
- computes metrics (accuracy, F1-score, recall, precision, AUC)

Usage:

```bash
python process_combinations.py
```

⚠️ Important:

```yaml
detailed_metrics: True
```

must be enabled in `multiclass_classification_conf.yaml` before executing `multiclass_classification.py` in order to save patient predictions for `process_combinations.py` usage.

---

## 2. Metrics summarization (`make_summary.py`)

Aggregates results across folds or runs by computing:
- mean
- standard deviation

Usage:

```bash
python make_summary.py
```

---

# Features

## Acoustic features available

- ComParE 2016 (energy, spectral, MFCC, voicing, rasta)
- spafe features (MFCC, CQCC, GFCC, LFCC, PLP)

## SSL embeddings available

- WavLM
- HuBERT
- wav2vec 2.0

---

# SSL Models

The SSL embeddings used in this repository rely on the following pretrained models:

- wav2vec 2.0 → Facebook AI pretrained model
- WavLM and HuBERT (fine-tuned for pathological speech):  
  https://huggingface.co/morenolq/SSL4PR-hubert-base

If you use these models, please cite:

```bibtex
@inproceedings{moreno_ssl4pr,
  title={Self-Supervised Learning for Pathological Speech Representation},
  author={Moreno et al.},
  year={2024}
}

```

---

# Outputs

After execution:

- Trained models → saved in `model_folder`
- Evaluation metrics (CSV) → saved in `save_metrics_path`
- Per-sample predictions (optional) → saved when `detailed_metrics: True`
- Extracted features → saved in `path_extracted_features` (HDF5 `.h5` format)

---

# Notes

- Update all paths in the YAML file before running
- GPU usage requires compatible CUDA and drivers
- Utility scripts use hardcoded paths and may need adaptation

---

# Citation

```
@inproceedings{GTM_improving_respiratory_disease,
  title   = {Improving Respiratory Disease Detection Through SSL-Enhanced Acoustic Analysis and Exercise-Rest Measurements},
  author  = {Vera López, Álvaro and Tilves Santiago, Darío and Ramírez Sánchez, José Manuel and Docío-Fernández, Laura and García-Mateo, Carmen and García-Caballero, Alejandro and Bustillo Casado, María},
  journal = {Frontiers},
  year    = {2026}
}
```

---

# Acknowledgments

- Multimedia Technologies Group (GTM)  
- atlanTTic Research Center  
- Universidade de Vigo  