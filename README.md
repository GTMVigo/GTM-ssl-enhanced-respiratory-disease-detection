# GTM Respiratory Disease Detection

Code accompanying the paper **Improving Respiratory Disease Detection Through SSL-Enhanced Acoustic Analysis and Exercise-Rest Measurements**.

This repository provides a pipeline to:
- extract **SSL embeddings** and **acoustic features** from audio recordings.
- configure experiments via a YAML file.
- train and evaluate **classification models**.

It is intended for users who want to reproduce or adapt the methodology described in the paper.

---

# Reproducibility Notes
This repository is designed to support the experimental setup reported in the manuscript.

- Participant-level grouping: the pipeline groups recordings by participant when `which_features: aggregated` is used, concatenating per-participant audio features in canonical order.
- The main classification pipeline uses stratified K-fold cross-validation via `kfold_splits`. Fixed seeds are used for dataset shuffling, fold generation and model initialitation.

---

# Repository Workflow

These are the **core scripts required to run experiments**:

- `extract_embeddings.py` → Extract SSL embeddings from audio (Acoustic features are computed during classification by executing `multiclass_classification.py`, using the `AudioProcessor` module).  
- `multiclass_classification_conf.yaml` → Configure experiment.
- `multiclass_classification.py` → Train & evaluate models.  

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
│   ├── exceptions/
│   ├── features/
│   ├── files/
│   ├── logger/
│   ├── model/
│   └── common_classification.py
├── extract_embeddings.py
├── multiclass_classification.py
├── multiclass_classification_conf.yaml
├── process_combinations.py
├── make_summary.py
├── requirements.txt
└── README.md
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

Configured in `multiclass_classification_conf.yaml` via:

```yaml
global_transcript_file: path/to/metadata.csv
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

## 1. Prepare data

1. Place `.wav` files in `wav_folder`  
2. Prepare labels and metadata CSVs  

## 2. Extract SSL embeddings

```bash
python extract_embeddings.py
```

This step extracts embeddings using:
- WavLM
- HuBERT
- wav2vec 2.0

You may need to adjust paths inside the script depending on your setup.

---

## 3. Configure experiment

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
---

## 4. Run classification

```bash
python multiclass_classification.py
```

This will:
- load configuration.
- load and/or compute features.
- train models.
- evaluate performance.
- save results.

---

## 5. Outputs

After execution:

- Trained models → saved in `model_folder`.
- Evaluation metrics (CSV) → saved in `save_metrics_path`.
- Per-sample predictions (optional) → saved when `detailed_metrics: True`.
- Extracted features → saved in `path_extracted_features` (HDF5 `.h5` format).

---

## 6. Post-classification Utilities

### 6.1. Combination-based aggregation (`process_combinations.py`)

This script implements late fusion by combining predictions from multiple experiments.

It:
- loads prediction CSVs.
- aggregates predictions at subject level.
- computes metrics (accuracy, F1-score, recall, precision, AUC).

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

### 6.2. Metrics summarization (`make_summary.py`)

Aggregates results across folds or runs by computing:
- mean
- standard deviation

Usage:

```bash
python make_summary.py
```

---

### 6.3. ROC curve plotting (`plot_roc_curves_cv_folds.py`)

Use this script to generate ROC curves from saved model predictions.

Usage:

```bash
python plot_roc_curves.py
```

---

### 6.4. Notebook-based manuscript analysis (`coperia_analysis.ipynb`)

Use this notebook for complementary analysis. It includes:
- cross-fold confidence interval computation,
- logistic regression coefficient stability,
- top-feature identification,


---

# Features

## SSL embedding

- WavLM
- HuBERT
- wav2vec 2.0

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

## Acoustic features

- ComParE 2016 (energy, spectral, MFCC, voicing, rasta)
- spafe features (MFCC, CQCC, GFCC, LFCC, PLP)

Acoustic features extraction is configured through the `audioprocessor_data` section in `multiclass_classification_conf.yaml`. These are the available parameters:

Available parameters:

1. `feature_type` (list): Types of features to extract. Options include:  
   `compare_2016_energy`, `compare_2016_voicing`,  
   `compare_2016_spectral`, `compare_2016_rasta`,  
   `compare_2016_basic_spectral`,  
   `spafe_mfcc`

2. `resampling_rate` (int): Target sampling rate for audio signals  
3. `top_db` (float): Threshold (in dB) for trimming silence  
4. `pre_emphasis_coefficient` (float): Pre-emphasis filter coefficient  
5. `f_min` (int): Minimum frequency  
6. `f_max` (int): Maximum frequency  
7. `window_size` (int): Window size in milliseconds  
8. `hop_length` (int): Hop length in milliseconds  
9. `n_mels` (int): Number of Mel bands  
10. `n_mfcc` (int): Number of MFCC coefficients  
11. `plp_order` (int): Order of PLP coefficients  
12. `conversion_approach` (str): Audio conversion strategy  
13. `normalize` (bool): Apply normalization to features  
14. `use_energy` (bool): Include energy in feature computation  
15. `apply_mean_norm` (bool): Apply mean normalization  
16. `apply_vari_norm` (bool): Apply variance normalization  
17. `compute_deltas_feats` (bool): Compute first-order derivatives  
18. `compute_deltas_deltas_feats` (bool): Compute second-order derivatives  
19. `compute_opensmile_extra_features` (bool): Include additional OpenSMILE features

## Feature Dimensionality and Preprocessing

Prior to feature extraction, recordings were resampled to 16 kHz. For the acoustic features, a voice activity detection procedure based on energy thresholding was applied to remove silent and non-vocal segments. The resulting signals were subsequently processed using pre-emphasis filtering (coefficient = 0.97) and amplitude normalization. First-order and second-order derivatives were computed for these features. 


Dimensionality (per recording) of the feature representations evaluated in this study:

- energy-based descriptors: 132
- voicing features: 198
- RASTA spectral descriptors: 858
- basic spectral descriptors: 495
- MFCCs: 429
- Wav2vec 2.0 embeddings: 1024
- WavLM embeddings: 768
- HuBERT embeddings: 768

Common feature configurations:

- `acoustic` total dimension: 2112
- `acoustic + Wav2vec 2.0 + WavLM` total dimension: 3904

---

# Citation

The paper is currently under review.

---

# Acknowledgments

- Multimedia Technologies Group (GTM)  
- atlanTTic Research Center  
- Universidade de Vigo  