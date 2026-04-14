# GTM Respiratory Disease Detection

Code accompanying the paper **Improving Respiratory Disease Detection Through SSL-Enhanced Acoustic Analysis and Exercise-Rest Measurements**.

This repository provides a minimal pipeline to:
- extract **SSL embeddings** from audio recordings,
- configure experiments (via a YAML file),
- train and evaluate **classification models**.

It is intended for users who want to reproduce or adapt the methodology described in the paper.

---

# Repository Workflow

The **three main files** are:

- `extract_embeddings.py` → Extract SSL embeddings from audio  
- `multiclass_classification_conf.yaml` → Configure experiment  
- `multiclass_classification.py` → Train & evaluate models  

---

# Installation

```pip install -r requirements.txt
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
  - a **condition CSV** (labels)
  - a **global CSV** (metadata)

---

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

## Condition CSV (labels)

Patient-level labels:

```
participant_id,label
1848560680d,1
983475920ab,0
```

Configured via:
```yaml
condition_path: path/to/labels.csv
condition_id_column: <column_index>
condition_label_column: <column_index>
```

---

## Global CSV (metadata)

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

## Linking audio to labels

Audio files are linked to labels using a simple convention: the **patient ID is the prefix of the filename**.

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

**Note:**  
- On the **first run**, set `load_extracted_features: False` to compute and save the acoustic features and embeddings. After that, you can set it to `True` to reuse them and speed up experiments.  
- Acoustic feature extraction (e.g., MFCCs, ComParE subsets, etc.) is configured in the YAML file under the corresponding feature extraction section (`feature_type` and related parameters).

#### Model configuration
```yaml
model_list: [LogisticRegression]
kfold_splits: 5
```

---

## 3. Run classification

```bash
python multiclass_classification.py
```

This will:
- load config
- load or compute features
- train models
- evaluate performance
- save results

---

# Typical Usage Modes

## A. Using precomputed embeddings (recommended)

```yaml
load_extracted_features: True
```
---

## B. Full pipeline from raw audio

1. Place `.wav` files in `wav_folder`
2. Prepare labels and metadata CSVs
3. Run:
   ```   
   python extract_embeddings.py
   ```
4. Configure YAML
5. Run:
   ```
   python multiclass_classification.py
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

# Outputs

After execution:

- Models → `model_folder`
- Metrics → `save_metrics_path`
- Optional detailed predictions

---

# Notes

- Update all paths in the YAML file before running.
- GPU requires compatible CUDA + drivers.

---

# Citation

```bibtex
@inproceedings{GTM_improving_respiratory_disease,
  title   = {Improving Respiratory Disease Detection Through SSL-Enhanced Acoustic Analysis and Exercise-Rest Measurements},
  author  = {Vera López, Álvaro and Tilves Santiago, Darío and Ramírez Sánchez, José Manuel and Docío-Fernández, Laura and García-Mateo, Carmen and García-Caballero, Alejandro and Bustillo Casado, María},
  journal = {Frontiers},
  year    = {2026}
}
```

---

# Acknowledgments

Multimedia Technologies Group (GTM)  
atlanTTic Research Center  
Universidade de Vigo