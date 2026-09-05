# ViSPe — Vietnamese Speech-PTE Evaluator (v1)

> **End-to-End Speech Evaluation Pipeline combining Automatic Speech Recognition (ASR) and Predict Emotion from Text (PTE) for Vietnamese Audio**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![ASR](https://img.shields.io/badge/ASR-Automatic%20Speech%20Recognition-orange.svg)](#2-features--how-it-works)
[![PTE](https://img.shields.io/badge/PTE-Predict%20Emotion%20from%20Text-purple.svg)](#2-features--how-it-works)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#9-license--citation)

---

## 1. Introduction

### About the Author
Hello! I am **Kadu** (Pham Thai Dang Minh), an Information Technology (IT) student at the **University of Information Technology, Vietnam National University Ho Chi Minh City (UIT - VNU-HCM)**.

### About the Project
**ViSPe v1 (Vietnamese Speech-PTE Evaluator)** is an open-source, modular Python evaluation system designed to benchmark joint speech recognition and text emotion recognition pipelines specifically tailored for Vietnamese audio datasets. 

Speech emotion analysis systems often suffer from error propagation when transcribing raw audio into text before classifying emotion. ViSPe v1 provides a unified framework to quantitatively measure both transcription fidelity and emotion classification performance simultaneously, storing experiment histories in structured JSON format.

---

## 2. Features & How It Works

ViSPe v1 operates through an integrated two-stage pipeline combined with automated text normalization and metric benchmarking:

```text
 Audio Input (.wav)
       │
       ▼
 ┌────────────────────────────────────────────────────────┐
 │ 1. Automatic Speech Recognition (ASR)                  │
 │    • Ingests audio files via HuggingFace ASR models    │
 │    • Transcribes speech to Vietnamese text             │
 └─────────────────────────┬──────────────────────────────┘
                           │ Transcribed Text
                           ▼
 ┌────────────────────────────────────────────────────────┐
 │ 2. Text Normalization                                  │
 │    • Lowercases text, strips punctuation & whitespace  │
 │    • Calculates Word Error Rate (WER) vs Ground Truth  │
 └─────────────────────────┬──────────────────────────────┘
                           │ Cleaned Text
                           ▼
 ┌────────────────────────────────────────────────────────┐
 │ 3. Predict Emotion from Text (PTE)                     │
 │    • Ingests transcribed text into NLP classifiers     │
 │    • Predicts emotion labels & confidence distributions│
 └─────────────────────────┬──────────────────────────────┘
                           │ Predicted Emotion + Scores
                           ▼
 ┌────────────────────────────────────────────────────────┐
 │ 4. Metrics Aggregation & JSON Tracking                 │
 │    • Computes Accuracy, Macro F1, and Weighted F1      │
 │    • Appends experiment runs to predict.json history   │
 └────────────────────────────────────────────────────────┘
```

### Core Features
* **Dual-Stage Processing**: Evaluate ASR and PTE individually or end-to-end.
* **Automated Speech Transcription (ASR)**: Native integration with HuggingFace ASR pipelines (e.g., OpenAI Whisper models) to convert speech into text.
* **Emotion Classification from Text (PTE)**: Classifies recognized text into target emotional states (e.g., joy, sadness, anger, fear) and outputs probability distributions across all labels.
* **Standardized Metric Suite**: Calculates Word Error Rate (WER), Accuracy, Macro F1-Score, and Weighted F1-Score.
* **Zero-Database JSON History**: All experiment logs, configurations, and per-sample outputs are logged directly into structured JSON files for complete reproducibility.

---

## 3. Installation & Setup

### Prerequisites
* Python 3.8 or higher
* `pip` package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/dminhwa/ViSPe.git
cd ViSPe
```

### Step 2: Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Project Structure

```text
ViSPe/
├── data/
│   └── sound.wav          # Input audio sample files
├── src/
│   ├── asr.py             # Standalone ASR engine class
│   ├── pte.py             # Standalone PTE emotion classification engine class
│   ├── config.json        # Pipeline configuration parameters
│   ├── sound.json         # Dataset manifest (audio paths, reference text & emotion)
│   ├── predict.json       # Combined pipeline execution logs and history
│   ├── wer.json           # Standalone ASR benchmark history
│   └── pte.json           # Standalone PTE benchmark history
├── pipeline.py            # Main entry point for end-to-end evaluation
├── README.md              # Project documentation
└── requirements.txt       # Dependencies list
```

### Manifest Schema Specification: `src/sound.json`

The dataset input file contains audio metadata, reference ground-truth text, and ground-truth emotion labels:

```json
{
  "sound-list": [
    {
      "name": "sample-1",
      "dir": "data/sound.wav",
      "text": "Xin chào Tôi là Minh là sinh viên Trường Đại học Công nghệ Thông tin",
      "emotion": "joy"
    }
  ]
}
```

| Field Key | Type | Description |
| :--- | :--- | :--- |
| `sound-list` | `Array` | List of audio sample objects to process. |
| `name` | `String` | Unique sample identifier or alias. |
| `dir` | `String` | Relative or absolute path to the audio file. |
| `text` | `String` | Ground-truth reference text for ASR WER evaluation. |
| `emotion` | `String` | Ground-truth reference emotion label for PTE evaluation. |

---

## 5. Usage

### 1. End-to-End Pipeline Evaluation
To run the full ASR + PTE workflow over all samples in `src/sound.json`:

```bash
python pipeline.py
```

### 2. Standalone ASR Evaluation
To benchmark speech recognition only:

```bash
python src/asr.py
```

### 3. Standalone PTE Evaluation
To benchmark text emotion classification only:

```bash
python src/pte.py
```

---

## 6. Running & Output Tracking

### Evaluation Metrics Calculation

1. **Text Normalization**: Strips punctuation, lowers case, and collapses whitespace before WER calculation.
2. **Word Error Rate (WER)**: 
   $$WER = \frac{S + D + I}{N}$$
   *(Substitutions $S$, Deletions $D$, Insertions $I$, Reference Words $N$)*.
3. **Accuracy**: Ratio of correctly predicted emotions over total samples.
4. **Macro F1-Score**: Unweighted mean of F1-scores across all emotion classes:
   $$Macro\ F1 = \frac{1}{|C|} \sum_{c \in C} F1_c$$
5. **Weighted F1-Score**: F1-score average weighted by the support of each emotion class:
   $$Weighted\ F1 = \sum_{c \in C} \left( \frac{N_c}{N_{total}} \times F1_c \right)$$

### Output Tracking History: `src/predict.json`

Results are automatically persisted after execution:

```json
{
  "tracking_history": [
    {
      "version": "v1.0.0-vispe",
      "timestamp": "2026-08-27 21:38:37",
      "models": {
        "asr": "openai/whisper-small",
        "pte": "bhadresh-savani/distilbert-base-uncased-emotion"
      },
      "total_samples": 1,
      "overall_metrics": {
        "wer": 0.0667,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "weighted_f1": 0.0
      },
      "details": [
        {
          "id": 1,
          "name": "sample-1",
          "audio_dir": "data/sound.wav",
          "asr_result": {
            "predicted_text": " Xin chào, tôi là Minh, là sinh viên trường đại học công ngựa thông tin.",
            "ground_truth_text": "Xin chào Tôi là Minh là sinh viên Trường Đại học Công nghệ Thông tin",
            "wer": 0.0667
          },
          "pte_result": {
            "predicted_emotion": "anger",
            "ground_truth_emotion": "joy",
            "score": 0.6405,
            "all_scores": {
              "anger": 0.6405,
              "fear": 0.2738,
              "joy": 0.0512
            }
          }
        }
      ]
    }
  ]
}
```

---

## 7. Workflow Overview

```text
┌───────────────────────────┐
│     Audio Input Data      │
│      (data/*.wav)         │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  Stage 1: ASR Engine      │
│  (src/asr.py)             │
└─────────────┬─────────────┘
              │ Transcribed Text
              ▼
┌───────────────────────────┐
│  Stage 2: PTE Engine      │
│  (src/pte.py)             │
└─────────────┬─────────────┘
              │ Predicted Emotion
              ▼
┌───────────────────────────┐
│  Stage 3: Evaluation      │
│  • WER Calculation        │
│  • Accuracy & F1 Scores   │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  Stage 4: JSON Logging    │
│  (src/predict.json)       │
└───────────────────────────┘
```

---

## 8. Author

* **Pham Thai Dang Minh**
* **Institution**: University of Information Technology (UIT - VNU-HCM)
* **Major**: Information Technology
* **GitHub**: [@dminhwa](https://github.com/dminhwa)
* **Repository**: [https://github.com/dminhwa/ViSPe.git](https://github.com/dminhwa/ViSPe.git)

---

## 9. License & Citation

### License
This project is open-source software licensed under the **MIT License**.

### Citation / Attribution
If you use, adapt, or build upon **ViSPe v1** in your research, academic work, or software projects, please cite and credit the project as follows:

```text
PHAM THAI DANG MINH - ViSPe - UITVNUHCM
```

#### BibTeX Citation

```bibtex
@misc{vispe2026,
  author = {Pham Thai Dang Minh},
  title = {ViSPe: Vietnamese Speech-PTE Evaluator},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/dminhwa/ViSPe.git}},
  institution = {University of Information Technology (UIT - VNU-HCM)}
}
```
