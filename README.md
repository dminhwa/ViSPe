# ViSPe — Vietnamese Speech-PTE Evaluator (v1)

> **End-to-End Speech Evaluation Pipeline combining Automatic Speech Recognition (ASR) and Predict Emotion from Text (PTE) for Vietnamese Audio**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![ASR](https://img.shields.io/badge/ASR-Automatic%20Speech%20Recognition-orange.svg)](#modules-overview)
[![PTE](https://img.shields.io/badge/PTE-Predict%20Emotion%20from%20Text-purple.svg)](#modules-overview)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

**ViSPe (Vietnamese Speech-PTE Evaluator)** is an open-source, modular Python framework designed to evaluate speech recognition (ASR) and text-based emotion classification (PTE) joint systems on Vietnamese speech datasets.

The system transcribes Vietnamese audio into text using ASR models, passes the recognized text into PTE models to infer speaker emotions, benchmarks performance against ground-truth labels using standard NLP/Speech metrics, and records detailed execution logs in structured JSON format.

---

## Key Features

* **Modular Architecture**: Run ASR independently, PTE independently, or end-to-end via `pipeline.py`.
* **Automatic Speech Recognition (ASR)**: Transcribe Vietnamese speech to text and compute standardized Word Error Rate (WER) with automated text normalization.
* **Predict Emotion from Text (PTE)**: Ingest transcribed text to predict multi-class emotion distributions and compute Accuracy, Macro F1, and Weighted F1 scores.
* **Structured JSON Experiment Tracking**: Save setup configs, dataset manifests, and fine-grained benchmarking history into JSON format without needing an external database.
* **Open-Source & Unrestricted**: Free to modify, distribute, and integrate into open-source, academic, or commercial software.

---

## Modules Overview

ViSPe is structured into independent components inside `src/` and orchestrated via root entry scripts:

### 1. `src/asr.py` (`ASR` Engine)
* **Purpose**: Handles Speech-to-Text inference and speech recognition quality evaluation.
* **Key Functions**:
  * `init_model()`: Loads HuggingFace ASR models (e.g., `openai/whisper-small`).
  * `normalize_text(text)`: Standardizes text by converting to lowercase, removing punctuation, and collapsing whitespace for fair metric comparison.
  * `compute_wer(preds, refs)`: Uses HuggingFace `evaluate` (`wer` metric) to compute Word Error Rate.
  * `log_history(ver, total_wer, details)`: Logs execution metadata and sample-level WER records into `src/wer.json`.

### 2. `src/pte.py` (`PTE` Engine)
* **Purpose**: Performs text classification to infer emotional states from transcribed text.
* **Key Functions**:
  * `init_model()`: Loads HuggingFace text-classification models (e.g., `bhadresh-savani/distilbert-base-uncased-emotion`).
  * `compute_metrics(preds, refs)`: Computes Accuracy, Macro F1-Score, and Weighted F1-Score using HuggingFace `evaluate`.
  * `log_history(ver, metrics, details)`: Saves prediction probability distributions and metric summaries into `src/pte.json`.

### 3. `pipeline.py` (`ViSPePipeline` Orchestrator)
* **Purpose**: Manages the complete end-to-end pipeline across ASR and PTE modules.
* **Execution Flow**:
  1. Reads sample entries from `src/sound.json`.
  2. Executes ASR (`Audio -> Transcribed Text`) and calculates sample-level WER.
  3. Passes ASR output directly to PTE (`Transcribed Text -> Predicted Emotion`) and captures confidence scores for all emotion classes.
  4. Calculates overall dataset-level evaluation metrics (WER, Accuracy, Macro F1, Weighted F1).
  5. Stores comprehensive experiment results in `src/predict.json`.

---

## Calculation & Evaluation Methodology

### 1. Text Normalization
Before calculating speech recognition error rates, both predicted text ($\hat{y}_{text}$) and reference ground-truth text ($y_{text}$) undergo automated preprocessing:
* Conversion to lowercase.
* Stripping all punctuation marks (`[\!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]`).
* Collapsing extra whitespace into a single space.

### 2. Word Error Rate (WER)
Word Error Rate measures speech recognition accuracy at the word level using Levenshtein distance:

$$\text{WER} = \frac{S + D + I}{N} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Number of Reference Words}}$$

* A lower WER score represents higher transcription accuracy.

### 3. Classification Accuracy
Evaluates the ratio of correctly predicted emotion labels over total valid labeled samples ($N_{emo}$):

$$\text{Accuracy} = \frac{\sum_{i=1}^{N_{emo}} \mathbb{I}(\hat{y}_{emo,i} = y_{emo,i})}{N_{emo}}$$

### 4. Macro F1-Score
Calculates the unweighted average of F1-scores across all emotion categories $C$:

$$\text{Macro F1} = \frac{1}{\vert{}C\vert{}} \sum_{c \in C} \text{F1}_c$$

$$\text{F1}_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

* Evaluates performance evenly across all emotion classes regardless of dataset balance.

### 5. Weighted F1-Score
Calculates the average F1-score weighted by the support (true sample count $N_c$) of each class:

$$\text{Weighted F1} = \sum_{c \in C} \left( \frac{N_c}{N_{emo}} \times \text{F1}_c \right)$$

---

## Directory & File Structure

```text
ViSPe/
├── data/
│   └── sound.wav          # Input audio sample files
├── src/
│   ├── asr.py             # ASR evaluation engine class
│   ├── pte.py             # PTE emotion classification engine class
│   ├── config.json        # Pipeline configuration settings
│   ├── sound.json         # Dataset manifest (paths, transcripts, emotion labels)
│   ├── predict.json       # End-to-end pipeline evaluation history
│   ├── wer.json           # Standalone ASR tracking history
│   └── pte.json           # Standalone PTE tracking history
├── pipeline.py            # Main entry point for end-to-end pipeline
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies
Dataset Format & Schema: src/sound.jsonAudio samples, target transcriptions, and ground-truth emotion labels are defined in src/sound.json.Manifest JSON ExampleJSON{
  "sound-list": [
    {
      "name": "sample-1",
      "dir": "data/sound.wav",
      "text": "Xin chào Tôi là Minh là sinh viên Trường Đại học Công nghệ Thông tin",
      "emotion": "joy"
    },
    {
      "name": "sample-2",
      "dir": "data/sample_02.wav",
      "text": "Tôi thực sự rất buồn về chuyện này.",
      "emotion": "sadness"
    }
  ]
}
Manifest Fields DescriptionField KeyData TypeRequiredDescriptionsound-listArrayYesList of audio sample records to process.nameStringYesUnique identifier/alias for the audio entry.dirStringYesPath to the audio file (relative to root or absolute).textStringYesReference text transcript used for ASR WER evaluation.emotionStringOptionalReference ground-truth emotion label used for PTE evaluation.Installation & Setup1. Clone RepositoryBashgit clone [https://github.com/dminhwa/ViSPe.git](https://github.com/dminhwa/ViSPe.git)
cd ViSPe
2. Set Up Virtual EnvironmentWindowsBashpython -m venv .venv
.venv\Scripts\activate
Linux / macOSBashpython3 -m venv .venv
source .venv/bin/activate
3. Install DependenciesBashpip install -r requirements.txt
Usage Guide1. End-to-End Evaluation PipelineTo execute full ASR transcription and PTE emotion classification:Bashpython pipeline.py
2. Standalone ASR BenchmarkingTo test speech recognition quality only:Bashpython src/asr.py
3. Standalone PTE BenchmarkingTo test emotion classification performance on raw text:Bashpython src/pte.py
Output & Tracking Format: src/predict.jsonAll execution results are automatically written to src/predict.json:JSON{
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
LicenseThis project is licensed under the MIT License. It is open-source software — you are free to use, modify, distribute, and integrate this project without restrictions or credit requirements.