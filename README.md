# ViSPe — Vietnamese Speech-PTE Evaluator

> **End-to-End Speech Evaluation Pipeline combining ASR and PTE for Vietnamese audio**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![ASR](https://img.shields.io/badge/ASR-Automatic%20Speech%20Recognition-orange.svg)](#features)
[![PTE](https://img.shields.io/badge/PTE-Predict%20Emotion%20from%20Text-purple.svg)](#features)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

**ViSPe (Vietnamese Speech-PTE Evaluator)** is an end-to-end evaluation system that combines **Automatic Speech Recognition (ASR)** with **Predict Emotion from Text (PTE)** to analyze Vietnamese speech.

The pipeline converts audio into text, predicts the speaker's emotion, evaluates recognition and classification performance, and stores experimental results in JSON format for later analysis.

**Author:** Pham Thai Dang Minh
**Affiliation:** University of Information Technology (UIT - VNU-HCM), Class K21

**Repository:** https://github.com/dminhwa/ViSPe.git

---

## Features

* **ASR Evaluation**

  * Convert Vietnamese speech into text.
  * Calculate **Word Error Rate (WER)** to evaluate transcription quality.

* **PTE Emotion Evaluation**

  * Predict emotions from recognized text.
  * Calculate **Accuracy** and **F1-Score**.
  * Support both **Macro F1-Score** and **Weighted F1-Score**.

* **Experiment Tracking**

  * Store experiment configurations and evaluation results in JSON files.
  * Keep prediction and evaluation history for reproducibility and comparison.

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/dminhwa/ViSPe.git
cd ViSPe
```

### 2. Create a Virtual Environment

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

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
ViSPe/
├── data/
│   └── audio files
│
├── src/
│   ├── config.json
│   ├── sound.json
│   └── predict.json
│
├── pipeline.py
├── README.md
└── requirements.txt
```

### File Description

| File / Directory   | Description                                                                      |
| ------------------ | -------------------------------------------------------------------------------- |
| `data/`            | Stores input audio files used for evaluation.                                    |
| `src/config.json`  | Stores system and pipeline configuration.                                        |
| `src/sound.json`   | Contains metadata for audio samples, including transcription and emotion labels. |
| `src/predict.json` | Stores prediction and evaluation results.                                        |
| `pipeline.py`      | Main entry point for the complete ASR + PTE evaluation pipeline.                 |
| `requirements.txt` | Lists required Python dependencies.                                              |
| `README.md`        | Project documentation.                                                           |

---

## Usage

### 1. Add Audio Files

Place your Vietnamese audio files inside the `data/` directory.

For example:

```text
data/
├── sample_01.wav
├── sample_02.wav
└── sample_03.wav
```

The audio filenames and paths should correspond to the entries defined in `src/sound.json`.

### 2. Configure `sound.json`

The `src/sound.json` file contains metadata for each audio sample.

Example:

```json
{
  "sound-list": [
    {
      "name": "sample_01",
      "dir": "data/sample_01.wav",
      "text": "Xin chào, hôm nay bạn cảm thấy thế nào?",
      "emotion": "happy"
    },
    {
      "name": "sample_02",
      "dir": "data/sample_02.wav",
      "text": "Tôi thực sự rất buồn về chuyện này.",
      "emotion": "sad"
    },
    {
      "name": "sample_03",
      "dir": "data/sample_03.wav",
      "text": "Tại sao bạn lại làm như vậy?",
      "emotion": "angry"
    }
  ]
}
```

### JSON Fields

| Field     | Description                                                     |
| --------- | --------------------------------------------------------------- |
| `name`    | Unique identifier for the audio sample.                         |
| `dir`     | Path to the corresponding audio file.                           |
| `text`    | Reference transcription (ground-truth text) for ASR evaluation. |
| `emotion` | Ground-truth emotion label for PTE evaluation.                  |

---

## Running the Pipeline

After completing the setup and adding the required data, run:

```bash
python pipeline.py
```

The pipeline will process the configured audio samples and perform the following workflow:

```text
Audio Input
    ↓
Automatic Speech Recognition (ASR)
    ↓
Transcribed Text
    ↓
Predict Emotion from Text (PTE)
    ↓
Evaluation
    ├── WER
    ├── Accuracy
    ├── Macro F1-Score
    └── Weighted F1-Score
    ↓
JSON Experiment Tracking
```

---

## Evaluation Metrics

### Word Error Rate (WER)

WER measures the difference between the reference transcription and the ASR-generated transcription.

A lower WER indicates better speech recognition performance.

### Accuracy

Accuracy measures the percentage of correctly predicted emotion labels:

```text
Accuracy = Correct Predictions / Total Predictions
```

### Macro F1-Score

Macro F1 calculates the F1-score independently for each emotion class and then takes their unweighted average.

This metric gives every class equal importance.

### Weighted F1-Score

Weighted F1 calculates the F1-score for each class and averages them according to the number of samples in each class.

This is useful when the dataset contains imbalanced emotion classes.

---

## Output & Experiment Tracking

ViSPe stores prediction and evaluation information in JSON format, allowing experiments to be:

* Reproduced.
* Compared across different runs.
* Analyzed after pipeline execution.
* Tracked without requiring an external database.

The generated or updated prediction information is stored in:

```text
src/predict.json
```

---

## Workflow Overview

```text
┌─────────────────┐
│  Audio Dataset  │
│     data/       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│       ASR       │
│ Speech → Text   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│       PTE       │
│ Text → Emotion  │
└────────┬────────┘
         │
         ▼
┌────────────────────────────┐
│      Evaluation            │
│ WER / Accuracy / F1-Score  │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│   JSON Experiment History  │
│       predict.json         │
└────────────────────────────┘
```

---

## Author

**Pham Thai Dang Minh**

University of Information Technology (UIT - VNU-HCM)
Class K21

GitHub: [@dminhwa](https://github.com/dminhwa)

Repository: [ViSPe](https://github.com/dminhwa/ViSPe.git)

---

## Citation

If you use ViSPe in an academic project, research project, or other publication, please acknowledge the project and its author:

> **Pham Thai Dang Minh. ViSPe: Vietnamese Speech-PTE Evaluator. University of Information Technology (UIT - VNU-HCM), K21.**

Repository:

```text
https://github.com/dminhwa/ViSPe.git
```

---

## License

This project is intended for educational and research purposes.

See the repository for the applicable license and usage terms.
