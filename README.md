# ViSPe: Vietnamese Speech-PTE Evaluator

**ViSPe** is an end-to-end evaluation pipeline that benchmark Automatic Speech Recognition (ASR) and Predict Emotion from Text (PTE) models. It processes raw audio to text, predicts emotion categories, computes performance metrics (WER & F1-Score), and tracks experiment history into unified JSON reports.

## 🚀 Features
- **ASR Evaluation:** Computes Word Error Rate (WER).
- **PTE Evaluation:** Computes Accuracy, Macro F1-Score, and Weighted F1-Score.
- **Experiment Tracking:** Stores full prediction history and metrics into `src/predict.json`.

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/ViSPe.git](https://github.com/YOUR_USERNAME/ViSPe.git)
cd ViSPe
2. Create a Virtual Environment & Install Dependencies
Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/MacOS:
source venv/bin/activate

pip install -r requirements.txt
🏃 Usage
1. Prepare Audio Data
Place your target audio files into the data/ folder and configure ground truth labels in src/sound.json:

JSON
{
  "sound-list": [
    {
      "name": "sample-1",
      "dir": "data/sound.wav",
      "text": "Trường Đại học Công nghệ Thông tin",
      "emotion": "joy"
    }
  ]
}
2. Run Evaluation Pipeline
Execute the main script:

Bash
python pipeline.py
Follow the interactive prompt to specify models or press Enter to use default models (openai/whisper-small & bhadresh-savani/distilbert-base-uncased-emotion).