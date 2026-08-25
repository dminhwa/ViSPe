from transformers import pipeline
import os, warnings
import time 
import torch, json 
import evaluate
import re
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = CURRENT_DIR 

class ViSPePipeline:
    def __init__(self, config_rel_path: str = os.path.join("src", "config.json")):
        config_full_path = os.path.join(BASE_DIR, config_rel_path)
        self.config = self.load_config(config_full_path)
        
        self.asr_model_name = None
        self.pte_model_name = None
        self.device = 0
        self.verbose = True
        
        self.asr_pipeline = None
        self.pte_pipeline = None
        
        self.wer_metric = evaluate.load("wer")
        self.f1_metric = evaluate.load("f1")
        self.accuracy_metric = evaluate.load("accuracy")
        
        self.setup_env()

    def load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        return {
            "paths": {
                "default_sound_text_data": os.path.join("src", "sound.json"),
                "predict_history_file": os.path.join("src", "predict.json")
            },
            "default_asr": {
                "model": "openai/whisper-small"
            },
            "default_pte": {
                "model": "bhadresh-savani/distilbert-base-uncased-emotion"
            }
        }

    def setup_env(self):
        warnings.filterwarnings("ignore")
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_SYMBOLS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def import_models(self):
        while True:
            try:
                def_asr = self.config.get("default_asr", {}).get("model", "openai/whisper-small")
                def_pte = self.config.get("default_pte", {}).get("model", "bhadresh-savani/distilbert-base-uncased-emotion")

                asr_inp = input(f"Input ASR Model (Default: {def_asr}): ").strip()
                pte_inp = input(f"Input PTE Model (Default: {def_pte}): ").strip()
                device_inp = input("Select Device (0: GPU, 1: CPU) [Default: 0]: ").strip()
                verbose_inp = input("Show detailed progress? (True/False) [Default: True]: ").strip().lower()

                self.asr_model_name = asr_inp if asr_inp else def_asr
                self.pte_model_name = pte_inp if pte_inp else def_pte
                self.device = 0 if device_inp != "1" and torch.cuda.is_available() else -1
                self.verbose = False if verbose_inp == "false" else True

                print(f"\n[ViSPe] Loading ASR Model '{self.asr_model_name}'...")
                self.asr_pipeline = pipeline(
                    task="automatic-speech-recognition",
                    model=self.asr_model_name,
                    device=self.device
                )

                print(f"[ViSPe] Loading PTE Model '{self.pte_model_name}'...")
                self.pte_pipeline = pipeline(
                    task="text-classification",
                    model=self.pte_model_name,
                    top_k=None,
                    device=self.device
                )

                print("[ViSPe] Both models loaded successfully.\n")
                break
            except Exception as e:
                print(f"Failed to load models ({e}), please try again!\n")

    def save_predict_history(self, version: str, overall_metrics: dict, details: list):
        rel_path = self.config["paths"].get("predict_history_file", os.path.join("src", "predict.json"))
        tracking_path = os.path.join(BASE_DIR, rel_path)

        tracking_data = {"tracking_history": []}
        if os.path.exists(tracking_path):
            try:
                with open(tracking_path, "r", encoding="utf-8") as f:
                    tracking_data = json.load(f)
            except Exception:
                pass

        new_entry = {
            "version": version,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": {
                "asr": self.asr_model_name,
                "pte": self.pte_model_name
            },
            "total_samples": len(details),
            "overall_metrics": overall_metrics,
            "details": details
        }

        tracking_data["tracking_history"].append(new_entry)

        with open(tracking_path, "w", encoding="utf-8") as f:
            json.dump(tracking_data, f, ensure_ascii=False, indent=2)
        print(f"\n--> Saved ViSPe prediction history to '{tracking_path}'")

    def process(self, version: str = "v1.0.0-vispe"):
        rel_data_path = self.config["paths"].get("default_sound_text_data", os.path.join("src", "sound.json"))
        data_path = os.path.join(BASE_DIR, rel_data_path)

        if not self.asr_pipeline or not self.pte_pipeline:
            self.import_models()

        if not os.path.exists(data_path):
            print(f"Error: Data file not found at '{data_path}'")
            return

        predictions_text = []
        references_text = []
        predictions_emotion = []
        references_emotion = []
        details_log = []

        with open(data_path, "r", encoding="utf-8") as f:
            sound_list = json.load(f)["sound-list"]

            for id_idx, sound_detail in enumerate(sound_list, 1):
                sound_name = sound_detail.get("name", f"sample_{id_idx}")
                raw_audio_dir = sound_detail["dir"]
                audio_dir = os.path.join(BASE_DIR, raw_audio_dir) if not os.path.isabs(raw_audio_dir) else raw_audio_dir
                
                target_text = sound_detail.get("text", "")
                target_emotion = sound_detail.get("emotion", None)

                if self.verbose:
                    print(f"[{id_idx}/{len(sound_list)}] Processing: {sound_name}")

                if not os.path.exists(audio_dir):
                    if self.verbose:
                        print(f"Skip: Audio file missing at '{audio_dir}'\n")
                    continue

                # 1. ASR Execution
                asr_raw_out = self.asr_pipeline(audio_dir)["text"]
                norm_predicted_text = self.normalize_text(asr_raw_out)
                norm_target_text = self.normalize_text(target_text)

                sample_wer = round(self.wer_metric.compute(
                    predictions=[norm_predicted_text], 
                    references=[norm_target_text]
                ), 4)

                # 2. PTE Execution
                pte_raw_out = self.pte_pipeline(asr_raw_out)[0]
                top_emotion_item = max(pte_raw_out, key=lambda x: x["score"])
                predicted_emotion = top_emotion_item["label"]
                emotion_score = round(float(top_emotion_item["score"]), 4)

                formatted_scores = {
                    pred["label"]: round(float(pred["score"]), 4) for pred in pte_raw_out
                }

                if self.verbose:
                    print(f"  ASR Predicted : {asr_raw_out}")
                    print(f"  ASR Target    : {target_text}")
                    print(f"  Sample WER    : {round(sample_wer * 100, 2)}%")
                    print(f"  PTE Predicted : {predicted_emotion} ({round(emotion_score * 100, 2)}%)")
                    if target_emotion:
                        print(f"  PTE Target    : {target_emotion}")
                    print("-" * 40)

                predictions_text.append(norm_predicted_text)
                references_text.append(norm_target_text)

                if target_emotion:
                    predictions_emotion.append(predicted_emotion)
                    references_emotion.append(target_emotion)

                details_log.append({
                    "id": id_idx,
                    "name": sound_name,
                    "audio_dir": raw_audio_dir,
                    "asr_result": {
                        "predicted_text": asr_raw_out,
                        "ground_truth_text": target_text,
                        "wer": sample_wer
                    },
                    "pte_result": {
                        "predicted_emotion": predicted_emotion,
                        "ground_truth_emotion": target_emotion,
                        "score": emotion_score,
                        "all_scores": formatted_scores
                    }
                })

                time.sleep(0.1)

        # 3. Computing Metrics
        overall_metrics = {}
        if references_text and predictions_text:
            overall_wer = self.wer_metric.compute(
                predictions=predictions_text, 
                references=references_text
            )
            overall_metrics["wer"] = round(overall_wer, 4)

        if references_emotion and predictions_emotion:
            labels = sorted(list(set(references_emotion + predictions_emotion)))
            label2id = {lbl: idx for idx, lbl in enumerate(labels)}

            preds_id = [label2id[p] for p in predictions_emotion]
            refs_id = [label2id[r] for r in references_emotion]

            acc = self.accuracy_metric.compute(predictions=preds_id, references=refs_id)["accuracy"]
            macro_f1 = self.f1_metric.compute(predictions=preds_id, references=refs_id, average="macro")["f1"]
            weighted_f1 = self.f1_metric.compute(predictions=preds_id, references=refs_id, average="weighted")["f1"]

            overall_metrics["accuracy"] = round(acc, 4)
            overall_metrics["macro_f1"] = round(macro_f1, 4)
            overall_metrics["weighted_f1"] = round(weighted_f1, 4)

        if self.verbose and overall_metrics:
            print("\n==========================================")
            print("      ViSPe OVERALL EVALUATION RESULTS    ")
            print("==========================================")
            if "wer" in overall_metrics:
                print(f" OVERALL WER   : {round(overall_metrics['wer'] * 100, 2)}%")
            if "macro_f1" in overall_metrics:
                print(f" ACCURACY      : {round(overall_metrics['accuracy'] * 100, 2)}%")
                print(f" MACRO F1      : {round(overall_metrics['macro_f1'] * 100, 2)}%")
                print(f" WEIGHTED F1   : {round(overall_metrics['weighted_f1'] * 100, 2)}%")
            print("==========================================")

        if details_log:
            self.save_predict_history(version=version, overall_metrics=overall_metrics, details=details_log)

if __name__ == "__main__":
    app = ViSPePipeline()
    app.import_models()
    app.process(version="v1.0.0-vispe") 