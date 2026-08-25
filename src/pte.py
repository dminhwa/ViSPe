from transformers import pipeline
import os, warnings
import time 
import torch, json 
import evaluate
import re
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../"))

class PTE:
    def __init__(self, config_rel_path: str = "config.json"):
        config_full_path = os.path.join(BASE_DIR, config_rel_path)
        self.config = self.load_config(config_full_path)
        self.model = None
        self.device = 0
        self.ignoreFlags = True
        self.verbose = True
        self.pipeline = None
        self.f1_metric = evaluate.load("f1")
        self.accuracy_metric = evaluate.load("accuracy")
        self.setupEnv()

    def load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "paths": {
                "default_sound_text_data": "src/sound-textData.json",
                "pte_tracking_file": "src/pte.json"
            },
            "default_pte": {
                "model": "bhadresh-savani/distilbert-base-uncased-emotion"
            }
        }

    def compute_metrics(self, predictions: list, references: list) -> dict:
        labels = sorted(list(set(references + predictions)))
        label2id = {label: i for i, label in enumerate(labels)}
        
        preds_id = [label2id[p] for p in predictions]
        refs_id = [label2id[r] for r in references]

        acc = self.accuracy_metric.compute(predictions=preds_id, references=refs_id)["accuracy"]
        macro_f1 = self.f1_metric.compute(predictions=preds_id, references=refs_id, average="macro")["f1"]
        weighted_f1 = self.f1_metric.compute(predictions=preds_id, references=refs_id, average="weighted")["f1"]

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4)
        }

    def save_pte_log(self, version: str, metrics: dict, details: list):
        rel_path = self.config["paths"].get("pte_tracking_file", "src/pte.json")
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
            "model_name": self.model,
            "total_samples": len(details),
            "metrics": metrics,
            "details": details
        }
        
        tracking_data["tracking_history"].append(new_entry)

        with open(tracking_path, "w", encoding="utf-8") as f:
            json.dump(tracking_data, f, ensure_ascii=False, indent=2)
        print(f"\n--> Saved PTE tracking result with F1-Score to '{tracking_path}'")

    def setupEnv(self):
        if self.ignoreFlags: 
            warnings.filterwarnings("ignore")
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["HF_HUB_DISABLE_SYMBOLS_WARNING"] = "1"
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    def importModel(self):
        while True: 
            try:
                default_model = self.config.get("default_pte", {}).get("model", "bhadresh-savani/distilbert-base-uncased-emotion")
                model_inp = input(f"Input Model (Default: {default_model}): ").strip()
                device_inp = input("Select Device (0: GPU, 1: CPU) [Default: 0]: ").strip()
                verbose_inp = input("Show detailed progress? (True/False) [Default: True]: ").strip().lower()

                self.model = model_inp if model_inp else default_model
                self.device = 0 if device_inp != "1" and torch.cuda.is_available() else -1
                self.verbose = False if verbose_inp == "false" else True

                print(f"\nLoading text emotion model '{self.model}' on {'GPU' if self.device == 0 else 'CPU'}...")
                self.pipeline = pipeline(
                    task = "text-classification",
                    model = self.model,
                    top_k = None,
                    device = self.device
                )
                print("Model loaded successfully.\n")
                break
            except Exception as e:
                print(f"Get Model Failed ({e}), Please try again!\n") 

    def process(self, version: str = "v1.0.0-baseline"):
        rel_data_path = self.config["paths"]["default_sound_text_data"]
        dir_path = os.path.join(BASE_DIR, rel_data_path)

        print("PTE is processing text emotion prediction & computing F1-Score...\n")
        
        if not self.pipeline:
            print("Model not loaded. Please load the model first.")
            self.importModel()
            
        if not os.path.exists(dir_path): 
            print(f"Error [101]: Cannot find the data file [.json] at '{dir_path}'")
            return None

        references = []
        predictions = []
        details_log = []

        with open(dir_path, "r", encoding="utf-8") as file_data:
            sound_list = json.load(file_data)["sound-list"]
            
            for id_idx, sound_detail in enumerate(sound_list, 1):
                sound_name = sound_detail.get("name", f"sample_{id_idx}")
                input_text = sound_detail.get("text", "")
                ground_truth = sound_detail.get("emotion", None)  # Nhãn cảm xúc thực tế (nếu có trong dataset)

                if self.verbose:
                    print(f"File {id_idx} | {sound_name}")
                
                if not input_text.strip():
                    if self.verbose:
                        print(f"Skip: Input text is empty\n")
                    continue

                preds = self.pipeline(input_text)[0]
                
                formatted_scores = {
                    pred["label"]: round(float(pred["score"]), 4) for pred in preds
                }
                top_emotion = max(preds, key=lambda x: x["score"])["label"]
                top_score = max(preds, key=lambda x: x["score"])["score"]

                if self.verbose:
                    print(f"INPUT : {input_text}")
                    if ground_truth:
                        print(f"TARGET: {ground_truth}")
                    print(f"OUTPUT: {top_emotion} ({round(top_score * 100, 2)}%)\n")

                if ground_truth:
                    references.append(ground_truth)
                    predictions.append(top_emotion)

                details_log.append({
                    "id": id_idx,
                    "name": sound_name,
                    "text": input_text,
                    "ground_truth": ground_truth,
                    "top_emotion": top_emotion,
                    "top_score": round(float(top_score), 4),
                    "all_scores": formatted_scores
                })
                
                time.sleep(0.1)

        metrics = {}
        if references and predictions:
            metrics = self.compute_metrics(predictions=predictions, references=references)
            print("==========================================")
            print(f" ACCURACY    : {round(metrics['accuracy'] * 100, 2)}%")
            print(f" MACRO F1    : {round(metrics['macro_f1'] * 100, 2)}%")
            print(f" WEIGHTED F1 : {round(metrics['weighted_f1'] * 100, 2)}%")
            print("==========================================")

        if details_log:
            self.save_pte_log(version=version, metrics=metrics, details=details_log)

if __name__ == "__main__":
    app = PTE()
    app.importModel()
    app.process(version="v1.0.0-baseline")