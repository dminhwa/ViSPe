from transformers import pipeline
import os, warnings, time, torch, json, evaluate
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(ROOT_DIR, "../"))

class PTE:
    """
    Predict Text Emotion (PTE) engine for analyzing text data, 
    predicting emotion labels, and benchmarking metrics (Accuracy, F1-Score).
    """
    
    def __init__(self, cfg_path: str = "config.json"):
        """
        Initialize PTE component and load system configuration.
        
        Args:
            cfg_path (str): Relative path to configuration file.
        """
        cfg_full = os.path.join(BASE_DIR, cfg_path)
        self.cfg = self._load_cfg(cfg_full)
        self.model = None
        self.device = 0
        self.ignore_flags = True
        self.verbose = True
        self.pipeline = None
        self.f1_metric = evaluate.load("f1")
        self.acc_metric = evaluate.load("accuracy")
        self.setup_env()

    def _load_cfg(self, path: str) -> dict:
        """
        Load configuration from JSON file or return fallback structure.
        
        Args:
            path (str): Absolute file path to config JSON.
            
        Returns:
            dict: Parsed configuration dictionary.
        """
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

    def compute_metrics(self, preds: list, refs: list) -> dict:
        """
        Compute Accuracy and F1-Scores (Macro, Weighted) for predictions.
        
        Args:
            preds (list): List of predicted emotion labels.
            refs (list): List of ground truth reference labels.
            
        Returns:
            dict: Dictionary containing accuracy, macro_f1, and weighted_f1 scores.
        """
        labels = sorted(list(set(refs + preds)))
        label2id = {label: i for i, label in enumerate(labels)}
        
        preds_id = [label2id[p] for p in preds]
        refs_id = [label2id[r] for r in refs]

        acc = self.acc_metric.compute(predictions=preds_id, references=refs_id)["accuracy"]
        macro_f1 = self.f1_metric.compute(predictions=preds_id, references=refs_id, average="macro")["f1"]
        weight_f1 = self.f1_metric.compute(predictions=preds_id, references=refs_id, average="weighted")["f1"]

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weight_f1, 4)
        }

    def log_history(self, ver: str, metrics: dict, details: list):
        """
        Append experiment results and metadata into tracking history JSON file.
        
        Args:
            ver (str): Current experiment version string.
            metrics (dict): Computed overall performance metrics.
            details (list): Sample-level prediction logs.
        """
        rel_path = self.cfg["paths"].get("pte_tracking_file", "src/pte.json")
        out_path = os.path.join(BASE_DIR, rel_path)
        
        hist = {"tracking_history": []}
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    hist = json.load(f)
            except Exception:
                pass

        entry = {
            "version": ver,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.model,
            "total_samples": len(details),
            "metrics": metrics,
            "details": details
        }
        
        hist["tracking_history"].append(entry)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
        print(f"\n--> Saved PTE tracking result with F1-Score to '{out_path}'")

    def setup_env(self):
        """
        Suppress warnings and disable unnecessary logs from HuggingFace/TensorFlow.
        """
        if self.ignore_flags: 
            warnings.filterwarnings("ignore")
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["HF_HUB_DISABLE_SYMBOLS_WARNING"] = "1"
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    def init_model(self):
        """
        Interactively trigger model loading routines for the text classification engine.
        """
        while True: 
            try:
                def_model = self.cfg.get("default_pte", {}).get("model", "bhadresh-savani/distilbert-base-uncased-emotion")
                m_inp = input(f"Input Model (Default: {def_model}): ").strip()
                dev_inp = input("Select Device (0: GPU, 1: CPU) [Default: 0]: ").strip()
                verb_inp = input("Show detailed progress? (True/False) [Default: True]: ").strip().lower()

                self.model = m_inp if m_inp else def_model
                self.device = 0 if dev_inp != "1" and torch.cuda.is_available() else -1
                self.verbose = False if verb_inp == "false" else True

                print(f"\nLoading text emotion model '{self.model}' on {'GPU' if self.device == 0 else 'CPU'}...")
                self.pipeline = pipeline(
                    task="text-classification",
                    model=self.model,
                    top_k=None,
                    device=self.device
                )
                print("Model loaded successfully.\n")
                break
            except Exception as e:
                print(f"Get Model Failed ({e}), Please try again!\n") 

    def run(self, ver: str = "v1.0.0-baseline"):
        """
        Execute PTE pipeline over data sample batch and compute overall metrics.
        
        Args:
            ver (str): Release tag or experiment version.
        """
        rel_data = self.cfg["paths"]["default_sound_text_data"]
        data_path = os.path.join(BASE_DIR, rel_data)

        print("PTE is processing text emotion prediction & computing F1-Score...\n")
        
        if not self.pipeline:
            print("Model not loaded. Please load the model first.")
            self.init_model()
            
        if not os.path.exists(data_path): 
            print(f"Error [101]: Cannot find the data file [.json] at '{data_path}'")
            return None

        refs, preds, details = [], [], []

        with open(data_path, "r", encoding="utf-8") as f:
            samples = json.load(f).get("sound-list", [])
            
            for idx, item in enumerate(samples, 1):
                s_name = item.get("name", f"sample_{idx}")
                inp_txt = item.get("text", "")
                gt_emo = item.get("emotion", None) 

                if self.verbose:
                    print(f"File {idx} | {s_name}")
                
                if not inp_txt.strip():
                    if self.verbose:
                        print(f"Skip: Input text is empty\n")
                    continue

                res = self.pipeline(inp_txt)[0]
                
                fmt_scores = {p["label"]: round(float(p["score"]), 4) for p in res}
                top_emo = max(res, key=lambda x: x["score"])["label"]
                top_score = max(res, key=lambda x: x["score"])["score"]

                if self.verbose:
                    print(f"INPUT : {inp_txt}")
                    if gt_emo:
                        print(f"TARGET: {gt_emo}")
                    print(f"OUTPUT: {top_emo} ({round(top_score * 100, 2)}%)\n")

                if gt_emo:
                    refs.append(gt_emo)
                    preds.append(top_emo)

                details.append({
                    "id": idx,
                    "name": s_name,
                    "text": inp_txt,
                    "ground_truth": gt_emo,
                    "top_emotion": top_emo,
                    "top_score": round(float(top_score), 4),
                    "all_scores": fmt_scores
                })
                
                time.sleep(0.1)

        metrics_res = {}
        if refs and preds:
            metrics_res = self.compute_metrics(preds=preds, refs=refs)
            print("==========================================")
            print(f" ACCURACY    : {round(metrics_res['accuracy'] * 100, 2)}%")
            print(f" MACRO F1    : {round(metrics_res['macro_f1'] * 100, 2)}%")
            print(f" WEIGHTED F1 : {round(metrics_res['weighted_f1'] * 100, 2)}%")
            print("==========================================")

        if details:
            self.log_history(ver=ver, metrics=metrics_res, details=details)

if __name__ == "__main__":
    app = PTE()
    app.init_model()
    app.run(ver="v1.0.0-baseline")