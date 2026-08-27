import os
import json
import time
from datetime import datetime

from src.asr import ASR
from src.pte import PTE

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class ViSPePipeline:
    """
    Main evaluation pipeline orchestrating ASR (Speech-to-Text) 
    and PTE (Text-to-Emotion) modules to benchmark joint system metrics.
    """
    
    def __init__(self, cfg_path: str = os.path.join("src", "config.json")):
        """
        Initialize pipeline components and load general configuration.
        
        Args:
            cfg_path (str): Relative path to configuration file.
        """
        self.cfg_path = cfg_path
        self.cfg = self._load_cfg(os.path.join(ROOT_DIR, cfg_path))
        
        # Instantiate base module engines
        self.asr_mod = ASR(config_rel_path=cfg_path)
        self.pte_mod = PTE(config_rel_path=cfg_path)
        self.verbose = True

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
                "default_sound_text_data": os.path.join("src", "sound.json"),
                "predict_history_file": os.path.join("src", "predict.json")
            }
        }

    def init_models(self):
        """
        Interactively trigger model loading routines for both ASR and PTE engines.
        """
        print("=== [ViSPe] STEP 1: Setting up ASR Model ===")
        self.asr_mod.importModel()
        
        print("\n=== [ViSPe] STEP 2: Setting up PTE Model ===")
        self.pte_mod.importModel()
        
        self.verbose = self.asr_mod.verbose

    def log_history(self, ver: str, metrics: dict, details: list):
        """
        Append detailed benchmark results and metadata into tracking history JSON file.
        
        Args:
            ver (str): Current experiment version string.
            metrics (dict): Evaluated metrics summary (WER, F1, Accuracy).
            details (list): Sample-level prediction logs.
        """
        rel_path = self.cfg.get("paths", {}).get("predict_history_file", os.path.join("src", "predict.json"))
        out_path = os.path.join(ROOT_DIR, rel_path)

        hist_data = {"tracking_history": []}
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    hist_data = json.load(f)
            except Exception:
                pass

        entry = {
            "version": ver,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": {
                "asr": self.asr_mod.model,
                "pte": self.pte_mod.model
            },
            "total_samples": len(details),
            "overall_metrics": metrics,
            "details": details
        }

        hist_data["tracking_history"].append(entry)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(hist_data, f, ensure_ascii=False, indent=2)
        print(f"\n--> Saved ViSPe evaluation history to '{out_path}'")

    def run(self, ver: str = "v1.0.0-vispe"):
        """
        Execute end-to-end processing pipeline over data sample batch.
        
        Args:
            ver (str): Release tag or experiment version.
        """
        rel_data_path = self.cfg.get("paths", {}).get("default_sound_text_data", os.path.join("src", "sound.json"))
        data_path = os.path.join(ROOT_DIR, rel_data_path)

        # Lazy loading fallback
        if not self.asr_mod.pipeline or not self.pte_mod.pipeline:
            self.init_models()

        if not os.path.exists(data_path):
            print(f"Error: Data file not found at '{data_path}'")
            return

        txt_preds, txt_refs = [], []
        emo_preds, emo_refs = [], []
        details = []

        print("\n=== [ViSPe] Running End-to-End Pipeline Evaluation ===")
        
        with open(data_path, "r", encoding="utf-8") as f:
            samples = json.load(f).get("sound-list", [])

            for idx, item in enumerate(samples, 1):
                s_name = item.get("name", f"sample_{idx}")
                raw_path = item["dir"]
                aud_path = os.path.join(ROOT_DIR, raw_path) if not os.path.isabs(raw_path) else raw_path
                
                gt_txt = item.get("text", "")
                gt_emo = item.get("emotion", None)

                if self.verbose:
                    print(f"\n[{idx}/{len(samples)}] Processing: {s_name}")

                if not os.path.exists(aud_path):
                    if self.verbose:
                        print(f"Skip: Audio missing at '{aud_path}'")
                    continue

                # 1. ASR Stage: Audio -> Transcribed Text
                asr_out = self.asr_mod.pipeline(
                    aud_path,
                    generate_kwargs={"task": "transcribe", "language": "vietnamese"},
                    return_timestamps=False
                )
                pred_txt = asr_out["text"]
                wer = round(self.asr_mod.compute_wer(predictions=[pred_txt], references=[gt_txt]), 4)

                # 2. PTE Stage: Transcribed Text -> Emotion Classification
                pte_out = self.pte_mod.pipeline(pred_txt)[0]
                top_item = max(pte_out, key=lambda x: x["score"])
                pred_emo = top_item["label"]
                emo_score = round(float(top_item["score"]), 4)

                scores_map = {res["label"]: round(float(res["score"]), 4) for res in pte_out}

                if self.verbose:
                    print(f"  ASR Predicted : {pred_txt}")
                    print(f"  ASR Target    : {gt_txt}")
                    print(f"  Sample WER    : {round(wer * 100, 2)}%")
                    print(f"  PTE Predicted : {pred_emo} ({round(emo_score * 100, 2)}%)")
                    if gt_emo:
                        print(f"  PTE Target    : {gt_emo}")
                    print("-" * 40)

                txt_preds.append(pred_txt)
                txt_refs.append(gt_txt)

                if gt_emo:
                    emo_preds.append(pred_emo)
                    emo_refs.append(gt_emo)

                rec = {
                    "id": idx,
                    "name": s_name,
                    "audio_dir": raw_path,
                    "asr_result": {
                        "predicted_text": pred_txt,
                        "ground_truth_text": gt_txt,
                        "wer": wer
                    },
                    "pte_result": {
                        "predicted_emotion": pred_emo,
                        "ground_truth_emotion": gt_emo,
                        "score": emo_score,
                        "all_scores": scores_map
                    }
                }
                details.append(rec)
                time.sleep(0.1)

        # 3. Overall Metrics Aggregation Stage
        metrics = {}
        if txt_refs and txt_preds:
            metrics["wer"] = round(self.asr_mod.compute_wer(predictions=txt_preds, references=txt_refs), 4)

        if emo_refs and emo_preds:
            metrics.update(self.pte_mod.compute_metrics(predictions=emo_preds, references=emo_refs))

        if self.verbose and metrics:
            print("\n==========================================")
            print("      ViSPe OVERALL EVALUATION RESULTS    ")
            print("==========================================")
            if "wer" in metrics:
                print(f" OVERALL WER   : {round(metrics['wer'] * 100, 2)}%")
            if "macro_f1" in metrics:
                print(f" ACCURACY      : {round(metrics['accuracy'] * 100, 2)}%")
                print(f" MACRO F1      : {round(metrics['macro_f1'] * 100, 2)}%")
                print(f" WEIGHTED F1   : {round(metrics['weighted_f1'] * 100, 2)}%")
            print("==========================================")

        if details:
            self.log_history(ver=ver, metrics=metrics, details=details)


if __name__ == "__main__":
    app = ViSPePipeline()
    app.init_models()
    app.run(ver="v1.0.0-vispe")