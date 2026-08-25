from transformers import pipeline
import os, warnings
import time 
import torch, json 
import evaluate
import re
from datetime import datetime

# Xử lý đường dẫn gốc (Root Directory: NLP project 1)
# Vì asr.py nằm trong src/asr/ nên đi ngược lên 2 cấp để lấy thư mục gốc NLP project 1
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

class ASR:
    def __init__(self, config_rel_path: str = "config.json"):
        config_full_path = os.path.join(BASE_DIR, config_rel_path)
        self.config = self.load_config(config_full_path)
        self.model = None
        self.device = 0
        self.verbose = True
        self.pipeline = None
        self.wer_metric = evaluate.load("wer")
        self.setupEnv()

    def load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "paths": {
                "default_sound_text_data": "src/sound-textData.json",
                "wer_tracking_file": "src/wer.json"
            },
            "default_asr": {"model": "openai/whisper-small", "language": "vietnamese"}
        }

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def compute_wer(self, predictions: list, references: list) -> float:
        norm_preds = [self.normalize_text(p) for p in predictions]
        norm_refs = [self.normalize_text(r) for r in references]
        return self.wer_metric.compute(predictions=norm_preds, references=norm_refs)

    def save_wer_log(self, version: str, overall_wer: float, details: list):
        rel_path = self.config["paths"]["wer_tracking_file"]
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
            "overall_wer": round(overall_wer, 4),
            "details": details
        }
        
        tracking_data["tracking_history"].append(new_entry)

        with open(tracking_path, "w", encoding="utf-8") as f:
            json.dump(tracking_data, f, ensure_ascii=False, indent=2)
        print(f"\n--> Saved WER tracking result to '{tracking_path}'")

    def setupEnv(self):
        warnings.filterwarnings("ignore")
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_SYMBOLS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    def importModel(self):
        while True: 
            try:
                default_model = self.config["default_asr"]["model"]
                model_inp = input(f"Input Model (Default: {default_model}): ").strip()
                device_inp = input("Select Device (0: GPU, 1: CPU) [Default: 0]: ").strip()
                verbose_inp = input("Show detailed progress? (True/False) [Default: True]: ").strip().lower()

                self.model = model_inp if model_inp else default_model
                self.device = 0 if device_inp != "1" and torch.cuda.is_available() else -1
                self.verbose = False if verbose_inp == "false" else True

                print(f"\nLoading model '{self.model}' on {'GPU' if self.device == 0 else 'CPU'}...")
                self.pipeline = pipeline(
                    task = "automatic-speech-recognition",
                    model = self.model,
                    device = self.device
                )
                print("Model loaded successfully.\n")
                break
            except Exception as e:
                print(f"Get Model Failed ({e}), Please try again!\n") 

    def process(self, version: str = "v1.0.0-baseline"):
        rel_data_path = self.config["paths"]["default_sound_text_data"]
        dir_path = os.path.join(BASE_DIR, rel_data_path)
        language = self.config["default_asr"].get("language", "vietnamese")

        print("ASR is processing & computing WER...\n")
        
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
                raw_audio_dir = sound_detail["dir"]
                audio_dir = os.path.join(BASE_DIR, raw_audio_dir) if not os.path.isabs(raw_audio_dir) else raw_audio_dir
                sound_name = sound_detail["name"]
                ground_truth = sound_detail["text"]

                if self.verbose:
                    print(f"File {id_idx} | {sound_name} | DIRECTORY: {audio_dir}")
                
                if not os.path.exists(audio_dir):
                    if self.verbose:
                        print(f"Skip: Audio file not found at '{audio_dir}'\n")
                    continue

                result = self.pipeline(
                    audio_dir,
                    generate_kwargs={"task": "transcribe", "language": language},
                    return_timestamps=False
                )
                output_text = result['text']

                single_wer = self.compute_wer(predictions=[output_text], references=[ground_truth])

                if self.verbose:
                    print(f"INPUT : {ground_truth}")
                    print(f"OUTPUT: {output_text}")
                    print(f"--> Single WER: {round(single_wer * 100, 2)}%\n")

                references.append(ground_truth)
                predictions.append(output_text)
                
                details_log.append({
                    "id": id_idx,
                    "name": sound_name,
                    "dir": raw_audio_dir,
                    "ground_truth": ground_truth,
                    "prediction": output_text,
                    "single_wer": round(single_wer, 4)
                })
                
                time.sleep(0.1)

        if references:
            total_wer = self.compute_wer(predictions=predictions, references=references)
            print("==========================================")
            print(f" OVERALL DATASET WER: {round(total_wer * 100, 2)}%")
            print("==========================================")
            
            self.save_wer_log(version=version, overall_wer=total_wer, details=details_log)

if __name__ == "__main__":
    app = ASR()
    app.importModel()
    app.process(version="v1.0.0-baseline")