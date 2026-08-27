from transformers import pipeline
import os, warnings, time, torch, json, evaluate, re, string
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(ROOT_DIR, "../../"))

class ASR:
    """
    Automatic Speech Recognition (ASR) engine for processing audio files, 
    transcribing text, and benchmarking Word Error Rate (WER).
    """
    
    def __init__(self, cfg_path: str = "config.json"):
        """
        Initialize ASR component and load system configuration.
        
        Args:
            cfg_path (str): Relative path to configuration file.
        """
        cfg_full = os.path.join(BASE_DIR, cfg_path)
        self.cfg = self._load_cfg(cfg_full)
        self.model = None
        self.device = 0
        self.verbose = True
        self.pipeline = None
        self.wer_metric = evaluate.load("wer")
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
                "wer_tracking_file": "src/wer.json"
            },
            "default_asr": {"model": "openai/whisper-small", "language": "vietnamese"}
        }

    def normalize_text(self, text: str) -> str:
        """
        Standardize text for fair WER evaluation:
        - Lowercase all characters.
        - Remove punctuation marks.
        - Collapse multiple spaces into one.
        
        Args:
            text (str): Raw input text string.
            
        Returns:
            str: Normalized text string.
        """
        if not text:
            return ""
        text = text.lower()
        # Remove all punctuations
        text = re.sub(r'[' + re.escape(string.punctuation) + r']', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def compute_wer(self, preds: list, refs: list) -> float:
        """
        Compute Word Error Rate (WER) after applying text normalization.
        
        Args:
            preds (list): List of predicted text strings.
            refs (list): List of ground truth reference strings.
            
        Returns:
            float: Calculated WER score.
        """
        norm_preds = [self.normalize_text(p) for p in preds]
        norm_refs = [self.normalize_text(r) for r in refs]
        return self.wer_metric.compute(predictions=norm_preds, references=norm_refs)

    def log_history(self, ver: str, total_wer: float, details: list):
        """
        Append experiment results and metadata into tracking history JSON file.
        
        Args:
            ver (str): Current experiment version string.
            total_wer (float): Overall WER score for the dataset.
            details (list): Sample-level prediction logs.
        """
        rel_path = self.cfg["paths"].get("wer_tracking_file", "src/wer.json")
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
            "overall_wer": round(total_wer, 4),
            "details": details
        }
        
        hist["tracking_history"].append(entry)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
        print(f"\n--> Saved WER tracking result to '{out_path}'")

    def setup_env(self):
        """
        Suppress warnings and disable unnecessary logs from HuggingFace/TensorFlow.
        """
        warnings.filterwarnings("ignore")
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_SYMBOLS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    def init_model(self):
        """
        Interactively trigger model loading routines for the ASR engine.
        """
        while True: 
            try:
                def_model = self.cfg["default_asr"]["model"]
                m_inp = input(f"Input Model (Default: {def_model}): ").strip()
                dev_inp = input("Select Device (0: GPU, 1: CPU) [Default: 0]: ").strip()
                verb_inp = input("Show detailed progress? (True/False) [Default: True]: ").strip().lower()

                self.model = m_inp if m_inp else def_model
                self.device = 0 if dev_inp != "1" and torch.cuda.is_available() else -1
                self.verbose = False if verb_inp == "false" else True

                print(f"\nLoading model '{self.model}' on {'GPU' if self.device == 0 else 'CPU'}...")
                self.pipeline = pipeline(
                    task="automatic-speech-recognition",
                    model=self.model,
                    device=self.device
                )
                print("Model loaded successfully.\n")
                break
            except Exception as e:
                print(f"Get Model Failed ({e}), Please try again!\n") 

    def run(self, ver: str = "v1.0.0-baseline"):
        """
        Execute ASR pipeline over data sample batch and compute overall WER.
        
        Args:
            ver (str): Release tag or experiment version.
        """
        rel_data = self.cfg["paths"]["default_sound_text_data"]
        data_path = os.path.join(BASE_DIR, rel_data)
        lang = self.cfg["default_asr"].get("language", "vietnamese")

        print("ASR is processing & computing WER...\n")
        
        if not self.pipeline:
            print("Model not loaded. Please load the model first.")
            self.init_model()
            
        if not os.path.exists(data_path): 
            print(f"Error [101]: Cannot find data file at '{data_path}'")
            return None

        refs, preds, details = [], [], []

        with open(data_path, "r", encoding="utf-8") as f:
            samples = json.load(f).get("sound-list", [])
            
            for idx, item in enumerate(samples, 1):
                raw_path = item["dir"]
                aud_path = os.path.join(BASE_DIR, raw_path) if not os.path.isabs(raw_path) else raw_path
                s_name = item["name"]
                gt_txt = item["text"]

                if self.verbose:
                    print(f"File {idx} | {s_name} | DIR: {aud_path}")
                
                if not os.path.exists(aud_path):
                    if self.verbose:
                        print(f"Skip: Audio missing at '{aud_path}'\n")
                    continue

                res = self.pipeline(
                    aud_path,
                    generate_kwargs={"task": "transcribe", "language": lang},
                    return_timestamps=False
                )
                pred_txt = res['text']
                wer = self.compute_wer(preds=[pred_txt], refs=[gt_txt])

                if self.verbose:
                    print(f"INPUT : {gt_txt}")
                    print(f"OUTPUT: {pred_txt}")
                    print(f"--> Single WER: {round(wer * 100, 2)}%\n")

                refs.append(gt_txt)
                preds.append(pred_txt)
                
                details.append({
                    "id": idx,
                    "name": s_name,
                    "dir": raw_path,
                    "ground_truth": gt_txt,
                    "prediction": pred_txt,
                    "single_wer": round(wer, 4)
                })
                
                time.sleep(0.1)

        if refs and preds:
            total_wer = self.compute_wer(preds=preds, refs=refs)
            print("==========================================")
            print(f" OVERALL DATASET WER: {round(total_wer * 100, 2)}%")
            print("==========================================")
            
            self.log_history(ver=ver, total_wer=total_wer, details=details)

if __name__ == "__main__":
    app = ASR()
    app.init_model()
    app.run(ver="v1.0.0-baseline")