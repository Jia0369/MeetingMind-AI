from pathlib import Path
from typing import Dict, Any
import soundfile as sf
import numpy as np
import librosa
import random
import torch
import os

# Hugging Face / Pyannote imports
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from pyannote.audio import Pipeline

# ==== CONFIG ====
ROOT = Path(r"C:\Users\USER\Documents\meeting-asr\data\asr_eval")
HYP_ROOT = ROOT / "hyps_multi_robust_test_set" 
HYP_ROOT.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")

DATASETS = {
    "ascend_en":       ROOT / "ascend" / "test" / "en",
    "ascend_zh":       ROOT / "ascend" / "test" / "zh",
    "ascend_mixed":    ROOT / "ascend" / "test" / "mixed_enzh",
    "malcsc_ms":       ROOT / "malcsc" / "test" / "ms",
    "fleurs_ms_my":    ROOT / "fleurs" / "test" / "ms_my",
}

TARGET_SR = 16000
LIMIT_PER_DATASET = 100   # Set to None for full run

MODELS_TO_TEST = [
    ("whisper_small", "openai/whisper-small"),
    ("whisper_medium", "openai/whisper-medium"),
    ("ms_small_v3", "mesolitica/malaysian-whisper-small-v3"),
    ("whisper_small_lora_tri", r"C:\Users\USER\Documents\meeting-asr\models\whisper-small-lora-trilingual"),
]

DATASET_DECODING_CONFIG: Dict[str, Dict[str, Any]] = {
    "ascend_en":      {"lang": "en", "max_new_tokens": 256},
    "ascend_zh":      {"lang": "zh", "max_new_tokens": 256},
    "ascend_mixed":   {"lang": None, "max_new_tokens": 256}, 
    "malcsc_ms":      {"lang": "ms", "max_new_tokens": 400}, 
    "fleurs_ms_my":   {"lang": "ms", "max_new_tokens": 300},
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(wav_path: Path) -> np.ndarray:
    """Robust audio loader that handles multi-channel and resampling"""
    try:
        y, sr = sf.read(str(wav_path))
        if y.ndim > 1: y = y.mean(axis=1)
        y = y.astype(np.float32)
        if np.max(np.abs(y)) > 0: y /= np.max(np.abs(y)) + 1e-8
        if sr != TARGET_SR: y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        return y
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        return np.array([])

def transcribe_with_sliding_window(wav_path, processor, model, lang_hint, max_new_tokens):
    """
    Robust transcription using manual 30s sliding window.
    This is your specific robust logic.
    """
    audio = load_audio(wav_path)
    if len(audio) == 0: return ""
    
    chunk_samples = int(30.0 * TARGET_SR)
    transcriptions = []
    
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        
        # Skip silence chunks
        if len(chunk) < 1000 or np.max(np.abs(chunk)) < 0.001: 
            continue
        
        inputs = processor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
        input_features = inputs.input_features.to(device=device, dtype=model.dtype)
        
        gen_kwargs = {
            "task": "transcribe",
            "max_new_tokens": min(max_new_tokens, 440),
            "condition_on_prev_tokens": False, # Crucial for robustness
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3
        }
        if lang_hint: gen_kwargs["language"] = lang_hint
        
        with torch.no_grad():
            generated_ids = model.generate(input_features=input_features, **gen_kwargs)
        
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if text: transcriptions.append(text)
        
    return " ".join(transcriptions)

def run_diarization(pipeline, wav_path, rttm_path):
    """Run Pyannote Pipeline and save RTTM"""
    try:
        # Pyannote expects a path string
        diarization = pipeline(str(wav_path))
        
        with open(rttm_path, "w") as f:
            diarization.write_rttm(f)
            
    except Exception as e:
        print(f"  [Diarization Error] {wav_path.name}: {e}")

def main():
    print(f"--- Starting ASR + Diarization on {device} ---")
    
    # 1. Initialize Diarization Pipeline (Loaded ONCE, reused for all models)
    print("Loading Pyannote Diarization Pipeline...")
    try:
        diar_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=HF_TOKEN
        ).to(torch.device(device))
    except Exception as e:
        print(f"❌ Could not load Pyannote pipeline: {e}")
        print("   Make sure you have accepted the user agreement on Hugging Face hub and set HF_TOKEN.")
        return

    # 2. Get list of files
    eval_files = {}
    rng = random.Random(42)
    for ds, path in DATASETS.items():
        if not path.exists(): 
            print(f"Dataset path not found: {path}")
            continue
        files = sorted(path.glob("*.wav"))
        if LIMIT_PER_DATASET and len(files) > LIMIT_PER_DATASET: 
            rng.shuffle(files)
            files = files[:LIMIT_PER_DATASET]
        eval_files[ds] = files

    # 3. Iterate Models
    for short_name, model_id in MODELS_TO_TEST:
        print(f"\n=== Processing Model: {short_name} ===")
        
        # Load ASR Model
        try:
            if "lora" in short_name:
                print(f"  Loading LoRA adapter: {model_id}")
                # For LoRA, we load the base model then apply the adapter
                proc = WhisperProcessor.from_pretrained(model_id) # Often config is saved with adapter
                base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
                model = PeftModel.from_pretrained(base, model_id).merge_and_unload()
            else:
                print(f"  Loading HF Model: {model_id}")
                proc = WhisperProcessor.from_pretrained(model_id)
                model = WhisperForConditionalGeneration.from_pretrained(model_id)
            
            model.to(device).eval()
        except Exception as e:
            print(f"❌ Skipping model {short_name} due to load error: {e}")
            continue

        # Process Datasets
        for ds_name, wav_files in eval_files.items():
            if not wav_files: continue
            
            # Setup Output Directory
            # We save outputs to: hyps_multi_robust_test_set/<model_name>/<dataset_name>/
            out_dir = HYP_ROOT / short_name / ds_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  [{ds_name}] Processing {len(wav_files)} files -> {out_dir}")
            
            for i, wav_path in enumerate(wav_files):
                txt_path = out_dir / f"{wav_path.stem}.txt"
                rttm_path = out_dir / f"{wav_path.stem}.rttm"
                
                # A. Run ASR (Text)
                if not txt_path.exists():
                    cfg = DATASET_DECODING_CONFIG.get(ds_name, {})
                    try:
                        text = transcribe_with_sliding_window(
                            wav_path, proc, model, 
                            cfg.get("lang"), 
                            cfg.get("max_new_tokens", 256)
                        )
                        txt_path.write_text(text, encoding="utf-8")
                    except Exception as e:
                        print(f"    Error transcribing {wav_path.name}: {e}")
                
                # B. Run Diarization (Speakers) -> RTTM
                # Note: We run this inside the model loop so the RTTMs end up 
                # in the same folder structure expected by the eval script.
                if not rttm_path.exists():
                    run_diarization(diar_pipeline, wav_path, rttm_path)

                if (i + 1) % 10 == 0: 
                    print(f"    {i + 1}/{len(wav_files)} processed...")

        # Cleanup ASR model to free VRAM for next iteration
        # (Pyannote pipeline stays in memory as it is small and reused)
        del model, proc
        torch.cuda.empty_cache()

    print("\n✅ Transcription & Diarization Complete!")

if __name__ == "__main__":
    main()