from pathlib import Path
import torch
import os
from pyannote.audio import Pipeline

# ==== CONFIG ====
ROOT = Path(r"C:\Users\USER\Documents\meeting-asr\data\asr_eval")
OUTPUT_ROOT = ROOT / "hyps_diarization"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")

# Points to the LONG-FORM audio folders
DIAR_DATASETS = {
    "malay_diarization":     ROOT / "malay_diarization" / "audio",
    "english_diarization":   ROOT / "english_diarization" / "audio",
    "mandarin_diarization":  ROOT / "mandarin_diarization" / "audio",
}

MODELS = [
    "whisper_small",
    "whisper_medium",
    "ms_small_v3",
    "whisper_small_lora_tri"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"--- Running Diarization Pipeline on {device} ---")
    
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=HF_TOKEN
        ).to(torch.device(device))
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        print("   Ensure you have accepted the Pyannote user agreement on Hugging Face.")
        return

    for model_name in MODELS:
        print(f"\nGeneratings RTTMs for 'pseudo-model' folder: {model_name}")
        
        for ds_name, audio_dir in DIAR_DATASETS.items():
            if not audio_dir.exists(): 
                print(f"  [Skip] Audio dir not found: {audio_dir}")
                continue
            
            wav_files = list(audio_dir.glob("*.wav"))
            out_dir = OUTPUT_ROOT / model_name / ds_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  Processing {ds_name} ({len(wav_files)} files) -> {out_dir}")
            
            for i, wav_path in enumerate(wav_files):
                rttm_path = out_dir / f"{wav_path.stem}.rttm"
                
                # Skip if already exists
                if rttm_path.exists(): continue
                
                try:
                    # Run Diarization
                    diarization = pipeline(str(wav_path))
                    
                    # Save to RTTM
                    with open(rttm_path, "w") as f:
                        diarization.write_rttm(f)
                except Exception as e:
                    print(f"    Error on {wav_path.name}: {e}")
                    
                if i % 5 == 0: print(f"    {i}/{len(wav_files)} done...")

    print("\n✅ Diarization Hypotheses Generated.")

if __name__ == "__main__":
    main()