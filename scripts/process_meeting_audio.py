import os
import gc
import torch
import librosa
import re  
import langid
from pathlib import Path

# ================= MEMORY OPTIMIZATION =================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ================= NUCLEAR FIX FOR TORCHCODEC =================
import transformers.utils.import_utils
transformers.utils.import_utils.is_torchcodec_available = lambda: False
try:
    import transformers.pipelines.automatic_speech_recognition
    transformers.pipelines.automatic_speech_recognition.is_torchcodec_available = lambda: False
except (ImportError, AttributeError):
    pass

from pyannote.audio import Pipeline
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    pipeline
)
from peft import PeftModel

# ================= CONFIGURATION =================

AUDIO_PATH = Path(r"C:\Users\USER\Documents\meeting-asr\data\youtube_clips\mandarin_news_sample.wav")
LORA_PATH = r"C:\Users\USER\Documents\meeting-asr\models\whisper-small-lora-trilingual"
BASE_MODEL = "openai/whisper-small"
HF_TOKEN = os.environ.get("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= HELPER: Text Cleaning =================

def clean_text(text):
    """
    Removes characters that are not English (Latin), Malay (Latin), 
    Chinese (CJK), numbers, or standard punctuation.
    """
    # 1. Define allowed ranges:
    # \u0000-\u007F  : Basic Latin (English/Malay/Numbers/Punctuation)
    # \u4e00-\u9fff  : CJK Unified Ideographs (Common Chinese)
    # \u3400-\u4dbf  : CJK Extension A
    # \u3000-\u303f  : CJK Symbols and Punctuation
    # \uff00-\uffef  : Full-width characters (often used in Chinese text)
    allowed_pattern = re.compile(r'[^\u0000-\u007F\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+')
    
    # 2. Replace disallowed characters with empty string
    cleaned = allowed_pattern.sub('', text)
    return cleaned.strip()

# ================= 1. Diarization (CPU) =================

def run_diarization_on_cpu(wav_path):
    print("\n[1/2] Step 1: Diarization (VAD) on CPU...")
    try:
        dia_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=HF_TOKEN
        )
        dia_pipeline.to(torch.device("cpu")) 
    except Exception as e:
        print(f"Error loading Pyannote: {e}")
        return []

    print("      Scanning audio for speech segments...")
    diarization = dia_pipeline(str(wav_path))
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    
    print(f"      Found {len(segments)} valid speech segments.")
    del dia_pipeline
    gc.collect()
    return segments

# ================= 2. VAD-Guided Transcription (GPU) =================

def transcribe_segments_on_gpu(wav_path, segments):
    print("\n[2/2] Step 2: VAD-Guided Transcription on GPU...")
    
    y, sr = librosa.load(str(wav_path), sr=16000)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("      Loading Whisper LoRA...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, 
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 
    )
    
    try:
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        model = model.merge_and_unload()
        model.eval()
    except Exception:
        print("      Warning: LoRA load failed, using Base.")
        model = base_model

    model = model.to(device)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True, 
        torch_dtype=torch.float16
    )

    final_transcript = []
    
    print(f"      Transcribing {len(segments)} segments...")
    
    for i, seg in enumerate(segments):
        start_sec = seg["start"]
        end_sec = seg["end"]
        speaker = seg["speaker"]
       
        # INCREASED MINIMUM DURATION:
        # Segments < 0.5s are usually just noise/breaths which cause hallucinations
        if (end_sec - start_sec) < 0.5:
            continue
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        audio_slice = y[start_sample:end_sample]
       
        try:
            result = asr_pipe(
                audio_slice,
                batch_size=1,
                generate_kwargs={
                    "language": "ms",
                    "task": "transcribe",
                    "repetition_penalty": 1.2,
                    "no_repeat_ngram_size": 3,
                    "condition_on_prev_tokens": False
                    # Note: We do NOT set "language" here to allow code-switching  # <-- This comment seems misplaced; "language": "ms" is set above
                }
            )
            raw_text = result["text"].strip()
           
            # --- HYBRID LANGUAGE DETECTION ---
            if raw_text:  # Only classify if there's text
                lang, confidence = langid.classify(raw_text)
                if lang == 'zh' and confidence > 0.9:  # Re-transcribe if strongly Mandarin (ignores en/ms)
                    print(f"Re-transcribing segment {i} as Mandarin...")
                    result = asr_pipe(
                        audio_slice,
                        batch_size=1,
                        generate_kwargs={
                            "language": "zh",  # Switch for this segment only
                            "task": "transcribe",
                            "repetition_penalty": 1.2,
                            "no_repeat_ngram_size": 3,
                            "condition_on_prev_tokens": False
                        }
                    )
                    raw_text = result["text"].strip()
           
            # --- APPLY FILTER HERE ---
            cleaned_text = clean_text(raw_text)
           
            # Only append if there is actual text left after cleaning
            if cleaned_text and len(cleaned_text) > 1:
                final_transcript.append({
                    "start": start_sec,
                    "end": end_sec,
                    "speaker": speaker,
                    "text": cleaned_text
                })
               
        except Exception as e:
            print(f" Error on segment {i}: {e}")
       
        if i % 10 == 0:
            print(f" Processed {i}/{len(segments)}...")
            torch.cuda.empty_cache()
            gc.collect()
    return final_transcript

# ================= Main =================

def main():
    if not AUDIO_PATH.exists():
        print(f"Error: Audio file not found at {AUDIO_PATH}")
        return

    segments = run_diarization_on_cpu(AUDIO_PATH)
    if not segments:
        print("No speech detected.")
        return

    transcript = transcribe_segments_on_gpu(AUDIO_PATH, segments)
    
    print("\n" + "="*50)
    print("FINAL TRANSCRIPT PREVIEW")
    print("="*50)
    
    output_lines = []
    for item in transcript:
        line = f"[{item['start']:.2f}s - {item['end']:.2f}s] {item['speaker']}: {item['text']}"
        output_lines.append(line)
        print(line)
        
    out_file = AUDIO_PATH.with_suffix(".lora_diarized.txt")
    out_file.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\n✅ Saved to: {out_file}")

if __name__ == "__main__":
    main()