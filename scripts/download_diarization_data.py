from pathlib import Path
from datasets import load_dataset, Value, Features
import soundfile as sf
import io
import tarfile
import urllib.request
from tqdm import tqdm

# ==== CONFIG ====
ROOT = Path(r"C:\Users\USER\Documents\meeting-asr\data\asr_eval")

# 1. MALAY CONFIG
MALAY_ROOT = ROOT / "malay_diarization"
MALAY_ROOT.mkdir(parents=True, exist_ok=True)

# 2. ENGLISH CONFIG (VoxConverse)
ENGLISH_ROOT = ROOT / "english_diarization"
ENGLISH_ROOT.mkdir(parents=True, exist_ok=True)

# 3. MANDARIN CONFIG (AISHELL-4)
MANDARIN_ROOT = ROOT / "mandarin_diarization"
MANDARIN_ROOT.mkdir(parents=True, exist_ok=True)


def generate_rttm_line(rec_id, start_time, duration, speaker_id):
    return f"SPEAKER {rec_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA>"

# ==========================================
# PART 1: MALAY (Hugging Face)
# ==========================================
def process_malay():
    print("\n[1/3] Processing Malay Data (Hugging Face)...")
    try:
        ds = load_dataset("malaysia-ai/malay-conversational-speech-corpus", split="train")
        
        # FIX: Rename and cast to bypass torchcodec
        if "filename" in ds.column_names:
            ds = ds.rename_column("filename", "audio")
            
        new_features = ds.features.copy()
        new_features["audio"] = {"bytes": Value("binary"), "path": Value("string")}
        ds = ds.cast(Features(new_features))

        sessions = {}
        for row in tqdm(ds, desc="Grouping Malay Sessions"):
            rec_id = "conversation_unknown"
            if "original_filename" in row and row["original_filename"]:
                rec_id = Path(row["original_filename"]).stem
            elif "audio" in row and row["audio"].get("path"):
                 rec_id = Path(row["audio"]["path"]).stem.split("_")[0]

            if rec_id not in sessions: sessions[rec_id] = []
            sessions[rec_id].append(row)

        for rec_id, rows in tqdm(sessions.items(), desc="Writing Malay RTTMs"):
            rttm_lines = []
            full_audio = []
            current_time = 0.0
            
            # Paths
            audio_out = MALAY_ROOT / "audio" / f"{rec_id}.wav"
            rttm_out = MALAY_ROOT / "rttm" / f"{rec_id}.rttm"
            audio_out.parent.mkdir(parents=True, exist_ok=True)
            rttm_out.parent.mkdir(parents=True, exist_ok=True)

            if audio_out.exists(): continue # Skip if done

            for row in rows:
                if not row["audio"]: continue
                try:
                    audio_entry = row["audio"]
                    if audio_entry.get("bytes"):
                        audio_array, sr = sf.read(io.BytesIO(audio_entry["bytes"]))
                    elif audio_entry.get("path"):
                        audio_array, sr = sf.read(audio_entry["path"])
                    else: continue
                except: continue

                duration = len(audio_array) / sr
                full_audio.extend(audio_array)
                
                spk = str(row.get("speaker_id", row.get("id", "Unknown")))
                rttm_lines.append(generate_rttm_line(rec_id, current_time, duration, spk))
                current_time += duration

            if full_audio:
                sf.write(str(audio_out), full_audio, 16000)
                with open(rttm_out, "w") as f: f.write("\n".join(rttm_lines))

        print(f"✅ Malay Done: {len(sessions)} sessions.")

    except Exception as e:
        print(f"❌ Malay Failed: {e}")

# ==========================================
# PART 2: ENGLISH (VoxConverse via HF)
# ==========================================
def process_english():
    print("\n[2/3] Processing English Data (VoxConverse)...")
    try:
        # FIX 1: Enable 'streaming=True' to prevent downloading all 7GB at once
        ds = load_dataset("diarizers-community/voxconverse", "default", split="test", streaming=True)
        
        # FIX 2: Limit to just 20 files (like your Mandarin/Malay sets)
        ds = ds.take(20)
        
        # FIX 3: Cast to raw struct to bypass torchcodec & auto-decoding
        # This keeps the audio as raw bytes so we can save them directly
        new_features = ds.features.copy()
        new_features["audio"] = {"bytes": Value("binary"), "path": Value("string")}
        ds = ds.cast(Features(new_features))

        count = 0
        for row in tqdm(ds, desc="Writing English RTTMs (Limit 20)"):
            audio_entry = row["audio"]
            rec_id = Path(audio_entry["path"]).stem
            
            audio_out = ENGLISH_ROOT / "audio" / f"{rec_id}.wav"
            rttm_out = ENGLISH_ROOT / "rttm" / f"{rec_id}.rttm"
            
            audio_out.parent.mkdir(parents=True, exist_ok=True)
            rttm_out.parent.mkdir(parents=True, exist_ok=True)

            # Skip if done
            if audio_out.exists() and rttm_out.exists():
                count += 1
                continue

            # 1. Write Audio
            if not audio_out.exists():
                if audio_entry.get("bytes"):
                    audio_array, sr = sf.read(io.BytesIO(audio_entry["bytes"]))
                else:
                    audio_array, sr = sf.read(audio_entry["path"])
                sf.write(str(audio_out), audio_array, sr)

            # 2. Write RTTM
            starts = row['timestamps_start']
            ends = row['timestamps_end']
            speakers = row['speakers']
            
            rttm_lines = []
            for s, e, spk in zip(starts, ends, speakers):
                duration = e - s
                rttm_lines.append(generate_rttm_line(rec_id, s, duration, spk))
            
            with open(rttm_out, "w") as f:
                f.write("\n".join(rttm_lines))
            
            count += 1

        print(f"✅ English Done: {count} sessions.")

    except Exception as e:
        print(f"❌ English Failed: {e}")# ==========================================
# PART 3: MANDARIN (AISHELL-4 Manual Download)
# ==========================================
def process_mandarin():
    print("\n[3/3] Processing Mandarin Data (AISHELL-4)...")
    
    # URL for AISHELL-4 Test Set (OpenSLR mirror)
    # We use the test set (5.2GB) because it's manageable and has Ground Truth
    URL = "https://www.openslr.org/resources/111/test.tar.gz"
    tar_path = MANDARIN_ROOT / "test.tar.gz"
    extract_path = MANDARIN_ROOT / "raw"
    
    if not extract_path.exists():
        print(f"   Downloading AISHELL-4 Test Set (approx 5GB)...")
        try:
            # Download with progress bar
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading") as t:
                def hook(b=1, bsize=1, tsize=None):
                    if tsize is not None: t.total = tsize
                    t.update(b * bsize - t.n)
                urllib.request.urlretrieve(URL, tar_path, reporthook=hook)
            
            print("   Extracting...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=extract_path)
            
            # Cleanup tar
            tar_path.unlink()
        except Exception as e:
            print(f"❌ Mandarin Download Failed: {e}")
            return

    # Process extracted files
    # Structure: raw/test/wav/*.flac and raw/test/TextGrid/*.rttm
    wav_dir = extract_path / "test" / "wav"
    rttm_source_dir = extract_path / "test" / "TextGrid" # AISHELL sometimes puts RTTMs here
    
    # If RTTMs are not in TextGrid folder, check root
    if not list(rttm_source_dir.glob("*.rttm")):
        rttm_source_dir = extract_path / "test" 

    print("   Organizing Mandarin files...")
    
    # Move/Copy to standard folders
    final_audio_dir = MANDARIN_ROOT / "audio"
    final_rttm_dir = MANDARIN_ROOT / "rttm"
    final_audio_dir.mkdir(exist_ok=True)
    final_rttm_dir.mkdir(exist_ok=True)

    count = 0
    for wav_file in wav_dir.glob("*.flac"):
        rec_id = wav_file.stem
        
        # Convert FLAC to WAV (optional, but good for consistency)
        out_wav = final_audio_dir / f"{rec_id}.wav"
        if not out_wav.exists():
            data, sr = sf.read(str(wav_file))
            sf.write(str(out_wav), data, sr)
        
        # Copy RTTM
        # AISHELL RTTMs often named same as wav
        src_rttm = rttm_source_dir / f"{rec_id}.rttm"
        if src_rttm.exists():
            dst_rttm = final_rttm_dir / f"{rec_id}.rttm"
            with open(src_rttm, 'r') as f_in, open(dst_rttm, 'w') as f_out:
                f_out.write(f_in.read())
            count += 1
            
    print(f"✅ Mandarin Done: {count} sessions prepared.")


def main():
    process_malay()
    process_english()
    process_mandarin()
    
    print("\n\n=== FINAL SUMMARY ===")
    print(f"Malay:    {MALAY_ROOT}")
    print(f"English:  {ENGLISH_ROOT}")
    print(f"Mandarin: {MANDARIN_ROOT}")

if __name__ == "__main__":
    main()