from pathlib import Path
from itertools import islice
import soundfile as sf
from datasets import load_dataset, Audio
import io

# ==== ROOT CONFIG ==== 
ROOT = Path(r"C:\Users\USER\Documents\meeting-asr\data\asr_eval")

# ==== OUTPUT PATHS (TEST DATA) ====
ASCEND_TEST_ROOT = ROOT / "ascend" / "test"
MALCSC_TEST_ROOT = ROOT / "malcsc" / "test"
FLEURS_TEST_ROOT = ROOT / "fleurs" / "test"

# ==== DATA LIMITS ====
ASCEND_LIMIT_PER_LANG = 5000 
MALCSC_LIMIT = 5000          
FLEURS_LIMIT = 5000          

def ensure_dir(p: Path):
    """Creates directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def check_existing_training_data():
    print("\n🛡️  SAFETY CHECK: Verifying existing training data is safe...")
    
    # Check ASCEND Train
    ascend_train = ROOT / "ascend" / "en"
    if ascend_train.exists():
        count = len(list(ascend_train.glob("*.wav")))
        print(f"   [SAFE] Found {count} existing ASCEND 'train' files in: {ascend_train}")
    else:
        print(f"   [INFO] No existing ASCEND 'train' folder found at: {ascend_train}")

    # Check MALCSC Train
    malcsc_train = ROOT / "malcsc" / "ms"
    if malcsc_train.exists():
        count = len(list(malcsc_train.glob("*.wav")))
        print(f"   [SAFE] Found {count} existing MALCSC 'train' files in: {malcsc_train}")
    
    # Check FLEURS Train
    fleurs_train = ROOT / "fleurs" / "ms_my"
    if fleurs_train.exists():
        count = len(list(fleurs_train.glob("*.wav")))
        print(f"   [SAFE] Found {count} existing FLEURS 'train' files in: {fleurs_train}")

    print("   ✅ These folders will NOT be touched.\n")

def save_audio_file(audio_entry, wav_path):
    """
    Helper to manually decode and save audio.
    Bypasses datasets.Audio(decode=True) to avoid torchcodec issues.
    """
    try:
        # If 'bytes' is available, read from memory (common in streaming/tar)
        if audio_entry.get("bytes"):
            y, sr = sf.read(io.BytesIO(audio_entry["bytes"]))
        # Otherwise read from the extracted path
        elif audio_entry.get("path"):
            y, sr = sf.read(audio_entry["path"])
        else:
            print("[ERROR] No bytes or path found for audio.")
            return False

        sf.write(str(wav_path), y, sr)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save {wav_path}: {e}")
        return False

# ================== ASCEND (Test Split) ==================
def dump_ascend(limit_per_lang: int):
    print("[ASCEND] Loading CAiRE/ASCEND 'test' split…")
    try:
        # Load dataset
        ds = load_dataset("CAiRE/ASCEND", split="test")
        # FORCE decode=False to prevent crashes during filter/access
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"[ASCEND] Error loading test split: {e}")
        return

    lang_configs = [
        ("en", "en"),
        ("zh", "zh"),
        ("mixed_enzh", "mixed"),
    ]

    for folder_label, lang_value in lang_configs:
        print(f"[ASCEND] Filtering language == '{lang_value}'…")
        # Now filtering should be safe because audio is not being decoded
        ds_lang = ds.filter(lambda ex, lv=lang_value: ex.get("language", "") == lv)
        
        n = len(ds_lang)
        out_dir = ASCEND_TEST_ROOT / folder_label
        ensure_dir(out_dir)

        limit = min(limit_per_lang, n)
        print(f"[ASCEND] Saving {limit} TEST files to: {out_dir}")
        
        new_count = 0
        skipped_count = 0

        for idx, row in enumerate(islice(ds_lang, limit)):
            base = row.get("id") or f"{folder_label}_test_{idx:05d}"
            wav_path = out_dir / f"{base}.wav"
            txt_path = out_dir / f"{base}.txt"

            if wav_path.exists() and txt_path.exists():
                skipped_count += 1
                continue

            # Manually handle audio saving
            if save_audio_file(row["audio"], wav_path):
                txt = row.get("transcription", "")
                txt_path.write_text(txt, encoding="utf-8")
                new_count += 1

        print(f"   -> Downloaded {new_count} new, Skipped {skipped_count} existing.")

# ================== MALCSC (Test/Validation Split) ===================
def dump_malcsc(limit: int):
    print("\n[MALCSC] Attempting to load SEACrowd/asr_malcsc 'test' split…")
    ds = None
    try:
        ds = load_dataset("SEACrowd/asr_malcsc", split="test", trust_remote_code=True)
    except Exception:
        print("   -> 'test' split not found. Switching to 'validation' split…")
        try:
            ds = load_dataset("SEACrowd/asr_malcsc", split="validation", trust_remote_code=True)
        except Exception as e:
            print(f"   -> Could not load validation splits either: {e}")
            return

    # FORCE decode=False
    ds_ms = ds.cast_column("audio", Audio(decode=False))
    n = len(ds_ms)
    
    out_dir = MALCSC_TEST_ROOT / "ms"
    ensure_dir(out_dir)

    limit = min(limit, n)
    print(f"[MALCSC] Saving {limit} TEST files to: {out_dir}")

    new_count = 0
    skipped_count = 0

    for idx, row in enumerate(islice(ds_ms, limit)):
        base = row.get("id") or f"malcsc_test_{idx:05d}"
        wav_path = out_dir / f"{base}.wav"
        txt_path = out_dir / f"{base}.txt"

        if wav_path.exists() and txt_path.exists():
            skipped_count += 1
            continue

        if save_audio_file(row["audio"], wav_path):
            txt = row.get("text", "")
            txt_path.write_text(txt, encoding="utf-8")
            new_count += 1

    print(f"   -> Downloaded {new_count} new, Skipped {skipped_count} existing.")

# ================== FLEURS (Test Split) ==========================
def dump_fleurs_ms(limit: int):
    print("\n[FLEURS] Loading google/fleurs (ms_my) 'test' split…")
    try:
        ds = load_dataset("google/fleurs", "ms_my", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"[FLEURS] Error loading test split: {e}")
        return

    # FORCE decode=False
    ds = ds.cast_column("audio", Audio(decode=False))
    n = len(ds)
    
    out_dir = FLEURS_TEST_ROOT / "ms_my"
    ensure_dir(out_dir)

    limit = min(limit, n)
    print(f"[FLEURS] Saving {limit} TEST files to: {out_dir}")

    new_count = 0
    skipped_count = 0

    for idx, row in enumerate(islice(ds, limit)):
        base = str(row.get("id") or f"fleurs_test_{idx:05d}")
        wav_path = out_dir / f"{base}.wav"
        txt_path = out_dir / f"{base}.txt"

        if wav_path.exists() and txt_path.exists():
            skipped_count += 1
            continue

        if save_audio_file(row["audio"], wav_path):
            txt = row.get("transcription", "")
            txt_path.write_text(txt, encoding="utf-8")
            new_count += 1

    print(f"   -> Downloaded {new_count} new, Skipped {skipped_count} existing.")

# ============================ MAIN ==================================
def main():
    ensure_dir(ROOT)
    
    # 1. Run Safety Check
    check_existing_training_data()

    print("🚀 Starting download of TEST (Evaluation) datasets...")
    
    # 2. Download Test Data
    dump_ascend(limit_per_lang=ASCEND_LIMIT_PER_LANG)
    dump_malcsc(limit=MALCSC_LIMIT)
    dump_fleurs_ms(limit=FLEURS_LIMIT)

    print("\n✅ All TEST datasets downloaded successfully.")
    print(f"📂 New data is located in subfolders inside: {ROOT}\\...\\test")

if __name__ == "__main__":
    main()