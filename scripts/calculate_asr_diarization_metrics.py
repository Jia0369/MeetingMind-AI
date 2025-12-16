from pathlib import Path
import re
import warnings

# Suppress pyannote approximation warnings
warnings.filterwarnings("ignore", message=".*'uem' was approximated.*")

# Try to import pyannote for DER.
try:
    from pyannote.core import Segment, Annotation
    from pyannote.metrics.diarization import DiarizationErrorRate
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("⚠️  [WARNING] 'pyannote.metrics' or 'pandas' not found. Diarization (DER) will be skipped.")

# ===================== CONFIG =====================
ROOT = Path(r"C:\Users\USER\Documents\meeting-asr\data\asr_eval")

# Path for ASR results
ASR_HYP_ROOT = ROOT / "hyps_multi_robust_test_set"

# Path for Diarization results
DIAR_HYP_ROOT = ROOT / "hyps_diarization"

# 1. ASR Datasets
ASR_DATASETS = {
    "ascend_en":      ROOT / "ascend" / "test" / "en",
    "ascend_zh":      ROOT / "ascend" / "test" / "zh",
    "ascend_mixed":   ROOT / "ascend" / "test" / "mixed_enzh",
    "malcsc_ms":      ROOT / "malcsc" / "test" / "ms",
    "fleurs_ms_my":   ROOT / "fleurs" / "test" / "ms_my",
}

# 2. Diarization Datasets
DIAR_DATASETS = {
    "malay_diarization":     ROOT / "malay_diarization" / "rttm",
    "english_diarization":   ROOT / "english_diarization" / "rttm",
    "mandarin_diarization":  ROOT / "mandarin_diarization" / "rttm",
}

MODELS = [
    "whisper_small",
    "whisper_medium",
    "ms_small_v3",
    "whisper_small_lora_tri"
]

# ===================== ASR METRICS =====================

def norm_cer(text: str) -> str:
    """Removes spaces entirely for CER calc"""
    return "".join(text.strip().split()).lower()

_punct_re = re.compile(r"[^\w]+", flags=re.UNICODE)

def norm_wer(text: str) -> str:
    text = text.strip().lower()
    text = _punct_re.sub(" ", text)
    text = " ".join(text.split())
    return text

def levenshtein(a, b):
    n, m = len(a), len(b)
    if n > m: a, b = b, a; n, m = m, n
    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]: change = change + 1
            current[j] = min(add, delete, change)
    return current[n]

def calculate_cer(ref, hyp):
    ref_n, hyp_n = norm_cer(ref), norm_cer(hyp)
    if not ref_n: return 0.0 if not hyp_n else 1.0
    return levenshtein(ref_n, hyp_n) / len(ref_n)

def calculate_wer(ref, hyp):
    ref_t = norm_wer(ref).split()
    hyp_t = norm_wer(hyp).split()
    if not ref_t: return 0.0 if not hyp_t else 1.0
    return levenshtein(ref_t, hyp_t) / len(ref_t)

# ===================== DIARIZATION METRICS =====================

def load_rttm(file_path):
    annotation = Annotation(uri=file_path.stem)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER": continue
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker = parts[7]
            annotation[Segment(start, end)] = speaker
    return annotation

def safe_get_scalar(df, row, col_pattern):
    """
    Safely extracts a float looking for a column that matches a pattern (case-insensitive).
    """
    try:
        # Find matching column
        col_name = None
        for c in df.columns:
            if col_pattern.lower() in str(c).lower():
                col_name = c
                break
        
        if col_name is None:
            # Debug: Uncomment if you still see 0.000
            # print(f"DEBUG: Could not find column for '{col_pattern}'. Available: {df.columns.tolist()}")
            return 0.0

        val = df.loc[row, col_name]
        
        if hasattr(val, 'item'): return float(val.item())
        if hasattr(val, 'iloc'): return float(val.iloc[0])
        return float(val)
    except Exception:
        return 0.0

def evaluate_diarization(ref_dir, hyp_dir):
    if not PYANNOTE_AVAILABLE: return None, 0

    der_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    
    ref_files = {p.stem: p for p in ref_dir.glob("*.rttm")}
    hyp_files = {p.stem: p for p in hyp_dir.glob("*.rttm")}
    
    common = sorted(set(ref_files.keys()) & set(hyp_files.keys()))
    if not common: return None, 0

    count = 0
    for fid in common:
        try:
            ref = load_rttm(ref_files[fid])
            hyp = load_rttm(hyp_files[fid])
            der_metric(ref, hyp)
            count += 1
        except Exception:
            pass

    try:
        report = der_metric.report(display=False)
        
        # 'total' is usually the duration (seconds)
        total_dur = safe_get_scalar(report, 'TOTAL', 'total') 
        
        if total_dur == 0: return None, count

        # Extract values using fuzzy matching
        miss_sec = safe_get_scalar(report, 'TOTAL', 'miss') # 'missed detection'
        fa_sec = safe_get_scalar(report, 'TOTAL', 'false')  # 'false alarm'
        conf_sec = safe_get_scalar(report, 'TOTAL', 'conf') # 'confusion'
        der_percent = safe_get_scalar(report, 'TOTAL', 'error') # 'diarization error rate'

        # Convert to percentages (0-100 scale) for display
        # Note: If pyannote returns seconds for components, we divide by total duration * 100
        return {
            "DER": der_percent * 100 if der_percent < 1.0 else der_percent, # Auto-scale
            "Miss": (miss_sec / total_dur) * 100,
            "FA": (fa_sec / total_dur) * 100,
            "Conf": (conf_sec / total_dur) * 100
        }, count

    except Exception as e:
        print(f"Error generating report: {e}")
        return None, count

# ===================== MAIN =====================

def main():
    print(f"{'MODEL':<25} | {'DATASET':<20} | {'METRIC':<6} | {'SCORE':<6} | {'FILES'}")
    print("-" * 85)

    for model in MODELS:
        # --- ASR ---
        for ds_name, ref_dir in ASR_DATASETS.items():
            hyp_dir = ASR_HYP_ROOT / model / ds_name
            if not hyp_dir.exists(): continue

            ref_files = {p.stem: p for p in ref_dir.glob("*.txt")}
            hyp_files = {p.stem: p for p in hyp_dir.glob("*.txt")}
            common = sorted(set(ref_files.keys()) & set(hyp_files.keys()))
            if not common: continue

            scores = []
            metric_name = "WER"
            for fid in common:
                ref = ref_files[fid].read_text(encoding="utf-8")
                hyp = hyp_files[fid].read_text(encoding="utf-8")
                
                if ds_name in ["ascend_zh", "ascend_mixed"]:
                    scores.append(calculate_cer(ref, hyp))
                    metric_name = "CER"
                else:
                    scores.append(calculate_wer(ref, hyp))

            avg_score = sum(scores) / len(scores)
            print(f"{model:<25} | {ds_name:<20} | {metric_name:<6} | {avg_score:.3f}  | {len(common)}")

        # --- DIARIZATION ---
        if PYANNOTE_AVAILABLE:
            for ds_name, ref_dir in DIAR_DATASETS.items():
                hyp_dir = DIAR_HYP_ROOT / model / ds_name
                if not hyp_dir.exists(): continue

                metrics, count = evaluate_diarization(ref_dir, hyp_dir)
                if metrics:
                    print(f"{model:<25} | {ds_name:<20} | {'DER':<6} | {metrics['DER']:.3f}% | {count}")
                    # Breakdown (Indented)
                    print(f"{'':<25} | {'':<20} | {'Miss':<6} | {metrics['Miss']:.3f}% |")
                    print(f"{'':<25} | {'':<20} | {'FA':<6}   | {metrics['FA']:.3f}%   |")
                    print(f"{'':<25} | {'':<20} | {'Conf':<6} | {metrics['Conf']:.3f}% |")
                    print("-" * 40)

    print("-" * 85)

if __name__ == "__main__":
    main()