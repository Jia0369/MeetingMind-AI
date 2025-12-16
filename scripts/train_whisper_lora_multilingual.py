from pathlib import Path
import os
import random
import numpy as np
import librosa
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass

# =============================================================================
# Config
# =============================================================================
ASR_EVAL_ROOT = Path(r"C:\Users\USER\Documents\meeting-asr\data\asr_eval")
LOCAL_DATASETS = {
    "ascend_en": ASR_EVAL_ROOT / "ascend" / "en",
    "ascend_zh": ASR_EVAL_ROOT / "ascend" / "zh",
    "ascend_mixed": ASR_EVAL_ROOT / "ascend" / "mixed_enzh",
    #"malcsc_ms": ASR_EVAL_ROOT / "malcsc" / "ms",
    "fleurs_ms_my": ASR_EVAL_ROOT / "fleurs" / "ms_my",
}

MAX_FILES_PER_DATASET = {
    "ascend_en": 1000,
    "ascend_zh": 1000,
    "ascend_mixed": 1000,
    #"malcsc_ms": 20,
    "fleurs_ms_my": 455,
}

TARGET_SR = 16000
SEED = 42
MODEL_ID = "openai/whisper-small"
OUTPUT_DIR = r"C:\Users\USER\Documents\meeting-asr\models\whisper-small-lora-trilingual"

NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# Data Loading
# =============================================================================
def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    audio = audio.astype(np.float32)
    if len(audio) > 0:
        audio /= np.max(np.abs(audio)) + 1e-8
    return audio

def collect_examples():
    examples = []
    for name, folder in LOCAL_DATASETS.items():
        if not folder.exists():
            print(f"[SKIP] {folder}")
            continue
        txt_files = list(folder.glob("*.txt"))
        max_n = MAX_FILES_PER_DATASET.get(name, len(txt_files))
        txt_files = random.sample(txt_files, min(len(txt_files), max_n))
        print(f"[LOAD] {name}: {len(txt_files)}")
        for txt in txt_files:
            wav = folder / f"{txt.stem}.wav"
            if wav.exists():
                text = txt.read_text(encoding="utf-8").strip()
                if text:
                    examples.append({"audio_path": str(wav), "text": text})
    print(f"[LOAD] Total: {len(examples)}")
    return examples

# =============================================================================
# Collator
# =============================================================================
@dataclass
class Collator:
    processor: WhisperProcessor
    def __call__(self, batch):
        feats = [torch.tensor(b["input_features"], dtype=torch.float32) for b in batch]
        feats = torch.stack(feats)
        labels = [b["labels"] for b in batch]
        padded = self.processor.tokenizer.pad({"input_ids": labels}, padding=True, return_tensors="pt")
        labels = padded["input_ids"].masked_fill(padded.attention_mask.ne(1), -100)
        return {"input_features": feats, "labels": labels}

# =============================================================================
# Main
# =============================================================================
def main():
    examples = collect_examples()
    ds = Dataset.from_list(examples)

    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # Freeze encoder
    for p in model.model.encoder.parameters():
        p.requires_grad = False

    # LoRA
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM"
    ))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(device)

    # Preprocess
    def preprocess(ex):
        audio = load_audio(ex["audio_path"])
        feat = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features[0]
        labels = processor.tokenizer(ex["text"], truncation=True, max_length=448).input_ids
        return {"input_features": feat, "labels": labels}

    print("[PREPROCESS] Processing audio features (this takes ~2-4 minutes)...")
    ds = ds.map(preprocess, remove_columns=["audio_path", "text"], num_proc=None, keep_in_memory=True)
    ds = ds.train_test_split(test_size=0.05, seed=SEED)

    train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=Collator(processor))
    eval_loader = DataLoader(ds["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=Collator(processor))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"\nSTARTING TRAINING — {len(ds['train'])} samples")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with model._enable_peft_forward_hooks():
                outputs = model.base_model(input_features=batch["input_features"], labels=batch["labels"])
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            total_loss += loss.item()
            if i % GRAD_ACCUM_STEPS == 0 or i == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            if i % 10 == 0:
                print(f"  Step {i} | Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f}")
        print(f"\nEpoch {epoch} — Avg Loss: {total_loss/len(train_loader):.4f}")

        # Eval
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with model._enable_peft_forward_hooks():
                    outputs = model.base_model(input_features=batch["input_features"], labels=batch["labels"])
                eval_loss += outputs.loss.item()
        print(f"  Eval Loss: {eval_loss / len(eval_loader):.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\nDONE! Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()