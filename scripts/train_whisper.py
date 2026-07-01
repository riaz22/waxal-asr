"""Step 4: fine-tune openai/whisper-large-v3-turbo jointly on lug/lin/sna.

One multilingual model, per-sample language prefix tokens. Whisper's subword
decoder handles code-switching (French/English words in Lingala/Shona) and
punctuation far better than char-CTC -- this is our lever on Lingala, the 44%
test slice where MMS struggles.

Notes:
  * Whisper truncates audio to 30s, so prepare_data already drops >30s clips.
    At inference, route any >30s test clip to MMS instead.
  * Full fine-tune with fp16 + gradient checkpointing fits P100/T4 (turbo is
    ~0.8B). If you OOM, pass --lora to switch to PEFT-LoRA, or lower --bs.

    !python scripts/train_whisper.py            # full FT
    !python scripts/train_whisper.py --lora     # low-memory
"""

from __future__ import annotations

import os
# Single GPU (avoid HF Trainer's nn.DataParallel OOM on T4 x2). Set before torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import torch
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src.text_norm import score_normalize  # noqa: E402
from src.metrics import wer as _wer, cer as _cer  # noqa: E402


@dataclass
class DataCollatorWhisper:
    """Build log-mel features + per-sample language-tokened labels ON THE FLY.

    Not pre-materialized to disk: Whisper log-mels are ~30GB for this data and
    would overflow /kaggle/working. Only the current batch is realized.
    """
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        fe, tok = self.processor.feature_extractor, self.processor.tokenizer
        audios = [f["audio"]["array"] for f in features]
        batch = fe(audios, sampling_rate=C.SAMPLE_RATE, return_tensors="pt")
        label_ids = []
        for f in features:
            tok.set_prefix_tokens(language=C.WHISPER_LANG[f["lang"]],
                                  task="transcribe")
            label_ids.append(
                {"input_ids": tok(score_normalize(f["target"])).input_ids[:448]})
        lab = tok.pad(label_ids, return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (labels[:, 0] == tok.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def load_split(split: str) -> datasets.Dataset:
    parts = []
    for lang in C.LANGS:
        ds = datasets.load_from_disk(str(C.clean_ds_path(lang, split)))
        ds = ds.add_column("lang", [lang] * len(ds))
        parts.append(ds)
    return datasets.concatenate_datasets(parts).shuffle(seed=42)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=4)         # turbo on 16GB
    ap.add_argument("--grad_accum", type=int, default=8)   # eff. batch 32
    ap.add_argument("--lora", action="store_true")
    args = ap.parse_args()
    C.require_prepared()
    out = C.whisper_dir()

    processor = WhisperProcessor.from_pretrained(
        C.WHISPER_MODEL, task="transcribe")
    tok = processor.tokenizer

    # Keep only audio + target + lang; features built per-batch in the collator.
    keep = ["audio", "target", "lang"]
    train = load_split("train")
    val = load_split("validation")
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    val = val.remove_columns([c for c in val.column_names if c not in keep])

    model = WhisperForConditionalGeneration.from_pretrained(C.WHISPER_MODEL)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none"))
        model.print_trainable_parameters()
    else:
        model.gradient_checkpointing_enable()

    collator = DataCollatorWhisper(processor=processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tok.pad_token_id
        pred_str = [score_normalize(s) for s in
                    tok.batch_decode(pred_ids, skip_special_tokens=True)]
        ref_str = [score_normalize(s) for s in
                   tok.batch_decode(label_ids, skip_special_tokens=True)]
        return {"wer": _wer(ref_str, pred_str), "cer": _cer(ref_str, pred_str)}

    targs = Seq2SeqTrainingArguments(
        output_dir=str(out), per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        warmup_ratio=0.05, lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(), gradient_checkpointing=not args.lora,
        remove_unused_columns=False,   # collator needs raw audio + target + lang
        dataloader_num_workers=2,      # parallelize on-the-fly audio decode
        predict_with_generate=True, generation_max_length=400,
        eval_strategy="steps", eval_steps=500, save_steps=500, logging_steps=50,
        save_total_limit=1, save_only_model=True, load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False,
        weight_decay=0.0, report_to="none",
    )
    proc_key = ("processing_class"
                if "processing_class" in inspect.signature(Seq2SeqTrainer).parameters
                else "tokenizer")
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train, eval_dataset=val,
        data_collator=collator, compute_metrics=compute_metrics,
        **{proc_key: processor.feature_extractor})
    trainer.train()
    model.save_pretrained(str(out))
    processor.save_pretrained(str(out))
    print(f"Saved Whisper to {out}")


if __name__ == "__main__":
    main()
