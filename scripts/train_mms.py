"""Step 3: fine-tune facebook/mms-1b-all (CTC) for one language.

Uses the official MMS *adapter* fine-tuning recipe: freeze the 1B base, train
only the per-language adapter + a fresh CTC head sized to our character vocab.
This fits a single P100/T4 (16 GB) with fp16 + gradient checkpointing and is
what makes MMS-1b tractable on Kaggle. SpecAugment is enabled via model config
for regularization / Phase-2 robustness.

Run once per language:
    !python scripts/train_mms.py --lang lin
    !python scripts/train_mms.py --lang sna
    !python scripts/train_mms.py --lang lug
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import torch
from transformers import (
    Trainer, TrainingArguments, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC, Wav2Vec2Processor,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src.text_norm import score_normalize  # noqa: E402
from src.metrics import wer as _wer, cer as _cer  # noqa: E402


@dataclass
class DataCollatorCTC:
    """Extract features from raw audio + tokenize targets ON THE FLY, per batch.

    Feature extraction is NOT pre-materialized to disk (that overflowed Kaggle's
    ~20GB /kaggle/working). The dataset keeps only compressed audio + text; the
    heavy float arrays exist just for the current batch.
    """
    processor: Wav2Vec2Processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audios = [f["audio"]["array"] for f in features]
        texts = [score_normalize(f["target"]) for f in features]
        batch = self.processor.feature_extractor(
            audios, sampling_rate=C.SAMPLE_RATE, padding=True,
            return_tensors="pt")
        lab = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
        batch["labels"] = lab["input_ids"].masked_fill(
            lab.attention_mask.ne(1), -100)
        return batch


def _write_mms_vocab(lang: str, out_dir: Path) -> Path:
    """MMS tokenizer wants a nested {lang: {char: id}} vocab.json."""
    flat = json.loads(C.vocab_path(lang).read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "vocab.json"
    p.write_text(json.dumps({lang: flat}, ensure_ascii=False))
    return p


def build_processor(lang: str, out_dir: Path) -> Wav2Vec2Processor:
    _write_mms_vocab(lang, out_dir)
    tokenizer = Wav2Vec2CTCTokenizer(
        str(out_dir / "vocab.json"), unk_token="<unk>", pad_token="<pad>",
        word_delimiter_token="|", target_lang=lang)
    tokenizer.save_pretrained(str(out_dir))
    fe = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=C.SAMPLE_RATE, padding_value=0.0,
        do_normalize=True, return_attention_mask=True)
    return Wav2Vec2Processor(feature_extractor=fe, tokenizer=tokenizer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=C.LANGS)
    ap.add_argument("--epochs", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=1e-3)   # adapters like higher lr
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    args = ap.parse_args()
    C.require_prepared()
    lang, out = args.lang, C.mms_dir(args.lang)

    proc = build_processor(lang, out)

    # Keep only audio + target; features are built per-batch in the collator
    # (no disk materialization -> no /kaggle/working overflow).
    train = datasets.load_from_disk(str(C.clean_ds_path(lang, "train")))
    val = datasets.load_from_disk(str(C.clean_ds_path(lang, "validation")))
    keep = ["audio", "target"]
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    val = val.remove_columns([c for c in val.column_names if c not in keep])

    model = Wav2Vec2ForCTC.from_pretrained(
        C.MMS_MODEL, target_lang=lang, ignore_mismatched_sizes=True,
        vocab_size=len(proc.tokenizer),
        ctc_loss_reduction="mean", pad_token_id=proc.tokenizer.pad_token_id,
        # SpecAugment for regularization.
        apply_spec_augment=True, mask_time_prob=0.05, mask_feature_prob=0.05,
    )
    model.init_adapter_layers()
    model.freeze_base_model()
    # Unfreeze only adapter weights + CTC head.
    for name, p in model.named_parameters():
        if "adapter" in name or "lm_head" in name:
            p.requires_grad = True
    model.gradient_checkpointing_enable()

    collator = DataCollatorCTC(processor=proc)

    def compute_metrics(pred):
        logits = pred.predictions
        ids = logits.argmax(-1)
        labels = pred.label_ids
        labels[labels == -100] = proc.tokenizer.pad_token_id
        pred_str = proc.batch_decode(ids)
        ref_str = proc.batch_decode(labels, group_tokens=False)
        pred_str = [score_normalize(s) for s in pred_str]
        ref_str = [score_normalize(s) for s in ref_str]
        return {"wer": _wer(ref_str, pred_str), "cer": _cer(ref_str, pred_str)}

    targs = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        warmup_ratio=0.1, lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(), gradient_checkpointing=True,
        remove_unused_columns=False,   # collator needs raw audio + target
        eval_strategy="steps", eval_steps=400, save_steps=400, logging_steps=50,
        save_total_limit=1, save_only_model=True, load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False,
        weight_decay=0.005, report_to="none",
    )
    # transformers v5 renamed Trainer's `tokenizer` arg to `processing_class`.
    proc_key = ("processing_class"
                if "processing_class" in inspect.signature(Trainer).parameters
                else "tokenizer")
    trainer = Trainer(
        model=model, args=targs, train_dataset=train, eval_dataset=val,
        data_collator=collator, compute_metrics=compute_metrics,
        **{proc_key: proc.feature_extractor})
    trainer.train()
    # Persist the adapter weights explicitly (small, easy to reload for infer).
    model.save_pretrained(str(out))
    proc.save_pretrained(str(out))
    print(f"Saved MMS-{lang} to {out}")


if __name__ == "__main__":
    main()
