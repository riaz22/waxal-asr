"""Step 6a: run one model over a split and write per-id predictions.

    !python scripts/infer.py --model mms     --split test        # + KenLM
    !python scripts/infer.py --model whisper --split test
    # use --split validation to estimate the board score before submitting.

Routing uses the LID parquet (scripts/lid.py) so this path is identical in
Phase 2 (audio only): MMS picks the per-language model by LID; Whisper forces
the decoder language by LID. Falls back to the id prefix if LID is absent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src.text_norm import train_normalize  # noqa: E402


def load_split(split: str) -> datasets.Dataset:
    parts = []
    for lang in C.LANGS:
        ds = datasets.load_from_disk(str(C.clean_ds_path(lang, split)))
        cols = ["id", "audio"] + (["target"] if "target" in ds.column_names else [])
        ds = ds.remove_columns([c for c in ds.column_names if c not in cols])
        parts.append(ds)
    return datasets.concatenate_datasets(parts)


def lang_map(ds: datasets.Dataset, split: str) -> dict[str, str]:
    lid = C.ART_DIR / f"lid_{split}.parquet"
    if lid.exists():
        df = pd.read_parquet(lid)
        return dict(zip(df["id"].astype(str), df["lang"]))
    # Fallback: id prefix (Phase 1 only).
    return {str(i): str(i).split("_")[0] for i in ds["id"]}


def _mms_decoder(proc, lang):
    """pyctcdecode decoder with KenLM if a model is present, else greedy."""
    try:
        from pyctcdecode import build_ctcdecoder
    except Exception:
        return None
    labels = [t for t, _ in sorted(proc.tokenizer.get_vocab().items(),
                                   key=lambda kv: kv[1])]
    # pyctcdecode wants "" for CTC blank/special and " " for the word delimiter.
    labels = ["" if t in ("<pad>", "<unk>", "<s>", "</s>") else
              (" " if t == "|" else t) for t in labels]
    for cand in (C.lm_path(lang), C.lm_path(lang).with_suffix(".arpa")):
        if cand.exists():
            return build_ctcdecoder(labels, kenlm_model_path=str(cand),
                                    alpha=0.5, beta=1.5)
    return build_ctcdecoder(labels)  # no LM, still beam-capable


def infer_mms(ds, l_map, split):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds: dict[str, str] = {}
    for lang in C.LANGS:
        ids = [str(i) for i in ds["id"] if l_map.get(str(i), "") == lang]
        if not ids:
            continue
        mdir = C.mms_dir(lang)
        proc = Wav2Vec2Processor.from_pretrained(str(mdir))
        proc.tokenizer.set_target_lang(lang)
        # Fine-tuned adapter + head are in the saved weights; target_lang wires
        # the right adapter path. (No separate load_adapter needed.)
        model = Wav2Vec2ForCTC.from_pretrained(
            str(mdir), target_lang=lang,
            ignore_mismatched_sizes=True).to(device).eval()
        decoder = _mms_decoder(proc, lang)
        idset = set(ids)
        for row in ds:
            rid = str(row["id"])
            if rid not in idset:
                continue
            au = row["audio"]
            iv = proc(au["array"], sampling_rate=au["sampling_rate"],
                      return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                logits = model(iv).logits[0].cpu().numpy()
            if decoder is not None:
                text = decoder.decode(logits)
            else:
                ids_arg = logits.argmax(-1)
                text = proc.decode(ids_arg)
            preds[rid] = train_normalize(text)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return preds


def infer_whisper(ds, l_map, split):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wdir = C.whisper_dir()
    proc = WhisperProcessor.from_pretrained(str(wdir))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(wdir)).to(device).eval()
    preds: dict[str, str] = {}
    for row in ds:
        rid = str(row["id"])
        lang = l_map.get(rid, rid.split("_")[0])
        au = row["audio"]
        feats = proc.feature_extractor(
            au["array"], sampling_rate=au["sampling_rate"],
            return_tensors="pt").input_features.to(device)
        forced = proc.get_decoder_prompt_ids(
            language=C.WHISPER_LANG[lang], task="transcribe")
        with torch.no_grad():
            out = model.generate(feats, forced_decoder_ids=forced,
                                 num_beams=5, max_new_tokens=400)
        text = proc.batch_decode(out, skip_special_tokens=True)[0]
        preds[rid] = train_normalize(text)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mms", "whisper"])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ds = load_split(args.split)
    l_map = lang_map(ds, args.split)
    if args.model == "mms":
        preds = infer_mms(ds, l_map, args.split)
    else:
        preds = infer_whisper(ds, l_map, args.split)

    rows = [{"id": k, "pred": v, "lang": l_map.get(k, k.split("_")[0])}
            for k, v in preds.items()]
    out = C.ART_DIR / f"preds_{args.model}_{args.split}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    print(f"wrote {out}  ({len(rows)} preds)")


if __name__ == "__main__":
    main()
