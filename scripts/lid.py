"""Spoken language identification (lug/lin/sna) from audio.

Phase 2 ships audio with NO language label, so per-language routing must come
from the signal. We use facebook/mms-lid-256 (covers all three) restricted to
our 3 classes -> argmax. On Phase 1 we validate against the id prefix, which is
free ground truth; if accuracy is imperfect we can fine-tune a small classifier.

Outputs artifacts/lid_<split>.parquet with columns [id, lang, p_lug, p_lin, p_sna].

    !python scripts/lid.py --split test
    !python scripts/lid.py --split validation   # to measure LID accuracy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import datasets
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

# MMS-LID label codes (ISO 639-3) for our languages.
MMS_LID_CODE = {"lug": "lug", "lin": "lin", "sna": "sna"}


def load_all(split: str) -> datasets.Dataset:
    parts = []
    for lang in C.LANGS:
        ds = datasets.load_from_disk(str(C.clean_ds_path(lang, split)))
        if "id" not in ds.column_names:
            ds = ds.add_column("id", [f"{lang}_{i}" for i in range(len(ds))])
        parts.append(ds)
    return datasets.concatenate_datasets(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    C.require_prepared()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(C.LID_MODEL)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        C.LID_MODEL).to(device).eval()

    # Column indices for our 3 languages within the 256-way head.
    l2i = model.config.label2id
    idx = {lang: l2i[MMS_LID_CODE[lang]] for lang in C.LANGS}

    ds = load_all(args.split)
    rows = []
    for start in range(0, len(ds), args.batch):
        chunk = ds[start:start + args.batch]
        arrays = [a["array"] for a in chunk["audio"]]
        inputs = fe(arrays, sampling_rate=C.SAMPLE_RATE,
                    return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        sub = torch.stack([logits[:, idx[l]] for l in C.LANGS], dim=1)
        probs = sub.softmax(-1).cpu().numpy()
        picks = probs.argmax(1)
        for j, cid in enumerate(chunk["id"]):
            rows.append({"id": str(cid), "lang": C.LANGS[picks[j]],
                         "p_lug": float(probs[j][0]),
                         "p_lin": float(probs[j][1]),
                         "p_sna": float(probs[j][2])})

    df = pd.DataFrame(rows)
    out = C.ART_DIR / f"lid_{args.split}.parquet"
    df.to_parquet(out)
    print(f"wrote {out}  ({len(df)} rows)")

    # Free validation via the id prefix.
    if df["id"].str.contains("_").all():
        df["true"] = df["id"].str.split("_").str[0]
        acc = (df["true"] == df["lang"]).mean()
        print(f"LID accuracy vs id-prefix: {acc:.4f}")
        print(pd.crosstab(df["true"], df["lang"]))


if __name__ == "__main__":
    main()
