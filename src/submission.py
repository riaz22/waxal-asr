"""Build + validate the Zindi submission (ID,Target) from id->prediction dicts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .text_norm import train_normalize


def build_submission(preds: dict[str, str], sample_sub_csv: Path,
                     out_csv: Path, fallback: str = "") -> pd.DataFrame:
    """Assemble a submission that matches SampleSubmission's ids and order.

    - Every SampleSubmission ID must appear exactly once (Zindi rejects
      otherwise). Missing predictions get `fallback`.
    - Targets are passed through train_normalize for a consistent orthography.
    """
    sample = pd.read_csv(sample_sub_csv)
    id_col = sample.columns[0]           # "ID"
    tgt_col = sample.columns[1]          # "Target"
    ids = sample[id_col].astype(str).tolist()

    missing = [i for i in ids if i not in preds]
    extra = [k for k in preds if k not in set(ids)]
    rows = []
    for i in ids:
        text = preds.get(i, fallback)
        text = train_normalize(text)
        if not text:
            text = fallback or "a"       # never submit empty -> avoids NaN rows
        rows.append({id_col: i, tgt_col: text})
    out = pd.DataFrame(rows, columns=[id_col, tgt_col])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    print(f"submission: {len(out)} rows -> {out_csv}")
    print(f"  missing preds (filled with fallback): {len(missing)}")
    print(f"  extra preds not in sample (ignored): {len(extra)}")
    assert len(out) == len(ids), "row count must equal SampleSubmission"
    assert out[id_col].is_unique, "duplicate ids in submission"
    return out


def validate_submission(out_csv: Path, sample_sub_csv: Path) -> bool:
    sub = pd.read_csv(out_csv)
    sample = pd.read_csv(sample_sub_csv)
    ok = True
    if list(sub.columns) != list(sample.columns):
        print(f"BAD columns: {list(sub.columns)} != {list(sample.columns)}")
        ok = False
    if set(sub[sub.columns[0]].astype(str)) != set(
            sample[sample.columns[0]].astype(str)):
        print("BAD: ID set differs from SampleSubmission")
        ok = False
    if sub[sub.columns[1]].isna().any():
        print("BAD: empty/NaN targets present")
        ok = False
    print("submission valid" if ok else "submission INVALID")
    return ok
