"""Step 1 (Kaggle): download the 3 WAXAL ASR configs, clean, and cache.

Run this ONCE with internet enabled, then save the notebook's /kaggle/working
output as a Kaggle Dataset named e.g. `waxal-clean` and attach it to the
training notebooks (so you never re-download ~10-15 GB per session).

For each language (lug/lin/sna) it:
  * loads train + validation (labeled) and applies text + audio cleaning
  * loads the test split and keeps ONLY the 4,253 Zindi test ids (no cleaning)
  * writes cleaned splits with save_to_disk, rebuilds CTC vocab + LM corpus

Cleaning follows WAXAL-NET: drop <1.5s, >30s, >4 words/sec, empty/punct-only.

Usage (Kaggle cell):
    !python scripts/prepare_data.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import datasets
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src.text_norm import score_normalize  # noqa: E402
from src.data_clean import clean_target, audio_ok, CleanStats  # noqa: E402

PAD, UNK, DELIM = "<pad>", "<unk>", "|"


def _zindi_test_ids() -> set[str]:
    return set(pd.read_csv(C.TEST_CSV)["ID"].astype(str))


def _clean_labeled(ds: datasets.Dataset, stats: CleanStats) -> datasets.Dataset:
    """Apply text + audio filters, attach normalized `target`, drop bad rows."""
    def _map(batch):
        keep_target, keep_flag = [], []
        for tr, au in zip(batch["transcription"], batch["audio"]):
            t = clean_target(tr)
            if t is None:
                keep_target.append(""); keep_flag.append(False)
                stats.empty += 1
                continue
            dur = len(au["array"]) / au["sampling_rate"]
            ok, reason = audio_ok(dur, len(t.split()))
            if not ok:
                keep_target.append(""); keep_flag.append(False)
                setattr(stats, reason, getattr(stats, reason) + 1)
                continue
            keep_target.append(t); keep_flag.append(True)
            stats.kept += 1
        return {"target": keep_target, "_keep": keep_flag}

    ds = ds.map(_map, batched=True, batch_size=64)
    ds = ds.filter(lambda b: b["_keep"], batched=True)
    return ds.remove_columns([c for c in ("_keep",) if c in ds.column_names])


def _build_vocab(targets: list[str]) -> dict:
    chars: Counter = Counter()
    for t in targets:
        chars.update(score_normalize(t).replace(" ", ""))
    ordered = sorted(chars, key=lambda c: (-chars[c], c))
    vocab = {DELIM: 0, UNK: 1, PAD: 2}
    for c in ordered:
        vocab[c] = len(vocab)
    return vocab


def prepare_language(lang: str, test_ids: set[str]) -> None:
    cfg = f"{lang}_asr"
    print(f"\n=== {lang} ===")
    stats = CleanStats()

    for split in ("train", "validation"):
        ds = datasets.load_dataset(C.DATASET_ID, name=cfg, split=split)
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=C.SAMPLE_RATE))
        ds = _clean_labeled(ds, stats)
        out = C.clean_ds_path(lang, split)
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(out))
        print(f"  {split}: kept {len(ds)} -> {out}")

    # Test: keep every Zindi id, no filtering (we must predict all of them).
    test = datasets.load_dataset(C.DATASET_ID, name=cfg, split="test")
    test = test.cast_column("audio", datasets.Audio(sampling_rate=C.SAMPLE_RATE))
    before = len(test)
    test = test.filter(lambda b: [str(i) in test_ids for i in b["id"]],
                       batched=True)
    keep_cols = [c for c in ("id", "audio") if c in test.column_names]
    test = test.remove_columns([c for c in test.column_names if c not in keep_cols])
    test.save_to_disk(str(C.clean_ds_path(lang, "test")))
    print(f"  test: {before} -> kept {len(test)} matching Zindi ids")

    # Rebuild vocab + LM corpus from the CLEANED train targets.
    train = datasets.load_from_disk(str(C.clean_ds_path(lang, "train")))
    targets = train["target"]
    C.vocab_path(lang).parent.mkdir(parents=True, exist_ok=True)
    C.vocab_path(lang).write_text(json.dumps(_build_vocab(targets),
                                             ensure_ascii=False, indent=2))
    corpus = "\n".join(score_normalize(t) for t in targets) + "\n"
    C.lm_corpus_path(lang).write_text(corpus)
    print(f"  filter stats: {stats.as_dict()}")


def main() -> None:
    if str(C.DATA_DIR).startswith("/kaggle/input"):
        raise SystemExit(
            f"DATA_DIR={C.DATA_DIR} is a read-only attached dataset -- the "
            "cleaned data already exists, so skip this step and go straight to "
            "build_lm / training. (Detach the 'waxal-clean' dataset if you want "
            "to regenerate it into /kaggle/working.)")
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    test_ids = _zindi_test_ids()
    print(f"Zindi test ids: {len(test_ids)}")
    missing = dict.fromkeys(C.LANGS, 0)
    for lang in C.LANGS:
        prepare_language(lang, test_ids)
    # Cross-check we found audio for every Zindi id.
    found = set()
    for lang in C.LANGS:
        found |= set(datasets.load_from_disk(
            str(C.clean_ds_path(lang, "test")))["id"])
    missing_ids = test_ids - {str(i) for i in found}
    print(f"\nTest ids with NO audio found: {len(missing_ids)}")
    if missing_ids:
        print("  e.g.", list(missing_ids)[:10])
        print("  -> check other splits (validation/unlabeled) for these ids")


if __name__ == "__main__":
    main()
