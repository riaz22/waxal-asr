"""Text-level EDA + artifact generation from Train.csv (runs on the Mac).

Produces, per language (lug / lin / sna):
  * artifacts/vocab_<lang>.json   -> CTC character vocabulary for MMS/wav2vec2
  * artifacts/lm_corpus_<lang>.txt -> normalized text, one utterance per line,
                                      to train a KenLM n-gram for CTC decoding
  * artifacts/train_clean.csv     -> id, language, split, target (train_normalize)
  * artifacts/eda_report.md       -> human-readable summary

Audio-duration filters are NOT applied here (no waveforms locally); they run on
Kaggle. This step handles text cleaning, vocab, and the LM corpus.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.text_norm import train_normalize, score_normalize  # noqa: E402
from src.data_clean import clean_target  # noqa: E402

TRAIN_CSV = Path(
    "/Users/razihamdi/Downloads/google-waxal-asr-challenge20260630-10570-elxebu/Train.csv"
)
ART = ROOT / "artifacts"
LANGS = ["lug", "lin", "sna"]

# CTC special tokens (wav2vec2/MMS convention).
PAD, UNK, WORD_DELIM = "<pad>", "<unk>", "|"


def build_vocab(targets: list[str]) -> dict:
    """Char vocab for CTC from score-normalized text (lowercase, no punct)."""
    chars: Counter = Counter()
    for t in targets:
        chars.update(score_normalize(t).replace(" ", ""))
    # deterministic order: by frequency desc then codepoint
    ordered = sorted(chars, key=lambda c: (-chars[c], c))
    vocab = {WORD_DELIM: 0, UNK: 1, PAD: 2}
    for c in ordered:
        vocab[c] = len(vocab)
    return {"vocab": vocab, "char_freq": dict(chars.most_common())}


def main() -> None:
    ART.mkdir(exist_ok=True)
    df = pd.read_csv(TRAIN_CSV, escapechar="\\")
    df = df[df["language"].isin(LANGS)].copy()

    df["target"] = df["transcription"].map(clean_target)
    dropped = int(df["target"].isna().sum())
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    df["nwords"] = df["target"].str.split().map(len)
    df["nchars"] = df["target"].str.len()

    report = ["# WAXAL text EDA\n",
              f"Rows after text cleaning: **{len(df)}** "
              f"(dropped {dropped} empty/punct-only)\n"]

    for lang in LANGS:
        sub = df[df["language"] == lang]
        targets = sub["target"].tolist()
        vinfo = build_vocab(targets)
        (ART / f"vocab_{lang}.json").write_text(
            json.dumps(vinfo, ensure_ascii=False, indent=2))
        corpus = "\n".join(score_normalize(t) for t in targets) + "\n"
        (ART / f"lm_corpus_{lang}.txt").write_text(corpus)

        vocab_chars = [c for c in vinfo["vocab"]
                       if c not in (PAD, UNK, WORD_DELIM)]
        rare = [c for c, n in vinfo["char_freq"].items() if n <= 5]
        report += [
            f"\n## {lang}  (n={len(sub)})",
            f"- words/utt: med={sub.nwords.median():.0f} "
            f"p95={sub.nwords.quantile(.95):.0f} max={sub.nwords.max()}",
            f"- chars/utt: med={sub.nchars.median():.0f} "
            f"p95={sub.nchars.quantile(.95):.0f} max={sub.nchars.max()}",
            f"- vocab size (CTC, ex-specials): {len(vocab_chars)}",
            f"- alphabet: `{''.join(sorted(vocab_chars))}`",
            f"- rare chars (<=5 occ), review for noise: "
            f"`{''.join(sorted(rare))}`" if rare else "- rare chars: none",
        ]

    df[["id", "language", "original_split", "target"]].to_csv(
        ART / "train_clean.csv", index=False)
    (ART / "eda_report.md").write_text("\n".join(report) + "\n")

    print(f"Wrote artifacts to {ART}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
