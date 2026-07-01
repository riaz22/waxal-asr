"""Text normalization for WAXAL ASR (Luganda / Lingala / Shona).

The Zindi metric is 0.5*WER + 0.5*CER between our predicted `Target` and the
hidden reference. We do NOT know Zindi's exact normalization, so we control the
only thing we can: keep training targets and predictions in one consistent
orthography, and provide `score_normalize` matching the common ASR convention
(lowercase, strip punctuation, collapse whitespace) for local validation.

Two normalizers, on purpose:

  * train_normalize  -> what the model learns to output and what we submit.
    Conservative: unify quotes/dashes, drop control chars / stray markup,
    collapse whitespace. KEEP case and sentence punctuation (HF references keep
    them; Whisper models them natively). If offline checks show Zindi strips
    them, set SUBMIT_LOWER_NOPUNCT = True and resubmit.

  * score_normalize  -> lowercased, punctuation-removed, whitespace-collapsed.
    Used only for our local WER/CER estimate (mirrors the WAXAL-NET paper's
    "lowercasing and punctuation removal" for metric computation).

Pure-Python, no third-party deps: runs identically on the Mac, on Kaggle, and
inside the data collator.
"""

from __future__ import annotations

import re
import unicodedata

# Flip to True only if we confirm Zindi lowercases + strips punctuation before
# scoring. Default False keeps the richer orthography (safer, reversible).
SUBMIT_LOWER_NOPUNCT = False

# Unify fancy glyphs -> ascii. Keys use \u escapes to avoid invisible chars.
_UNIFY = {
    "‘": "'", "’": "'", "ʼ": "'", "′": "'", "`": "'",
    "“": '"', "”": '"', "″": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...",
    " ": " ", "​": "", "﻿": "",
}
_UNIFY_RE = re.compile("|".join(re.escape(k) for k in _UNIFY))

# C0 controls (except \t\n\r), C1 controls, and DEL.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WS_RE = re.compile(r"\s+")

# For scoring: keep word chars (incl. accented letters), intra-word apostrophe,
# digits and spaces; drop the rest. Apostrophe is orthographic in Luganda /
# Lingala (g'ennyanja, n'ebbali) so we keep it between letters only.
_PUNCT_FOR_SCORE_RE = re.compile(r"[^\w'\s]", re.UNICODE)
_EDGE_APOS_RE = re.compile(r"(?<!\w)'|'(?!\w)")


def _base_clean(text: str) -> str:
    """Shared pass: NFC, unify glyphs, kill control chars, newlines -> space."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = _UNIFY_RE.sub(lambda m: _UNIFY[m.group(0)], text)
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    text = _CTRL_RE.sub("", text)
    return text


def train_normalize(text: str) -> str:
    """Normalization for training targets and submitted predictions.

    Keeps case + sentence punctuation; only removes noise and unifies glyphs.
    """
    text = _base_clean(text)
    text = _WS_RE.sub(" ", text).strip()
    if SUBMIT_LOWER_NOPUNCT:
        return score_normalize(text)
    return text


def score_normalize(text: str) -> str:
    """Aggressive normalization for local WER/CER estimation only."""
    text = _base_clean(text).lower()
    text = _PUNCT_FOR_SCORE_RE.sub(" ", text)
    text = _EDGE_APOS_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


if __name__ == "__main__":
    samples = [
        "Ekyuma ekyakolebwa Bamagulumeeru nga kiri mu makkati g'ennyanja.",
        'iyo ine chikwangwani chakanyorwa kuti "WHITEHILL".',
        "  ya  liboso   eza\tlangi\n ya motane. ",
    ]
    for s in samples:
        print("RAW  :", repr(s))
        print("TRAIN:", repr(train_normalize(s)))
        print("SCORE:", repr(score_normalize(s)))
        print()
