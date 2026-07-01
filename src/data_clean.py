"""Data cleaning + filtering for WAXAL ASR training.

Two groups of filters:

  Text-only (runnable anywhere, incl. the Mac from Train.csv):
    - normalize with train_normalize
    - drop empty / whitespace-only targets
    - drop targets that are pure punctuation/markup residue
    - flag suspiciously long or malformed rows (the CSV had a few broken lines)

  Audio-dependent (applied on Kaggle where waveforms exist):
    - drop clips shorter than MIN_DURATION_S (default 1.5s) -> too little signal
    - drop clips whose implied speaking rate exceeds MAX_WORDS_PER_SEC
      (default 4.0) -> label/audio mismatch or truncated audio
    - drop clips longer than MAX_DURATION_S (default 30s) -> Whisper's window

These thresholds follow the WAXAL-NET recipe (drop <1.5s and >4 words/sec),
which materially improved their WER by removing mislabeled/misaligned pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .text_norm import train_normalize, score_normalize

MIN_DURATION_S = 1.5
MAX_DURATION_S = 30.0
MAX_WORDS_PER_SEC = 4.0
MIN_CHARS = 2


@dataclass
class CleanStats:
    kept: int = 0
    empty: int = 0
    too_short_text: int = 0
    too_short_audio: int = 0
    too_long_audio: int = 0
    too_fast: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def clean_target(raw: str) -> str | None:
    """Return the normalized training target, or None if the row is unusable."""
    t = train_normalize(raw)
    if not t:
        return None
    # If nothing survives aggressive normalization, it was pure punctuation.
    if not score_normalize(t):
        return None
    if len(t.strip()) < MIN_CHARS:
        return None
    return t


def audio_ok(duration_s: float, n_words: int,
             min_dur: float = MIN_DURATION_S,
             max_dur: float = MAX_DURATION_S,
             max_wps: float = MAX_WORDS_PER_SEC) -> tuple[bool, str]:
    """Audio-level filter. Returns (keep, reason_if_dropped)."""
    if duration_s < min_dur:
        return False, "too_short_audio"
    if duration_s > max_dur:
        return False, "too_long_audio"
    if n_words > 0 and (n_words / duration_s) > max_wps:
        return False, "too_fast"
    return True, ""


def clean_row(raw_target: str, duration_s: float | None,
              stats: CleanStats) -> str | None:
    """Full per-row cleaning. duration_s=None skips audio filters (text-only)."""
    t = clean_target(raw_target)
    if t is None:
        stats.empty += 1
        return None
    if duration_s is not None:
        keep, reason = audio_ok(duration_s, len(t.split()))
        if not keep:
            setattr(stats, reason, getattr(stats, reason) + 1)
            return None
    stats.kept += 1
    return t
