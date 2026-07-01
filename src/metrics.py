"""Local WER / CER / blended-score computation for WAXAL ASR.

Mirrors how ASR challenges score: corpus-level (micro-averaged) edit distance,
i.e. sum of edits over the whole set divided by total reference length -- NOT
the mean of per-utterance rates. This matches jiwer.wer(refs, preds) when refs
and preds are lists.

Zindi's leaderboard "score" is the weighted error 0.5*WER + 0.5*CER, and the
public board shows it as (1 - error) so higher is better (top is ~0.83). We
report both the raw error and the (1 - error) board-style score.

Uses jiwer if installed (fast C-ish path); otherwise a pure-Python Levenshtein
fallback so this runs on the Mac with no extra install.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import jiwer  # type: ignore
    _HAVE_JIWER = True
except Exception:  # pragma: no cover
    _HAVE_JIWER = False


def _edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two token sequences (O(len(a)*len(b)))."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = cur
    return prev[-1]


def _corpus_rate(refs: list[str], preds: list[str], char: bool) -> float:
    edits = 0
    length = 0
    for r, p in zip(refs, preds):
        rt = list(r) if char else r.split()
        pt = list(p) if char else p.split()
        edits += _edit_distance(rt, pt)
        length += len(rt)
    return edits / max(length, 1)


def wer(refs: list[str], preds: list[str]) -> float:
    if _HAVE_JIWER:
        return float(jiwer.wer(list(refs), list(preds)))
    return _corpus_rate(list(refs), list(preds), char=False)


def cer(refs: list[str], preds: list[str]) -> float:
    if _HAVE_JIWER:
        return float(jiwer.cer(list(refs), list(preds)))
    return _corpus_rate(list(refs), list(preds), char=True)


@dataclass
class Score:
    wer: float
    cer: float

    @property
    def error(self) -> float:
        """Weighted error 0.5*WER + 0.5*CER (lower is better)."""
        return 0.5 * self.wer + 0.5 * self.cer

    @property
    def board(self) -> float:
        """Leaderboard-style score = 1 - error (higher is better)."""
        return 1.0 - self.error

    def __str__(self) -> str:
        return (f"WER={self.wer:.4f}  CER={self.cer:.4f}  "
                f"error={self.error:.4f}  board={self.board:.4f}")


def score(refs: list[str], preds: list[str]) -> Score:
    """Compute the full Score. Caller is responsible for normalizing text."""
    return Score(wer=wer(refs, preds), cer=cer(refs, preds))


def score_by_language(ids: list[str], refs: list[str], preds: list[str]
                      ) -> dict[str, Score]:
    """Per-language + overall breakdown. Language is the id prefix (lug/lin/sna)."""
    buckets: dict[str, tuple[list, list]] = {}
    for i, r, p in zip(ids, refs, preds):
        lang = i.split("_", 1)[0]
        buckets.setdefault(lang, ([], []))
        buckets[lang][0].append(r)
        buckets[lang][1].append(p)
    out = {lang: score(rs, ps) for lang, (rs, ps) in buckets.items()}
    out["OVERALL"] = score(refs, preds)
    return out


if __name__ == "__main__":
    refs = ["mu makkati g'ennyanja", "iyo ine chikwangwani"]
    preds = ["mu makati g'ennyanja", "iyo ine chikwangwani chakanyorwa"]
    print("jiwer available:", _HAVE_JIWER)
    print(score(refs, preds))
