"""Shared configuration + path resolution for the WAXAL ASR pipeline.

Works on the Mac and on Kaggle by detecting the environment and picking paths
that are actually writable where writes happen.

Key ideas:
  * CHALLENGE_DIR — the raw Zindi CSVs. Shipped inside the repo under
    `challenge/`, so nothing extra needs attaching. (An attached
    `/kaggle/input/waxal-challenge` dataset is also honored if present.)
  * DATA_DIR — the cleaned-data bundle that `prepare_data.py` produces (cleaned
    per-language splits + vocab + LM corpus). On Kaggle this is a WRITABLE
    working dir by default, or an attached `waxal-clean` dataset if you saved one
    from a previous session (so you don't re-download).
  * WORK_DIR — writable scratch for trained models / LMs / predictions.
"""

from __future__ import annotations

from pathlib import Path

DATASET_ID = "google/WaxalNLP"
LANGS = ("lug", "lin", "sna")
SAMPLE_RATE = 16_000

MMS_MODEL = "facebook/mms-1b-all"
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
LID_MODEL = "facebook/mms-lid-256"
WHISPER_LANG = {"lug": "luganda", "lin": "lingala", "sna": "shona"}

_ROOT = Path(__file__).resolve().parents[1]
ON_KAGGLE = Path("/kaggle").exists()


def _first_existing(cands: list, default: Path) -> Path:
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(default)


if ON_KAGGLE:
    WORK_DIR = Path("/kaggle/working")
    CHALLENGE_DIR = _first_existing(
        [_ROOT / "challenge", "/kaggle/input/waxal-challenge"],
        _ROOT / "challenge")
    # Prefer an attached cleaned dataset; else a writable working dir that
    # prepare_data.py fills this session.
    DATA_DIR = _first_existing(
        ["/kaggle/input/waxal-clean"], WORK_DIR / "waxal-clean")
else:
    WORK_DIR = _ROOT / "artifacts"
    CHALLENGE_DIR = _first_existing(
        [_ROOT / "challenge",
         "/Users/razihamdi/Downloads/"
         "google-waxal-asr-challenge20260630-10570-elxebu"],
        _ROOT / "challenge")
    DATA_DIR = _ROOT / "data"

ART_DIR = WORK_DIR
TRAIN_CSV = CHALLENGE_DIR / "Train.csv"
TEST_CSV = CHALLENGE_DIR / "Test.csv"
SAMPLE_SUB = CHALLENGE_DIR / "SampleSubmission.csv"


def clean_ds_path(lang: str, split: str) -> Path:
    return DATA_DIR / f"{lang}_{split}"


def vocab_path(lang: str) -> Path:
    return DATA_DIR / f"vocab_{lang}.json"


def lm_corpus_path(lang: str) -> Path:
    return DATA_DIR / f"lm_corpus_{lang}.txt"


def lm_path(lang: str) -> Path:
    return WORK_DIR / f"lm_{lang}.binary"


def mms_dir(lang: str) -> Path:
    return WORK_DIR / f"mms_{lang}"


def whisper_dir() -> Path:
    return WORK_DIR / "whisper_turbo"


def require_prepared() -> None:
    """Fail fast with a clear message if prepare_data.py hasn't run yet."""
    missing = [lang for lang in LANGS
               if not clean_ds_path(lang, "train").exists()]
    if missing:
        raise SystemExit(
            "Cleaned data not found for: " + ", ".join(missing) +
            f"\nLooked in DATA_DIR={DATA_DIR}"
            "\n-> Run STEP 1 first:  python scripts/prepare_data.py"
            "\n   (it downloads + cleans the audio; needs Internet ON).")
