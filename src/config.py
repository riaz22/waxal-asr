"""Shared configuration + path resolution for the WAXAL ASR pipeline.

Works on the Mac (local paths) and on Kaggle (/kaggle/... paths) by detecting
the environment. Import this everywhere so paths are defined in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

DATASET_ID = "google/WaxalNLP"
LANGS = ("lug", "lin", "sna")
SAMPLE_RATE = 16_000

# HF model backbones.
MMS_MODEL = "facebook/mms-1b-all"
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
LID_MODEL = "facebook/mms-lid-256"  # covers lug/lin/sna; fallback to trained LID

# Whisper language tokens (Whisper's own codes). lug/lin/sna ARE in Whisper's
# tokenizer as "luganda"/"lingala"/"shona".
WHISPER_LANG = {"lug": "luganda", "lin": "lingala", "sna": "shona"}

ON_KAGGLE = Path("/kaggle").exists()

if ON_KAGGLE:
    # Raw challenge CSVs: attach the Zindi challenge files as a Kaggle Dataset.
    CHALLENGE_DIR = Path(
        os.environ.get("WAXAL_CHALLENGE_DIR", "/kaggle/input/waxal-challenge"))
    # Cleaned data produced by prepare_data (a Kaggle Dataset you create once).
    DATA_DIR = Path(os.environ.get("WAXAL_DATA_DIR", "/kaggle/input/waxal-clean"))
    WORK_DIR = Path("/kaggle/working")
else:
    _ROOT = Path(__file__).resolve().parents[1]
    CHALLENGE_DIR = Path(
        "/Users/razihamdi/Downloads/"
        "google-waxal-asr-challenge20260630-10570-elxebu")
    DATA_DIR = _ROOT / "data"
    WORK_DIR = _ROOT / "artifacts"

ART_DIR = WORK_DIR                      # trained models / LMs / preds land here
TRAIN_CSV = CHALLENGE_DIR / "Train.csv"
TEST_CSV = CHALLENGE_DIR / "Test.csv"
SAMPLE_SUB = CHALLENGE_DIR / "SampleSubmission.csv"


def clean_ds_path(lang: str, split: str) -> Path:
    """save_to_disk location for a cleaned per-language split."""
    return DATA_DIR / f"{lang}_{split}"


def lm_path(lang: str) -> Path:
    return ART_DIR / f"lm_{lang}.binary"


def lm_corpus_path(lang: str) -> Path:
    return ART_DIR / f"lm_corpus_{lang}.txt"


def vocab_path(lang: str) -> Path:
    return ART_DIR / f"vocab_{lang}.json"


def mms_dir(lang: str) -> Path:
    return ART_DIR / f"mms_{lang}"


def whisper_dir() -> Path:
    return ART_DIR / "whisper_turbo"
