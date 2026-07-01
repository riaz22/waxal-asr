"""Step 2: build a per-language KenLM 5-gram from the cleaned LM corpus.

The n-gram is used for CTC shallow fusion (pyctcdecode) when decoding MMS,
which is one of the cheapest low-resource WER wins available.

KenLM's `lmplz` binary is required. On Kaggle:

    !apt-get -qq install -y build-essential cmake libboost-all-dev \
        libeigen3-dev zlib1g-dev
    !git clone -q https://github.com/kpu/kenlm.git /kaggle/working/kenlm
    !cd /kaggle/working/kenlm && mkdir -p build && cd build && \
        cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j4 >/dev/null
    !pip -q install https://github.com/kpu/kenlm/archive/master.zip pyctcdecode

Then:
    !python scripts/build_lm.py --kenlm_bin /kaggle/working/kenlm/build/bin
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402


def build_one(lang: str, kenlm_bin: Path, order: int) -> None:
    corpus = C.lm_corpus_path(lang)
    if not corpus.exists():
        print(f"skip {lang}: no corpus at {corpus}")
        return
    arpa = C.ART_DIR / f"lm_{lang}.arpa"
    lmplz = kenlm_bin / "lmplz"
    build_binary = kenlm_bin / "build_binary"
    binary = C.lm_path(lang)

    # --discount_fallback: small low-resource corpora often lack high n-gram
    # counts for Kneser-Ney; this keeps lmplz from erroring out.
    with open(corpus) as fin, open(arpa, "w") as fout:
        subprocess.run([str(lmplz), "-o", str(order), "--discount_fallback"],
                       stdin=fin, stdout=fout, check=True)
    if build_binary.exists():
        subprocess.run([str(build_binary), str(arpa), str(binary)], check=True)
        print(f"{lang}: {binary}")
    else:
        # pyctcdecode also accepts the .arpa directly.
        shutil.copy(arpa, binary.with_suffix(".arpa"))
        print(f"{lang}: {arpa} (no build_binary; arpa is fine for pyctcdecode)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kenlm_bin", type=Path, required=True,
                    help="dir containing lmplz / build_binary")
    ap.add_argument("--order", type=int, default=5)
    args = ap.parse_args()
    for lang in C.LANGS:
        build_one(lang, args.kenlm_bin, args.order)


if __name__ == "__main__":
    main()
