"""Step 6b: combine model predictions, route per language, build submission.

On the VALIDATION split (which has targets) it measures each model's board
score per language and picks the winner per language automatically, then prints
the expected blended board score. On TEST it applies that routing and writes
submission.csv.

    # 1) decide routing on validation (needs preds_*_validation.parquet)
    !python scripts/ensemble.py --split validation
    # 2) apply the learned routing to test and write the submission
    !python scripts/ensemble.py --split test --route artifacts/route.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import datasets
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src.text_norm import score_normalize  # noqa: E402
from src.metrics import score, score_by_language  # noqa: E402
from src.submission import build_submission, validate_submission  # noqa: E402

MODELS = ["mms", "whisper"]


def load_preds(split: str) -> dict[str, pd.DataFrame]:
    out = {}
    for m in MODELS:
        p = C.ART_DIR / f"preds_{m}_{split}.parquet"
        if p.exists():
            out[m] = pd.read_parquet(p).set_index("id")
    if not out:
        raise SystemExit(f"no preds_*_{split}.parquet found in {C.ART_DIR}")
    return out


def val_targets() -> dict[str, tuple[str, str]]:
    """id -> (language, target) for the validation split."""
    m = {}
    for lang in C.LANGS:
        ds = datasets.load_from_disk(str(C.clean_ds_path(lang, "validation")))
        ids = ds["id"] if "id" in ds.column_names else [
            f"{lang}_{i}" for i in range(len(ds))]
        for i, t in zip(ids, ds["target"]):
            m[str(i)] = (lang, t)
    return m


def choose_route(preds: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Per language, pick the model with the lower blended error on validation."""
    truth = val_targets()
    route = {}
    print("\nPer-language model comparison (board = higher better):")
    for lang in C.LANGS:
        ids = [i for i, (l, _) in truth.items() if l == lang]
        best_m, best_board = None, -1.0
        for m, df in preds.items():
            common = [i for i in ids if i in df.index]
            if not common:
                continue
            refs = [score_normalize(truth[i][1]) for i in common]
            hyp = [score_normalize(str(df.loc[i, "pred"])) for i in common]
            s = score(refs, hyp)
            print(f"  {lang:4s} {m:8s} {s}  (n={len(common)})")
            if s.board > best_board:
                best_board, best_m = s.board, m
        route[lang] = best_m or MODELS[0]
    print(f"\nchosen route: {route}")
    return route


def apply_route(preds: dict[str, pd.DataFrame], route: dict[str, str],
                all_ids: list[str]) -> dict[str, str]:
    """For each id, take the prediction from its language's chosen model,
    falling back to any available model if that one lacks the id."""
    final = {}
    for i in all_ids:
        lang = None
        for m, df in preds.items():
            if i in df.index:
                lang = df.loc[i, "lang"] if "lang" in df.columns else \
                    str(i).split("_")[0]
                break
        chosen = route.get(lang, MODELS[0])
        if chosen in preds and i in preds[chosen].index:
            final[i] = str(preds[chosen].loc[i, "pred"])
        else:
            for m, df in preds.items():
                if i in df.index:
                    final[i] = str(df.loc[i, "pred"]); break
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--route", type=Path, default=None,
                    help="route.json to apply; if omitted on validation it is "
                         "learned and saved")
    args = ap.parse_args()
    preds = load_preds(args.split)

    if args.route and args.route.exists():
        route = json.loads(args.route.read_text())
    elif args.split == "validation":
        route = choose_route(preds)
        (C.ART_DIR / "route.json").write_text(json.dumps(route, indent=2))
    else:
        route = {lang: MODELS[0] for lang in C.LANGS}

    all_ids = sorted(set().union(*[set(df.index) for df in preds.values()]))
    final = apply_route(preds, route, all_ids)

    if args.split == "validation":
        truth = val_targets()
        ids = [i for i in all_ids if i in truth]
        refs = [score_normalize(truth[i][1]) for i in ids]
        hyp = [score_normalize(final[i]) for i in ids]
        print("\n=== ensemble on validation ===")
        for lang, s in score_by_language(ids, refs, hyp).items():
            print(f"  {lang:8s} {s}")
    else:
        out = C.ART_DIR / "submission.csv"
        build_submission(final, C.SAMPLE_SUB, out)
        validate_submission(out, C.SAMPLE_SUB)


if __name__ == "__main__":
    main()
