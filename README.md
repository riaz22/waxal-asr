# WAXAL ASR — Kaggle runbook

Beat the Zindi WAXAL leaderboard (Luganda / Lingala / Shona ASR) with an
MMS-1b + Whisper-turbo + KenLM + LID ensemble. Read **STRATEGY.md** first for
the *why*; this file is the *how*.

Your Mac can't train these models (no CUDA). Everything below runs on **Kaggle
free GPU** (Settings → Accelerator → GPU P100 or T4 x2; enable Internet). Budget
~30 GPU-h/week; the plan fits comfortably.

## What already exists (built + verified on the Mac)

```
src/            text_norm.py  metrics.py  data_clean.py  submission.py  config.py
scripts/        eda_and_prep.py  prepare_data.py  build_lm.py  train_mms.py
                train_whisper.py  lid.py  infer.py  ensemble.py
artifacts/      vocab_{lug,lin,sna}.json   lm_corpus_{lug,lin,sna}.txt
                train_clean.csv   eda_report.md
```

`src/` is pure-Python and unit-tested locally (normalization, WER/CER scorer,
submission format). The `scripts/` are the Kaggle GPU steps.

## One-time setup on Kaggle

1. Create a **Kaggle Dataset** from the Zindi challenge folder (the 3 CSVs +
   SampleSubmission). Attach it; it mounts at `/kaggle/input/waxal-challenge`.
   (Or set `WAXAL_CHALLENGE_DIR` to wherever you put them.)
2. New Notebook → attach that dataset, this repo (as a dataset or `git clone`),
   GPU on, Internet on.
3. Accept the model licenses on Hugging Face (Gemma not needed; MMS/Whisper are
   open) and add your HF token via `huggingface_hub.login()` if prompted.

## Pipeline (run in order)

```bash
# 0. Cache the cleaned data ONCE (internet on), then save the notebook output
#    as a Kaggle Dataset called `waxal-clean` and attach it to later notebooks.
python scripts/prepare_data.py

# 1. Language model for CTC shallow fusion (see build_lm.py header for the
#    kenlm build commands to run first).
python scripts/build_lm.py --kenlm_bin /kaggle/working/kenlm/build/bin

# 2. Train the three MMS-1b adapters (one per language).
python scripts/train_mms.py --lang lin       # biggest lever (44% of test)
python scripts/train_mms.py --lang sna
python scripts/train_mms.py --lang lug

# 3. Train the joint Whisper-turbo (Lingala / code-switch lever).
python scripts/train_whisper.py              # add --lora if you OOM

# 4. Language ID (needed for Phase 2; validate it on Phase 1 for free).
python scripts/lid.py --split validation     # prints LID accuracy vs prefix
python scripts/lid.py --split test

# 5. Predict with each model on validation, then choose per-language routing.
python scripts/infer.py --model mms     --split validation
python scripts/infer.py --model whisper --split validation
python scripts/ensemble.py --split validation      # prints local board score,
                                                   # writes artifacts/route.json

# 6. If the local board score looks good, predict on test and build submission.
python scripts/infer.py --model mms     --split test
python scripts/infer.py --model whisper --split test
python scripts/ensemble.py --split test --route artifacts/route.json
#   -> artifacts/submission.csv   (upload to Zindi)
```

## Golden rules

- **Never submit blind.** Step 5 estimates your board score on the held-out
  validation split with the exact metric. Only spend a Zindi submission
  (5/day, 200 total) when the local number improves.
- **Change one thing per submission** and log it in `experiments.md`.
- **Don't touch the Phase-1 test labels on HF** — using them is a rules breach
  and destroys Phase-2 generalization anyway.
- **Phase 2 = audio only.** The LID step makes the exact same pipeline work with
  no metadata. Keep it honest and it transfers.

## Tuning levers (in priority order)

1. Lingala: more Whisper epochs / beam size; try a Lingala-only Whisper.
2. KenLM `alpha`/`beta` in `infer._mms_decoder` (grid 0.3–0.7 / 1.0–2.0).
3. MMS: unfreeze top encoder layers (drop `freeze_base_model`) if GPU allows.
4. Augmentation: raise `mask_time_prob`; add speed perturbation to prepare_data.
5. Checkpoint averaging across seeds for stability (Phase-2 robustness).
```
