# WAXAL ASR — Strategy to beat the leaderboard (and win Phase 2)

## The target

Leaderboard "score" = `1 - (0.5*WER + 0.5*CER)`, higher is better. Current top
is **0.830**; the pack sits **0.77–0.83**. From the [WAXAL-NET paper](https://arxiv.org/html/2606.02375v1),
fine-tuned **MMS-300M** gets:

| Lang | test rows | share | WER | CER |
|------|-----------|-------|-----|-----|
| Luganda (lug) | 638  | 15% | 16.9% | 3.4% |
| Lingala (lin) | 1866 | 44% | **42.6%** | **18.9%** |
| Shona  (sna)  | 1749 | 41% | 25.0% | 4.3% |

Blend those by test share → error ≈ 0.21 → board ≈ **0.79**. That is exactly
where the leaderboard is. **Conclusion: the leaders are at "fine-tuned compact
MMS/Whisper-small" level. To pass 0.83 we must beat that recipe, not match it.**

Where the points live: **Lingala (44%) + Shona (41%) = 85% of the test set.**
Luganda is already good and only 15% — deprioritize it. **Lingala is the
bottleneck** (42% WER) and the biggest slice → it is the #1 lever.

## The Phase-2 trap (this decides the prize)

Phase 2 releases **audio only — no language, speaker, or gender metadata.**
Phase 1 test IDs start with `lug_/lin_/sna_`; it is tempting to route per-language
models by that prefix. **That code path does not exist in Phase 2.** So:

1. **Language ID (LID) from audio is mandatory**, not optional. With only 3
   phonetically distinct Bantu languages it is near-perfect (~99%+). We validate
   LID for free on Phase 1 (the prefix is ground truth).
2. **Generalization > leaderboard overfitting.** Augmentation, weight decay,
   early stopping on val, and checkpoint averaging matter more than squeezing
   the public 30%. We never train on the Phase-1 test labels (a rules breach
   anyway).

## Architecture (full ensemble)

```
             audio (16kHz)
                  │
        ┌─────────▼──────────┐
        │  LID (MMS-LID-256  │  → lug / lin / sna   (Phase 1: sanity-check vs prefix)
        │  or small trained) │
        └─────────┬──────────┘
      ┌───────────┼────────────┐
      ▼           ▼            ▼
 MMS-1b-all   Whisper-lv3-   (per-lang routing)
  + KenLM      turbo (FT)
 (CTC, FT)     seq2seq
      │           │
      └─────►  ENSEMBLE  ◄─────┘
        per-language best model, then
        utterance-level agreement / ROVER
                  │
                  ▼
            submission.csv
```

**Model A — `facebook/mms-1b-all`, per-language CTC fine-tune + KenLM.**
MMS-1b (not the paper's 300M) is best-in-class for Bantu languages and beat
Whisper on CER in 17/19 languages. CTC + a 5-gram KenLM (via `pyctcdecode`)
is a cheap, large WER win in low-resource settings. This is our accuracy anchor,
especially for Luganda and Shona (clean CER).

**Model B — `openai/whisper-large-v3-turbo`, fine-tune (all 3 langs).**
Whisper's subword decoder + implicit LM handles **code-switching** (French/English
words appear in Lingala/Shona) and punctuation far better than char-CTC. This is
our lever on **Lingala** — the exact place MMS struggles. Whisper turbo fits a
T4/P100 with fp16 + gradient checkpointing + LoRA.

**Ensemble.** Start simple and robust: pick the best model *per language* on the
validation set (likely MMS for lug/sna, Whisper for lin). Then optionally
utterance-level ROVER / confidence selection for the last fraction of a point.

## Why this beats 0.83

| Lever | Expected effect |
|-------|-----------------|
| MMS-1b vs MMS-300M | −3 to −6 WER pts across langs |
| KenLM shallow fusion (CTC) | −2 to −5 WER pts, low-resource |
| Whisper on Lingala (code-switch) | −5 to −10 WER pts on the 44% slice |
| Data cleaning (drop <1.5s, >4 wps, misaligned) | −1 to −3 WER pts |
| Augmentation (speed perturb, SpecAugment, noise) | robustness + Phase-2 |
| Checkpoint averaging / seed ensemble | −0.5 to −1 WER pt, stability |

Even conservatively, Lingala 42→32 and Shona 25→20 moves the blended board from
~0.79 to **~0.84–0.86**, past the current top — with Phase-2-robust design.

## Execution order (Kaggle, free P100/T4x2, 30 h/week)

0. **Local (done on Mac):** text clean, CTC vocabs, KenLM corpora, local scorer.
   Artifacts in `artifacts/`.
1. **`prepare_data`** — download `lug_asr`/`lin_asr`/`sna_asr` (train+val+test),
   apply audio+text cleaning, save cleaned parquet as a **Kaggle Dataset** (so we
   never re-download). Extract the 4,253 Zindi test clips by id.
2. **`build_lm`** — KenLM 5-gram per language from the cleaned corpus.
3. **`train_mms`** — per-language MMS-1b CTC fine-tune. 3 runs.
4. **`train_whisper`** — Whisper-turbo fine-tune (joint, language-tokened).
5. **`train_lid`** (small) or wire MMS-LID — validate on Phase-1 prefixes.
6. **`infer` + `ensemble`** — decode test with each model, LM-rescore MMS,
   route by LID, ensemble → `submission.csv`. Validate the whole chain on the
   held-out validation split with the local scorer **before** spending a Zindi
   submission (5/day, 200 total).

## Submission discipline

- Never submit blind. Estimate the board score locally on `validation` first.
- Change one thing per submission; log it (`experiments.md`).
- Keep the richer orthography (case+punct) unless a controlled A/B shows Zindi
  strips it (`SUBMIT_LOWER_NOPUNCT` flag in `src/text_norm.py`).
- Phase 1 is a dev sandbox. Optimize for a model that *generalizes*, because
  Phase 2 (audio-only, unseen speakers) decides the prizes.
