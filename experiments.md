# Experiment log

One row per Zindi submission. Change ONE thing at a time. Local board = the
score from `ensemble.py --split validation`; LB = the Zindi public score.

| # | date | change | local WER | local CER | local board | Zindi LB | notes |
|---|------|--------|-----------|-----------|-------------|----------|-------|
| 0 | — | baseline plan: MMS-1b/lang + Whisper-turbo + KenLM + LID | | | | | target: beat 0.830 |
| 1 | | first MMS-only submission (no LM) | | | | | sanity check the pipeline end-to-end |
| 2 | | + KenLM shallow fusion | | | | | |
| 3 | | + Whisper on Lingala (route) | | | | | |
| 4 | | tune alpha/beta | | | | | |

## Known baselines to beat (WAXAL-NET, MMS-300M)
- lug 16.9 WER / 3.4 CER · lin 42.6 / 18.9 · sna 25.0 / 4.3 → blended board ≈ 0.79
