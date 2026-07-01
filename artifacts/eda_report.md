# WAXAL text EDA

Rows after text cleaning: **38198** (dropped 1 empty/punct-only)


## lug  (n=6119)
- words/utt: med=27 p95=52 max=90
- chars/utt: med=193 p95=371 max=650
- vocab size (CTC, ex-specials): 34
- alphabet: `'0146abcdefghijklmnopqrstuvwxyzàŋᵑ`
- rare chars (<=5 occ), review for noise: `0146qà`

## lin  (n=16243)
- words/utt: med=26 p95=48 max=102
- chars/utt: med=146 p95=257 max=512
- vocab size (CTC, ex-specials): 60
- alphabet: `'0123456789abcdefghijklmnopqrstuvwxyzàáâçèéêìíîïñòóôùûüþĝķĺœ`
- rare chars (<=5 occ), review for noise: `áíòóùûüþĝķĺœ`

## sna  (n=15836)
- words/utt: med=23 p95=38 max=59
- chars/utt: med=185 p95=306 max=488
- vocab size (CTC, ex-specials): 45
- alphabet: `'012345789abcdefghijklmnopqrstuvwxyzàáéíñòóúā`
- rare chars (<=5 occ), review for noise: `58áéòā`
