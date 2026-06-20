#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source bioevo/bin/activate

OTHER_TAG="hydra_1e8_no_chen_tyson_v3"

python hydra/wsbw_merge_hydra_chunks_streaming.py \
  --tag "$OTHER_TAG" \
  --models kholodenko2000 leloup1999 locke2005 ueda2001 vilar2002

cp \
  results/bruteforce_cloud_stats/tyson_bfc_1e8/tyson1991_complexity_frequency_tyson_bfc_1e8.json \
  "results/tyson1991_complexity_frequency_${OTHER_TAG}_merged.json"

python hydra/wsbw_merge_bruteforce_cloud.py \
  --tag chen_bfc_1e9 \
  --model chen2004 \
  --neutral-cutoff 15 \
  --max-point-sample 100000 \
  --max-neutral-sample 100000 \
  --max-wt-phenotype-sample 100000 \
  --max-plot-points 50000 \
  --max-pairwise-points 3000 \
  --skip-sample-npz \
  --skip-frequency-json \
  --pipeline-other-tag "$OTHER_TAG" \
  --pipeline-freqcomp FreqCompChen1e9_Tyson1e8_Other5_1e8.png

cp plots/FreqCompChen1e9_Tyson1e8_Other5_1e8.png figures/pipeline/
cp plots/FreqCompChen1e9_Tyson1e8_Other5_1e8_grid.png figures/pipeline/ 2>/dev/null || true
