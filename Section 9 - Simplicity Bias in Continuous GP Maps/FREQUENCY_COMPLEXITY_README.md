# Frequency--complexity figure reproduction

This directory contains the code used for the continuous ODE
genotype--phenotype-map sampling and the thesis frequency--complexity figures.

## Sampling

Each varied parameter is sampled independently and uniformly from
`[0, 2 * theta_WT]`. Simulated output trajectories are converted to binary
up--down strings and grouped by Lempel--Ziv complexity. The main computational
entry points are:

- `wsbw_pipeline.py`: model definitions, phenotype encoding, complexity and plotting.
- `wsbw_pipeline_parallel.py`: local parallel sampling.
- `wsbw_freqcomp_chunk.py`: compact cluster chunk generation.
- `wsbw_merge_hydra_chunks_streaming.py`: streaming merge of compact chunks.
- `run_pipeline_freqcomp_chen1e9_tyson1e8_other1e8.sh`: original-model pipeline.
- `render_modern_fixed_axes_grid.py`: nine-model fixed-axis grid renderer.

The raw sample/chunk outputs are intentionally not stored in this thesis-code
repository because they are many gigabytes. The scripts expect those results in
the `WhySystemsBiologyWorks/results` tree.

## Thesis figures

Run:

```bash
python split_frequency_complexity_figures.py --outdir /path/to/thesis/Figures
```

This creates:

- `FreqCompChen1e10.png`: Chen 2004, estimated from `N = 10^10` samples.
- `FreqCompOther9_1e8_grid3x3.png`: nine models, each estimated from `N = 10^8` samples.

The 3x3 panel order is:

1. ii: Rodenfels 2019
2. iii: Almeida 2020
3. iv: Novak 2022
4. v: Kholodenko 2000
5. vi: Vilar 2002
6. vii: Tyson 1991
7. viii: Leloup 1999
8. ix: Locke 2005
9. x: Ueda 2001

This order is deliberate and is not the same as the model-table order in the
thesis. The figure caption therefore lists it explicitly.
