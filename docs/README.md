# Documentation

Start here:

- `PROJECT_STATUS.md` — current milestone and validated conclusions.
- `NAV2_INTEGRATION_NOTES.md` — C++/Nav2 architecture and porting seam.
- `REPOSITORY_LAYOUT.md` — source, experiment, result, and Git boundaries.
- `amoeba_mathematics_guide.tex` — focused geodesic-body mathematics.
- `amoeba_mppi_progress.tex` — complete thesis progress document.
- `experiments/` — versioned benchmark and ablation contracts.
- `experiments/FINAL_BENCHMARK_CONTRACT.md` — frozen narrow-navigation thesis comparison.
- `experiments/INTERACTION_RISK_ABLATION.md` — reciprocal peer-prediction
  worst/mean/CVaR results and the probability-weighted next step.

## LaTeX progress notes

The main document is `amoeba_mppi_progress.tex`.

Compile it from the repository root with:

```bash
pdflatex -output-directory=docs docs/amoeba_mppi_progress.tex
pdflatex -output-directory=docs docs/amoeba_mppi_progress.tex
```

The second pass fills the table of contents and references. A TeX distribution
such as TeX Live is required.
