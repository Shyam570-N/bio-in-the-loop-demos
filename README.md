# Biologist-in-the-loop demos: Bayesian optimization and assay QC

Two small, self-contained Python demos that show how a biologist's judgment can be encoded as explicit rules around data-driven tools. Both use **simulated or example data only**. They contain no real experimental data, no proprietary information, and make no biological claims.

Authorship note: I designed the problems, the experimental variables, the objectives, and the biological rules. The code was written with Claude's help (AI-assisted coding), and I ran and checked it.

## 1. Bayesian optimization of a simulated barrier-protection screen

**Scenario (simulated).** An epithelial barrier is injured, and a protective compound is added at some dose and at some time relative to injury. The readout is barrier recovery (TEER, percent of uninjured baseline) at 72 h. The aim is to find good conditions with as few wells as possible.

**Session 1 (`bayesopt_session1.py`).**
- A hidden, noisy dose-by-timing landscape with a decoy local optimum and a toxicity cliff (`fig1_hidden_truth.png`).
- A Gaussian process surrogate (plain numpy) and expected improvement to choose each new plate (`fig2_model_over_rounds.png`).
- Comparison against random selection with the same budget, 30 repeats (`fig3_bo_vs_random.png`).

Result in this simulation (median true TEER of the chosen condition):

| Wells | Bayesian optimization | Random |
|---|---|---|
| 8 | 61.7 | 62.1 |
| 20 | 87.4 | 81.4 |
| 32 | 88.8 | 81.8 |
| 44 | 89.0 | 82.8 |

The simulated optimum is about 89.7. Early on (8 wells) Bayesian optimization is no better than random, and it pulls ahead as the model learns. This depends entirely on the simulated landscape and the noise level, so it should not be read as evidence about real assays.

**Session 2 (`bayesopt_session2.py`).** Three objectives at once: higher barrier recovery, lower LDH release (cytotoxicity), and lower dose. Selection uses ParEGO-style scalarization with a Gaussian process and expected improvement. A **biologist rules layer** then rejects conditions that are Pareto-optimal on paper but biologically unacceptable (for example, LDH release above 20 percent or barrier recovery below 60 percent). The rules are examples and the thresholds are illustrative (`fig4_pareto.png`).

Run:

```
python3 bayesopt_session1.py
python3 bayesopt_session2.py
```

Session 2 imports Session 1, so keep the two files together. Figures are regenerated next to the scripts. Runtime is about 30 seconds in total.

## 2. Two-tier assay QC

A small script that checks in vitro pharmacology results (`assay_results.csv`, invented example data) at two levels:

1. **Syntactic QC.** Data hygiene that needs no biology: non-numeric IC50, impossible Emax (outside 0 to 100 percent), missing control status.
2. **Semantic QC.** Rules that a pharmacologist has to define in advance, which the script can then execute but not invent:
   - Near-identical IC50 across cell types with very different Emax (same potency, different maximal effect).
   - A potent IC50 with no paired cytotoxicity readout (signal may be cell death, not target inhibition).
   - IC50 comparisons across different genetic backgrounds that are not directly comparable without normalization.

Run:

```
python3 assay_qc_example.py assay_results.csv
```

On the example data it flags 3 syntactic and 6 semantic issues.

## Limitations

- Everything here is simulated or invented. Results show the method working on a toy problem, not performance on a real screen.
- The Gaussian process is written in plain numpy for clarity, not speed or numerical robustness.
- The QC rules and thresholds are examples. Real thresholds depend on the assay.

## Requirements

Python 3.9 or later with numpy, pandas, and matplotlib (`pip install -r requirements.txt`).



