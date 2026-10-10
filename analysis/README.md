# Analysis (Python)

Python code for every experiment in the paper: the descriptive tables,
the DI series and trend tests, the hindsight share, the Shapley partitions on both targets, the
comparisons across jurisdictions, agencies and completeness, the amendment forecast, the
robustness checks, and the supplementary analyses of Appendix D.

`replication_report.md` sets each value printed in the paper beside the value produced here.

## Run

Place `tender_export.csv` in the repository root, then from the root:

```
python3 -m venv analysis/.venv
analysis/.venv/bin/pip install -r requirements.txt
mkdir -p analysis/cache
PY=analysis/.venv/bin/python
$PY analysis/01_end_date_revision.py        > analysis/01_end_date_revision_output.txt
$PY analysis/02_build_frames.py
$PY analysis/03_supplementary_models.py     > analysis/03_supplementary_models_output.txt
$PY analysis/04_term_by_amendment_status.py > analysis/04_term_by_amendment_status_output.txt
$PY analysis/05_make_supp_tables.py
$PY analysis/06_value_target_main.py        > analysis/06_value_target_main_output.txt
$PY analysis/07_shapley.py                  > analysis/07_shapley_output.txt
$PY analysis/08_group_comparisons.py        > analysis/08_group_comparisons_output.txt
$PY analysis/09_amendment_target.py         > analysis/09_amendment_target_output.txt
$PY analysis/10_robustness.py               > analysis/10_robustness_output.txt
$PY analysis/11_descriptives.py             > analysis/11_descriptives_output.txt
$PY analysis/12_figures.py
$PY analysis/13_replication_report.py
```

Scripts 08, 10, 12 and 13 read the outputs of earlier scripts, so keep the order. On ten cores,
script 07 (about 900 model fits) takes two to three hours; the others take between one and
thirty minutes each. Random seeds are fixed.

## Scripts

| Script | Purpose | Paper |
|---|---|---|
| `lib.py` | Shared functions: encodings, learner, DI, cluster bootstrap, trend tests | |
| `01_end_date_revision.py` | Differences in end date and value between successive notices, by year; overwriting of parent notices; reporting threshold; completeness of the export | Section 4.2 |
| `02_build_frames.py` | Builds the notice table and the one-row-per-contract table in `cache/` | Section 3.1 |
| `03_supplementary_models.py` | Notice frame and unit-by-classification table; floor on the first-published value; suppliers absent from training; amendment forecast with first-published duration | Sections 4.2, 4.3, 4.6; Appendix D |
| `04_term_by_amendment_status.py` | Contribution of the contract-term block for contracts never amended and later amended | Section 4.2 |
| `05_make_supp_tables.py` | Writes six table fragments to `Manuscript_new/tables/` | Appendix D tables |
| `06_value_target_main.py` | DI at eleven origins for six field sets, confidence intervals, balanced panel, permuted-target check, hindsight share, trend tests | Sections 4.1, 4.2 |
| `07_shapley.py` | Exact Shapley partition over 256 coalitions at three capacities and over 128 coalitions on the amendment target; leave-one-out; interactions; cross-target agreement | Sections 4.4, 4.6 |
| `08_group_comparisons.py` | Jurisdictions, common core of fields, award procedure, completeness, agencies, specification curve, value target by amendment status | Section 4.5 |
| `09_amendment_target.py` | Amendment forecast by origin, duration test, training without the gap, strata, Platt scaling, compliance across agencies | Section 4.6 |
| `10_robustness.py` | Other learners; placeholder values retained | Section 4.7 |
| `11_descriptives.py` | Sample construction, notices by year, amendment recording, volume, dispersion, changes by amendment step, duration by notice type | Section 3; Appendix A |
| `12_figures.py` | Draws the 21 figures from the result tables into `figures/` | all figures |
| `13_replication_report.py` | Writes `replication_report.md` | |

## Outputs included

- `*_output.txt`: the printed log of each script.
- `results/*.csv`: result tables. Prefixes A to F (except D1 to D7): supplementary analyses;
  D1 to D7: descriptives; M: value target; S: Shapley; G: group comparisons; T: amendment target;
  R: robustness; Z: replication table.
- `figures/*.png`: figures drawn from those tables, with the file names used in the paper.

## Implementation details

- Categorical fields enter as smoothed target means and counts, and the text as the prediction of
  a ridge regression on term frequencies, both fitted out of fold on training rows only.
- Learner: 400 trees, learning rate 0.08, depth 6, row and column subsampling 0.8, minimum child
  weight 5. Lower capacities: depth 4 and depth 3. Shallow learner: depth 2.
- Compliance index: share of nine optional fields populated (description of at least 20
  characters, category code, parent category code, contract start, contract end, procurement
  method, supplier state, publisher portfolio, publisher industry).
- Group comparisons pool the evaluation sets of the three most recent origins (2023 to 2025).
- Adjusted DI across agencies: the residual of agency DI regressed on the dispersion of the
  agency's contract values.
