# Disclosure Informativeness of Public Procurement Records for Forecasting under Data Leakage

Code, logs and results for the article submitted to *Forecasting* (MDPI), revised version. This Python
implementation produces every table, figure and number in the article and its Supplementary Materials.

## Data

`tender_export.csv` is an export of Australian government contract notices (Commonwealth, state and
territory buyers, 2010 to 2025; 1,048,575 rows, about 2 GB). It is not included because of its size.
Place it in the repository root.

## Run

```
python3 -m venv analysis/.venv
analysis/.venv/bin/pip install -r requirements.txt
bash analysis/run_all.sh
```

On a 10-core machine the whole pipeline takes about 45 minutes; script 07 (about 900 model fits) is the
longest. Seeds are fixed (20261005), and a clean rerun reproduces every result (all files of scripts 01 to 14 byte for byte).
`analysis/run_log/status*.tsv` records the run times and exit codes of the run that produced the files here.

## Scripts

| Script | Purpose |
|---|---|
| `lib.py` | Shared settings and functions: encodings, learner, DI, cluster bootstrap, trend tests |
| `01_end_date_revision.py` | End dates and values across successive notices; overwriting of parent notices |
| `02_build_frames.py` | Notice table and one-row-per-contract table (`analysis/cache/`) |
| `03_supplementary_models.py` | Notice frame against contract frame; first-published value; unseen suppliers; duration test (Table 6, panel A) |
| `04_term_by_amendment_status.py` | Contribution of the contract term by amendment status |
| `05_make_supp_tables.py` | Nine supplement tables (with script 22) |
| `06_value_target_main.py` | DI at eleven origins, intervals, balanced panel, permuted-target check, hindsight share, trend tests |
| `07_shapley.py` | Exact Shapley partitions (value and amendment targets), capacities, leave-one-out, interactions, cross-target agreement |
| `08_group_comparisons.py` | Jurisdictions, common core, award procedure, completeness, agencies, specification curve |
| `09_amendment_target.py` | Amendment forecast, duration test, no-gap sensitivity, strata, calibration |
| `10_robustness.py` | Other learners; placeholder values retained |
| `11_descriptives.py` | Sample construction and descriptive tables |
| `12_figures.py` | Figures 1, 2 and 4 of the article and Figures S1 to S18 (PDF and PNG) |
| `14_robust_error.py` | Absolute-error measures with placeholder values excluded and retained |
| `15_training_reference.py` | DI with reference forecasts from training data only (revision) |
| `16_hindsight_tiers.py` | Identified and inferred hindsight; sensitivity to unreported extensions (revision) |
| `17_duration_equal_window.py` | Duration test with equal observation windows (revision) |
| `18_simple_baselines.py` | Group-mean lookup, ridge and logistic regression (revision) |
| `19_subsample_seeds.py` | Ten further subsample and learner seeds; no subsampling (revision) |
| `20_implementation_table.py` | Implementation table generated from the code (revision) |
| `21_revision_figures.py` | Figure 3 of the article and Figure S19 |
| `22_supplement_tables.py` | Every other numeric table of the supplement, as LaTeX, from `results/` |
| `23_check_manuscript_numbers.py` | Checks the numbers in the article text against `results/` |
| `24_graphical_abstract.py` | Graphical abstract (numbers read from `results/`) |

## Where each result comes from

Article

| Item | Script | Result file |
|---|---|---|
| Table 3 (trend tests) | 06 | `M2_trend_tests.csv` |
| Table 4 (hindsight share) | 06 | `M3_hindsight.csv` |
| Table 5 (jurisdictions) | 08 | `G1_contrasts.csv`, `G3_jurisdiction.csv` |
| Table 6, panel A | 03 | `E2_duration_first_published.csv` |
| Table 6, panel B | 17 | `N3_equal_window.csv` |
| Figure 1 | 06, 12 | `M1_di_by_origin.csv`; `figures/R2_composition.png` |
| Figure 2 | 06, 07, 12 | `M3_hindsight.csv`, `S4_leave_one_out.csv`; `figures/R3_hindsight.png` |
| Figure 3 | 16, 21 | `N2_sensitivity_q.csv`, `N2_q_anchor.csv`; `figures/R11_hindsight_sensitivity.png` |
| Figure 4 | 07, 12 | `S2_shapley_value.csv`; `figures/E1_shapley_partition.png` |
| Section 4.7, reference forecasts | 15 | `N1_*.csv` |
| Section 4.7, simpler models | 18 | `N4_*.csv` |
| Section 4.7, seeds and subsampling | 19 | `N5_*.csv` |

Supplementary Materials (LaTeX in `analysis/supplement_tables/`, written by script 22 unless stated)

| Table | Source | Table | Source |
|---|---|---|---|
| S2 sample | 11: `D1_sample.csv` | S24 capacity | 07: `S3_shapley_capacity.csv`, `S1b_*` |
| S3 end dates by year | 05 (notices) | S25 term by amendment status | 05: `F_term_by_amendment_status.csv` |
| S4 unit of analysis | 03: `A_*`, `B_notice_frame.csv` | S26 two by two | 05: `A_*`, `B_*` |
| S5 Shapley, value | 07: `S2_shapley_value.csv` | S27 unseen suppliers | 05: `D_unseen_supplier.csv` |
| S6 notices by year | 11: `D2_*` | S28 robust trend | 05: `R3_robust_trend.csv` |
| S7 recording | 11: `D3_*` | S29 robust jurisdiction | 05: `R3_robust_jurisdiction.csv` |
| S9 DI by origin | 06: `M1_di_by_origin.csv` | S30 size-matched Shapley | 05: `S9_*` |
| S10 successive notices | 11: `D6_*` | S31-S33 reference forecasts | 15: `N1_*` |
| S11 duration by notice type | 11: `D7_*` | S34 three tiers | 16: `N2_tiers_summary.csv` |
| S12 leave-one-out | 07: `S4_*` | S35 tier (ii), value | 16: `N2_tier2_value.csv` |
| S13 overwritten target | 08: `G11_*` | S36 sensitivity to q | 16: `N2_sensitivity_q.csv` |
| S14 availability | 08: `G2_*` | S37 anchor for q | 16: `N2_q_anchor.csv` |
| S15 matched completeness | 08: `G7_*` | S38 balance | 17: `N3_balance.csv` |
| S16 compliance | 08: `G9_compliance_agencies.csv` | S39 simpler models, value | 18: `N4_value_summary.csv` |
| S17 specification curve | 08: `G10_*` | S40 simpler models, amendment | 18: `N4_amend_by_origin.csv` |
| S18 completeness bins | 08: `G6_*` | S41 duration test, logistic | 18: `N4_duration_test.csv` |
| S19 amendment by origin | 09: `T1_*` | S42 seeds | 19: `N5_seeds_summary.csv` |
| S20 duration and amendment | 09: `T2_duration_test.csv` | S43 no subsampling | 19: `N5_no_subsample.csv` |
| S21 Shapley, amendment | 07: `S7_*` | S44 implementation | 20: `N6_implementation.*` |
| S22 cross-target | 07: `S8_*` | | |
| S23 amendment strata | 09: `T4_*` | | |

Tables S1 and S8 contain no results. Figures S1 to S18 are drawn by script 12 under the file names in
`analysis/figures/`, and Figure S19 by script 21; each figure is saved as a vector PDF (used in the article)
and as a PNG.

## Implementation

Gradient-boosted trees (xgboost; 400 trees, learning rate 0.08, depth 6, subsampling 0.8, minimum child
weight 5), fixed in advance and never tuned. Five-year training windows, subsampled to 100,000 contracts.
Categorical fields as smoothed out-of-fold target means and counts; text as an out-of-fold ridge
prediction on TF-IDF features. Full list: `analysis/results/N6_implementation.csv` (Table S44).
