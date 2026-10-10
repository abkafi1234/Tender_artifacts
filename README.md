# Hindsight in the Record

Code and result artefacts for the paper

> Md Abdullah Al Kafi and Walayat Hussain. *Hindsight in the Record: Bounding How Much of Public
> Procurement Disclosure Is Added After the Fact.*

The paper measures how well published procurement fields forecast contract value and later
amendment, and bounds the share of that forecastability which was added to the record after a
contract notice was first published. The data are 655,347 Australian contract notices published
from 2010 to 2025, describing 485,068 contracts.

## Contents

| Folder | Contents |
|---|---|
| `Manuscript_new/` | LaTeX sources (`src_revised/`), tables, figures, bibliography, build script and the built PDF (`main_revised.pdf`) |
| `analysis/` | Code for every experiment in the paper, with logs, result tables, figures and a table of reported and reproduced values |
| `requirements.txt` | Pinned Python package versions |

## Data

The dataset is not included because of its size (about 2 GB). The scripts expect a file named
`tender_export.csv` in the repository root: an export of Australian government contract notices
with 36 columns and 1,048,575 rows. The contract notice data are published by the Australian
Government.

## Reproducing the results

- All experiments: follow `analysis/README.md`.
- Reported and reproduced values: `analysis/replication_report.md`.
- Manuscript: from `Manuscript_new/`, run `python3 build_revised.py` and then
  `latexmk -pdf main_revised.tex`.
