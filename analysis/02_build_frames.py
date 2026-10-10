"""Build the notice table and the contract table used by the supplementary analyses.

Run from the project root:  analysis/.venv/bin/python analysis/02_build_frames.py
Input: tender_export.csv.  Output: analysis/cache/notices.pkl, analysis/cache/contracts.pkl,
        analysis/cache/contracts_all_values.pkl (before the exclusion of placeholder values)
"""
import pandas as pd, numpy as np

cols = ["recid", "title", "description", "published_date", "value", "is_consultancy_services",
        "contract_start", "contract_end", "disposition", "procurement_method", "amendment_value",
        "category_code", "parent_category_code", "supplier_name", "supplier_state", "supplier_country",
        "publisher_name", "publisher_portfolio", "publisher_entity_type", "publisher_employees",
        "publisher_gov_type", "publisher_industry", "publisher_cofog_code"]
d = pd.read_csv("tender_export.csv", usecols=cols, dtype=str, keep_default_na=False, na_values=["N/A", ""])
d = d[d.disposition == "Contract Notice"].drop(columns="disposition")
for c in ["published_date", "contract_start", "contract_end"]:
    d[c] = pd.to_datetime(d[c], format="%A, %d %B %Y", errors="coerce")
for c in ["value", "amendment_value"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["publisher_employees"] = pd.to_numeric(d.publisher_employees.str.replace(",", ""), errors="coerce")
d["yr"] = d.published_date.dt.year
d = d[d.yr.between(2010, 2025) & d.value.notna()].drop_duplicates("recid").copy()
d["yr"] = d.yr.astype(int)
d["fed"] = d.publisher_gov_type == "FED"
d["k"] = pd.to_numeric(d.recid.str.extract(r"-A(\d+)$")[0]).fillna(0).astype(int)
d.loc[~d.fed, "k"] = 0
d["parent"] = np.where(d.k > 0, d.recid.str.replace(r"-A\d+$", "", regex=True), d.recid)
d["text"] = (d.title.fillna("") + " " + d.description.fillna("")).str.lower()
d["desc_len"] = d.description.fillna("").str.len()
d["duration"] = (d.contract_end - d.contract_start).dt.days.astype(float)
d["has_end"] = d.contract_end.notna().astype(int)
d["pub_lag"] = (d.published_date - d.contract_start).dt.days.astype(float)
d["month"] = d.published_date.dt.month
d["quarter"] = d.published_date.dt.quarter
d["fy_pos"] = (d.month - 7) % 12
d["consult"] = (d.is_consultancy_services == "Yes").astype(int)
d["am_flag"] = d.amendment_value.notna().astype(int)          # the amendment indicator in the export
d = d.drop(columns=["title", "description", "is_consultancy_services"])
print("notices:", len(d), "| amendment notices:", (d.k > 0).sum())
d.to_pickle("analysis/cache/notices.pkl")

par = d[d.k == 0].set_index("recid")
am = d[(d.k > 0) & d.parent.isin(par.index)].sort_values(["parent", "k"])
am = am.join(par.published_date.rename("ppub"), on="parent")
am["lag"] = (am.published_date - am.ppub).dt.days
g = am.groupby("parent")
h = pd.DataFrame({"n_amend": g.size(), "n_amend24": g.apply(lambda t: int((t.lag <= 730).sum())),
                  "first_lag": g.lag.min(), "last_value": g.value.last(),
                  "a1_value": g.value.first(), "a1_av": g.amendment_value.first(), "a1_k": g.k.first(),
                  "a1_end": g.contract_end.first(), "a1_yr": g.yr.first()})
c = par.join(h)
c["n_amend"] = c.n_amend.fillna(0); c["n_amend24"] = c.n_amend24.fillna(0)
c["ever_amended"] = (c.n_amend > 0).astype(int)
c["amended24"] = (c.n_amend24 > 0).astype(int)
c["overwritten"] = np.where(c.ever_amended == 1, (c.value == c.last_value), np.nan)
# first-published value: as shown for never-amended; value on first amendment less its amendment value otherwise
fp = np.where(c.ever_amended == 0, c.value, np.nan)
ok = (c.ever_amended == 1) & (c.a1_k == 1) & c.a1_av.notna() & (c.a1_av != 0) & ((c.a1_value - c.a1_av) > 1)
fp = np.where(ok, c.a1_value - c.a1_av, fp)
c["fp_value"] = fp
c.to_pickle("analysis/cache/contracts_all_values.pkl")    # kept for the placeholder sensitivity (script 10)
c = c[c.value > 1].copy()                                   # placeholder values of A$1 or less excluded
print("contracts:", len(c), "| Commonwealth:", int(c.fed.sum()), "| later amended:", int(c.ever_amended.sum()),
      "| first-published value recoverable among amended: %.3f" % c[c.ever_amended == 1].fp_value.notna().mean())
print("Commonwealth parents with a duration:", int((c.fed & c.duration.notna()).sum()))
c.to_pickle("analysis/cache/contracts.pkl")
