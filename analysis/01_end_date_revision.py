"""End-date revision across successive AusTender notices of one contract.

Run from the project root:  python3 analysis/01_end_date_revision.py > analysis/01_end_date_revision_output.txt
Input: tender_export.csv. Commonwealth contract notices first published 2010-2025.
"""
import pandas as pd, numpy as np

cols = ["recid", "published_date", "value", "contract_start", "contract_end", "disposition",
        "amendment_value", "publisher_gov_type"]
d = pd.read_csv("tender_export.csv", usecols=cols, dtype=str, keep_default_na=False, na_values=["N/A", ""])
print("rows in file:", len(d))
for c in ["published_date", "contract_start", "contract_end"]:
    d[c] = pd.to_datetime(d[c], format="%A, %d %B %Y", errors="coerce")
for c in ["value", "amendment_value"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["yr"] = d.published_date.dt.year
cn = d[(d.disposition == "Contract Notice") & d.yr.between(2010, 2025)]
print("contract notices 2010-2025:", len(cn), "| after dropping duplicate ids:", cn.recid.nunique())
cn = cn[cn.publisher_gov_type == "FED"].drop_duplicates("recid").copy()
cn["parent"] = cn.recid.str.replace(r"-A\d+$", "", regex=True)
cn["k"] = pd.to_numeric(cn.recid.str.extract(r"-A(\d+)$")[0]).fillna(0).astype(int)
par = cn[cn.k == 0].set_index("recid")
am = cn[(cn.k > 0) & cn.parent.isin(par.index)].sort_values(["parent", "k"]).copy()
g = am.groupby("parent")
print("Commonwealth parents:", len(par), "| amendment notices with parent:", len(am), "| amended contracts:", am.parent.nunique())

# 1. Steps between consecutive amendment notices
am["pk"] = g.k.shift(1); am["pe"] = g.contract_end.shift(1); am["pv"] = g.value.shift(1)
st = am[(am.k > 1) & (am.pk == am.k - 1) & am.contract_end.notna() & am.pe.notna()].copy()
st["end_changed"] = st.contract_end != st.pe
st["value_changed"] = st.value != st.pv
print("\n1. Consecutive amendment-to-amendment steps:", len(st),
      "| end date changed %.4f | value changed %.4f" % (st.end_changed.mean(), st.value_changed.mean()))
print("   by publication year of the later notice:")
print(st.groupby("yr").agg(steps=("end_changed", "size"), end_changed=("end_changed", "mean"),
                           value_changed=("value_changed", "mean")).round(4).to_string())
s25 = st[st.yr == 2025]
print("   2025 by quarter:")
print(s25.groupby(s25.published_date.dt.quarter).end_changed.agg(["size", "mean"]).round(3).to_string())

# 2. Parent to first amendment, split by whether the parent still shows a value other than the latest
a1 = am[am.k == 1].set_index("parent"); P = par.loc[a1.index]
ok = P.contract_end.notna() & a1.contract_end.notna() & P.value.notna() & a1.value.notna()
P, a1 = P[ok], a1[ok]
lastv = g.value.last().loc[P.index]
t = pd.DataFrame({"A1yr": a1.yr.values, "overwritten": (P.value.values == lastv.values),
                  "end_changed": (P.contract_end.values != a1.contract_end.values),
                  "extended": (P.contract_end.values < a1.contract_end.values),
                  "ext_days": (a1.contract_end.values - P.contract_end.values).astype("timedelta64[D]").astype(float),
                  "orig_dur": (P.contract_end - P.contract_start).dt.days.values,
                  "new_dur": (a1.contract_end.values - P.contract_start.values).astype("timedelta64[D]").astype(float)},
                 index=P.index)
print("\n2. Parent to first amendment (n=%d)" % len(t))
print("   parent value equals latest amendment value (parent overwritten): %.4f" % t.overwritten.mean())
print("   end date differs | parent overwritten:     %.4f (n=%d)" % (t[t.overwritten].end_changed.mean(), t.overwritten.sum()))
print("   end date differs | parent not overwritten: %.4f (n=%d)" % (t[~t.overwritten].end_changed.mean(), (~t.overwritten).sum()))
print("   not-overwritten parents, by year the first amendment was published:")
print(t[~t.overwritten].groupby("A1yr").agg(n=("end_changed", "size"), end_changed=("end_changed", "mean"),
                                             extended=("extended", "mean")).round(3).to_string())
print("   share of parents not overwritten, by year of first amendment:")
print((1 - t.groupby("A1yr").overwritten.mean()).round(3).to_string())

# 3. Planned duration where it is observable: 2025 first amendments with a parent not overwritten
u = t[(~t.overwritten) & (t.A1yr == 2025)]
par["dur"] = (par.contract_end - par.contract_start).dt.days
never = par[~par.index.isin(am.parent)]
print("\n3. First amendments published in 2025, parent not overwritten (n=%d)" % len(u))
print("   median duration on parent (as first published): %.0f days" % u.orig_dur.median())
print("   median duration after first amendment:          %.0f days" % u.new_dur.median())
print("   median extension where extended:                %.0f days" % u[u.extended].ext_days.median())
print("   for comparison, median duration of never-amended parents: all years %.0f, published 2024-2025 %.0f"
      % (never.dur.median(), never[never.yr >= 2024].dur.median()))
print("   median duration on parent, all later-amended contracts (mostly overwritten): %.0f"
      % par[par.index.isin(am.parent)].dur.median())

# 4. Reported amendment values
av = am.amendment_value.dropna()
print("\n4. amendment_value on amendment notices (n=%d): |v| >= 10,000 %.3f | zero %.3f | non-zero under 10,000 %.3f"
      % (len(av), (av.abs() >= 10000).mean(), (av == 0).mean(), ((av != 0) & (av.abs() < 10000)).mean()))
a1v = am[am.k == 1].dropna(subset=["value", "amendment_value"])
print("   first amendments with an amendment value: %d of %d | implied first-published value positive: %.3f"
      % (len(a1v), (am.k == 1).sum(), ((a1v.value - a1v.amendment_value) > 0).mean()))
k2 = st.dropna(subset=["value", "amendment_value", "pv"])
print("   later steps where value minus amendment_value equals the previous notice's value: %.3f"
      % ((k2.value - k2.amendment_value - k2.pv).abs() <= 1).mean())

# 5. Completeness of the export
mx, n = g.k.max(), g.size()
print("\n5. Amended contracts with a gap in amendment numbering: %.3f | notices implied missing: %d of %d"
      % ((mx > n).mean(), int((mx - n).sum()), int(mx.sum())))
