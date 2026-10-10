"""Descriptive tables (Section 3; Appendix A). No model is fitted.

Run from the project root:
    analysis/.venv/bin/python analysis/11_descriptives.py > analysis/11_descriptives_output.txt
Output: analysis/results/D1 ... D7 (CSV).
"""
import numpy as np, pandas as pd
from lib import *

N = pd.read_pickle(CACHE + "notices.pkl")
A = pd.read_pickle(CACHE + "contracts_all_values.pkl")
C = load_contracts()

# ------------------------------------------------------------------ D1 from notices to contracts
par = set(N[N.k == 0].recid)
orphan = int(((N.k > 0) & ~N.parent.isin(par)).sum())
S = pd.DataFrame([("Contract notices published 2010-2025 with a recorded value", len(N)),
                  ("of which Commonwealth amendment notices", int((N.k > 0).sum())),
                  ("amendment notices whose parent is outside the study window (dropped)", orphan),
                  ("Contracts (one row per parent notice)", len(A)),
                  ("recorded value of zero or less (no logarithm)", int((A.value <= 0).sum())),
                  ("recorded value above zero and at most A$1 (placeholders, excluded)", int(((A.value > 0) & (A.value <= 1)).sum())),
                  ("Contracts analysed", len(C)),
                  ("of which Commonwealth", int(C.fed.sum())),
                  ("Agencies", C.publisher_name.nunique())], columns=["step", "records"])
S.to_csv(RES + "D1_sample.csv", index=False); print("D1. From notices to contracts"); print(S.to_string(index=False))
print("   Commonwealth share of contracts: %.1f%% | Commonwealth contracts never amended: %.1f%%" % (100 * C.fed.mean(), 100 * (1 - C[C.fed].ever_amended.mean())))

# ------------------------------------------------------------------ D2 Commonwealth notices and contracts by year
F = N[N.fed]; W = C[C.fed]
T = pd.DataFrame({"notices": F.groupby("yr").size(), "amendment_notices": F[F.k > 0].groupby("yr").size(),
                  "contracts": W.groupby("yr").size(), "later_amended": W.groupby("yr").ever_amended.mean(),
                  "amended_within_24m": W.groupby("yr").amended24.mean()})
T["share_amendment_notices"] = T.amendment_notices / T.notices
T.to_csv(RES + "D2_notices_by_year.csv"); print("\nD2. Commonwealth notices and contracts by publication year"); print(T.round(3).to_string())
lag = W.first_lag.dropna()
print("   first amendment: median %d days after the parent notice, 95th percentile %d days" % (lag.median(), lag.quantile(0.95)))
am = F[F.k > 0].merge(F[F.k == 0][["recid", "publisher_name", "supplier_name"]], left_on="parent", right_on="recid", suffixes=("", "_p"))
print("   amendment notices repeating the parent's agency: %.1f%%; supplier: %.1f%%"
      % (100 * (am.publisher_name == am.publisher_name_p).mean(), 100 * (am.supplier_name == am.supplier_name_p).mean()))
ow = W[W.ever_amended == 1].overwritten
print("   amended contracts whose parent displays the value on the latest amendment notice: %.1f%%" % (100 * ow.mean()))

# ------------------------------------------------------------------ D3 amendment recording by jurisdiction
C["recorded_amended"] = np.where(C.fed, C.ever_amended, C.am_flag)
J = C.groupby("publisher_gov_type").agg(contracts=("value", "size"), recorded_as_amended=("recorded_amended", "sum"))
J["rate"] = J.recorded_as_amended / J.contracts; J = J.sort_values("contracts", ascending=False)
J.to_csv(RES + "D3_amendment_recording.csv"); print("\nD3. Amendment recording by jurisdiction"); print(J.round(4).to_string())
ag = C.groupby("publisher_name").agg(n=("value", "size"), amended=("recorded_amended", "sum")); big = ag[ag.n >= 200]
print("   jurisdictions recording no amendment: %d of %d | agencies with 200+ contracts: %d, never recording one: %d"
      % ((J.recorded_as_amended == 0).sum(), len(J), len(big), (big.amended == 0).sum()))

# ------------------------------------------------------------------ D4 volume, D5 value distribution and dispersion
V = C.assign(j=np.where(C.fed, "Commonwealth", "State and territory")).groupby(["yr", "j"]).size().unstack(fill_value=0)
V["total"] = V.sum(axis=1); V.to_csv(RES + "D4_volume_by_year.csv")
g = np.polyfit(V.index, np.log(V.total), 1)[0]; tau, p, s = mann_kendall(V.index, V.total)
print("\nD4. Contracts by year of first publication"); print(V.to_string())
print("   growth %.1f%% a year (log-linear fit); Mann-Kendall p %.4f" % (100 * (np.exp(g) - 1), p))
q = C.ly.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
D = C.groupby("yr").ly.agg(["mean", "std", "count"]); D.to_csv(RES + "D5_target_dispersion_by_year.csv")
print("\nD5. Log contract value: 10th to 90th percentile spread is %.0f-fold; median A$%s" % (np.exp(q[0.9] - q[0.1]), format(int(np.exp(q[0.5])), ",")))
print(D.round(3).to_string())

# ------------------------------------------------------------------ D6 changes between successive notices, by amendment step
F2 = F.sort_values(["parent", "k"]).copy()
F2["prev_end"] = F2.groupby("parent").contract_end.shift(); F2["prev_val"] = F2.groupby("parent").value.shift(); F2["prev_k"] = F2.groupby("parent").k.shift()
st = F2[(F2.k > 0) & (F2.prev_k == F2.k - 1)].copy()
st["step"] = np.where(st.k == 1, "Parent to first amendment", np.where(st.k == 2, "First to second", np.where(st.k == 3, "Second to third", "Later steps")))
both = st.contract_end.notna() & st.prev_end.notna()
st["end_changed"] = np.where(both, st.contract_end != st.prev_end, np.nan); st["chg"] = (st.contract_end - st.prev_end).dt.days.abs()
Rv = st.groupby("step").agg(steps=("k", "size"), end_date_changed=("end_changed", "mean"), value_changed=("value", lambda v: (v != st.loc[v.index, "prev_val"]).mean()))
Rv["median_change_days"] = st[st.end_changed == 1].groupby("step").chg.median()
Rv = Rv.reindex(["Parent to first amendment", "First to second", "Second to third", "Later steps"])
Rv.to_csv(RES + "D6_revision_by_step.csv"); print("\nD6. Differences between successive notices of a contract"); print(Rv.round(4).to_string())

# ------------------------------------------------------------------ D7 recorded duration by notice type
F3 = F.copy(); amended = set(W.index[W.ever_amended == 1])
F3["type"] = np.where(F3.k > 0, "Amendment notice", np.where(F3.recid.isin(amended), "Parent notice, contract later amended", "Parent notice, contract never amended"))
Dt = F3[F3.duration.notna()].groupby("type").duration.agg(n="size", median_days="median")
Dt.loc["All notices with the amendment indicator set"] = [int((F3.am_flag == 1).sum()), F3[F3.am_flag == 1].duration.median()]
Dt.loc["All notices with the amendment indicator not set"] = [int((F3.am_flag == 0).sum()), F3[F3.am_flag == 0].duration.median()]
Dt.to_csv(RES + "D7_duration_by_notice_type.csv"); print("\nD7. Recorded duration by notice type, Commonwealth"); print(Dt.to_string())
na = W[W.ever_amended == 0]
print("   correlation of duration with log value: never amended %.2f, all Commonwealth %.2f"
      % (na[["duration", "ly"]].corr(method="spearman").iloc[0, 1], W[["duration", "ly"]].corr(method="spearman").iloc[0, 1]))
