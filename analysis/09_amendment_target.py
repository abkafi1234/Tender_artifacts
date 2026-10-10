"""Main experiments on the amendment target (Section 4.6; Appendix C).

Run from the project root:
    analysis/.venv/bin/python analysis/09_amendment_target.py > analysis/09_amendment_target_output.txt
Target: an amendment notice published within 730 days of first publication. Commonwealth contracts.
Training window Y-7 to Y-3, so that every training label is observed when the origin year begins.
Output: analysis/results/T1_amendment_by_origin.csv, T2_duration_test.csv, T3_no_gap.csv,
        T4_strata.csv, T5_calibration.csv, T6_compliance_agencies.csv,
        analysis/cache/preds_amend.pkl
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from lib import *

C = load_contracts(); W = C[C.fed].copy()
N = pd.read_pickle(CACHE + "notices.pkl"); N = N[N.fed]
same = N.groupby("parent").contract_end.agg(lambda s: s.nunique(dropna=False) == 1)
W["same_end"] = same.reindex(W.index).fillna(True).values
W["proc_class"] = method_class(W.procurement_method)
W["compliance"] = compliance_fields(W).mean(axis=1)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

rows, dur, cal, preds = [], [], [], []
for Y in AM_ORIGINS:
    tr = sample(W[W.yr.between(Y - 7, Y - 3)], SEED + Y); ev = W[W.yr == Y]
    ytr, yev = tr.amended24.values.astype(float), ev.amended24.values.astype(float)
    Xtr, Xev, cols = encode(tr, ev, ytr, {})
    six, seven = use_cols(cols, SIX), use_cols(cols, SEVEN)
    p7 = fit_predict(Xtr, ytr, Xev, seven, True); p6 = fit_predict(Xtr, ytr, Xev, six, True)
    p6d = fit_predict(Xtr, ytr, Xev, six + ["duration"], True)
    rg = np.random.default_rng(SEED + Y)
    Xtr["perm"], Xev["perm"] = rg.permutation(Xtr.duration.values), rg.permutation(Xev.duration.values)
    p6p = fit_predict(Xtr, ytr, Xev, six + ["perm"], True)
    # permuted-target sentinel, whole pipeline rebuilt
    yp = rg.permutation(ytr); Xp, Xpe, cp = encode(tr, ev, yp, {}); ps = fit_predict(Xp, yp, Xpe, use_cols(cp, SEVEN), True)
    cs = ClusterStats(ev.agency_id.values, yev, {"p7": p7}); lo, hi = ci(cs.draws().p7)
    b = yev.mean()
    rows.append({"origin": Y, "training": "%d-%d" % (Y - 7, Y - 3), "n": len(ev), "base_rate": b, "p1p": b * (1 - b),
                 "auc": roc_auc_score(yev, p7), "DI": di(yev, p7), "di_lo": lo, "di_hi": hi, "cal_slope": cal_slope(yev, p7),
                 "sentinel_di": di(yev, ps), "sentinel_auc": roc_auc_score(yev, ps)})
    s = ev.same_end.values
    dur.append({"origin": Y, "base_rate": b, "auc_six": roc_auc_score(yev, p6), "auc_six_duration": roc_auc_score(yev, p6d),
                "gain": roc_auc_score(yev, p6d) - roc_auc_score(yev, p6), "gain_permuted": roc_auc_score(yev, p6p) - roc_auc_score(yev, p6),
                "gain_same_end": roc_auc_score(yev[s], p6d[s]) - roc_auc_score(yev[s], p6[s]), "share_same_end": s.mean()})
    # Platt scaling fitted on a held-out fifth of the training sample
    h = np.random.default_rng(SEED + Y).random(len(tr)) < 0.2
    ph = fit_predict(Xtr[~h], ytr[~h], Xtr[h], seven, True); pe = fit_predict(Xtr[~h], ytr[~h], Xev, seven, True)
    pl = LogisticRegression(C=1e6, max_iter=1000).fit(logit(ph).reshape(-1, 1), ytr[h])
    pc = pl.predict_proba(logit(pe).reshape(-1, 1))[:, 1]
    cal.append({"origin": Y, "di_raw": di(yev, pe), "di_platt": di(yev, pc), "change": di(yev, pc) - di(yev, pe),
                "slope_raw": cal_slope(yev, pe), "slope_platt": cal_slope(yev, pc)})
    preds.append(pd.DataFrame({"origin": Y, "y": yev, "p": p7}, index=ev.index))
    print(Y, {k: round(float(v), 4) for k, v in rows[-1].items() if k not in ("origin", "training")}, flush=True)

R, Dn, K = pd.DataFrame(rows), pd.DataFrame(dur), pd.DataFrame(cal)
PA = pd.concat(preds); PA.to_pickle(CACHE + "preds_amend.pkl")
R.to_csv(RES + "T1_amendment_by_origin.csv", index=False); Dn.to_csv(RES + "T2_duration_test.csv", index=False); K.to_csv(RES + "T5_calibration.csv", index=False)
print("\nT1. Amendment forecast, seven at-publication blocks"); print(R.round(4).to_string(index=False))
for nm, col in [("AUC", "auc"), ("DI", "DI"), ("base rate", "base_rate")]:
    tau, p, s = mann_kendall(R.origin, R[col]); print("   trend in %s: Mann-Kendall p %.3f, Sen slope %+.4f" % (nm, p, s))
print("   AUC range %.3f to %.3f, mean %.3f | sentinel: max |DI| %.4f, max |AUC - 0.5| %.4f"
      % (R.auc.min(), R.auc.max(), R.auc.mean(), R.sentinel_di.abs().max(), (R.sentinel_auc - 0.5).abs().max()))
print("\nT2. What recorded duration adds to the six descriptive blocks (AUC)"); print(Dn.round(4).to_string(index=False))
print("   mean: six %.4f | with duration %.4f | gain %+.4f | permuted %+.4f | same end date %+.4f"
      % (Dn.auc_six.mean(), Dn.auc_six_duration.mean(), Dn.gain.mean(), Dn.gain_permuted.mean(), Dn.gain_same_end.mean()))
print("   paired t-test, gain from duration against gain from the permuted column: p = %.2g" % stats.ttest_rel(Dn.gain, Dn.gain_permuted).pvalue)
print("\nT5. Platt scaling"); print(K.round(4).to_string(index=False))

# ------------------------------------------------------------------ T3 sensitivity: no gap between training and evaluation
rows = []
for Y in AM_ORIGINS:
    tr = sample(W[W.yr.between(Y - 5, Y - 1)], SEED + Y); ev = W[W.yr == Y]
    ytr, yev = tr.amended24.values.astype(float), ev.amended24.values.astype(float)
    Xtr, Xev, cols = encode(tr, ev, ytr, {})
    a = roc_auc_score(yev, fit_predict(Xtr, ytr, Xev, use_cols(cols, SEVEN), True))
    rows.append({"origin": Y, "auc_gap": R.set_index("origin").auc[Y], "auc_no_gap": a, "difference": a - R.set_index("origin").auc[Y]})
G = pd.DataFrame(rows); G.to_csv(RES + "T3_no_gap.csv", index=False)
print("\nT3. Training up to the year before the origin (labels as recorded today)"); print(G.round(4).to_string(index=False))
print("   difference: %.3f to %.3f, mean %.3f" % (G.difference.min(), G.difference.max(), G.difference.mean()))

# ------------------------------------------------------------------ T4 strata, three most recent origins pooled
Z = W.loc[PA.index].copy(); Z["y"], Z["p"], Z["origin"] = PA.y.values, PA.p.values, PA.origin.values
Zr = Z[Z.origin >= AM_ORIGINS[-3]]
S = []
for grp, col in [("Award procedure", "proc_class"), ("Entity type", "publisher_entity_type")]:
    for g, sub in Zr.assign(**{col: Zr[col].fillna("Missing")}).groupby(col):
        b = sub.y.mean()
        if len(sub) < 1000 or b < 0.02 or g in ("other", "missing"): continue
        lo, hi = ci(ClusterStats(sub.agency_id.values, sub.y.values, {"p": sub.p.values}).draws().p)
        S.append({"grouping": grp, "stratum": g, "n": len(sub), "base_rate": b, "auc": roc_auc_score(sub.y, sub.p), "DI": di(sub.y, sub.p), "lo": lo, "hi": hi})
S = pd.DataFrame(S); S.to_csv(RES + "T4_strata.csv", index=False)
print("\nT4. Amendment forecast by stratum, origins %d to %d pooled" % (AM_ORIGINS[-3], AM_ORIGINS[-1])); print(S.round(4).to_string(index=False))

# ------------------------------------------------------------------ T6 compliance and amendment DI across agencies
ag = Z.groupby("agency_id").apply(lambda t: pd.Series({"n": len(t), "base": t.y.mean(), "DI": di(t.y, t.p) if 0 < t.y.mean() < 1 else np.nan,
                                                       "auc": roc_auc_score(t.y, t.p) if 0 < t.y.mean() < 1 else np.nan, "compliance": t.compliance.mean()}))
ag = ag[(ag.n >= 200) & ag.DI.notna()].copy(); ag["var"] = ag.base * (1 - ag.base)
ag["DI_adj"] = ag.DI - np.polyval(np.polyfit(ag["var"], ag.DI, 1), ag["var"])
ag.to_csv(RES + "T6_compliance_agencies.csv")
r1, p1 = stats.spearmanr(ag.compliance, ag.DI_adj); r2, p2 = stats.spearmanr(ag.compliance, ag.DI)
print("\nT6. Compliance and amendment DI across %d agencies: adjusted %+.3f (p %.3f); raw %+.3f (p %.3f)" % (len(ag), r1, p1, r2, p2))
