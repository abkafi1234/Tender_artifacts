"""Comparisons across jurisdictions, award procedures, agencies and completeness (Section 4.5;
Appendix B), and the value target scored by amendment status.

Run from the project root, after script 06:
    analysis/.venv/bin/python analysis/08_group_comparisons.py > analysis/08_group_comparisons_output.txt
Evaluation-set predictions of the seven at-publication blocks are pooled over the three most recent
origins (2023 to 2025), the window in which state and territory records are numerous enough to
compare. G12 repeats the jurisdiction comparison over all eleven origins as a sensitivity.
A group's DI uses that group's own mean as the reference forecast.
Output: analysis/results/G1 ... G11 (CSV).
"""
import numpy as np, pandas as pd
from scipy import stats
from lib import *

C = load_contracts()
PV = pd.read_pickle(CACHE + "preds_value.pkl")
E = C.loc[PV.index].copy(); E["y"] = PV.y.values; E["p"] = PV.p_seven.values; E["p8"] = PV.p_all8.values; E["origin"] = PV.origin.values
E["juris"] = np.where(E.fed, "Commonwealth", "State and territory")
E["proc_class"] = method_class(E.procurement_method)
FLD = compliance_fields(E); E["n_fields"] = FLD.sum(axis=1); E["compliance"] = FLD.mean(axis=1)
rng = np.random.default_rng(SEED)
POOL = ORIGINS[-3:]
E_all = E; E = E[E.origin.isin(POOL)].copy()


def group_stats(df, label, pcol="p", B=NBOOT):
    """Point values and cluster-bootstrap draws (agencies resampled jointly) of DI and residual SD by group."""
    codes = pd.factorize(df.agency_id.values)[0]; K = codes.max() + 1
    W = np.stack([np.bincount(rng.integers(0, K, K), minlength=K) for _ in range(B)]).astype(float)
    out, draws = [], {}
    for g, m in df.groupby(label).indices.items():
        y, p, c = df.y.values[m], df[pcol].values[m], codes[m]
        n = np.bincount(c, minlength=K).astype(float); sy = np.bincount(c, weights=y, minlength=K)
        syy = np.bincount(c, weights=y * y, minlength=K); sse = np.bincount(c, weights=(y - p) ** 2, minlength=K)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = 1 - (W @ sse) / (W @ syy - (W @ sy) ** 2 / (W @ n)); r = np.sqrt((W @ sse) / (W @ n))
        draws[g] = (d, r)
        out.append({"group": g, "n": len(y), "DI": di(y, p), "lo": ci(d)[0], "hi": ci(d)[1], "total_sd": y.std(), "resid_sd": rsd(y, p)})
    return pd.DataFrame(out), draws


def contrast(draws, a, b, point):
    d = draws[a][0] - draws[b][0]
    return {"contrast": "%s minus %s" % (a, b), "diff": point, "lo": ci(d)[0], "hi": ci(d)[1], "p": boot_p(d)}


# ------------------------------------------------------------------ G1 jurisdiction, full field set
J, dj = group_stats(E, "juris")
cj = contrast(dj, "Commonwealth", "State and territory", J.DI[0] - J.DI[1])
print("G1. Jurisdiction comparison, seven at-publication blocks, origins %d to %d pooled" % (POOL[0], POOL[-1])); print(J.round(4).to_string(index=False))

# ------------------------------------------------------------------ G2 field availability
raw = [c for cs in CATS.values() for c in cs] + ["publisher_employees", "contract_start", "contract_end"]
A = pd.DataFrame({"field": raw + ["description (20+ characters)"],
                  "Commonwealth": [C[C.fed][c].notna().mean() for c in raw] + [(C[C.fed].desc_len >= 20).mean()],
                  "State and territory": [C[~C.fed][c].notna().mean() for c in raw] + [(C[~C.fed].desc_len >= 20).mean()]})
A["gap"] = A.Commonwealth - A["State and territory"]
A = A.reindex(A.gap.abs().sort_values(ascending=False).index); A.to_csv(RES + "G2_field_availability.csv", index=False)
print("\nG2. Field availability by jurisdiction"); print(A.round(3).to_string(index=False))

# ------------------------------------------------------------------ G3 common core (fields populated >= 90% in both groups)
keep = set(A[(A.Commonwealth >= 0.9) & (A["State and territory"] >= 0.9)].field)
cats_cc = {b: [c for c in cs if c in keep] for b, cs in CATS.items()}; cats_cc = {b: cs for b, cs in cats_cc.items() if cs}
nums_cc = {"cal": NUMS["cal"], "prod": ["consult"], "text": ["desc_len"], "term": NUMS["term"]}
print("\nG3. Common core keeps:", sorted(keep), "| categorical blocks kept:", list(cats_cc))
pcc = []
for Y in POOL:
    tr = sample(C[C.yr.between(Y - 5, Y - 1)], SEED + Y); ev = C[C.yr == Y]
    Xtr, Xev, cols = encode(tr, ev, tr.ly.values, {}, cats=cats_cc, nums=nums_cc)
    pcc.append(pd.Series(fit_predict(Xtr, tr.ly.values, Xev, [c for cs in cols.values() for c in cs]), index=ev.index))
E["p_cc"] = pd.concat(pcc).reindex(E.index).values
Jc, djc = group_stats(E, "juris", "p_cc")
ccj = contrast(djc, "Commonwealth", "State and territory", Jc.DI[0] - Jc.DI[1])
print(Jc.round(4).to_string(index=False))
print("   difference shrinks by %.1f%% on the common core" % (100 * (1 - ccj["diff"] / cj["diff"])))

# ------------------------------------------------------------------ G4 individual jurisdictions and award procedure
G, _ = group_stats(E[E.groupby("publisher_gov_type").y.transform("size") >= 2000], "publisher_gov_type")
G.to_csv(RES + "G4_by_jurisdiction.csv", index=False)
print("\nG4. DI by jurisdiction (at least 2,000 evaluation contracts)"); print(G.round(4).to_string(index=False))
Pm, dp = group_stats(E[E.proc_class.isin(["open", "limited"])], "proc_class")
cp = contrast(dp, "open", "limited", Pm.set_index("group").DI["open"] - Pm.set_index("group").DI["limited"])
Pm.to_csv(RES + "G5_by_award_procedure.csv", index=False)
print("\nG5. DI by award procedure"); print(Pm.round(4).to_string(index=False))

# ------------------------------------------------------------------ G6 completeness: dose-response and matched comparison
E["bin"] = pd.cut(E.n_fields, [-1, 5, 6, 7, 8, 9], labels=["5 or fewer", "6", "7", "8", "9"]).astype(str)
Dz, dd = group_stats(E, "bin"); Dz = Dz.set_index("group").loc[["5 or fewer", "6", "7", "8", "9"]].reset_index()
cd = contrast(dd, "9", "5 or fewer", Dz.DI.iloc[-1] - Dz.DI.iloc[0])
Dz.to_csv(RES + "G6_dose_response.csv", index=False)
print("\nG6. DI by number of optional fields populated"); print(Dz.round(4).to_string(index=False))
M = []
for k, sub in E.groupby("n_fields"):
    if (sub.fed.sum() >= 500) and ((~sub.fed).sum() >= 500):
        a, b = sub[sub.fed], sub[~sub.fed]
        M.append({"fields": k, "n_cw": len(a), "n_st": len(b), "di_cw": di(a.y, a.p), "di_st": di(b.y, b.p),
                  "diff": di(a.y, a.p) - di(b.y, b.p), "rsd_cw": rsd(a.y, a.p), "rsd_st": rsd(b.y, b.p)})
M = pd.DataFrame(M); M.to_csv(RES + "G7_matched_completeness.csv", index=False)
print("\nG7. Completeness-matched jurisdiction comparison"); print(M.round(4).to_string(index=False))

# ------------------------------------------------------------------ contrasts with Holm correction
K = pd.DataFrame([dict(cj, test="Jurisdiction, full field set"), dict(ccj, test="Jurisdiction, common core"),
                  dict(cp, test="Award procedure"), dict(cd, test="Completeness, best minus worst documented")])
K["holm_p"] = holm(K.p.values); K.to_csv(RES + "G1_contrasts.csv", index=False)
pd.concat([J.assign(spec="Full field set"), Jc.assign(spec="Common core")]).to_csv(RES + "G3_jurisdiction.csv", index=False)
print("\nG1/G3/G5/G6 contrasts"); print(K.round(4).to_string(index=False))

# ------------------------------------------------------------------ G8 compliance index over time
FY = compliance_fields(C).assign(yr=C.yr.values).groupby("yr").mean(); FY["index"] = FY.mean(axis=1)
FY.to_csv(RES + "G8_fill_rate_by_year.csv")
tau, p, s = mann_kendall(FY.index[FY.index >= 2015], FY["index"][FY.index >= 2015])
print("\nG8. Compliance index 2015-2025: Sen slope %+.4f per year, Mann-Kendall p %.3f" % (s, p))
print(FY.loc[[2015, 2020, 2025]].round(3).to_string())

# ------------------------------------------------------------------ G9 compliance across agencies
ag = E.groupby("agency_id").apply(lambda t: pd.Series({"n": len(t), "DI": di(t.y, t.p), "resid_sd": rsd(t.y, t.p), "total_sd": t.y.std(),
                                                        "compliance": t.compliance.mean(), "fed": float(t.fed.mean() > 0.5)}))
ag = ag[ag.n >= 200].copy()
b = np.polyfit(ag.total_sd, ag.DI, 1); ag["DI_adj"] = ag.DI - np.polyval(b, ag.total_sd)
ag.to_csv(RES + "G9_agencies.csv")
def partial(x, y, z):
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r = np.corrcoef(ex, ey)[0, 1]; n = len(x); t = r * np.sqrt((n - 3) / (1 - r * r))
    return r, 2 * stats.t.sf(abs(t), n - 3)
rows = []
def add(label, r, p): rows.append({"association": label, "estimate": r, "p": p, "shared_var": r * r})
add("Compliance and adjusted DI (primary)", *stats.spearmanr(ag.compliance, ag.DI_adj))
add("Compliance and DI (raw)", *stats.spearmanr(ag.compliance, ag.DI))
add("Compliance and DI (Pearson)", *stats.pearsonr(ag.compliance, ag.DI))
add("Compliance and residual dispersion (raw)", *stats.spearmanr(ag.compliance, ag.resid_sd))
add("Compliance and DI, given target dispersion", *partial(ag.compliance, ag.DI, ag.total_sd))
add("Compliance and residual dispersion, given target dispersion", *partial(ag.compliance, ag.resid_sd, ag.total_sd))
add("Guard: DI and target dispersion", *stats.spearmanr(ag.DI, ag.total_sd))
add("Guard: residual dispersion and target dispersion", *stats.spearmanr(ag.resid_sd, ag.total_sd))
add("Guard: adjusted DI and target dispersion", *stats.spearmanr(ag.DI_adj, ag.total_sd))
add("Compliance and target dispersion", *stats.spearmanr(ag.compliance, ag.total_sd))
Q = pd.DataFrame(rows); Q.to_csv(RES + "G9_compliance_agencies.csv", index=False)
print("\nG9. Compliance and DI across %d agencies with at least 200 evaluation contracts (%d with DI below zero)" % (len(ag), (ag.DI < 0).sum()))
print(Q.round(4).to_string(index=False))

# specification curve: residual dispersion regressed on compliance
def ols(y, X, names):
    X = np.column_stack([np.ones(len(y))] + X); beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta; s2 = e @ e / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t = beta[1] / se[1]; return beta[1], se[1], 2 * stats.t.sf(abs(t), len(y) - X.shape[1])
S = []
for nm, X in [("Compliance only", [ag.compliance.values]),
              ("+ target dispersion", [ag.compliance.values, ag.total_sd.values]),
              ("+ target dispersion + log n", [ag.compliance.values, ag.total_sd.values, np.log(ag.n.values)]),
              ("+ target dispersion + jurisdiction", [ag.compliance.values, ag.total_sd.values, ag.fed.values]),
              ("+ target dispersion + log n + jurisdiction", [ag.compliance.values, ag.total_sd.values, np.log(ag.n.values), ag.fed.values])]:
    c, se, p = ols(ag.resid_sd.values, X, nm)
    S.append({"specification": nm, "coef": c, "se": se, "p": p, "direction": "more informative" if c < 0 else "less informative"})
S = pd.DataFrame(S); S.to_csv(RES + "G10_specification_curve.csv", index=False)
print("\nG10. Specification curve (dependent variable: residual dispersion; coefficient on compliance)"); print(S.round(4).to_string(index=False))

# ------------------------------------------------------------------ G11 value target by amendment status (overwritten target)
F = E_all[E_all.fed & E_all.origin.isin(FULL)]
O = []
for fs, pc in [("Seven at-publication blocks", "p"), ("All eight blocks", "p8")]:
    for nm, sub in [("Never amended (value as first published)", F[F.ever_amended == 0]), ("Later amended (value overwritten)", F[F.ever_amended == 1]), ("All Commonwealth", F)]:
        O.append({"field_set": fs, "contracts": nm, "n": len(sub), "DI": di(sub.y, sub[pc]), "total_sd": sub.y.std(), "resid_sd": rsd(sub.y, sub[pc])})
O = pd.DataFrame(O); O.to_csv(RES + "G11_overwritten_target.csv", index=False)
print("\nG11. Value target by amendment status, Commonwealth, fully observed origins"); print(O.round(4).to_string(index=False))

# ------------------------------------------------------------------ G12 sensitivity: jurisdiction comparison over all eleven origins
E_all["juris"] = np.where(E_all.fed, "Commonwealth", "State and territory")
Ja, _ = group_stats(E_all, "juris"); Ja.to_csv(RES + "G12_jurisdiction_all_origins.csv", index=False)
print("\nG12. Sensitivity: jurisdiction comparison pooled over all eleven origins"); print(Ja.round(4).to_string(index=False))
sh = pd.Series({Y: 1 - C[C.yr.between(Y - 5, Y - 1)].fed.mean() for Y in ORIGINS})
print("   state and territory share of the training window: %.1f%% at the 2018 origin, %.1f%% at 2021, %.1f%% at 2025" % (100 * sh[2018], 100 * sh[2021], 100 * sh[2025]))
