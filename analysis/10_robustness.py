"""Robustness: other learners, and placeholder values retained (Section 4.7; Appendix C).

Run from the project root, after script 06:
    analysis/.venv/bin/python analysis/10_robustness.py > analysis/10_robustness_output.txt
Output: analysis/results/R1_learners.csv, R2_placeholders.csv
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from lib import *

C = load_contracts()
M1 = pd.read_csv(RES + "M1_di_by_origin.csv").set_index("origin")
NONTEXT = [b for b in EIGHT if b != "text"]
SHALLOW = dict(HP, max_depth=2)


def linear(Xtr, y, Xev, use):
    med = Xtr[use].median(); A, B = Xtr[use].fillna(med), Xev[use].fillna(med)
    mu, sd = A.mean(), A.std().replace(0, 1)
    return Ridge(alpha=1.0).fit((A - mu) / sd, y).predict((B - mu) / sd)


rows, jur = [], []
for Y in ORIGINS:
    tr = sample(C[C.yr.between(Y - 5, Y - 1)], SEED + Y); ev = C[C.yr == Y]
    ytr, yev = tr.ly.values, ev.ly.values
    Xtr, Xev, cols = encode(tr, ev, ytr, HIST)
    a8, nt = use_cols(cols, EIGHT), use_cols(cols, NONTEXT)
    ps = fit_predict(Xtr, ytr, Xev, a8, hp=SHALLOW)
    pl = linear(Xtr, ytr, Xev, a8)
    pb = fit_predict(Xtr, ytr, Xev, nt)
    med = Xtr[nt].median()
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=10, random_state=SEED)
    pr = rf.fit(Xtr[nt].fillna(med), ytr).predict(Xev[nt].fillna(med))
    rows.append({"origin": Y, "boosted_depth6": M1.all8[Y], "boosted_depth2": di(yev, ps), "penalised_linear": di(yev, pl),
                 "boosted_nontext": di(yev, pb), "random_forest_nontext": di(yev, pr)})
    if Y == ORIGINS[-1]:
        pm = fit_predict(Xtr, ytr, Xev, a8)
        for g, m in ev.groupby("publisher_gov_type").indices.items():
            if len(m) >= 500:
                jur.append({"jurisdiction": g, "n": len(m), "boosted_depth6": di(yev[m], pm[m]), "boosted_depth2": di(yev[m], ps[m])})
    print(Y, {k: round(float(v), 4) for k, v in rows[-1].items() if k != "origin"}, flush=True)
L = pd.DataFrame(rows); L.to_csv(RES + "R1_learners.csv", index=False); Jr = pd.DataFrame(jur)
print("\nR1. DI on all eight blocks under four learners"); print(L.round(4).to_string(index=False))
print("   depth 6 against depth 2: Pearson %.3f, rank correlation %.3f" % (stats.pearsonr(L.boosted_depth6, L.boosted_depth2)[0], stats.spearmanr(L.boosted_depth6, L.boosted_depth2)[0]))
for c in ["boosted_depth6", "boosted_depth2"]:
    print("   trend, %s: Mann-Kendall p %.3f" % (c, mann_kendall(L.origin, L[c])[1]))
d = L.boosted_depth6 - L.penalised_linear
print("   penalised linear model lower than depth 6 by %.3f to %.3f" % (d.min(), d.max()))
print("   random forest against boosted model on the same non-text blocks: largest absolute difference %.3f" % (L.random_forest_nontext - L.boosted_nontext).abs().max())
print("   jurisdictions at the %d origin (at least 500 contracts):" % ORIGINS[-1]); print(Jr.round(4).to_string(index=False))
print("   rank correlation of jurisdiction ordering between depth 6 and depth 2: %.3f" % stats.spearmanr(Jr.boosted_depth6, Jr.boosted_depth2)[0])

# ------------------------------------------------------------------ placeholder values retained
A = pd.read_pickle(CACHE + "contracts_all_values.pkl"); A = A[A.value > 0].copy()       # zero and negative values have no logarithm
A["ly"] = np.log(A.value); A["agency_id"] = A.publisher_name.fillna("NA"); A["ph"] = A.value <= 1
print("\nR2. Placeholder values (0 < value <= A$1) retained: %d contracts, %d of them on state and territory portals" % (A.ph.sum(), (A.ph & ~A.fed).sum()))
rows, pool = [], []
for Y in ORIGINS:
    tr = sample(A[A.yr.between(Y - 5, Y - 1)], SEED + Y); ev = A[A.yr == Y]
    ytr, yev = tr.ly.values, ev.ly.values
    Xtr, Xev, cols = encode(tr, ev, ytr, HIST)
    P = {nm: fit_predict(Xtr, ytr, Xev, use_cols(cols, b)) for nm, b in [("all8", EIGHT), ("seven", SEVEN), ("six", SIX)]}
    se = (yev - P["all8"]) ** 2; ph = ev.ph.values
    r = {"origin": Y, "n_placeholders": int(ph.sum()), "share_sq_error": se[ph].sum() / se.sum(), **{k: di(yev, v) for k, v in P.items()}}
    r["floor"] = (r["all8"] - r["seven"]) / r["all8"]; r["ceiling"] = (r["all8"] - r["six"]) / r["all8"]
    r["di_excluded"] = M1.all8[Y]; rows.append(r)
    pool.append(pd.DataFrame({"fed": ev.fed.values, "y": yev, "p": P["seven"]}))
    print(Y, {k: round(float(v), 4) for k, v in r.items() if k != "origin"}, flush=True)
Ph = pd.DataFrame(rows); Ph.to_csv(RES + "R2_placeholders.csv", index=False); Q = pd.concat(pool)
print(Ph.round(4).to_string(index=False))
print("   trend in DI with placeholders retained: Mann-Kendall p %.3f (excluded: %.3f)" % (mann_kendall(Ph.origin, Ph.all8)[1], mann_kendall(Ph.origin, Ph.di_excluded)[1]))
f = Ph[Ph.origin.isin(FULL)]
print("   hindsight share with placeholders retained: floor %.1f%%, ceiling %.1f%%" % (100 * f.floor.mean(), 100 * f.ceiling.mean()))
a, b = Q[Q.fed], Q[~Q.fed]
print("   jurisdiction difference with placeholders retained: %+.3f (Commonwealth %.3f, state and territory %.3f); residual SD %.3f and %.3f"
      % (di(a.y, a.p) - di(b.y, b.p), di(a.y, a.p), di(b.y, b.p), rsd(a.y, a.p), rsd(b.y, b.p)))
