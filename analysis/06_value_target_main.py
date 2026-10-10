"""Main experiments on the contract-value target (Sections 4.1 and 4.2; Appendix B).

Run from the project root:
    analysis/.venv/bin/python analysis/06_value_target_main.py > analysis/06_value_target_main_output.txt
Input:  analysis/cache/contracts.pkl
Output: analysis/results/M1_di_by_origin.csv, M2_trend_tests.csv, M3_hindsight.csv,
        analysis/cache/preds_value.pkl (evaluation-set predictions, reused by scripts 08, 10, 12)
"""
import numpy as np, pandas as pd
from lib import *

C = load_contracts()
SETS = {"all8": EIGHT, "seven": SEVEN, "six": SIX, "hist": ["hist"], "hist24": ["hist24"], "all8_24": SEVEN + ["hist24"]}
panel = set.intersection(*[set(C[C.yr == Y].agency_id) for Y in ORIGINS])
print("Balanced panel: %d agencies present in the evaluation year of all eleven origins" % len(panel))

rows, preds = [], []
for Y in ORIGINS:
    tr = sample(C[C.yr.between(Y - 5, Y - 1)], SEED + Y); ev = C[C.yr == Y]
    ytr, yev = tr.ly.values, ev.ly.values
    Xtr, Xev, cols = encode(tr, ev, ytr, HIST)
    P = {nm: fit_predict(Xtr, ytr, Xev, use_cols(cols, b)) for nm, b in SETS.items()}
    # permuted-target sentinel: the target is shuffled and the whole pipeline rebuilt
    yp = np.random.default_rng(SEED + Y).permutation(ytr)
    Xp, Xpe, cp = encode(tr, ev, yp, HIST)
    sent = di(yev, fit_predict(Xp, yp, Xpe, use_cols(cp, EIGHT)))
    # balanced panel: training and evaluation restricted to agencies present at every origin
    trb = sample(C[C.yr.between(Y - 5, Y - 1) & C.agency_id.isin(panel)], SEED + Y); evb = ev[ev.agency_id.isin(panel)]
    Xb, Xbe, cb = encode(trb, evb, trb.ly.values, HIST)
    dib = di(evb.ly.values, fit_predict(Xb, trb.ly.values, Xbe, use_cols(cb, EIGHT)))
    cs = ClusterStats(ev.agency_id.values, yev, P); D = cs.draws()
    lo, hi = ci(D.all8)
    r = {"origin": Y, "n_eval": len(ev), "total_sd": yev.std(), "sentinel": sent, "di_balanced": dib,
         "all8_lo": lo, "all8_hi": hi}
    for nm, p in P.items():
        r[nm] = di(yev, p); r[nm + "_rsd"] = rsd(yev, p)
    for nm, a, b in [("floor", "all8", "seven"), ("floor24", "all8_24", "seven"), ("ceiling", "all8", "six")]:
        num = D[a] - D[b]
        r[nm] = (r[a] - r[b]) / r[a]; r[nm + "_p"] = boot_p(num)
    r["term_increment"] = r["seven"] - r["six"]
    rows.append(r)
    q = pd.DataFrame({"origin": Y, "y": yev, **{"p_" + k: v for k, v in P.items()}}, index=ev.index)
    preds.append(q)
    print(Y, {k: round(float(r[k]), 4) for k in ["all8", "seven", "six", "hist24", "floor", "floor24", "ceiling", "sentinel", "di_balanced"]}, flush=True)

R = pd.DataFrame(rows); PV = pd.concat(preds); PV.to_pickle(CACHE + "preds_value.pkl")
R["floor_holm"] = holm(R.floor_p.values)
R.to_csv(RES + "M1_di_by_origin.csv", index=False)

# ------------------------------------------------------------------ hindsight share
H = R[["origin", "all8", "seven", "six", "floor", "floor24", "ceiling", "floor_holm"]].copy()
H.loc[~H.origin.isin(FULL), "floor24"] = np.nan
H.to_csv(RES + "M3_hindsight.csv", index=False)
F = R[R.origin.isin(FULL)]
print("\nHindsight share over the nine fully observed origins")
print("  floor   mean %.1f%% (range %.1f%% to %.1f%%)" % (100 * F.floor.mean(), 100 * F.floor.min(), 100 * F.floor.max()))
print("  floor at equal exposure (24-month form) mean %.1f%%; first %.1f%%, last %.1f%%" % (100 * F.floor24.mean(), 100 * F.floor24.iloc[0], 100 * F.floor24.iloc[-1]))
print("  ceiling mean %.1f%% (range %.1f%% to %.1f%%)" % (100 * F.ceiling.mean(), 100 * F.ceiling.min(), 100 * F.ceiling.max()))
print("  floors at censored origins: 2024 %.1f%%, 2025 %.1f%%" % tuple(100 * R[R.origin >= 2024].floor.values))

# ------------------------------------------------------------------ trend tests
T = []
def trend(label, sub, col, family=None):
    tau, p, s = mann_kendall(sub.origin, sub[col])
    T.append({"series": label, "origins": len(sub), "first": sub[col].iloc[0], "last": sub[col].iloc[-1], "tau": tau, "mk_p": p, "sen_slope": s, "family": family})
trend("DI, all eight blocks", R, "all8", "H")
trend("DI, seven at-publication blocks", R, "seven", "H")
trend("DI, amendment history alone (24-month form)", F, "hist24", "H")
trend("DI, six descriptive blocks", R, "six")
trend("Total SD of target", R, "total_sd")
trend("Residual SD, all eight blocks", R, "all8_rsd")
trend("Residual SD, seven blocks", R, "seven_rsd")
trend("Residual SD, amendment history alone (24-month form)", F, "hist24_rsd")
trend("DI, balanced agency panel", R, "di_balanced")
trend("Floor at equal exposure", F, "floor24")
trend("Ceiling", F, "ceiling")
trend("Increment of the term block (seven minus six)", F, "term_increment")
T = pd.DataFrame(T); h = T.family == "H"
T.loc[h, "holm_p"] = holm(T.loc[h, "mk_p"].values)

# difference between the first three and the last three origins, cluster bootstrap over agencies
meta = C.loc[PV.index, ["agency_id"]]
rng = np.random.default_rng(SEED)
def mean_di(orig):
    return np.mean([R.set_index("origin").all8[o] for o in orig])
first, last = ORIGINS[:3], ORIGINS[-3:]
ag = meta.agency_id.values; uniq, inv = np.unique(ag, return_inverse=True)
st = {o: ClusterStats(inv[(PV.origin == o).values], PV.y[(PV.origin == o).values].values, {"a": PV.p_all8[(PV.origin == o).values].values}) for o in first + last}
ids = {o: np.unique(inv[(PV.origin == o).values]) for o in first + last}
diffs = []
for _ in range(NBOOT):
    w_all = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
    v = {}
    for o in first + last:
        w = w_all[ids[o]]
        v[o] = st[o].di(w)["a"] if w.sum() > 0 else np.nan
    diffs.append(np.nanmean([v[o] for o in last]) - np.nanmean([v[o] for o in first]))
d0 = mean_di(last) - mean_di(first); lo, hi = ci(diffs)
T = pd.concat([T, pd.DataFrame([{"series": "DI all eight: last three origins minus first three", "first": mean_di(first), "last": mean_di(last),
                                 "tau": d0, "mk_p": boot_p(diffs), "sen_slope": np.nan, "ci_lo": lo, "ci_hi": hi}])], ignore_index=True)
T.to_csv(RES + "M2_trend_tests.csv", index=False)
print("\nTrend tests (Mann-Kendall p, Sen slope)"); print(T.round(4).to_string(index=False))
print("\nDI by origin"); print(R[["origin", "n_eval", "all8", "all8_lo", "all8_hi", "total_sd", "all8_rsd", "di_balanced", "sentinel"]].round(4).to_string(index=False))
print("\nMean DI all eight %.3f | sentinel max |DI| %.4f" % (R.all8.mean(), R.sentinel.abs().max()))
e = PV.y - PV.p_all8
print("Typical multiplicative error: x/÷ %.2f | share of predictions within a factor of two: %.1f%%" % (np.exp(np.median(np.abs(e))), 100 * np.mean(np.abs(e) <= np.log(2))))
