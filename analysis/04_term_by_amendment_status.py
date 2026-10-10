"""What the contract-term block adds to the value forecast, separately for contracts never amended
(recorded duration is the first-published term) and contracts later amended (duration may be revised).

Run from the project root:
    analysis/.venv/bin/python analysis/04_term_by_amendment_status.py > analysis/04_term_by_amendment_status_output.txt
Uses the functions of 03_supplementary_models.py (same learner, encodings and sampling).
"""
import importlib.util, io, re, numpy as np, pandas as pd
src = io.open("analysis/03_supplementary_models.py", encoding="utf-8").read().split("C = pd.read_pickle")[0]
exec(src)                                              # definitions only; no analysis is re-run
C = pd.read_pickle("analysis/cache/contracts.pkl"); C["ly"] = np.log(C.value)
rows = []
for Y in range(2015, 2024):
    tr = sample(C[C.yr.between(Y - 5, Y - 1)], SEED + Y); ev = C[C.yr == Y]
    ytr, yev = tr.ly.values, ev.ly.values
    Xtr, Xev, cols = encode(tr, ev, ytr, {})
    six = [c for b in SIX for c in cols[b]]; seven = six + cols["term"]
    p6, p7 = fit_predict(Xtr, ytr, Xev, six), fit_predict(Xtr, ytr, Xev, seven)
    for nm, m in [("never amended", ev.ever_amended.values == 0), ("later amended", ev.ever_amended.values == 1), ("all", np.ones(len(ev), bool))]:
        rows.append({"origin": Y, "group": nm, "n": int(m.sum()), "sse6": np.sum((yev[m] - p6[m]) ** 2), "sse7": np.sum((yev[m] - p7[m]) ** 2),
                     "sst": np.sum((yev[m] - yev[m].mean()) ** 2), "sst_pooled": np.sum((yev[m] - yev.mean()) ** 2)})
    print(Y, flush=True)
R = pd.DataFrame(rows); R.to_csv("analysis/results/F_term_by_amendment_status.csv", index=False)
g = R.groupby("group")[["n", "sse6", "sse7", "sst"]].sum()
g["DI six"] = 1 - g.sse6 / g.sst; g["DI seven"] = 1 - g.sse7 / g.sst; g["term increment"] = g["DI seven"] - g["DI six"]
g["resid SD six"] = np.sqrt(g.sse6 / g.n); g["resid SD seven"] = np.sqrt(g.sse7 / g.n)
print(g[["n", "DI six", "DI seven", "term increment", "resid SD six", "resid SD seven"]].round(4).to_string())
a = g.loc["all"]; nv = g.loc["never amended"]; am = g.loc["later amended"]
red = a.sse6 - a.sse7
print("\nReduction in squared error from the term block, pooled 2015-2023: %.0f" % red)
print("  share arising on never-amended contracts: %.3f | on later-amended contracts: %.3f"
      % ((nv.sse6 - nv.sse7) / red, (am.sse6 - am.sse7) / red))
print("  never-amended contracts are %.3f of evaluation contracts" % (nv.n / a.n))
