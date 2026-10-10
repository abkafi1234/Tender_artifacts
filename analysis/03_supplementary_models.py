"""Supplementary analyses on the rolling-origin design.

Run from the project root:
    analysis/.venv/bin/python analysis/03_supplementary_models.py > analysis/03_supplementary_models_output.txt
Inputs: analysis/cache/notices.pkl, analysis/cache/contracts.pkl (from 02_build_frames.py).
Outputs: the text log and CSV files in analysis/results/.

Design: five-year training window, 100,000 training rows, one frozen gradient-boosted learner,
encodings fitted on training rows only.
"""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
warnings.filterwarnings("ignore")
os.makedirs("analysis/results", exist_ok=True)
SEED, NTRAIN = 20261005, 100_000
HP = dict(n_estimators=400, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8,
          min_child_weight=5, tree_method="hist", n_jobs=10, random_state=SEED)   # frozen everywhere

CATS = {"prod": ["category_code", "parent_category_code"],
        "agency": ["publisher_name", "publisher_portfolio", "publisher_entity_type", "publisher_industry",
                   "publisher_cofog_code", "publisher_gov_type"],
        "proc": ["procurement_method"],
        "supp": ["supplier_name", "supplier_state", "supplier_country"]}
NUMS = {"cal": ["yr", "month", "quarter", "fy_pos"], "prod": ["consult"], "agency": ["publisher_employees"],
        "text": ["desc_len"], "term": ["duration", "has_end", "pub_lag"]}
SIX = ["cal", "prod", "agency", "proc", "supp", "text"]


def encode(tr, ev, y, extra):
    """Feature matrices for train and evaluation rows; every encoding is fitted on training rows only."""
    Xtr, Xev, cols = {}, {}, {}
    kf = list(KFold(5, shuffle=True, random_state=SEED).split(tr))
    prior, m = y.mean(), 20.0
    for b, cs in CATS.items():
        for c in cs:
            a, e = tr[c].fillna("NA").values, ev[c].fillna("NA").values
            s = pd.DataFrame({"a": a, "y": y}).groupby("a").y.agg(["sum", "count"])
            te_ev = ((s["sum"] + prior * m) / (s["count"] + m)).reindex(e).fillna(prior).values
            cnt_ev = np.log1p(s["count"].reindex(e).fillna(0).values)
            te_tr = np.empty(len(tr))
            for i, j in kf:                                        # out-of-fold on training rows
                si = pd.DataFrame({"a": a[i], "y": y[i]}).groupby("a").y.agg(["sum", "count"])
                te_tr[j] = ((si["sum"] + prior * m) / (si["count"] + m)).reindex(a[j]).fillna(prior).values
            cnt_tr = np.log1p(s["count"].reindex(a).values)
            for nm, vt, ve in [(c + "_te", te_tr, te_ev), (c + "_n", cnt_tr, cnt_ev)]:
                Xtr[nm], Xev[nm] = vt, ve; cols.setdefault(b, []).append(nm)
    for b, cs in {**NUMS, **extra}.items():
        for c in cs:
            Xtr[c], Xev[c] = tr[c].values.astype(float), ev[c].values.astype(float); cols.setdefault(b, []).append(c)
    tf = TfidfVectorizer(min_df=20, max_features=20000, sublinear_tf=True)
    Ttr, Tev = tf.fit_transform(tr.text), tf.transform(ev.text)
    rt = np.empty(len(tr))
    for i, j in kf:
        rt[j] = Ridge(alpha=1.0, solver="sparse_cg", random_state=SEED).fit(Ttr[i], y[i]).predict(Ttr[j])
    Xtr["text_score"] = rt
    Xev["text_score"] = Ridge(alpha=1.0, solver="sparse_cg", random_state=SEED).fit(Ttr, y).predict(Tev)
    cols.setdefault("text", []).append("text_score")
    return pd.DataFrame(Xtr), pd.DataFrame(Xev), cols


def fit_predict(Xtr, y, Xev, use, binary=False):
    M = xgb.XGBClassifier(**HP) if binary else xgb.XGBRegressor(**HP)
    M.fit(Xtr[use], y)
    return M.predict_proba(Xev[use])[:, 1] if binary else M.predict(Xev[use])


def di(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def sample(tr, seed):
    return tr.sample(NTRAIN, random_state=seed) if len(tr) > NTRAIN else tr


def rolling(df, target, extra, sets, origins, label):
    """DI of each field set at each origin. sets: name -> list of blocks. Returns table and predictions."""
    rows, preds = [], []
    for Y in origins:
        tr = sample(df[df.yr.between(Y - 5, Y - 1)], SEED + Y); ev = df[df.yr == Y]
        ytr, yev = tr[target].values, ev[target].values
        Xtr, Xev, cols = encode(tr, ev, ytr, extra)
        r = {"origin": Y, "n_eval": len(ev), "total_sd": yev.std()}
        for nm, blocks in sets.items():
            use = [c for b in blocks for c in cols[b]]
            p = fit_predict(Xtr, ytr, Xev, use)
            r[nm] = di(yev, p); r[nm + "_rsd"] = np.sqrt(np.mean((yev - p) ** 2))
            if nm == "seven":
                preds.append(pd.DataFrame({"origin": Y, "y": yev, "p": p,
                                           "seen": ev.supplier_name.fillna("NA").isin(set(tr.supplier_name.fillna("NA"))).values}))
        rows.append(r); print("   ", label, Y, {k: round(v, 4) for k, v in r.items() if k in sets}, flush=True)
    return pd.DataFrame(rows), (pd.concat(preds) if preds else None)


C = pd.read_pickle("analysis/cache/contracts.pkl")
C["ly"] = np.log(C.value)
FULL = list(range(2015, 2024))        # fully observed origins

# ------------------------------------------------------------------ A. contract frame, validation
print("A. Contract frame, value target (validation against the manuscript)")
sets = {"all8": SIX + ["term", "hist"], "seven": SIX + ["term"], "six": SIX}
A, PA = rolling(C, "ly", {"hist": ["ever_amended", "n_amend"]}, sets, FULL, "contract")
A["floor"] = (A.all8 - A.seven) / A.all8; A["ceiling"] = (A.all8 - A.six) / A.all8
A.to_csv("analysis/results/A_contract_frame.csv", index=False)
print(A[["origin", "all8", "seven", "six", "floor", "ceiling"]].round(4).to_string(index=False))
print("   mean DI all eight %.3f (manuscript 0.575) | floor %.1f%% (manuscript 4.4%%) | ceiling %.1f%% (manuscript 30.0%%)"
      % (A.all8.mean(), 100 * A.floor.mean(), 100 * A.ceiling.mean()))

# ------------------------------------------------------------------ B. notice frame, 2 x 2
print("\nB. Notice frame, value target: unit of analysis by classification of duration")
N = pd.read_pickle("analysis/cache/notices.pkl"); N = N[N.value > 1].copy(); N["ly"] = np.log(N.value)
setsN = {"all": SIX + ["term", "flag"], "noflag": SIX + ["term"], "six": SIX}
B, _ = rolling(N, "ly", {"flag": ["am_flag"]}, setsN, FULL, "notice")
B["share_flag_only"] = (B["all"] - B.noflag) / B["all"]; B["share_flag_and_term"] = (B["all"] - B.six) / B["all"]
ov = []
for Y in FULL:
    ev = N[N.yr == Y]; trp = set(N[N.yr.between(Y - 5, Y - 1)].parent)
    ov.append({"origin": Y, "amend_share": (ev.k > 0).mean(), "in_train": ev.parent.isin(trp).mean()})
B = B.merge(pd.DataFrame(ov)); B.to_csv("analysis/results/B_notice_frame.csv", index=False)
print(B[["origin", "all", "noflag", "six", "share_flag_only", "share_flag_and_term", "amend_share", "in_train"]].round(4).to_string(index=False))
t = pd.DataFrame({"Duration at publication": [B.share_flag_only.mean(), A.floor.mean()],
                  "Duration post-award": [B.share_flag_and_term.mean(), A.ceiling.mean()]},
                 index=["Notice as unit", "Contract as unit"])
print("   2 x 2, mean hindsight share over 2015-2023:"); print((100 * t).round(1).to_string())
print("   six-block DI: notice %.3f, contract %.3f" % (B.six.mean(), A.six.mean()))
t.to_csv("analysis/results/B_two_by_two.csv")

# ------------------------------------------------------------------ C. first-published value target
print("\nC. Floor on the current value and on the first-published value, same contracts")
F = C[C.fp_value.notna() & (C.fp_value > 1)].copy(); F["lfp"] = np.log(F.fp_value)
print("   contracts with a first-published value: %d (never amended %d, amended %d)"
      % (len(F), (F.ever_amended == 0).sum(), (F.ever_amended == 1).sum()))
s2 = {"all8": SIX + ["term", "hist"], "seven": SIX + ["term"]}
Cc, _ = rolling(F, "ly", {"hist": ["ever_amended", "n_amend"]}, s2, FULL, "current")
Cf, _ = rolling(F, "lfp", {"hist": ["ever_amended", "n_amend"]}, s2, FULL, "first-pub")
Cc["floor"] = (Cc.all8 - Cc.seven) / Cc.all8; Cf["floor"] = (Cf.all8 - Cf.seven) / Cf.all8
out = pd.DataFrame({"origin": FULL, "di_current": Cc.all8, "floor_current": Cc.floor, "di_firstpub": Cf.all8, "floor_firstpub": Cf.floor})
out.to_csv("analysis/results/C_first_published_target.csv", index=False); print(out.round(4).to_string(index=False))
print("   mean floor: current value %.1f%% | first-published value %.1f%%" % (100 * Cc.floor.mean(), 100 * Cf.floor.mean()))

# ------------------------------------------------------------------ D. unseen suppliers
print("\nD. Seven blocks, value target: suppliers present in or absent from the training sample")
rows = []
for nm, m in [("Supplier in training sample", PA.seen), ("Supplier not in training sample", ~PA.seen), ("All", PA.seen | True)]:
    q = PA[m]; rows.append({"group": nm, "n": len(q), "DI": di(q.y.values, q.p.values), "total_sd": q.y.std(),
                            "resid_sd": np.sqrt(np.mean((q.y - q.p) ** 2))})
D = pd.DataFrame(rows); D.to_csv("analysis/results/D_unseen_supplier.csv", index=False); print(D.round(4).to_string(index=False))

# ------------------------------------------------------------------ E. amendment target and duration
print("\nE. Amendment target: what recorded duration adds, and how much of it is revision")
W = C[C.fed].copy()
rows = []
for Y in range(2017, 2024):
    tr = sample(W[W.yr.between(Y - 7, Y - 3)], SEED + Y); ev = W[W.yr == Y]
    ytr, yev = tr.amended24.values.astype(float), ev.amended24.values
    Xtr, Xev, cols = encode(tr, ev, ytr, {})
    six = [c for b in SIX for c in cols[b]]
    a0 = roc_auc_score(yev, fit_predict(Xtr, ytr, Xev, six, True)); a1 = roc_auc_score(yev, fit_predict(Xtr, ytr, Xev, six + ["duration"], True))
    rows.append({"origin": Y, "base_rate": yev.mean(), "auc_six": a0, "auc_six_duration": a1, "gain": a1 - a0}); print("   ", rows[-1], flush=True)
E1 = pd.DataFrame(rows); E1.to_csv("analysis/results/E1_amendment_auc.csv", index=False)
print("   mean AUC six %.3f (manuscript 0.755) | with duration %.3f (0.833) | gain %+.3f (+0.077)" % (E1.auc_six.mean(), E1.auc_six_duration.mean(), E1.gain.mean()))

print("\n   E2. Contracts first published in 2024-2025; outcome: an amendment notice published by end of 2025.")
print("       Trained on contracts published 2019-2023 (outcome: amended within 24 months).")
tr = sample(W[W.yr.between(2019, 2023)], SEED); ev = W[W.yr >= 2024].copy()
ev = ev[(ev.ever_amended == 0) | (ev.a1_yr >= 2024)]
ytr = tr.amended24.values.astype(float)
Xtr, Xev, cols = encode(tr, ev, ytr, {})
six = [c for b in SIX for c in cols[b]]
M0 = xgb.XGBClassifier(**HP).fit(Xtr[six], ytr); M1 = xgb.XGBClassifier(**HP).fit(Xtr[six + ["duration"]], ytr)
p0 = M0.predict_proba(Xev[six])[:, 1]; p1 = M1.predict_proba(Xev[six + ["duration"]])[:, 1]
# the same contracts scored with the duration implied by the first amendment notice
Xrev = Xev.copy(); rev = ((ev.a1_end - ev.contract_start).dt.days).values.astype(float)
Xrev["duration"] = np.where((ev.ever_amended.values == 1) & ~np.isnan(rev), rev, Xev.duration.values)
p1rev = M1.predict_proba(Xrev[six + ["duration"]])[:, 1]
y = ev.ever_amended.values; never = y == 0
clean = never | (ev.overwritten.values == 0); over = never | (ev.overwritten.values == 1)
rng = np.random.default_rng(SEED)
def gain_ci(mask, pa, pb):
    idx = np.where(mask)[0]; g = roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx]); bs = []
    for _ in range(500):
        j = rng.choice(idx, len(idx)); bs.append(roc_auc_score(y[j], pb[j]) - roc_auc_score(y[j], pa[j]))
    return roc_auc_score(y[idx], pa[idx]), roc_auc_score(y[idx], pb[idx]), g, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
rows = []
for nm, mask, pb in [("Amended, parent not overwritten: duration as first published", clean, p1),
                     ("Same contracts: duration after the first amendment", clean, p1rev),
                     ("Amended, parent overwritten: recorded (current) duration", over, p1)]:
    a, b, g_, lo, hi = gain_ci(mask, p0, pb)
    rows.append({"evaluation set": nm, "n": int(mask.sum()), "n_amended": int(y[mask].sum()), "auc_six": a, "auc_six_duration": b, "gain": g_, "lo": lo, "hi": hi})
E2 = pd.DataFrame(rows); E2.to_csv("analysis/results/E2_duration_first_published.csv", index=False); print(E2.round(4).to_string(index=False))
d = ev.duration.values
for nm, mask in [("first-published duration (not overwritten)", clean), ("recorded duration (overwritten)", over)]:
    k = mask & ~np.isnan(d); print("       AUC of duration alone, %s: %.3f" % (nm, roc_auc_score(y[k], d[k])))
