"""Shared functions for the main experiments (scripts 06 to 13).

Same design as 03_supplementary_models.py: five-year training window, 100,000 training rows, one
frozen gradient-boosted learner, every encoding fitted on training rows only. Script 03 keeps its
own copy of these functions so that its published output stays byte-identical.
"""
import os, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")
RES, CACHE = "analysis/results/", "analysis/cache/"
os.makedirs(RES, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

SEED, NTRAIN, NBOOT = 20261005, 100_000, 2000
HP = dict(n_estimators=400, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8,
          min_child_weight=5, tree_method="hist", n_jobs=10, random_state=SEED)

CATS = {"prod": ["category_code", "parent_category_code"],
        "agency": ["publisher_name", "publisher_portfolio", "publisher_entity_type", "publisher_industry",
                   "publisher_cofog_code", "publisher_gov_type"],
        "proc": ["procurement_method"],
        "supp": ["supplier_name", "supplier_state", "supplier_country"]}
NUMS = {"cal": ["yr", "month", "quarter", "fy_pos"], "prod": ["consult"], "agency": ["publisher_employees"],
        "text": ["desc_len"], "term": ["duration", "has_end", "pub_lag"]}
SIX = ["cal", "prod", "agency", "proc", "supp", "text"]
SEVEN = SIX + ["term"]
EIGHT = SEVEN + ["hist"]
HIST = {"hist": ["ever_amended", "n_amend"], "hist24": ["amended24", "n_amend24"]}
BLOCK_NAMES = {"cal": "Calendar timing", "prod": "Product category", "agency": "Buying agency",
               "proc": "Award procedure", "supp": "Supplier identity", "text": "Free text",
               "term": "Contract term", "hist": "Amendment history"}
ORIGINS = list(range(2015, 2026))          # eleven annual origins
FULL = list(range(2015, 2024))             # origins at which the amendment history is fully observed
AM_ORIGINS = list(range(2017, 2024))       # origins of the amendment target


def load_contracts():
    C = pd.read_pickle(CACHE + "contracts.pkl")
    C["ly"] = np.log(C.value)
    C["agency_id"] = C.publisher_name.fillna("NA")
    return C


def encode(tr, ev, y, extra, cats=CATS, nums=NUMS, text=True):
    """Feature matrices for training and evaluation rows; all encodings fitted on training rows."""
    Xtr, Xev, cols = {}, {}, {}
    kf = list(KFold(5, shuffle=True, random_state=SEED).split(tr))
    prior, m = y.mean(), 20.0
    for b, cs in cats.items():
        for c in cs:
            a, e = tr[c].fillna("NA").values, ev[c].fillna("NA").values
            s = pd.DataFrame({"a": a, "y": y}).groupby("a").y.agg(["sum", "count"])
            te_ev = ((s["sum"] + prior * m) / (s["count"] + m)).reindex(e).fillna(prior).values
            cnt_ev = np.log1p(s["count"].reindex(e).fillna(0).values)
            te_tr = np.empty(len(tr))
            for i, j in kf:
                si = pd.DataFrame({"a": a[i], "y": y[i]}).groupby("a").y.agg(["sum", "count"])
                te_tr[j] = ((si["sum"] + prior * m) / (si["count"] + m)).reindex(a[j]).fillna(prior).values
            cnt_tr = np.log1p(s["count"].reindex(a).values)
            for nm, vt, ve in [(c + "_te", te_tr, te_ev), (c + "_n", cnt_tr, cnt_ev)]:
                Xtr[nm], Xev[nm] = vt, ve; cols.setdefault(b, []).append(nm)
    for b, cs in {**nums, **extra}.items():
        for c in cs:
            Xtr[c], Xev[c] = tr[c].values.astype(float), ev[c].values.astype(float); cols.setdefault(b, []).append(c)
    if text:
        tf = TfidfVectorizer(min_df=20, max_features=20000, sublinear_tf=True)
        Ttr, Tev = tf.fit_transform(tr.text), tf.transform(ev.text)
        rt = np.empty(len(tr))
        for i, j in kf:
            rt[j] = Ridge(alpha=1.0, solver="sparse_cg", random_state=SEED).fit(Ttr[i], y[i]).predict(Ttr[j])
        Xtr["text_score"] = rt
        Xev["text_score"] = Ridge(alpha=1.0, solver="sparse_cg", random_state=SEED).fit(Ttr, y).predict(Tev)
        cols.setdefault("text", []).append("text_score")
    return pd.DataFrame(Xtr), pd.DataFrame(Xev), cols


def fit_predict(Xtr, y, Xev, use, binary=False, hp=None):
    hp = hp or HP
    M = xgb.XGBClassifier(**hp) if binary else xgb.XGBRegressor(**hp)
    M.fit(Xtr[use], y)
    return M.predict_proba(Xev[use])[:, 1] if binary else M.predict(Xev[use])


def use_cols(cols, blocks):
    return [c for b in blocks for c in cols[b]]


def di(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def rsd(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def sample(tr, seed):
    return tr.sample(NTRAIN, random_state=seed) if len(tr) > NTRAIN else tr


# ---------------------------------------------------------------- cluster bootstrap over agencies
class ClusterStats:
    """Per-cluster sufficient statistics, so that DI can be recomputed for resampled clusters."""

    def __init__(self, groups, y, preds):
        """preds: dict name -> prediction vector, all on the same rows as y."""
        g = pd.factorize(np.asarray(groups))[0]; self.k = g.max() + 1
        y = np.asarray(y, float)
        self.n = np.bincount(g, minlength=self.k).astype(float)
        self.sy = np.bincount(g, weights=y, minlength=self.k)
        self.syy = np.bincount(g, weights=y * y, minlength=self.k)
        self.sse = {nm: np.bincount(g, weights=(y - np.asarray(p, float)) ** 2, minlength=self.k) for nm, p in preds.items()}

    def di(self, w=None):
        """DI of every prediction for cluster weights w (multiplicities); w=None gives the sample value."""
        w = np.ones(self.k) if w is None else w
        n, sy, syy = w @ self.n, w @ self.sy, w @ self.syy
        sst = syy - sy * sy / n
        return {nm: 1 - (w @ s) / sst for nm, s in self.sse.items()}

    def draws(self, B=NBOOT, seed=SEED):
        rng = np.random.default_rng(seed); out = []
        for _ in range(B):
            w = np.bincount(rng.integers(0, self.k, self.k), minlength=self.k).astype(float)
            out.append(self.di(w))
        return pd.DataFrame(out)


def ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def boot_p(v):
    """Two-sided bootstrap p-value for H0: the quantity is zero."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]; B = len(v)
    return float(min(1.0, 2 * (min((v <= 0).sum(), (v >= 0).sum()) + 1) / (B + 1)))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def mann_kendall(x, y):
    """Mann-Kendall trend test (Kendall's tau against time) and Sen's slope."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    tau, p = stats.kendalltau(x, y)
    slope = stats.theilslopes(y, x)[0]
    return float(tau), float(p), float(slope)


def cal_slope(y, p):
    """Calibration slope: logistic regression of the outcome on the logit of the forecast."""
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    return float(LogisticRegression(C=1e6, max_iter=1000).fit(z, y).coef_[0, 0])


def method_class(s):
    """Harmonised award procedure: open, limited or other."""
    t = s.fillna("").str.strip().str.lower()
    out = pd.Series("other", index=s.index)
    out[t.str.contains("open|public|request for tender|rft|expression of interest")] = "open"
    out[t.str.contains("limited|single|direct|non tender|non-tender|restricted|select|quotation|sole")] = "limited"
    out[t == ""] = "missing"
    return out


OPTIONAL_FIELDS = ["description", "category_code", "parent_category_code", "contract_start", "contract_end",
                   "procurement_method", "supplier_state", "publisher_portfolio", "publisher_industry"]


def compliance_fields(df):
    """Indicator matrix for the nine optional fields; a description under 20 characters counts as absent."""
    F = pd.DataFrame({c: df[c].notna().astype(int) for c in OPTIONAL_FIELDS if c != "description"}, index=df.index)
    F.insert(0, "description", (df.desc_len >= 20).astype(int))
    return F
