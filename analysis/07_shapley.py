"""Exact Shapley attribution across disclosure blocks (Section 4.4; Appendices B and C).

Run from the project root:
    analysis/.venv/bin/python analysis/07_shapley.py > analysis/07_shapley_output.txt
Value target: 2^8 = 256 coalitions at the 2023 origin, at three model capacities.
Amendment target: 2^7 = 128 coalitions at the 2023 origin (Commonwealth contracts).
Output: analysis/results/S1_coalitions_value.csv, S2_shapley_value.csv, S3_shapley_capacity.csv,
        S4_leave_one_out.csv, S5_interactions.csv, S6_coalitions_amend.csv, S7_shapley_amend.csv,
        S8_cross_target.csv
"""
import itertools, math
import numpy as np, pandas as pd
from scipy import stats
from lib import *

Y = 2023
C = load_contracts()


def coalitions(blocks):
    return [tuple(b for b, k in zip(blocks, bits) if k) for bits in itertools.product([0, 1], repeat=len(blocks))]


def key(S):
    return "+".join(S)


def shapley(blocks, v):
    """v: dict coalition key -> value (empty coalition "" = 0). Returns dict block -> Shapley value."""
    n = len(blocks); phi = {}
    for b in blocks:
        rest = [x for x in blocks if x != b]; tot = 0.0
        for r in range(n):
            wgt = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(rest, r):
                with_b = tuple(x for x in blocks if x in S or x == b)
                tot += wgt * (v[key(with_b)] - v[key(tuple(x for x in blocks if x in S))])
        phi[b] = tot
    return phi


def run(tr, ev, target, blocks, hp, binary, label):
    ytr, yev = tr[target].values.astype(float), ev[target].values.astype(float)
    Xtr, Xev, cols = encode(tr, ev, ytr, {"hist": HIST["hist"]} if "hist" in blocks else {})
    P = {}
    for i, S in enumerate(coalitions(blocks)):
        if S:
            P[key(S)] = fit_predict(Xtr, ytr, Xev, use_cols(cols, S), binary, hp).astype(np.float32)
        if i % 32 == 0: print("   ", label, "coalition", i, flush=True)
    cs = ClusterStats(ev.agency_id.values, yev, P)
    v = cs.di(); v[""] = 0.0
    D = cs.draws(); D[""] = 0.0
    phi = shapley(blocks, v)
    PB = pd.DataFrame([shapley(blocks, row) for row in D.to_dict("records")])
    full = v[key(blocks)]
    out = pd.DataFrame({"block": blocks, "name": [BLOCK_NAMES[b] for b in blocks], "shapley": [phi[b] for b in blocks]})
    out["share"] = out.shapley / full
    out["lo"] = [ci(PB[b])[0] for b in blocks]; out["hi"] = [ci(PB[b])[1] for b in blocks]
    out["holm_p"] = holm([boot_p(PB[b]) for b in blocks])
    out["rank"] = out.shapley.rank(ascending=False).astype(int)
    co = pd.DataFrame({"coalition": [S if S else "(empty)" for S in v], "size": [len(S.split("+")) if S else 0 for S in v], "DI": list(v.values())})
    return out, co, v, full, (Xtr, Xev, cols, ytr, yev)


# ------------------------------------------------------------------ value target, three capacities
tr = sample(C[C.yr.between(Y - 5, Y - 1)], SEED + Y); ev = C[C.yr == Y]
cap = {}
for depth in [6, 4, 3]:
    hp = dict(HP, max_depth=depth)
    out, co, v, full, enc = run(tr, ev, "ly", EIGHT, hp, False, "value depth %d" % depth)
    cap[depth] = (out, full)
    print("\nValue target, depth %d: full-model DI %.4f; Shapley values sum to %.4f" % (depth, full, out.shapley.sum()))
    print(out.round(4).to_string(index=False))
    if depth == 6:
        co.to_csv(RES + "S1_coalitions_value.csv", index=False); out.to_csv(RES + "S2_shapley_value.csv", index=False)
        v6, enc6, sv = v, enc, out
        I = [{"a": BLOCK_NAMES[a], "b": BLOCK_NAMES[b], "interaction": v[key(tuple(x for x in EIGHT if x in (a, b)))] - v[a] - v[b]}
             for a, b in itertools.combinations(EIGHT, 2)]
        pd.DataFrame(I).to_csv(RES + "S5_interactions.csv", index=False)
        marg = (full - v[key(SEVEN)]) / full
        print("   amendment history: Shapley share %.1f%%, marginal share %.1f%%; seven at-publication blocks carry %.1f%%"
              % (100 * out.set_index("block").share["hist"], 100 * marg, 100 * (1 - out.set_index("block").share["hist"])))
K = cap[6][0][["block", "name"]].copy()
for d in [6, 4, 3]:
    K["share_d%d" % d] = cap[d][0].share.values; K["rank_d%d" % d] = cap[d][0]["rank"].values
K.to_csv(RES + "S3_shapley_capacity.csv", index=False)
print("\nShapley shares by capacity (full-model DI: depth 6 %.3f, depth 4 %.3f, depth 3 %.3f)" % (cap[6][1], cap[4][1], cap[3][1]))
print(K.round(4).to_string(index=False))
for d in [4, 3]:
    print("   rank correlation of the partition with depth 6: depth %d %.3f" % (d, stats.spearmanr(K.share_d6, K["share_d%d" % d])[0]))

# ------------------------------------------------------------------ leave-one-out over term and amendment fields
Xtr, Xev, cols, ytr, yev = enc6
allc = use_cols(cols, EIGHT)
P = {"full": fit_predict(Xtr, ytr, Xev, allc)}
fields = ["duration", "has_end", "pub_lag", "ever_amended", "n_amend"]
for f in fields:
    P[f] = fit_predict(Xtr, ytr, Xev, [c for c in allc if c != f])
cs = ClusterStats(ev.agency_id.values, yev, P); v = cs.di(); D = cs.draws()
L = pd.DataFrame({"field": fields, "di_without": [v[f] for f in fields], "loss": [v["full"] - v[f] for f in fields]})
L["lo"] = [ci(D.full - D[f])[0] for f in fields]; L["hi"] = [ci(D.full - D[f])[1] for f in fields]
L["holm_p"] = holm([boot_p(D.full - D[f]) for f in fields])
L.to_csv(RES + "S4_leave_one_out.csv", index=False)
print("\nLeave-one-out at the %d origin (full-model DI %.4f)" % (Y, v["full"])); print(L.round(4).to_string(index=False))

# ------------------------------------------------------------------ amendment target
W = C[C.fed]
tra = sample(W[W.yr.between(Y - 7, Y - 3)], SEED + Y); eva = W[W.yr == Y]
outa, coa, va, fulla, _ = run(tra, eva, "amended24", SEVEN, HP, True, "amendment")
coa.to_csv(RES + "S6_coalitions_amend.csv", index=False); outa.to_csv(RES + "S7_shapley_amend.csv", index=False)
print("\nAmendment target: full-model DI (Brier skill) %.4f" % fulla); print(outa.round(4).to_string(index=False))

# ------------------------------------------------------------------ cross-target agreement
# value-target shares recomputed over the seven at-publication blocks only, for a like-for-like ranking
phi7 = shapley(SEVEN, {S: v6[S] for S in v6 if "hist" not in S.split("+")})
X = pd.DataFrame({"block": SEVEN, "name": [BLOCK_NAMES[b] for b in SEVEN],
                  "share_value": [phi7[b] / v6[key(SEVEN)] for b in SEVEN], "share_amend": outa.share.values})
X["rank_value"] = X.share_value.rank(ascending=False).astype(int); X["rank_amend"] = X.share_amend.rank(ascending=False).astype(int)
X.to_csv(RES + "S8_cross_target.csv", index=False)
six = X[X.block != "term"]
print("\nCross-target agreement"); print(X.round(4).to_string(index=False))
print("   Spearman over seven blocks %.3f | Pearson on shares %.3f | Spearman over six descriptive blocks %.3f"
      % (stats.spearmanr(X.share_value, X.share_amend)[0], stats.pearsonr(X.share_value, X.share_amend)[0],
         stats.spearmanr(six.share_value, six.share_amend)[0]))
