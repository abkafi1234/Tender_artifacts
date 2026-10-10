"""Figures from the result tables of scripts 03 to 11. Written to analysis/figures/.

Run from the project root, after scripts 06 to 11:
    analysis/.venv/bin/python analysis/12_figures.py
The file names follow the figures of the manuscript, so the two sets can be compared one to one.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib import *

OUT = "analysis/figures/"; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, MUTED, GRID, BAND = "#0b0b0b", "#898781", "#e6e5e1", "#f0efec"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})
rd = lambda f, **k: pd.read_csv(RES + f, **k)


def save(fig, name):
    fig.savefig(OUT + name + ".png"); plt.close(fig); print("wrote", name)


def shade(ax, lo=2023.5, hi=2025.5):
    ax.axvspan(lo, hi, color=BAND, zorder=0)


M1, M3 = rd("M1_di_by_origin.csv"), rd("M3_hindsight.csv")
yrs = M1.origin

# ---- D1 volume --------------------------------------------------------------------------------
V = rd("D4_volume_by_year.csv", index_col=0)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.bar(V.index, V["Commonwealth"], color=BLUE, width=0.72, label="Commonwealth")
ax.bar(V.index, V["State and territory"], bottom=V["Commonwealth"], color=ORANGE, width=0.72, label="State and territory", edgecolor="white", linewidth=1)
ax.set_ylabel("Contracts"); ax.set_title("Contracts by year of first publication"); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
save(fig, "D1_volume_by_jurisdiction")

# ---- D2 value distribution ---------------------------------------------------------------------
C = load_contracts()
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.hist(C.ly / np.log(10), bins=80, color=BLUE)
for q in [0.1, 0.9]:
    ax.axvline(C.ly.quantile(q) / np.log(10), color=INK, linewidth=1, linestyle=":")
ax.set_xlabel("Contract value, log10 A$"); ax.set_ylabel("Contracts"); ax.set_title("Distribution of contract value (dotted: 10th and 90th percentiles)")
save(fig, "D2_value_distribution")

# ---- D3 fill rate ------------------------------------------------------------------------------
FY = rd("G8_fill_rate_by_year.csv", index_col=0)
G = FY.drop(columns="index").T
fig, ax = plt.subplots(figsize=(7.0, 3.0))
im = ax.imshow(G.values, aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list("s", SEQ), vmin=0, vmax=1)
ax.set_xticks(range(len(G.columns))); ax.set_xticklabels(G.columns, rotation=45); ax.set_yticks(range(len(G.index))); ax.set_yticklabels([s.replace("_", " ") for s in G.index])
ax.grid(False); fig.colorbar(im, ax=ax, label="Share populated"); ax.set_title("Fill rate of each optional field by year")
save(fig, "D3_fill_rate_heatmap")

# ---- D4 dispersion -----------------------------------------------------------------------------
D5 = rd("D5_target_dispersion_by_year.csv", index_col=0)
fig, ax = plt.subplots(figsize=(5.2, 3.0)); ax.plot(D5.index, D5["std"], color=BLUE, marker="o")
ax.set_ylabel("Standard deviation of log value"); ax.set_title("Dispersion of the value target by year of first publication")
save(fig, "D4_outcome_dispersion")

# ---- D6 notice structure -----------------------------------------------------------------------
D2 = rd("D2_notices_by_year.csv", index_col=0); D6 = rd("D6_revision_by_step.csv", index_col=0)
fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.0))
a.plot(D2.index, 100 * D2.share_amendment_notices, color=BLUE, marker="o", label="Notices that are amendment notices")
a.plot(D2.index, 100 * D2.later_amended, color=ORANGE, marker="o", label="Contracts later amended")
a.set_ylabel("%"); a.set_title("(a) Commonwealth notices and contracts"); a.legend(loc="lower left", fontsize=8)
x = np.arange(len(D6)); w = 0.38
b.bar(x - w / 2, 100 * D6.value_changed, w, color=BLUE, label="Value differs"); b.bar(x + w / 2, 100 * D6.end_date_changed, w, color=ORANGE, label="End date differs")
b.set_xticks(x); b.set_xticklabels(["Parent\nto first", "First\nto second", "Second\nto third", "Later"], fontsize=8); b.set_ylabel("% of steps")
b.set_title("(b) Differences between successive notices"); b.legend(fontsize=8); b.grid(axis="x", visible=False)
save(fig, "D6_notice_structure")

# ---- D7 amendment recording --------------------------------------------------------------------
J = rd("D3_amendment_recording.csv", index_col=0)
fig, ax = plt.subplots(figsize=(5.2, 2.8)); ax.barh(J.index[::-1], 100 * J.rate[::-1], color=BLUE, height=0.6)
for i, (k, r) in enumerate(J[::-1].iterrows()):
    ax.text(100 * r.rate + 0.3, i, "%.1f%%" % (100 * r.rate), va="center", color=INK, fontsize=8)
ax.set_xlabel("Contracts recorded as amended, %"); ax.set_title("Amendment recording by jurisdiction"); ax.grid(axis="y", visible=False)
save(fig, "D7_amendment_recording")

# ---- R1 DI time series -------------------------------------------------------------------------
fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
shade(a); a.fill_between(yrs, M1.all8_lo, M1.all8_hi, color=BLUE, alpha=0.18, linewidth=0)
a.plot(yrs, M1.all8, color=BLUE, marker="o", label="All eight blocks (95% interval)"); a.plot(yrs, M1.di_balanced, color=ORANGE, marker="s", label="Balanced agency panel")
a.set_ylabel("DI"); a.legend(loc="lower right", fontsize=8); a.set_title("DI on contract value by forecast origin (shaded: amendment history right-censored)")
b.plot(FY.index[FY.index >= 2015], FY["index"][FY.index >= 2015], color=AQUA, marker="o"); b.set_ylabel("Compliance index"); b.set_ylim(0, 1)
save(fig, "R1_di_timeseries")

# ---- R2 composition ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.4)); shade(ax)
ax.fill_between(yrs, M1.seven, M1.all8, color=ORANGE, alpha=0.30, linewidth=0); ax.fill_between(yrs, M1.six, M1.seven, color=AQUA, alpha=0.30, linewidth=0)
for col, c, lab in [("all8", ORANGE, "All eight blocks"), ("seven", AQUA, "Seven at-publication blocks"), ("six", BLUE, "Six descriptive blocks")]:
    ax.plot(yrs, M1[col], color=c, marker="o", label=lab)
ax.set_ylabel("DI"); ax.legend(loc="lower right", fontsize=8); ax.set_title("DI by field set and forecast origin")
save(fig, "R2_composition")

# ---- R3 hindsight ------------------------------------------------------------------------------
L = rd("S4_leave_one_out.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.2)); shade(a)
a.plot(M3.origin, 100 * M3.ceiling, color=ORANGE, marker="o", label="Ceiling"); a.plot(M3.origin, 100 * M3.floor, color=BLUE, marker="o", label="Floor")
a.plot(M3.origin, 100 * M3.floor24, color=AQUA, marker="s", linestyle="--", label="Floor at equal exposure")
a.set_ylabel("Hindsight share, % of DI"); a.legend(fontsize=8); a.set_title("(a) Floor and ceiling by forecast origin")
b.barh(L.field[::-1], L.loss[::-1], color=BLUE, height=0.6, xerr=[(L.loss - L.lo)[::-1], (L.hi - L.loss)[::-1]], ecolor=MUTED)
b.set_xlabel("Loss in DI when the field is removed"); b.set_title("(b) Leave-one-out, 2023 origin"); b.grid(axis="y", visible=False)
save(fig, "R3_hindsight")

# ---- R4 jurisdictions and award procedure ------------------------------------------------------
G4, G5 = rd("G4_by_jurisdiction.csv"), rd("G5_by_award_procedure.csv")
F = pd.concat([G4.assign(kind="Jurisdiction"), G5.assign(kind="Award procedure")]).reset_index(drop=True)
fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=True); y = np.arange(len(F))[::-1]
a.errorbar(F.DI, y, xerr=[F.DI - F.lo, F.hi - F.DI], fmt="o", color=BLUE, ecolor=MUTED); a.set_yticks(y); a.set_yticklabels(F.group); a.set_xlabel("DI (95% interval)")
b.plot(F.resid_sd, y, "o", color=ORANGE); b.set_xlabel("Residual standard deviation"); a.set_title("DI by jurisdiction and award procedure"); b.set_title("Residual dispersion")
save(fig, "R4_regime_forest")

# ---- R5 learners -------------------------------------------------------------------------------
R1 = rd("R1_learners.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for col, c, lab, ls in [("boosted_depth6", BLUE, "Boosted trees, depth 6 (main)", "-"), ("boosted_depth2", ORANGE, "Boosted trees, depth 2", "-"),
                        ("random_forest_nontext", AQUA, "Random forest, non-text blocks", "--"), ("penalised_linear", YELLOW, "Penalised linear", "-")]:
    ax.plot(R1.origin, R1[col], color=c, marker="o", linestyle=ls, label=lab)
ax.set_ylabel("DI"); ax.legend(fontsize=8); ax.set_title("DI under four learners")
save(fig, "R5_learner_invariance")

# ---- R7 dose response --------------------------------------------------------------------------
G6 = rd("G6_dose_response.csv")
fig, ax = plt.subplots(figsize=(5.0, 3.0)); x = np.arange(len(G6))
ax.errorbar(x, G6.DI, yerr=[G6.DI - G6.lo, G6.hi - G6.DI], fmt="o-", color=BLUE, ecolor=MUTED); ax.set_xticks(x); ax.set_xticklabels(G6.group)
ax.set_xlabel("Optional fields populated"); ax.set_ylabel("DI (95% interval)"); ax.set_title("DI by number of optional fields populated")
save(fig, "R7_dose_response")

# ---- R8 amendment forecast, R9 calibration -----------------------------------------------------
T1, T5 = rd("T1_amendment_by_origin.csv"), rd("T5_calibration.csv")
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
ax[0].plot(T1.origin, T1.auc, color=BLUE, marker="o", label="AUC"); ax[0].plot(T1.origin, T1.sentinel_auc, color=MUTED, marker="s", linestyle="--", label="Permuted target")
ax[0].set_title("AUC"); ax[0].legend(fontsize=8)
ax[1].fill_between(T1.origin, T1.di_lo, T1.di_hi, color=BLUE, alpha=0.18, linewidth=0); ax[1].plot(T1.origin, T1.DI, color=BLUE, marker="o"); ax[1].set_title("DI (Brier-skill form, 95% interval)")
ax[2].plot(T1.origin, 100 * T1.base_rate, color=ORANGE, marker="o"); ax[2].set_title("Base rate, %")
save(fig, "R8_di_amendment")
fig, (a, b) = plt.subplots(1, 2, figsize=(8.0, 2.9))
a.plot(T5.origin, T5.slope_raw, color=BLUE, marker="o", label="Before Platt scaling"); a.plot(T5.origin, T5.slope_platt, color=ORANGE, marker="o", label="After")
a.axhline(1, color=INK, linewidth=1, linestyle=":"); a.set_title("Calibration slope"); a.legend(fontsize=8)
b.bar(T5.origin, T5.change, color=BLUE, width=0.6); b.axhline(0, color=INK, linewidth=1); b.set_title("Change in DI after Platt scaling"); b.grid(axis="x", visible=False)
save(fig, "R9_calibration")

# ---- R10 portal test ---------------------------------------------------------------------------
G2, G3 = rd("G2_field_availability.csv").head(6), rd("G3_jurisdiction.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.0)); y = np.arange(len(G2))[::-1]; h = 0.38
a.barh(y + h / 2, 100 * G2["Commonwealth"], h, color=BLUE, label="Commonwealth"); a.barh(y - h / 2, 100 * G2["State and territory"], h, color=ORANGE, label="State and territory")
a.set_yticks(y); a.set_yticklabels([s.replace("_", " ") for s in G2.field], fontsize=8); a.set_xlabel("Populated, %"); a.legend(fontsize=8, loc="lower right"); a.set_title("Field availability"); a.grid(axis="y", visible=False)
x = np.arange(2)
for i, (g, c) in enumerate([("Commonwealth", BLUE), ("State and territory", ORANGE)]):
    s = G3[G3.group == g]; b.bar(x + (i - 0.5) * 0.38, s.DI, 0.36, color=c, label=g)
b.set_xticks(x); b.set_xticklabels(["Full field set", "Common core"]); b.set_ylabel("DI"); b.set_title("Jurisdiction comparison"); b.grid(axis="x", visible=False)
save(fig, "R10_portal_test")

# ---- E1, E2 Shapley, E4 interactions -----------------------------------------------------------
S2 = rd("S2_shapley_value.csv").sort_values("shapley")
col = [ORANGE if b == "hist" else BLUE for b in S2.block]
fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.barh(S2.name, S2.shapley, color=col, height=0.62, xerr=[S2.shapley - S2.lo, S2.hi - S2.shapley], ecolor=MUTED)
for i, r in enumerate(S2.itertuples()): ax.text(r.hi + 0.004, i, "%.1f%%" % (100 * r.share), va="center", fontsize=8, color=INK)
ax.set_xlabel("Shapley value (DI)"); ax.set_title("Shapley contribution by block, value target\n(blue: at publication; orange: post-award)"); ax.grid(axis="y", visible=False)
save(fig, "E1_shapley_partition")
W = S2[::-1].reset_index(drop=True); cum = W.shapley.cumsum()
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.bar(W.name, W.shapley, bottom=cum - W.shapley, color=[ORANGE if b == "hist" else BLUE for b in W.block], width=0.62)
ax.axhline(cum.iloc[-1], color=INK, linewidth=1, linestyle=":"); ax.text(len(W) - 0.6, cum.iloc[-1], " full model %.3f" % cum.iloc[-1], fontsize=8, va="bottom", ha="right")
ax.set_xticklabels(W.name, rotation=30, ha="right", fontsize=8); ax.set_ylabel("Cumulative DI"); ax.set_title("Shapley values as a cumulative build-up"); ax.grid(axis="x", visible=False)
save(fig, "E2_shapley_waterfall")
I = rd("S5_interactions.csv"); names = list(dict.fromkeys(list(I.a) + list(I.b)))
Mx = pd.DataFrame(np.nan, index=names, columns=names)
for r in I.itertuples(): Mx.loc[r.a, r.b] = Mx.loc[r.b, r.a] = r.interaction
lim = np.nanmax(np.abs(Mx.values))
fig, ax = plt.subplots(figsize=(5.6, 4.2))
im = ax.imshow(Mx.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("d", ["#e34948", "#f0efec", "#2a78d6"]), vmin=-lim, vmax=lim)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.grid(False); fig.colorbar(im, ax=ax, label="DI of pair minus DI of each alone"); ax.set_title("Pairwise interaction between blocks (red: redundancy)")
save(fig, "E4_block_interactions")

# ---- E5 confound diagnostic --------------------------------------------------------------------
ag = rd("G9_agencies.csv")
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
for a, col, lab in zip(ax, ["DI", "resid_sd", "DI_adj"], ["(a) DI", "(b) Residual dispersion", "(c) Adjusted DI"]):
    a.scatter(ag.total_sd, ag[col], s=14, color=BLUE, edgecolor="white", linewidth=0.5); a.set_xlabel("Dispersion of the agency's contract values"); a.set_title(lab)
save(fig, "E5_confound_diagnostic")

# ---- E6 Shapley amendment, E7 cross-target -----------------------------------------------------
S7 = rd("S7_shapley_amend.csv").sort_values("shapley")
fig, ax = plt.subplots(figsize=(5.6, 2.9))
ax.barh(S7.name, S7.shapley, color=BLUE, height=0.62, xerr=[S7.shapley - S7.lo, S7.hi - S7.shapley], ecolor=MUTED)
ax.set_xlabel("Shapley value (DI, Brier-skill form)"); ax.set_title("Shapley contribution by block, amendment target"); ax.grid(axis="y", visible=False)
save(fig, "E6_shapley_amendment")
S8 = rd("S8_cross_target.csv").sort_values("share_value"); y = np.arange(len(S8)); h = 0.38
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.barh(y + h / 2, 100 * S8.share_value, h, color=BLUE, label="Value target"); ax.barh(y - h / 2, 100 * S8.share_amend, h, color=ORANGE, label="Amendment target")
ax.set_yticks(y); ax.set_yticklabels(S8.name); ax.set_xlabel("Share of DI, %"); ax.legend(fontsize=8, loc="lower right"); ax.set_title("Shapley share of each at-publication block under the two targets"); ax.grid(axis="y", visible=False)
save(fig, "E7_cross_outcome_agreement")
print("done")
