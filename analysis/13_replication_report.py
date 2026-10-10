"""Set each value printed in the paper beside the value produced by the scripts in this folder.

Run from the project root, after scripts 03 to 11:
    analysis/.venv/bin/python analysis/13_replication_report.py
Output: analysis/replication_report.md and analysis/results/Z_replication.csv
"""
import numpy as np, pandas as pd
from scipy import stats
from lib import *

rd = lambda f, **k: pd.read_csv(RES + f, **k)
M1, M2, T1, T2, T3 = rd("M1_di_by_origin.csv"), rd("M2_trend_tests.csv").set_index("series"), rd("T1_amendment_by_origin.csv"), rd("T2_duration_test.csv"), rd("T3_no_gap.csv")
S2, S3, S4, S7, S8 = rd("S2_shapley_value.csv").set_index("block"), rd("S3_shapley_capacity.csv"), rd("S4_leave_one_out.csv").set_index("field"), rd("S7_shapley_amend.csv").set_index("block"), rd("S8_cross_target.csv")
G3, GC, G4, G9, T4 = rd("G3_jurisdiction.csv"), rd("G1_contrasts.csv").set_index("test"), rd("G4_by_jurisdiction.csv").set_index("group"), rd("G9_compliance_agencies.csv").set_index("association"), rd("T4_strata.csv").set_index("stratum")
R1, R2, B, A, Cf, D, E2, F = rd("R1_learners.csv"), rd("R2_placeholders.csv"), rd("B_notice_frame.csv"), rd("A_contract_frame.csv"), rd("C_first_published_target.csv"), rd("D_unseen_supplier.csv"), rd("E2_duration_first_published.csv"), rd("F_term_by_amendment_status.csv")
D1, D2 = rd("D1_sample.csv").set_index("step").records, rd("D2_notices_by_year.csv", index_col=0)
Fu = M1[M1.origin.isin(FULL)]; o = M1.set_index("origin")
full = G3[G3.spec == "Full field set"].set_index("group"); six8 = S8[S8.block != "term"]
Fs = F.groupby("group")[["sse6", "sse7"]].sum(); red = Fs.sse6 - Fs.sse7
PV = pd.read_pickle(CACHE + "preds_value.pkl"); err = (PV.y - PV.p_all8).abs()

import re
JD = re.search(r"jurisdiction difference with placeholders retained: (\S+)", open("analysis/10_robustness_output.txt").read()).group(1)
rows = []
def add(section, quantity, manuscript, here, fmt="%.3f"):
    rows.append({"Section": section, "Quantity": quantity, "Paper": manuscript, "This code": (fmt % here) if isinstance(here, (int, float, np.floating, np.integer)) else here})
pct = lambda x: "%.1f%%" % (100 * x)

add("3.1", "Contracts analysed", "485,068", "%s" % format(int(D1["Contracts analysed"]), ","))
add("3.1", "Placeholder values excluded", "431", "%d" % D1["recorded value above zero and at most A$1 (placeholders, excluded)"])
add("3.1", "Agencies", "813", "%d" % D1["Agencies"])
add("3.6", "Contracts first published in 2025 already amended", "7.4%", pct(D2.later_amended[2025]))
add("4.1", "DI, all eight blocks: 2015, peak, 2025", "0.556, 0.618 (2022), 0.582", "%.3f, %.3f (%d), %.3f" % (o.all8[2015], M1.all8.max(), M1.origin[M1.all8.idxmax()], o.all8[2025]))
add("4.1", "Mean DI, all eight blocks", "0.577", M1.all8.mean())
add("4.1", "Trend in DI, Mann-Kendall p", "0.008", M2.mk_p["DI, all eight blocks"])
add("4.1", "Total SD of target, 2015 to 2025", "1.54 to 1.71", "%.2f to %.2f" % (o.total_sd[2015], o.total_sd[2025]))
add("4.1", "Trend in total SD, p", "0.0006", M2.mk_p["Total SD of target"], "%.4f")
add("4.1", "Residual SD: range; trend p", "1.03 to 1.10; 0.350", "%.2f to %.2f; %.3f" % (M1.all8_rsd.min(), M1.all8_rsd.max(), M2.mk_p["Residual SD, all eight blocks"]))
add("4.1", "Last three origins minus first three (95% CI; p)", "+0.042 (-0.008 to +0.072; 0.113)", "%+.3f (%+.3f to %+.3f; %.3f)" % tuple(M2.loc["DI all eight: last three origins minus first three", ["tau", "ci_lo", "ci_hi", "mk_p"]]))
add("4.1", "Balanced agency panel, trend p", "0.276", M2.mk_p["DI, balanced agency panel"])
add("4.2", "Floor: mean (range)", "4.4% (3.0% to 6.9%)", "%s (%s to %s)" % (pct(Fu.floor.mean()), pct(Fu.floor.min()), pct(Fu.floor.max())))
add("4.2", "Floor at equal exposure: mean; first to last", "3.8%; 4.2% to 2.9%", "%s; %s to %s" % (pct(Fu.floor24.mean()), pct(Fu.floor24.iloc[0]), pct(Fu.floor24.iloc[-1])))
add("4.2", "Ceiling: mean (range)", "30.0% (24.7% to 35.7%)", "%s (%s to %s)" % (pct(Fu.ceiling.mean()), pct(Fu.ceiling.min()), pct(Fu.ceiling.max())))
add("4.2", "Floors at censored origins, 2024 and 2025", "1.8% and -0.5%", "%s and %s" % (pct(o.floor[2024]), pct(o.floor[2025])))
add("4.2", "Trend in equal-exposure floor, p; in ceiling, p", "0.048; 0.001", "%.3f; %.3f" % (M2.mk_p["Floor at equal exposure"], M2.mk_p["Ceiling"]))
add("4.2", "Increment of the term block, first to last fully observed origin", "0.175 to 0.117", "%.3f to %.3f" % (Fu.term_increment.iloc[0], Fu.term_increment.iloc[-1]))
add("4.2", "Share of the term block's error reduction on never-amended contracts", "61%", pct(red["never amended"] / red["all"]))
add("4.2", "Floor on the first-published value", "1.5%", pct(Cf.floor_firstpub.mean()))
add("4.3", "Six descriptive blocks, notice frame and contract frame", "0.477 and 0.403", "%.3f and %.3f" % (B.six.mean(), A.six.mean()))
add("4.3", "Hindsight share in the notice frame, duration post-award", "25.9%", pct(B.share_flag_and_term.mean()))
add("4.4", "Shapley share, contract term", "27.6%", pct(S2.share["term"]))
add("4.4", "Shapley share, amendment history; marginal share", "9.5%; 3.0%", "%s; %s" % (pct(S2.share["hist"]), pct(o.floor[2023])))
add("4.4", "Seven at-publication blocks together", "90.5%", pct(1 - S2.share["hist"]))
add("4.4", "Blocks ranked second to sixth, each", "free text second; four blocks 10% to 12%", "; ".join("%s %s" % (r["name"], pct(r.share)) for _, r in S2.sort_values("share", ascending=False).iloc[1:6].iterrows()))
add("4.5", "DI, Commonwealth and state and territory", "0.557 and 0.388", "%.3f and %.3f" % (full.DI["Commonwealth"], full.DI["State and territory"]))
add("4.5", "Difference (95% CI)", "+0.168 (0.099 to 0.241)", "%+.3f (%.3f to %.3f)" % tuple(GC.loc["Jurisdiction, full field set", ["diff", "lo", "hi"]]))
add("4.5", "Residual SD, Commonwealth and state and territory", "1.089 and 1.087", "%.3f and %.3f" % (full.resid_sd["Commonwealth"], full.resid_sd["State and territory"]))
add("4.5", "Total SD, Commonwealth and state and territory", "1.638 and 1.401", "%.3f and %.3f" % (full.total_sd["Commonwealth"], full.total_sd["State and territory"]))
add("4.5", "Shrinkage of the difference on the common core", "4.5%", pct(1 - GC["diff"]["Jurisdiction, common core"] / GC["diff"]["Jurisdiction, full field set"]))
add("4.5", "New South Wales: DI; residual SD; total SD", "0.239; 1.045; 1.206", "%.3f; %.3f; %.3f" % (G4.DI["NSW"], G4.resid_sd["NSW"], G4.total_sd["NSW"]))
add("4.5", "Open minus limited award procedure", "+0.044 (Holm p 0.244)", "%+.3f (Holm p %.3f)" % tuple(GC.loc["Award procedure", ["diff", "holm_p"]]))
add("4.5", "Compliance and adjusted DI across agencies", "+0.058 (p 0.631), 71 agencies", "%+.3f (p %.3f), %d agencies" % (G9.estimate["Compliance and adjusted DI (primary)"], G9.p["Compliance and adjusted DI (primary)"], len(rd("G9_agencies.csv"))))
add("4.5", "Compliance and raw DI across agencies", "+0.329 (p 0.005)", "%+.3f (p %.3f)" % (G9.estimate["Compliance and DI (raw)"], G9.p["Compliance and DI (raw)"]))
add("4.5", "Best minus worst documented contracts", "+0.089 (p 0.039)", "%+.3f (p %.3f)" % tuple(GC.loc["Completeness, best minus worst documented", ["diff", "p"]]))
add("4.6", "AUC, six descriptive blocks; with recorded duration", "0.755; 0.833", "%.3f; %.3f" % (T2.auc_six.mean(), T2.auc_six_duration.mean()))
add("4.6", "AUC, seven blocks: range (mean); trend p", "0.819 to 0.863 (0.838); 0.230", "%.3f to %.3f (%.3f); %.3f" % (T1.auc.min(), T1.auc.max(), T1.auc.mean(), mann_kendall(T1.origin, T1.auc)[1]))
add("4.6", "Gain in AUC without the training gap: range (mean)", "0.011 to 0.034 (0.023)", "%.3f to %.3f (%.3f)" % (T3.difference.min(), T3.difference.max(), T3.difference.mean()))
add("4.6", "Gain from duration; on contracts with one end date; share of contracts", "+0.077; +0.077; 98.7% to 99.9%", "%+.3f; %+.3f; %s to %s" % (T2.gain.mean(), T2.gain_same_end.mean(), pct(T2.share_same_end.min()), pct(T2.share_same_end.max())))
add("4.6", "Gain from first-published, revised and overwritten duration", "-0.028; +0.036; +0.064", "%+.3f; %+.3f; %+.3f" % tuple(E2.gain))
add("4.6", "Shapley share of the contract term on the amendment target", "60.2%", pct(S7.share["term"]))
add("4.6", "Cross-target Spearman: seven blocks; six descriptive blocks", "0.929; 0.886", "%.3f; %.3f" % (stats.spearmanr(S8.share_value, S8.share_amend)[0], stats.spearmanr(six8.share_value, six8.share_amend)[0]))
add("4.6", "Buying agency: share on value target; on amendment target", "11.4%; none", "%s; %s" % (pct(S2.share["agency"]), pct(S7.share["agency"])))
add("4.7", "Permuted-target check, largest absolute DI (value target)", "0.006", M1.sentinel.abs().max())
add("4.7", "With placeholders: trend p; jurisdiction difference; floor; ceiling", "0.756; +0.288; 4.6%; 29.8%", "%.3f; %s; %s; %s" % (mann_kendall(R2.origin, R2.all8)[1], JD, pct(R2[R2.origin.isin(FULL)].floor.mean()), pct(R2[R2.origin.isin(FULL)].ceiling.mean())))
add("4.7", "DI for suppliers absent from training; present", "0.583; 0.567", "%.3f; %.3f" % (D.DI[1], D.DI[0]))
add("App. B", "Typical multiplicative error; predictions within a factor of two", "x/÷ 2.20; 56.0%", "x/÷ %.2f; %s" % (np.exp(err.mean()), pct((err <= np.log(2)).mean())))
add("App. B", "Leave-one-out loss for contract duration (2023)", "0.0824", S4.loss["duration"], "%.4f")
add("App. C", "Calibration slopes of the amendment model", "0.72 to 0.94", "%.2f to %.2f" % (T1.cal_slope.min(), T1.cal_slope.max()))
add("App. C", "Open and limited procedure: base rate; AUC", "30.2% and 10.6%; 0.820 and 0.815", "%s and %s; %.3f and %.3f" % (pct(T4.base_rate["open"]), pct(T4.base_rate["limited"]), T4.auc["open"], T4.auc["limited"]))
add("App. C", "Depth 6 against shallow model: Pearson; rank correlation", "0.991; 0.982", "%.3f; %.3f" % (stats.pearsonr(R1.boosted_depth6, R1.boosted_depth2)[0], stats.spearmanr(R1.boosted_depth6, R1.boosted_depth2)[0]))
add("App. C", "Penalised linear model lower by", "0.08 to 0.11", "%.2f to %.2f" % ((R1.boosted_depth6 - R1.penalised_linear).min(), (R1.boosted_depth6 - R1.penalised_linear).max()))
add("App. C", "Contract term share at depths 6 and 3; amendment history", "27.6% to 28.0%; 9.5% to 9.7%", "%s to %s; %s to %s" % (pct(S3.set_index("block").share_d6["term"]), pct(S3.set_index("block").share_d3["term"]), pct(S3.set_index("block").share_d6["hist"]), pct(S3.set_index("block").share_d3["hist"])))

Z = pd.DataFrame(rows); Z.to_csv(RES + "Z_replication.csv", index=False)
with open("analysis/replication_report.md", "w") as f:
    f.write("# Reported and reproduced values\n\n")
    f.write("Each value printed in the paper beside the value produced by the scripts in this folder. Generated by `13_replication_report.py`.\n\n")
    f.write("| Section | Quantity | Paper | This code |\n|---|---|---|---|\n")
    for r in rows:
        f.write("| %s | %s | %s | %s |\n" % (r["Section"], r["Quantity"], r["Paper"], r["This code"]))
print(Z.to_string(index=False))
