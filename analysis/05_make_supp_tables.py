"""Write the LaTeX fragments for the supplementary analyses into Manuscript_new/tables/.

Run from the project root, after 02, 03 and 04:
    analysis/.venv/bin/python analysis/05_make_supp_tables.py
"""
import numpy as np, pandas as pd
OUT = "Manuscript_new/tables/"
R = "analysis/results/"
pc = lambda x, d=1: f"{100*x:.{d}f}\\%"
com = lambda x: f"{int(x):,}".replace(",", "{,}")

def emit(name, caption, label, colspec, header, rows, note=None):
    L = [r"\begin{table}[H]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\small",
         r"\setlength{\tabcolsep}{3.4pt}", f"\\begin{{tabular}}{{{colspec}}}", r"\toprule",
         " & ".join(f"\\textbf{{{h}}}" for h in header) + r" \\", r"\midrule"]
    L += [r if isinstance(r, str) else " & ".join(r) + r" \\" for r in rows]
    L += [r"\bottomrule", r"\end{tabular}"]
    if note: L.append(r"\par\vspace{2pt}\noindent{\footnotesize " + note + r"\par}")
    L.append(r"\end{table}")
    open(OUT + name + ".tex", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("wrote", name)

# 1. end-date change rate by year of the later notice
N = pd.read_pickle("analysis/cache/notices.pkl")
par = set(N[(N.k == 0) & N.fed].recid)
am = N[(N.k > 0) & N.parent.isin(par)].sort_values(["parent", "k"]).copy()
g = am.groupby("parent"); am["pk"] = g.k.shift(1); am["pe"] = g.contract_end.shift(1)
st = am[(am.k > 1) & (am.pk == am.k - 1) & am.contract_end.notna() & am.pe.notna()].copy()
st["chg"] = st.contract_end != st.pe
rows = []
for lab, m in [("2010 to 2014", st.yr.between(2010, 2014)), ("2015 to 2018", st.yr.between(2015, 2018)),
               ("2019 to 2022", st.yr.between(2019, 2022)), ("2023 to 2024", st.yr.between(2023, 2024)),
               ("2025, first to third quarter", (st.yr == 2025) & (st.published_date.dt.quarter < 4)),
               ("2025, fourth quarter", (st.yr == 2025) & (st.published_date.dt.quarter == 4))]:
    rows.append([lab, com(m.sum()), com(st[m].chg.sum()), pc(st[m].chg.mean())])
emit("tab_revyear",
     "End-date differences between consecutive amendment notices of one contract, by the year in which the later "
     "notice was published. The rate is zero for early notices and rises steeply for the most recent, which "
     "indicates that end dates on earlier notices have been overwritten with the current date.",
     "tab:revyear", "lrrr", ["Later notice published", "Steps", "End date differs", "Rate"], rows,
     note="Commonwealth contracts; consecutive amendment notices with an end date on both. Computed directly "
          "from the export (Appendix~\\ref{app:supp}).")

# 2. two by two
A, B = pd.read_csv(R + "A_contract_frame.csv"), pd.read_csv(R + "B_notice_frame.csv")
emit("tab_twobytwo",
     "Hindsight share by unit of analysis and by the timing assigned to recorded duration, mean over the fully "
     "observed origins. The classification of duration moves the share; the unit of analysis does not.",
     "tab:twobytwo", "lrrr",
     ["Unit of analysis", "Duration at publication", "Duration post-award", "DI, six blocks"],
     [["Notice", pc(B.share_flag_only.mean()), pc(B.share_flag_and_term.mean()), f"{B.six.mean():.3f}"],
      ["Contract", pc(A.floor.mean()), pc(A.ceiling.mean()), f"{A.six.mean():.3f}"]],
     note="Independent re-implementation (Appendix~\\ref{app:supp}). In the notice frame the post-award content is "
          "the amendment indicator of the export; in the contract frame it is the amendment history.")

# 3. duration as first published
E = pd.read_csv(R + "E2_duration_first_published.csv")
lab = ["Parent not overwritten; duration as first published", "Same contracts; duration after first amendment",
       "Parent overwritten; recorded duration"]
emit("tab_durfp",
     "Contribution of duration to the amendment forecast when the first-published duration is observable. Contracts "
     "first published in 2024 and 2025; the outcome is an amendment notice published by the end of 2025. Each "
     "evaluation set contains every contract not amended and the amended contracts described.",
     "tab:durfp", "p{5.0cm}rrrrr",
     ["Amended contracts in evaluation set", "$n$", "Six", "+ dur.", "Gain", "95\\% CI"],
     [[l, com(r.n_amended), f"{r.auc_six:.3f}", f"{r.auc_six_duration:.3f}", f"\\textbf{{{r.gain:+.3f}}}",
       f"[{r.lo:+.3f}, {r.hi:+.3f}]"] for l, r in zip(lab, E.itertuples())],
     note="Six: AUC of the model with the six descriptive blocks. + dur.: AUC with duration added. Models trained on contracts published 2019 to 2023 with the 24-month outcome. Bootstrap interval "
          "over evaluation contracts, 500 replicates. Independent re-implementation (Appendix~\\ref{app:supp}).")

# 4. replication
E1 = pd.read_csv(R + "E1_amendment_auc.csv")
emit("tab_replication",
     "Headline quantities in the main analysis and in the independent re-implementation.",
     "tab:replication", "p{7.6cm}rr", ["Quantity", "Main analysis", "Re-implementation"],
     [["DI, all eight blocks, mean 2015 to 2023", "0.575", f"{A.all8.mean():.3f}"],
      ["DI, six descriptive blocks, contract frame", "0.403", f"{A.six.mean():.3f}"],
      ["DI, six descriptive blocks, notice frame", "0.477", f"{B.six.mean():.3f}"],
      ["Hindsight share, floor", "4.4\\%", pc(A.floor.mean())],
      ["Hindsight share, ceiling", "30.0\\%", pc(A.ceiling.mean())],
      ["Hindsight share, notice frame, duration post-award", "25.9\\%", pc(B.share_flag_and_term.mean())],
      ["Amendment AUC, six blocks, mean 2017 to 2023", "0.755", f"{E1.auc_six.mean():.3f}"],
      ["Amendment AUC, six blocks and duration", "0.833", f"{E1.auc_six_duration.mean():.3f}"]])

# 5. term block by amendment status
F = pd.read_csv(R + "F_term_by_amendment_status.csv").groupby("group")[["n", "sse6", "sse7", "sst"]].sum()
red = F.loc["all"].sse6 - F.loc["all"].sse7
rows = []
for k, lab in [("never amended", "Never amended"), ("later amended", "Later amended"), ("all", "All contracts")]:
    r = F.loc[k]
    rows.append([lab, com(r.n), f"{1 - r.sse6 / r.sst:.3f}", f"{1 - r.sse7 / r.sst:.3f}",
                 f"{np.sqrt(r.sse6 / r.n):.3f}", f"{np.sqrt(r.sse7 / r.n):.3f}", pc((r.sse6 - r.sse7) / red)])
emit("tab_termstatus",
     "Contribution of the contract-term block to the value forecast, by whether the contract was later amended, pooled "
     "over the 2015 to 2023 origins. The last column is the share of the block's reduction in squared error.",
     "tab:termstatus", "lrrrrrr",
     ["Contracts", "$n$", "DI six", "DI seven", "Res.\\ SD six", "Res.\\ SD seven", "Share"], rows,
     note="Six: descriptive blocks. Seven: with the contract term. Independent re-implementation.")

# 6. unseen suppliers
D = pd.read_csv(R + "D_unseen_supplier.csv")
emit("tab_unseen",
     "Value forecast from the seven at-publication blocks, by whether the supplier appears in the training sample, "
     "pooled over the 2015 to 2023 origins.",
     "tab:unseen", "lrrrr", ["Evaluation contracts", "$n$", "DI", "Total SD", "Residual SD"],
     [[r.group, com(r.n), f"{r.DI:.3f}", f"{r.total_sd:.3f}", f"{r.resid_sd:.3f}"] for r in D.itertuples()],
     note="Independent re-implementation.")

C = pd.read_csv(R + "C_first_published_target.csv")
print("first-published target: floor current %.1f%% (range %.1f-%.1f) | first-published %.1f%% (range %.1f-%.1f)" % (
    100*C.floor_current.mean(), 100*C.floor_current.min(), 100*C.floor_current.max(),
    100*C.floor_firstpub.mean(), 100*C.floor_firstpub.min(), 100*C.floor_firstpub.max()))
print("term share on amended: %.3f | narrowed ceiling (re-impl.): %.1f%% | with manuscript floor and ceiling: %.1f%%" % (
    (F.loc['later amended'].sse6-F.loc['later amended'].sse7)/red,
    100*(A.floor.mean()+ (A.ceiling.mean()-A.floor.mean())*(F.loc['later amended'].sse6-F.loc['later amended'].sse7)/red),
    4.4+(30.0-4.4)*(F.loc['later amended'].sse6-F.loc['later amended'].sse7)/red))
