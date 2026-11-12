"""Điền số liệu từ results/ vào reports/main_template.tex để sinh reports/main.tex.

    python src/make_report.py && cd reports && xelatex main.tex && xelatex main.tex

Văn bản nằm trong reports/main_template.tex (chỗ có @@TÊN@@ sẽ được thay bằng số thật).
Sau khi chạy lại thí nghiệm, chạy lại script này để báo cáo luôn khớp với results/.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, TARGET, load_data  # noqa: E402
from features import add_features  # noqa: E402
from models import MODEL_NAMES, MODELS  # noqa: E402

RES, REP = ROOT / "results", ROOT / "reports"


def esc(s):
    return str(s).replace("\\", "").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def n(x, d=3):
    """Số kiểu Việt Nam (dấu phẩy thập phân)."""
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=1):
    return n(100 * x, d) + r"\%"


def psum(model, params_json):
    p = json.loads(params_json)
    if model == "knn":
        return f"k={p['n_neighbors']}, {p['metric']}, {p['weights']}"
    if model == "dt":
        cw = "balanced" if p.get("class_weight") == "balanced" else "none"
        d = p.get("max_depth")
        return f"depth={'None' if d is None else int(d)}, {p['criterion']}, leaf={p['min_samples_leaf']}, cw={cw}"
    if model == "nb":
        return f"var\\_smoothing={p['var_smoothing']:g}"
    return ""


def dparams(model, params_json):
    p = json.loads(params_json)
    if model == "knn":
        return "k=5 (mặc định)"
    if model == "dt":
        return "mặc định"
    return "mặc định"


final = pd.read_csv(RES / "final.csv")
early = pd.read_csv(RES / "early_warning.csv")
info = json.load(open(RES / "run_info.json", encoding="utf-8"))
prof = pd.read_csv(RES / "error_profile.csv")
pair = pd.read_csv(RES / "confusion_best.csv", index_col=0)
imp = pd.read_csv(RES / "dt_B_importance_top20.csv")
df = load_data()


def g(stage, model, flow, col):
    r = final[(final.stage == stage) & (final.model == model) & (final.flow == flow)]
    return float(r[col].iloc[0])


def gp(stage, model, flow):
    r = final[(final.stage == stage) & (final.model == model) & (final.flow == flow)]
    return r["params"].iloc[0]


R = {}
# ---------------- dữ liệu / EDA
R["N"] = f"{len(df):,}".replace(",", ".")
R["NTR"] = f"{info['n_train']:,}".replace(",", ".")
R["NTE"] = f"{info['n_test']:,}".replace(",", ".")
vc = df[TARGET].value_counts(normalize=True)
for c in ["Dropout", "Enrolled", "Graduate"]:
    R[f"P_{c}"] = pct(vc[c])
    R[f"C_{c}"] = f"{(df[TARGET] == c).sum():,}".replace(",", ".")


def drop_rate(col, val):
    return pct((df[df[col] == val][TARGET] == "Dropout").mean())


R["FEE0"], R["FEE1"] = drop_rate("Tuition fees up to date", 0), drop_rate("Tuition fees up to date", 1)
R["DEB1"], R["DEB0"] = drop_rate("Debtor", 1), drop_rate("Debtor", 0)
R["SCH1"], R["SCH0"] = drop_rate("Scholarship holder", 1), drop_rate("Scholarship holder", 0)
R["GEN1"], R["GEN0"] = drop_rate("Gender", 1), drop_rate("Gender", 0)
tmp = df.assign(tc=df[TARGET].map({"Dropout": 0, "Enrolled": 1, "Graduate": 2}))
cor = tmp.drop(columns=TARGET).corr(numeric_only=True)["tc"].drop("tc")
R["COR_A2"] = n(cor["Curricular units 2nd sem (approved)"], 2)
R["COR_G2"] = n(cor["Curricular units 2nd sem (grade)"], 2)
R["COR_A1"] = n(cor["Curricular units 1st sem (approved)"], 2)
R["COR_G1"] = n(cor["Curricular units 1st sem (grade)"], 2)
R["COR_AGE"] = n(cor["Age at enrollment"], 2)
g1, e1 = "Curricular units 1st sem (grade)", "Curricular units 1st sem (enrolled)"
z = df[df[g1] == 0]
R["Z_G_N"] = str(len(z)); R["Z_G_DROP"] = pct((z[TARGET] == "Dropout").mean())
z = df[df[e1] == 0]
R["Z_E_N"] = str(len(z)); R["Z_E_DROP"] = pct((z[TARGET] == "Dropout").mean())
R["Z_E_GRAD"] = pct((z[TARGET] == "Graduate").mean())
a2 = df["Curricular units 2nd sem (approved)"] == 0
R["A2_ZERO_DROP"] = pct(a2[df[TARGET] == "Dropout"].mean())
R["A2_ZERO_GRAD"] = pct(a2[df[TARGET] == "Graduate"].mean())
Xf = add_features(df.drop(columns=TARGET))
R["COR_RATE2"] = n(Xf["approval_rate_2nd"].corr(tmp["tc"]), 2)

# ---------------- kết quả chính
for m in MODELS:
    for fl in ["A", "B1", "B2", "B"]:
        k = f"{m.upper()}_{fl}"
        R[f"CV_{k}"] = n(g("tuned", m, fl, "cv_f1_macro"))
        R[f"CVS_{k}"] = n(g("tuned", m, fl, "cv_std"))
        R[f"T_{k}"] = n(g("tuned", m, fl, "test_f1_macro"))
        R[f"ACC_{k}"] = n(g("tuned", m, fl, "test_acc"))
        R[f"D_{k}"] = n(g("default", m, fl, "test_f1_macro"))
        R[f"DCV_{k}"] = n(g("default", m, fl, "cv_f1_macro"))
        R[f"GAIN_{k}"] = n(g("tuned", m, fl, "cv_f1_macro") - g("tuned", m, "A", "cv_f1_macro"))
R["NB_B_DEF_ACC"] = n(g("default", "nb", "B", "test_acc"))
R["NB_B_DEF_RENR"] = n(g("default", "nb", "B", "recall_Enrolled"))
R["NB_B_DEF_F1"] = n(g("default", "nb", "B", "test_f1_macro"))
R["NB_B_PAR"] = json.loads(gp("tuned", "nb", "B"))["var_smoothing"]
R["NB_B_PAR"] = f"{R['NB_B_PAR']:g}".replace(".", ",")

best = info["best_by_cv"]
R["BEST_MODEL"] = MODEL_NAMES[best["model"]]
R["BEST_FLOW"] = best["flow"]
R["BEST_PARAMS"] = psum(best["model"], best["params"])
R["BEST_CV"] = n(best["cv_f1_macro"]); R["BEST_T"] = n(best["test_f1_macro"]); R["BEST_ACC"] = n(best["test_acc"])
bm = final[(final.stage == "tuned") & (final.model == best["model"]) & (final.flow == best["flow"])].iloc[0]
for c in ["Dropout", "Enrolled", "Graduate"]:
    R[f"BP_{c}"] = n(bm[f"prec_{c}"], 2); R[f"BR_{c}"] = n(bm[f"recall_{c}"], 2)
R["TEST_SE"] = n(0.5 / (info["n_test"] ** 0.5), 3)

# ---------------- bảng
rows = []
names = {"A": "A", "B": "B"}
for m in MODELS:
    for fl in ["A", "B"]:
        for stg in ["default", "tuned"]:
            cfg = dparams(m, gp(stg, m, fl)) if stg == "default" else psum(m, gp(stg, m, fl))
            rows.append(" & ".join([MODEL_NAMES[m] if (fl, stg) == ("A", "default") else "", fl,
                                    "mặc định" if stg == "default" else "tinh chỉnh", cfg,
                                    f"{n(g(stg, m, fl, 'cv_f1_macro'))}$\\pm${n(g(stg, m, fl, 'cv_std'), 3)}",
                                    n(g(stg, m, fl, "test_acc")), n(g(stg, m, fl, "test_f1_macro")),
                                    n(g(stg, m, fl, "recall_Dropout"), 2), n(g(stg, m, fl, "f1_Enrolled"), 2)]) + r" \\")
    rows.append(r"\midrule")
R["TABLE_MAIN"] = "\n".join(rows[:-1])

rows = []
for m in MODELS:
    cells = [MODEL_NAMES[m]]
    for fl in ["A", "B1", "B2", "B"]:
        cells.append(f"{n(g('tuned', m, fl, 'cv_f1_macro'))}$\\pm${n(g('tuned', m, fl, 'cv_std'), 3)}")
    for fl in ["A", "B1", "B2", "B"]:
        cells.append(n(g("tuned", m, fl, "test_f1_macro")))
    rows.append(" & ".join(cells) + r" \\")
R["TABLE_ABL"] = "\n".join(rows)

rows = []
ew_drop = []
for m in MODELS:
    for fl in ["A", "B"]:
        e = early[(early.model == m) & (early.flow == fl)].iloc[0]
        full_f1 = g("tuned", m, fl, "test_f1_macro")
        ew_drop.append(full_f1 - e["test_f1_macro"])
        rows.append(" & ".join([MODEL_NAMES[m] if fl == "A" else "", fl, n(e["cv_f1_macro"]), n(e["test_acc"]),
                                n(e["test_f1_macro"]), n(full_f1), n(full_f1 - e["test_f1_macro"]),
                                n(e["recall_Dropout"], 2)]) + r" \\")
R["TABLE_EW"] = "\n".join(rows)
R["EW_MIN"], R["EW_MAX"] = n(min(ew_drop), 2), n(max(ew_drop), 2)
ew_b = early[early.flow == "B"].sort_values("test_f1_macro", ascending=False).iloc[0]
R["EW_BEST"] = f"{MODEL_NAMES[ew_b['model']]} (Luồng {ew_b['flow']}), F1-macro {n(ew_b['test_f1_macro'])}"

cls = ["Dropout", "Enrolled", "Graduate"]
rows = [f"{c} & " + " & ".join(str(int(pair.loc[i, c])) for i in range(3)) + r" \\" if False else "" for c in []]
rows = []
for i, t in enumerate(cls):
    rows.append(f"\\textbf{{{t}}} & " + " & ".join(str(int(pair.iloc[i][c])) for c in cls) + r" \\")
R["TABLE_CONF"] = "\n".join(rows)

rows = []
sel = [("Dropout", "Dropout"), ("Dropout", "Enrolled"), ("Dropout", "Graduate"), ("Enrolled", "Enrolled"),
       ("Graduate", "Enrolled"), ("Graduate", "Graduate")]
P = {}
for t, p_ in sel:
    r = prof[(prof.true == t) & (prof.pred == p_)].iloc[0]
    P[(t, p_)] = r
    rows.append(" & ".join([f"{t} $\\to$ {p_}", str(int(r["n"])), n(r["Curricular units 2nd sem (approved)"], 2),
                            n(r["approval_rate_2nd"], 2), n(r["Tuition fees up to date"], 2),
                            n(r["Scholarship holder"], 2), n(r["Age at enrollment"], 1)]) + r" \\")
R["TABLE_ERR"] = "\n".join(rows)
de, dd, ge, gg = P[("Dropout", "Enrolled")], P[("Dropout", "Dropout")], P[("Graduate", "Enrolled")], P[("Graduate", "Graduate")]
R["DE_N"] = str(int(de["n"])); R["DE_RATE"] = n(de["approval_rate_2nd"], 2); R["DE_FEE"] = n(de["Tuition fees up to date"], 2)
R["DE_A2"] = n(de["Curricular units 2nd sem (approved)"], 1)
R["DD_RATE"] = n(dd["approval_rate_2nd"], 2); R["DD_A2"] = n(dd["Curricular units 2nd sem (approved)"], 1)
R["GE_N"] = str(int(ge["n"])); R["GE_RATE"] = n(ge["approval_rate_2nd"], 2); R["GG_RATE"] = n(gg["approval_rate_2nd"], 2)
R["GE_SCH"] = n(ge["Scholarship holder"], 2); R["GG_SCH"] = n(gg["Scholarship holder"], 2)
R["ERR_TOTAL"] = str(int(pair.values.sum() - sum(pair.iloc[i, i] for i in range(3))))
R["GRAD_TO_ENR"] = pct(pair.loc["Graduate", "Enrolled"] / pair.loc["Graduate"].sum(), 0)
R["DROP_TO_ENR"] = pct(pair.loc["Dropout", "Enrolled"] / pair.loc["Dropout"].sum(), 0)

rows = [f"\\texttt{{{esc(r['Unnamed: 0'])}}} & {n(r['importance'])} \\\\" for _, r in imp.head(8).iterrows()]
R["TABLE_IMP"] = "\n".join(rows)
R["IMP1"] = esc(imp.iloc[0]["Unnamed: 0"]); R["IMP1V"] = n(imp.iloc[0]["importance"], 2)
R["IMP2"] = esc(imp.iloc[1]["Unnamed: 0"]); R["IMP2V"] = n(imp.iloc[1]["importance"], 2)

# ---------------- điền vào mẫu
tpl = (REP / "main_template.tex").read_text(encoding="utf-8")
missing = []
for key in sorted(set(__import__("re").findall(r"@@([A-Za-z0-9_]+)@@", tpl))):
    if key not in R:
        missing.append(key)
        continue
    tpl = tpl.replace(f"@@{key}@@", str(R[key]))
(REP / "main.tex").write_text(tpl, encoding="utf-8")
print("Đã ghi reports/main.tex;", "THIẾU:" if missing else "đủ tất cả placeholder", missing)
