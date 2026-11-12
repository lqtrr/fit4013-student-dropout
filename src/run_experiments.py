"""Chạy toàn bộ thí nghiệm của đề tài 8 và ghi kết quả vào results/, models/, reports/figures/.

Cách chạy (từ thư mục gốc của repo):
    python src/run_experiments.py

Quy trình:
  1. Chia train/test 80/20 có stratify (random_state=42).
  2. Mô hình mặc định (baseline) trên 4 luồng tiền xử lý A, B1, B2, B.
  3. GridSearchCV (StratifiedKFold 5, thang đo F1-macro) cho K-NN, Decision Tree, Naive Bayes.
     Chọn tham số CHỈ dựa trên CV của tập train; tập test chỉ dùng để báo cáo.
  4. Biểu đồ so sánh, độ nhạy tham số, cây quyết định, độ quan trọng đặc trưng.
  5. Phân tích lỗi của mô hình tốt nhất (chọn theo CV).
  6. Cấu hình cảnh báo sớm: bỏ toàn bộ cột học kỳ 2.
"""
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CLASS_NAMES, RANDOM_STATE, ROOT, TARGET, get_cv, load_data, split_data  # noqa: E402
from evaluate import conf_matrix, evaluate, plot_confusion  # noqa: E402
from features import add_features  # noqa: E402
from models import DEFAULT_PARAMS, MODEL_NAMES, MODELS, PARAM_GRIDS, make_pipeline  # noqa: E402
from preprocess import FLOWS  # noqa: E402

warnings.filterwarnings("ignore")
RES, MOD, FIG = ROOT / "results", ROOT / "models", ROOT / "reports" / "figures"
for d in (RES, RES / "grid", MOD, FIG):
    d.mkdir(parents=True, exist_ok=True)

N_JOBS = 2
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def clean_params(p):
    return json.dumps({k.replace("model__", ""): (None if v is None else (float(v) if isinstance(v, np.floating) else v))
                       for k, v in p.items()}, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- 1. dữ liệu
df = load_data()
Xtr, Xte, ytr, yte = split_data(df)
cv = get_cv()
log(f"train={Xtr.shape}, test={Xte.shape}")

# --------------------------------------------------------------------------- 2. mặc định
rows = []
for flow in FLOWS:
    for m in MODELS:
        pipe = make_pipeline(flow, m, **DEFAULT_PARAMS[m])
        s = cross_validate(pipe, Xtr, ytr, cv=cv, scoring="f1_macro", n_jobs=N_JOBS)["test_score"]
        pipe.fit(Xtr, ytr)
        rows.append({"stage": "default", "model": m, "flow": flow,
                     "params": json.dumps(DEFAULT_PARAMS[m]),
                     "cv_f1_macro": s.mean(), "cv_std": s.std(),
                     **evaluate(yte, pipe.predict(Xte))})
default_df = pd.DataFrame(rows)
default_df.to_csv(RES / "default_models.csv", index=False)
default_df[default_df.flow == "A"].to_csv(RES / "baseline.csv", index=False)
log("xong mô hình mặc định")

# --------------------------------------------------------------------------- 3. GridSearch
tuned_rows, fitted = [], {}
for m in MODELS:
    for flow in FLOWS:
        gs = GridSearchCV(make_pipeline(flow, m), PARAM_GRIDS[m], cv=cv, scoring="f1_macro",
                          n_jobs=N_JOBS, refit=True)
        gs.fit(Xtr, ytr)
        cvr = pd.DataFrame(gs.cv_results_)
        cvr["params_clean"] = [clean_params(p) for p in cvr["params"]]
        if m in ("knn", "dt") and flow in ("A", "B"):  # chỉ lưu lưới dùng cho biểu đồ độ nhạy
            cvr[["params_clean", "mean_test_score", "std_test_score", "rank_test_score"]].sort_values(
                "rank_test_score").to_csv(RES / "grid" / f"{m}_{flow}.csv", index=False)
        best = gs.best_estimator_
        fitted[(m, flow)] = best
        tuned_rows.append({"stage": "tuned", "model": m, "flow": flow,
                           "params": clean_params(gs.best_params_),
                           "cv_f1_macro": gs.best_score_,
                           "cv_std": cvr.loc[gs.best_index_, "std_test_score"],
                           **evaluate(yte, best.predict(Xte))})
        log(f"grid {m}/{flow}: cv_f1={gs.best_score_:.4f} {clean_params(gs.best_params_)}")
tuned_df = pd.DataFrame(tuned_rows)
tuned_df.to_csv(RES / "tuned_models.csv", index=False)
final_df = pd.concat([default_df, tuned_df], ignore_index=True)
final_df.to_csv(RES / "final.csv", index=False)

# Bảng ablation (F1-macro trên test, cấu hình đã tinh chỉnh)
abl = tuned_df.pivot(index="model", columns="flow", values="test_f1_macro")[FLOWS]
abl_def = default_df.pivot(index="model", columns="flow", values="test_f1_macro")[FLOWS]
abl.to_csv(RES / "ablation_tuned_f1.csv")
abl_def.to_csv(RES / "ablation_default_f1.csv")

# --------------------------------------------------------------------------- mô hình tốt nhất (theo CV)
best_row = tuned_df.sort_values("cv_f1_macro", ascending=False).iloc[0]
bm, bf = best_row["model"], best_row["flow"]
best_pipe = fitted[(bm, bf)]
y_pred = best_pipe.predict(Xte)
joblib.dump(best_pipe, MOD / "best_model.joblib")
log(f"mô hình tốt nhất theo CV: {bm}/{bf} cv={best_row['cv_f1_macro']:.4f} test_f1={best_row['test_f1_macro']:.4f}")

# --------------------------------------------------------------------------- 4. biểu đồ
# 4a. so sánh mặc định vs tinh chỉnh, luồng A vs B
fig, ax = plt.subplots(figsize=(9, 4.5))
combos = [("default", "A", "Mặc định, Luồng A"), ("tuned", "A", "Tinh chỉnh, Luồng A"),
          ("default", "B", "Mặc định, Luồng B"), ("tuned", "B", "Tinh chỉnh, Luồng B")]
w = 0.2
for i, (stg, fl, lab) in enumerate(combos):
    vals = [final_df[(final_df.stage == stg) & (final_df.flow == fl) & (final_df.model == m)]["test_f1_macro"].iloc[0]
            for m in MODELS]
    bars = ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, label=lab)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels([MODEL_NAMES[m] for m in MODELS])
ax.set_ylabel("F1-macro (tập test)"); ax.set_ylim(0, 0.85)
ax.set_title("So sánh Luồng A và Luồng B"); ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(FIG / "res_01_compare_AB.png", dpi=150); plt.close(fig)

# 4b. ablation (tinh chỉnh)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, tab, ttl in [(axes[0], abl_def, "Mặc định"), (axes[1], abl, "Tinh chỉnh (GridSearch)")]:
    for i, fl in enumerate(FLOWS):
        ax.bar(np.arange(3) + (i - 1.5) * 0.2, [tab.loc[m, fl] for m in MODELS], 0.2, label=fl)
    ax.set_xticks(range(3)); ax.set_xticklabels([MODEL_NAMES[m] for m in MODELS])
    ax.set_title(ttl); ax.set_ylim(0, 0.85)
axes[0].set_ylabel("F1-macro (tập test)"); axes[1].legend(title="Luồng")
fig.suptitle("Ablation: A = cơ bản; B1 = + đặc trưng mới; B2 = + One-Hot; B = cả hai")
fig.tight_layout(); fig.savefig(FIG / "res_02_ablation.png", dpi=150); plt.close(fig)

# 4c. độ nhạy tham số từ kết quả grid
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for fl, ls in [("A", "-o"), ("B", "--s")]:
    g = pd.read_csv(RES / "grid" / f"knn_{fl}.csv")
    g["k"] = g["params_clean"].map(lambda s: json.loads(s)["n_neighbors"])
    s = g.groupby("k")["mean_test_score"].max()
    axes[0].plot(s.index, s.values, ls, label=f"Luồng {fl}")
    g = pd.read_csv(RES / "grid" / f"dt_{fl}.csv")
    g["d"] = g["params_clean"].map(lambda s: json.loads(s)["max_depth"])
    g["d"] = g["d"].fillna(-1)
    s = g.groupby("d")["mean_test_score"].max().sort_index()
    labels = [("None" if d == -1 else int(d)) for d in s.index]
    axes[1].plot(range(len(s)), s.values, ls, label=f"Luồng {fl}")
    axes[1].set_xticks(range(len(s))); axes[1].set_xticklabels(labels)
axes[0].set_xlabel("n_neighbors (k)"); axes[0].set_ylabel("F1-macro CV (tốt nhất theo k)"); axes[0].set_title("K-NN")
axes[1].set_xlabel("max_depth"); axes[1].set_title("Decision Tree"); axes[0].legend(); axes[1].legend()
fig.tight_layout(); fig.savefig(FIG / "res_03_sensitivity.png", dpi=150); plt.close(fig)

# 4d. confusion matrix: mô hình tốt nhất và lưới 3 thuật toán x 2 luồng
plot_confusion(yte, y_pred, f"Mô hình tốt nhất theo CV: {MODEL_NAMES[bm]}, Luồng {bf}",
               FIG / "res_04_confusion_best.png")
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for r, fl in enumerate(["A", "B"]):
    for c, m in enumerate(MODELS):
        cm = conf_matrix(yte, fitted[(m, fl)].predict(Xte))
        im = axes[r, c].imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                axes[r, c].text(j, i, cm[i, j], ha="center", va="center",
                                color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
        axes[r, c].set_xticks(range(3)); axes[r, c].set_xticklabels(CLASS_NAMES, fontsize=8)
        axes[r, c].set_yticks(range(3)); axes[r, c].set_yticklabels(CLASS_NAMES, fontsize=8)
        axes[r, c].set_title(f"{MODEL_NAMES[m]} - Luồng {fl}", fontsize=10)
fig.suptitle("Confusion matrix (tinh chỉnh; số trong ô là số lượng, màu là recall theo hàng)")
fig.tight_layout(); fig.savefig(FIG / "res_05_confusion_grid.png", dpi=150); plt.close(fig)

# 4e. cây quyết định nông (tên cột gốc) và độ quan trọng đặc trưng
shallow = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=RANDOM_STATE).fit(Xtr, ytr)
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(shallow, feature_names=[c.replace("Curricular units ", "CU ") for c in Xtr.columns],
          class_names=shallow.classes_, filled=True, rounded=True, fontsize=8, ax=ax, impurity=False)
fig.tight_layout(); fig.savefig(FIG / "res_06_tree_depth3.png", dpi=130); plt.close(fig)
(RES / "tree_depth3.txt").write_text(export_text(shallow, feature_names=list(Xtr.columns)), encoding="utf-8")

dtB = fitted[("dt", "B")]
imp = pd.Series(dtB.named_steps["model"].feature_importances_,
                index=dtB.named_steps["prep"].get_feature_names_out()).sort_values(ascending=False)
imp.head(20).to_csv(RES / "dt_B_importance_top20.csv", header=["importance"])
fig, ax = plt.subplots(figsize=(8, 5))
imp.head(12)[::-1].plot.barh(ax=ax)
ax.set_title("Độ quan trọng đặc trưng - Decision Tree tinh chỉnh, Luồng B"); ax.set_xlabel("Độ quan trọng")
fig.tight_layout(); fig.savefig(FIG / "res_07_importance_dtB.png", dpi=150); plt.close(fig)
log("xong biểu đồ")

# --------------------------------------------------------------------------- 5. phân tích lỗi
pair = pd.crosstab(yte.rename("thuc_te"), pd.Series(y_pred, index=yte.index, name="du_doan")).reindex(
    index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0)
pair.to_csv(RES / "confusion_best.csv")

Xf = add_features(Xte)
proba = pd.DataFrame(best_pipe.predict_proba(Xte), index=Xte.index, columns=best_pipe.classes_)
err = Xf.assign(true=yte, pred=y_pred, correct=(yte.values == y_pred), conf_pred=proba.max(axis=1))
profile_cols = ["Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)",
                "Curricular units 2nd sem (grade)", "approval_rate_2nd", "Tuition fees up to date",
                "Scholarship holder", "Debtor", "Age at enrollment", "Admission grade"]
prof = err.groupby(["true", "pred"])[profile_cols].mean().round(2)
prof.insert(0, "n", err.groupby(["true", "pred"]).size())
prof.to_csv(RES / "error_profile.csv")
show_cols = profile_cols + ["Course", "Gender"]
wrong = err[~err.correct].sort_values("conf_pred", ascending=False)
wrong[["true", "pred", "conf_pred"] + show_cols].to_csv(RES / "error_cases.csv")
log(f"phân tích lỗi: {len(wrong)} mẫu sai / {len(err)}")

# --------------------------------------------------------------------------- 6. cảnh báo sớm
ew_cols = [c for c in Xtr.columns if "2nd sem" not in c]
ew_rows = []
for m in MODELS:
    for flow in ["A", "B"]:
        gs = GridSearchCV(make_pipeline(flow, m), PARAM_GRIDS[m], cv=cv, scoring="f1_macro", n_jobs=N_JOBS)
        gs.fit(Xtr[ew_cols], ytr)
        ew_rows.append({"stage": "early_warning", "model": m, "flow": flow,
                        "params": clean_params(gs.best_params_), "cv_f1_macro": gs.best_score_,
                        **evaluate(yte, gs.predict(Xte[ew_cols]))})
ew_df = pd.DataFrame(ew_rows)
ew_df.to_csv(RES / "early_warning.csv", index=False)
log("xong cảnh báo sớm")

# --------------------------------------------------------------------------- thông tin lần chạy
info = {
    "random_state": RANDOM_STATE, "test_size": 0.2, "cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
    "scoring": "f1_macro", "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
    "class_counts_train": ytr.value_counts().to_dict(), "class_counts_test": yte.value_counts().to_dict(),
    "best_by_cv": {"model": bm, "flow": bf, "params": best_row["params"], "cv_f1_macro": float(best_row["cv_f1_macro"]),
                   "test_f1_macro": float(best_row["test_f1_macro"]), "test_acc": float(best_row["test_acc"])},
    "python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__,
    "numpy": np.__version__, "runtime_seconds": round(time.time() - t0),
}
(RES / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
log("HOÀN TẤT")
