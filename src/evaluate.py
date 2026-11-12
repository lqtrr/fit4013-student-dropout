"""Chỉ số đánh giá và hàm vẽ dùng chung."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from data import CLASS_NAMES


def evaluate(y_true, y_pred):
    """Trả về dict: accuracy, macro P/R/F1, và P/R/F1 từng lớp."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_NAMES, zero_division=0)
    out = {
        "test_acc": accuracy_score(y_true, y_pred),
        "test_prec_macro": p.mean(),
        "test_recall_macro": r.mean(),
        "test_f1_macro": f1_score(y_true, y_pred, average="macro", labels=CLASS_NAMES),
        "test_f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=CLASS_NAMES),
    }
    for i, c in enumerate(CLASS_NAMES):
        out[f"prec_{c}"], out[f"recall_{c}"], out[f"f1_{c}"] = p[i], r[i], f[i]
    return out


def conf_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)


def plot_confusion(y_true, y_pred, title, path):
    """Hai ma trận nhầm lẫn cạnh nhau: số lượng và chuẩn hóa theo hàng (recall)."""
    cm = conf_matrix(y_true, y_pred)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[0].set_title("Số lượng")
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, vmin=0, vmax=1)
    axes[1].set_title("Chuẩn hóa theo hàng (recall)")
    for ax in axes:
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
