"""Đặc trưng mới cho Luồng B.

Các đặc trưng được tính theo từng dòng (không học thống kê nào từ dữ liệu),
nên không gây data leakage. Hàm dùng được cả khi bỏ các cột học kỳ 2
(cấu hình cảnh báo sớm).
"""
import numpy as np
import pandas as pd

ENR = "Curricular units {s} sem (enrolled)"
EVA = "Curricular units {s} sem (evaluations)"
APP = "Curricular units {s} sem (approved)"
GRD = "Curricular units {s} sem (grade)"


def _safe_div(num, den):
    """num/den, trả về 0 khi den = 0 (sinh viên không đăng ký / không thi)."""
    return (num / den.where(den > 0)).fillna(0.0)


def add_features(X):
    """Thêm đặc trưng tỷ lệ vào bản sao của X (DataFrame)."""
    X = X.copy()
    approved_cols = []
    for s in ("1st", "2nd"):
        enr, eva, app = ENR.format(s=s), EVA.format(s=s), APP.format(s=s)
        if enr not in X.columns:  # cảnh báo sớm: không có học kỳ 2
            continue
        X[f"approval_rate_{s}"] = _safe_div(X[app], X[enr])      # môn đạt / môn đăng ký
        X[f"eval_pass_rate_{s}"] = _safe_div(X[app], X[eva])     # môn đạt / lượt thi
        approved_cols.append(app)
    g1, g2 = GRD.format(s="1st"), GRD.format(s="2nd")
    if g1 in X.columns and g2 in X.columns:
        X["grade_delta"] = X[g2] - X[g1]                          # điểm HK2 - HK1
    if approved_cols:
        X["total_approved"] = X[approved_cols].sum(axis=1)        # tổng môn đạt
    return X


def fe_feature_names_out(transformer, input_features):
    """Tên cột đầu ra của add_features (để Pipeline.get_feature_names_out hoạt động)."""
    out = add_features(pd.DataFrame(columns=list(input_features)).astype(float))
    return np.asarray(out.columns, dtype=object)
