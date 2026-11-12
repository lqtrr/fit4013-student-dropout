"""Dự đoán nguy cơ bỏ học cho một sinh viên bằng mô hình đã lưu (dùng cho GUI)."""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # để joblib tìm thấy module features/preprocess
from data import ROOT, TARGET, load_data  # noqa: E402

MODEL_PATH = ROOT / "models" / "best_model.joblib"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def default_row(df=None):
    """Hàng mặc định: trung vị của từng cột (các cột người dùng không nhập sẽ lấy giá trị này)."""
    df = load_data() if df is None else df
    return df.drop(columns=TARGET).median().to_frame().T


def predict_proba(model, overrides, base=None):
    """overrides: dict {tên cột: giá trị}. Trả về Series xác suất của 3 lớp."""
    row = default_row() if base is None else base.copy()
    for k, v in overrides.items():
        if k not in row.columns:
            raise KeyError(f"Không có cột '{k}'")
        row[k] = v
    return pd.Series(model.predict_proba(row)[0], index=model.classes_)
