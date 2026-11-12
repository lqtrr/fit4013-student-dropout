"""Tiền xử lý: Luồng A (cơ bản) và Luồng B (cải tiến).

flow:
  'A'  - cơ bản: giữ nguyên 36 cột, chỉ chuẩn hóa (khi thuật toán cần).
  'B1' - chỉ thêm đặc trưng mới (để tách đóng góp, dùng cho ablation).
  'B2' - chỉ One-Hot các cột mã định danh (để tách đóng góp, dùng cho ablation).
  'B'  - đầy đủ: đặc trưng mới + One-Hot cột mã định danh.

Mọi phép học tham số (StandardScaler, OneHotEncoder) nằm trong Pipeline nên
chỉ được fit trên tập train, tránh data leakage.
Các hàm chọn cột là hàm cấp module (không dùng lambda) để lưu được bằng joblib.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import add_features, fe_feature_names_out

# Cột mã số không có thứ tự (nominal)
ID_COLS = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]
FLOWS = ["A", "B1", "B2", "B"]
FLOW_DESC = {
    "A": "A: giữ nguyên cột",
    "B1": "B1: + đặc trưng mới",
    "B2": "B2: + One-Hot",
    "B": "B: + đặc trưng mới + One-Hot",
}


def _id_cols(df):
    return [c for c in ID_COLS if c in df.columns]


def _non_id_cols(df):
    return [c for c in df.columns if c not in ID_COLS]


def _all_cols(df):
    return list(df.columns)


def make_preprocessor(flow, scale):
    """Tạo bộ tiền xử lý cho `flow`; `scale=True` thì chuẩn hóa cột số (K-NN, Naive Bayes)."""
    if flow not in FLOWS:
        raise ValueError(f"flow phải thuộc {FLOWS}")
    use_fe = flow in ("B", "B1")
    use_onehot = flow in ("B", "B2")
    scaler = StandardScaler() if scale else "passthrough"

    steps = []
    if use_fe:
        steps.append(("features", FunctionTransformer(add_features, feature_names_out=fe_feature_names_out)))
    if use_onehot:
        ct = ColumnTransformer(
            [("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), _id_cols),
             ("num", scaler, _non_id_cols)],
            verbose_feature_names_out=False)
    else:
        ct = ColumnTransformer([("num", scaler, _all_cols)], verbose_feature_names_out=False)
    steps.append(("columns", ct))
    return Pipeline(steps)
