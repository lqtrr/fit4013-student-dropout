"""Tạo mô hình và pipeline đầy đủ (tiền xử lý + thuật toán)."""
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data import RANDOM_STATE
from preprocess import make_preprocessor

MODELS = ["knn", "dt", "nb"]
MODEL_NAMES = {"knn": "K-NN", "dt": "Decision Tree", "nb": "GaussianNB"}
NEEDS_SCALING = {"knn": True, "dt": False, "nb": True}  # K-NN dùng khoảng cách nên phải chuẩn hóa

DEFAULT_PARAMS = {"knn": {"n_neighbors": 5}, "dt": {}, "nb": {}}

# Lưới tham số (tiền tố model__ vì mô hình là bước 'model' của Pipeline)
PARAM_GRIDS = {
    "knn": {
        "model__n_neighbors": [3, 5, 7, 11, 15, 21],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    },
    "dt": {
        "model__max_depth": [3, 4, 5, 6, 8, 10, None],
        "model__criterion": ["gini", "entropy"],
        "model__min_samples_leaf": [1, 5, 10, 20],
        "model__class_weight": [None, "balanced"],
    },
    "nb": {"model__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 1.0]},
}


def make_model(name, **params):
    if name == "knn":
        return KNeighborsClassifier(**params)
    if name == "dt":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    if name == "nb":
        return GaussianNB(**params)
    raise ValueError(name)


def make_pipeline(flow, name, **params):
    """Pipeline = tiền xử lý theo luồng `flow` + mô hình `name`."""
    return Pipeline([
        ("prep", make_preprocessor(flow, NEEDS_SCALING[name])),
        ("model", make_model(name, **params)),
    ])
