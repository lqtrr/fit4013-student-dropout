from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

RANDOM_STATE = 42
TARGET = "Target"
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]  # thứ tự LabelEncoder (A-Z)

# Đường dẫn tính từ vị trí file này, chạy từ đâu cũng đúng
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.csv"
TEST_SIZE = 0.2
CV_FOLDS = 5


def load_data(path=DATA_PATH):
    """Đọc dữ liệu sinh viên, trả về DataFrame với tên cột đã làm sạch."""
    df = pd.read_csv(path, sep=";")
    # Một số cột có ký tự thừa (ví dụ 'Daytime/evening attendance\t')
    df.columns = df.columns.str.strip()
    return df


def split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Chia train/test 80/20 có stratify theo nhãn. Trả về X_train, X_test, y_train, y_test."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def get_cv(n_splits=CV_FOLDS, random_state=RANDOM_STATE):
    """StratifiedKFold dùng chung cho mọi GridSearchCV / cross_validate."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


if __name__ == "__main__":
    df = load_data()
    print("Kich thuoc:", df.shape)
    print("Thieu:", df.isna().sum().sum(), "| Trung:", df.duplicated().sum())
    print(df[TARGET].value_counts())
    print((df[TARGET].value_counts(normalize=True) * 100).round(1))