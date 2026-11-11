from pathlib import Path
import pandas as pd

RANDOM_STATE = 42
TARGET = "Target"
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]  # thứ tự LabelEncoder (A-Z)

# Đường dẫn tính từ vị trí file này, chạy từ đâu cũng đúng
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.csv"


def load_data(path=DATA_PATH):
    """Đọc dữ liệu sinh viên, trả về DataFrame với tên cột đã làm sạch."""
    df = pd.read_csv(path, sep=";")
    # Một số cột có ký tự thừa (ví dụ 'Daytime/evening attendance\t')
    df.columns = df.columns.str.strip()
    return df


if __name__ == "__main__":
    df = load_data()
    print("Kich thuoc:", df.shape)
    print("Thieu:", df.isna().sum().sum(), "| Trung:", df.duplicated().sum())
    print(df[TARGET].value_counts())
    print((df[TARGET].value_counts(normalize=True) * 100).round(1))