"""GUI demo (Streamlit): nhập thông tin sinh viên, xem xác suất Dropout / Enrolled / Graduate.

Chạy:  streamlit run app.py
Các cột không có trong form lấy giá trị trung vị của tập dữ liệu.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data import TARGET, load_data  # noqa: E402
from predict import default_row, load_model, predict_proba  # noqa: E402

st.set_page_config(page_title="Dự đoán nguy cơ bỏ học", layout="centered")
st.title("Dự đoán sinh viên có nguy cơ bỏ học")
st.caption("FIT4013 - Đề tài 8. Kết quả chỉ mang tính tham khảo, không dùng để ra quyết định đối với sinh viên.")


@st.cache_resource
def _model():
    return load_model()


@st.cache_data
def _data():
    df = load_data()
    return df, default_row(df)


df, base = _data()
model = _model()

with st.form("form"):
    st.subheader("Thông tin chung")
    c1, c2 = st.columns(2)
    age = c1.number_input("Tuổi khi nhập học", 16, 70, int(base["Age at enrollment"].iloc[0]))
    adm = c2.number_input("Điểm đầu vào (Admission grade)", 0.0, 200.0, float(base["Admission grade"].iloc[0]))
    course = c1.selectbox("Ngành (mã Course)", sorted(df["Course"].unique()),
                          index=sorted(df["Course"].unique()).index(int(base["Course"].iloc[0])))
    gender = c2.selectbox("Giới tính (mã Gender: 0/1)", [0, 1])
    c3, c4, c5 = st.columns(3)
    fees = c3.selectbox("Đóng học phí đúng hạn", [1, 0], format_func=lambda x: "Có" if x else "Không")
    debtor = c4.selectbox("Đang nợ", [0, 1], format_func=lambda x: "Có" if x else "Không")
    schol = c5.selectbox("Có học bổng", [0, 1], format_func=lambda x: "Có" if x else "Không")

    overrides = {}
    for s, label in (("1st", "Học kỳ 1"), ("2nd", "Học kỳ 2")):
        st.subheader(label)
        a, b, c, d = st.columns(4)
        overrides[f"Curricular units {s} sem (enrolled)"] = a.number_input("Môn đăng ký", 0, 30, int(base[f"Curricular units {s} sem (enrolled)"].iloc[0]), key=f"e{s}")
        overrides[f"Curricular units {s} sem (evaluations)"] = b.number_input("Lượt thi", 0, 60, int(base[f"Curricular units {s} sem (evaluations)"].iloc[0]), key=f"v{s}")
        overrides[f"Curricular units {s} sem (approved)"] = c.number_input("Môn đạt", 0, 30, int(base[f"Curricular units {s} sem (approved)"].iloc[0]), key=f"a{s}")
        overrides[f"Curricular units {s} sem (grade)"] = d.number_input("Điểm TB", 0.0, 20.0, float(base[f"Curricular units {s} sem (grade)"].iloc[0]), key=f"g{s}")
    submitted = st.form_submit_button("Dự đoán")

if submitted:
    overrides.update({"Age at enrollment": age, "Admission grade": adm, "Course": course, "Gender": gender,
                      "Tuition fees up to date": fees, "Debtor": debtor, "Scholarship holder": schol})
    p = predict_proba(model, overrides, base)
    st.subheader("Kết quả")
    st.bar_chart(p)
    st.write(f"Nhãn có xác suất cao nhất: **{p.idxmax()}** ({p.max():.0%})")
    st.info("Mô hình tốt nhất theo CV là Decision Tree nên xác suất lấy từ tỷ lệ mẫu trong lá của cây, "
            "không phải xác suất hiệu chỉnh.")
