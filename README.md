# FIT4013 - Đề tài 8: Dự đoán sinh viên có nguy cơ bỏ học

Bài tập lớn học phần Nhập môn Học máy (FIT4013), năm học 2026-2027. Nhóm 3 người. 
## Thành viên nhóm

| STT | Họ và tên | Lớp | Mã sinh viên |
|:---:|---|---|---|
| 1 | Lưu Quang Trường | CNTT 18-02 | 1871020606 |
| 2 | Phan Văn Thành Vinh | CNTT 18-02 | 1871020659 |
| 3 | Nguyễn Đình Trọng | CNTT 18-02 | 1871020588 |

`random_state = 42`.

Bài toán: phân loại 3 lớp (`Dropout`, `Enrolled`, `Graduate`) trên dữ liệu dạng bảng, so sánh hai cách tiền xử lý
(Luồng A cơ bản và Luồng B cải tiến) với K-NN, Cây quyết định và Naive Bayes.

## Dữ liệu
- Bộ dữ liệu "Predict Students' Dropout and Academic Success" (4.424 sinh viên, 36 đặc trưng, nhãn `Target`).
  Tác giả: V. Realinho, J. Machado, L. Baptista, M. V. Martins (Instituto Politécnico de Portalegre), công bố trong *Data* 7(11), 2022.
- File dùng trong bài: `data/data.csv` (phân tách bằng `;`), lấy từ liên kết giảng viên cung cấp. Không chỉnh sửa file gốc.

## Cấu trúc thư mục
```
data/data.csv            dữ liệu gốc
src/data.py              đọc dữ liệu, chia train/test, StratifiedKFold
src/features.py          đặc trưng mới (Luồng B)
src/preprocess.py        Luồng A, B1, B2, B (ColumnTransformer + Pipeline)
src/models.py            K-NN, Decision Tree, GaussianNB và lưới tham số
src/evaluate.py          Accuracy/Precision/Recall/F1, confusion matrix
src/run_experiments.py   chạy toàn bộ thí nghiệm, ghi results/, models/, reports/figures/
src/predict.py           dự đoán một sinh viên từ mô hình đã lưu
notebooks/               01_eda, 02_pipeline_A, 03_pipeline_B, 04_evaluation
results/                 kết quả CSV (final.csv, baseline.csv, grid/ cho biểu đồ độ nhạy, early_warning.csv, ...)
models/                  best_model.joblib (mô hình tốt nhất theo CV, dùng cho GUI)
reports/                 hình (figures/), báo cáo LaTeX
app.py                   GUI demo (Streamlit)
AI_USAGE.md              nhật ký sử dụng công cụ AI
```

## Cài đặt
Python 3.10 trở lên (đã chạy thử với 3.13).
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Cách chạy
Chạy từ thư mục gốc của repo.
```bash
python src/data.py                # kiểm tra nhanh dữ liệu
python src/run_experiments.py     # toàn bộ thí nghiệm, khoảng 2 phút trên 2 lõi CPU
jupyter notebook notebooks/       # xem EDA (01), Luồng A (02), Luồng B (03), đánh giá (04)
streamlit run app.py              # GUI demo (cần chạy run_experiments.py trước để có models/best_model.joblib)
```

### Biên dịch báo cáo
Số liệu trong báo cáo được điền tự động từ `results/`. Văn bản nằm trong `reports/main_template.tex`.
```bash
python src/make_report.py          # sinh reports/main.tex từ main_template.tex + results/
cd reports
xelatex main.tex && xelatex main.tex   # cần XeLaTeX (Overleaf: chọn compiler XeLaTeX)
```
Chạy lại `src/run_experiments.py` cho cùng kết quả (cùng `random_state`, cùng phiên bản thư viện).
Thông tin lần chạy được ghi trong `results/run_info.json`.

## Thiết lập thí nghiệm
- Chia train/test 80/20, có `stratify` theo nhãn, `random_state=42` (3.539 train, 885 test).
- Chọn tham số bằng `GridSearchCV` với `StratifiedKFold(5, shuffle=True, random_state=42)`, thang đo **F1-macro**.
  Tập test chỉ dùng để báo cáo, không dùng để chọn mô hình.
- Mọi bước học tham số (`StandardScaler`, `OneHotEncoder`) nằm trong `Pipeline`, chỉ fit trên tập train.
- Luồng tiền xử lý: **A** giữ nguyên cột; **B1** thêm đặc trưng tỷ lệ; **B2** One-Hot các cột mã định danh;
  **B** = B1 + B2 (B1, B2 dùng cho ablation).

## Kết quả chính (F1-macro trên tập test, mô hình đã tinh chỉnh)
| Thuật toán | Luồng A | Luồng B |
|---|---|---|
| K-NN | 0,621 | 0,685 |
| Decision Tree | 0,650 | 0,673 |
| GaussianNB | 0,557 | 0,669 |

Mô hình tốt nhất theo CV: Decision Tree với Luồng B1 (F1-macro CV 0,687, test 0,663). Số liệu đầy đủ trong `results/final.csv`.
