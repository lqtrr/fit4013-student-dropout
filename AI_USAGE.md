| Ngày | Người dùng | Công cụ | Mục đích | Phần nào của dự án | Đã tự kiểm tra thế nào |
|---|---|---|---|---|---|
| 2026-11-12 | Nhóm (điền tên người thực hiện) | Claude (Anthropic) | Lập kế hoạch và phân công; viết mã `src/features.py`, `preprocess.py`, `models.py`, `evaluate.py`, `run_experiments.py`, `predict.py`, `app.py`; tạo notebook 02-04; viết nháp README và báo cáo | Luồng A/B, thí nghiệm, đánh giá, phân tích lỗi, GUI, tài liệu | Chạy lại toàn bộ pipeline, đối chiếu số liệu trong báo cáo với `results/*.csv`; nhóm cần tự đọc và giải thích được từng đoạn mã, rồi bổ sung/chỉnh sửa cột này theo thực tế |
| 2026-11-11 | Lưu Quang Trường | Claude (Anthropic) | Hướng dẫn tạo repo GitHub, cấu trúc thư mục, quy trình Git cho nhóm | Hạ tầng repo | Tự làm theo từng bước, kiểm tra trên GitHub có đủ thư mục và commit |
| 2026-11-12 | Lưu Quang Trường | Claude (Anthropic) | Viết bản nháp notebook EDA, hướng dẫn tải dữ liệu và viết `src/data.py` | `notebooks/01_eda.ipynb`, `src/data.py` | Tự chạy notebook, đối chiếu kích thước 4424 dòng, nhãn 49,9/32,1/17,9%, xem lại 5 hình |
| 2026-11-13 | Lưu Quang Trường | Claude (Anthropic) | Giải thích `preprocess.py`, `features.py`, `run_experiments.py`; hướng dẫn chạy lại thí nghiệm | Luồng A, kiểm tra tái lập | Tự chạy lại `run_experiments.py` trên Python 3.14, so sánh với `results/final.csv`, chỉ khác cột thời gian |

Ghi chú: nếu các thành viên đã dùng AI cho phần EDA (`notebooks/01_eda.ipynb`, `src/data.py`) hoặc phần khác trước ngày 2026-11-12, hãy thêm dòng tương ứng.
