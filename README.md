### 1. Project

# Project: Phát hiện giao dịch gian lận thẻ tín dụng bằng XGBoost

---

### 2. Giới thiệu (Overview / Description)

Dự án xây dựng mô hình học máy nhằm phát hiện các giao dịch gian lận thẻ tín dụng (Credit Card Fraud Detection) trên bộ dữ liệu có mức độ mất cân bằng rất cao. Trong thực tế, số lượng giao dịch gian lận thường chỉ chiếm một tỷ lệ rất nhỏ so với các giao dịch hợp lệ, khiến việc huấn luyện mô hình trở nên khó khăn nếu chỉ tối ưu theo độ chính xác (Accuracy).

Trong dự án này, mô hình **Extreme Gradient Boosting (XGBoost)** được sử dụng để phân loại giao dịch bình thường và giao dịch gian lận. Hai phương pháp xử lý dữ liệu mất cân bằng được triển khai và so sánh:

- Sử dụng tham số `scale_pos_weight` của XGBoost.
- Kết hợp kỹ thuật **SMOTE (Synthetic Minority Over-sampling Technique)** để sinh thêm mẫu của lớp thiểu số trước khi huấn luyện.

Quy trình thực hiện bao gồm các bước tiền xử lý dữ liệu, chia tập Train/Validation/Test, chuẩn hóa dữ liệu, cân bằng dữ liệu bằng SMOTE (đối với mô hình tương ứng), huấn luyện mô hình XGBoost, lựa chọn ngưỡng phân loại (Threshold) và đánh giá hiệu năng thông qua nhiều chỉ số như Accuracy, Precision, Recall, F1-score, ROC-AUC, Precision-Recall Curve và Confusion Matrix.

Mục tiêu của dự án là xây dựng một mô hình có khả năng phát hiện được nhiều giao dịch gian lận nhất có thể trong khi vẫn duy trì tỷ lệ dự đoán chính xác cao, phù hợp với đặc thù của bài toán phát hiện gian lận trong lĩnh vực tài chính.

### Công nghệ sử dụng (Tech Stack)

- **XGBoost**  
  Sử dụng mô hình Gradient Boosting dựa trên cây quyết định để thực hiện bài toán phân loại. XGBoost được lựa chọn nhờ khả năng xử lý dữ liệu dạng bảng (tabular data), hiệu năng cao và hỗ trợ tốt trong các bài toán mất cân bằng dữ liệu.

- **Optuna**  
  Sử dụng thư viện tối ưu hóa siêu tham số (hyperparameter optimization) để tự động tìm kiếm bộ tham số phù hợp cho mô hình XGBoost, giúp cải thiện hiệu suất dự đoán và giảm thời gian thử nghiệm thủ công.

- **SMOTE (Synthetic Minority Over-sampling Technique)**  
  Áp dụng kỹ thuật sinh mẫu tổng hợp cho lớp thiểu số nhằm cân bằng dữ liệu huấn luyện. SMOTE giúp mô hình học tốt hơn các trường hợp hiếm gặp thay vì bị chi phối bởi lớp đa số.


### Demo / Result

Mô hình sau quá trình huấn luyện và tối ưu đạt được kết quả:

- **Precision:** đạt khoảng **90%**  
  Cho thấy mô hình có độ chính xác cao trong các dự đoán thuộc lớp mục tiêu, giảm số lượng dự đoán sai.

- **Recall:** đạt khoảng **85%** ở một số ngưỡng dự đoán  
  Cho thấy mô hình có khả năng phát hiện phần lớn các mẫu thuộc lớp thiểu số, phù hợp với bài toán dữ liệu mất cân bằng.

Kết quả được đánh giá dựa trên các chỉ số:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve / PR Curve



### Dataset

Sử dụng bộ dữ liệu **Credit Card Fraud Detection** (Creditcard).

Dataset chứa các giao dịch thẻ tín dụng với mục tiêu phát hiện các giao dịch gian lận (**fraud detection**).

Đặc điểm dữ liệu:

- Dữ liệu dạng bảng (**tabular data**)
- Bài toán: **Binary Classification**
  - `0`: Giao dịch bình thường
  - `1`: Giao dịch gian lận
- Dữ liệu có tính mất cân bằng cao (**imbalanced data**), số lượng giao dịch gian lận ít hơn rất nhiều so với giao dịch hợp lệ.

Tiền xử lý dữ liệu:
- Làm sạch và chuẩn hóa dữ liệu
- Chia dữ liệu thành tập train/test
- Áp dụng **SMOTE** trên tập huấn luyện để cân bằng lớp
- Huấn luyện mô hình bằng **XGBoost**


### Cấu trúc thư mục project

- `Optuna_XGBoost.ipynb`  
  Kết hợp **Optuna** để tối ưu siêu tham số (hyperparameter tuning) cho mô hình **XGBoost**.

- `EDA.ipynb`  
  Thực hiện **Exploratory Data Analysis (EDA)**:
  - Trực quan hóa dữ liệu
  - Khám phá đặc điểm dataset
  - Tiền xử lý dữ liệu

- `SMOTE_XGBoost.ipynb`  
  Áp dụng kỹ thuật **SMOTE (Synthetic Minority Over-sampling Technique)** để xử lý dữ liệu mất cân bằng trước khi huấn luyện XGBoost.

- `XGBoost.ipynb`  
  Huấn luyện mô hình **XGBoost** trên dữ liệu gốc và đánh giá hiệu quả khi dữ liệu chưa cân bằng.

- `requirements.txt`  
  Chứa danh sách các thư viện Python cần thiết để cài đặt môi trường chạy project.


### Cài đặt (Installation)

Cài đặt các thư viện cần thiết từ file `requirements.txt`:

bash:
pip install -r requirements.txt


