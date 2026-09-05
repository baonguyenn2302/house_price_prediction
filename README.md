# House Price Prediction System

## 1. Tổng quan hệ thống

Dự án này xây dựng hệ thống hồi quy dự đoán giá nhà dựa trên các thuộc tính của bất động sản ở Việt Nam, bao gồm diện tích, mặt tiền, đường vào, hướng nhà, hướng ban công, số tầng, số phòng ngủ, số phòng tắm, pháp lý, nội thất và địa chỉ. Mục tiêu là ước lượng giá nhà dưới dạng số thực (tỷ VNĐ) với độ tin cậy cao nhất có thể.

Quy trình thực hiện gồm: khởi tạo dữ liệu, kiểm tra chất lượng, tiền xử lý dữ liệu, huấn luyện 5 mô hình hồi quy, đánh giá bằng MAE, MSE, RMSE và R-squared, đồng thời trực quan hóa Actual vs Predicted và Residual Plot để phân tích sai số hệ thống.

## 2. Tiền xử lý dữ liệu

Bộ dữ liệu `houseprice.csv` có cả biến số và biến phân loại. Để chuẩn hóa quy trình, hệ thống:

- Xử lý giá trị thiếu bằng `SimpleImputer`.
- Chuẩn hóa các đặc trưng số bằng `StandardScaler`.
- Mã hóa các thuộc tính phân loại bằng `OneHotEncoder`.
- Tách địa điểm từ trường `Address` để tạo biến `Location` phù hợp với mô hình.
- Chia tập dữ liệu theo tỉ lệ 80/20 với `random_state=42` để đảm bảo kết quả ổn định.

## 3. Chi tiết 5 mô hình hồi quy

| Mô hình | Cơ chế học tập | Tham số cốt lõi | Ưu điểm | Nhược điểm |
| --- | --- | --- | --- | --- |
| Linear Regression | Tìm đường thẳng tối ưu để xấp xỉ giá nhà theo các đặc trưng đầu vào. | Hệ số hồi quy, intercept | Đơn giản, dễ diễn giải, rất hiệu quả nếu mối quan hệ gần tuyến tính | Kém khi dữ liệu có tương tác phi tuyến mạnh |
| Decision Tree Regressor | Chia không gian đầu vào theo các ngưỡng để tạo các nút quyết định. | Độ sâu cây, tiêu chí chia nhánh | Hiểu được quy tắc, xử lý tốt dữ liệu phi tuyến căn bản | Dễ overfit và không ổn định trên dữ liệu mới |
| Random Forest Regressor | Kết hợp nhiều cây quyết định học trên các bootstrap sample. | Số cây, max depth, min samples split | Bền vững, ít overfit hơn cây đơn, xử lý tốt dữ liệu phức tạp | Dễ lớn hơn về thời gian và tài nguyên |
| Gradient Boosting Regressor | Xây dựng dãy cây tuần tự, mỗi cây cố gắng sửa sai của cây trước. | Số cây, learning rate, depth | Chất lượng dự đoán cao trên dữ liệu thực tế, thích hợp cho dữ liệu cấu trúc | Đòi hỏi tinh chỉnh tham số và có thể mất thời gian |
| SVR | Tìm siêu phẳng tối ưu trong không gian ẩn sao cho sai số nằm trong vùng epsilon. | Kernel, C, epsilon, gamma | Hay khi có dữ liệu có nhiễu, hoạt động tốt trên không gian đặc trưng lớn | Tốn thời gian xử lý, nhạy với chuẩn hóa dữ liệu |

## 4. Thực nghiệm và phân tích hình ảnh

### 4.1. Actual vs Predicted Scatter Plot

Biểu đồ Actual vs Predicted cho thấy độ chênh lệch giữa giá thực tế và giá mô hình dự đoán. Nếu các điểm nằm gần đường chéo chính là sự khớp tốt. Trong bài toán giá nhà, sự phân tán quanh đường chéo phản ánh mức độ sai số của mô hình trên các mẫu. Nếu phân tán lớn ở góc giá cao, mô hình thường có xu hướng ước lượng thấp hơn ở các bất động sản đắt đỏ, cho thấy yếu tố chưa được biểu diễn hết như vị trí cụ thể, tiện ích, hoặc chất lượng dự án.

### 4.2. Residual Plot

Residual Plot biểu diễn phần dư `Actual - Predicted` theo giá dự đoán. Nếu các điểm phân tán ngẫu nhiên quanh đường 0, mô hình có độ sai số ổn định. Nếu có hình dạng fan-shape hoặc tăng dần theo mức giá, đây là dấu hiệu của heteroscedasticity: sai số không đồng đều giữa các vùng giá. Trong thực tế, điều này là hợp lý vì các căn nhà cao cấp và các dự án đặc thù thường có biến động giá rất lớn, khiến mô hình khó dự đoán chính xác tuyệt đối ở vùng giá cao.

## 5. Bảng so sánh hiệu suất tổng hợp

| Model | MAE | MSE | RMSE | R-squared |
| --- | ---: | ---: | ---: | ---: |
| SVR | 1.2651 | 2.6341 | 1.6230 | 0.4598 |
| Random Forest Regressor | 1.2420 | 2.6847 | 1.6385 | 0.4494 |
| Gradient Boosting Regressor | 1.3125 | 2.7294 | 1.6521 | 0.4402 |
| Linear Regression | 1.4492 | 3.2820 | 1.8116 | 0.3269 |
| Decision Tree Regressor | 1.5351 | 4.4031 | 2.0984 | 0.0970 |

Mô hình tốt nhất là SVR với RMSE thấp nhất và R-squared cao nhất. Lý do là dữ liệu giá nhà có nhiều biến `Location`, `Area`, `Frontage`, `Floors` và tính chất bất động sản có ảnh hưởng không tuyến tính. SVR có khả năng tìm ra mặt phẳng tối ưu trong không gian đặc trưng phức tạp, giúp mô hình thích ứng tốt với các quan hệ tỉ lệ và phi tuyến trong dữ liệu thực tế. Random Forest và Gradient Boosting cũng rất mạnh, nhưng SVR vẫn có ưu thế nhờ khả năng giữ sai số ở mức thấp hơn trên tập kiểm tra.

## 6. Kết luận

Hệ thống hồi quy giá nhà đã triển khai thành công với mô hình SVR là lựa chọn tốt nhất dựa trên độ đo RMSE và R². Mặc dù mức R² chưa chạm tới con số rất cao, kết quả này vẫn hợp lý với dữ liệu thị trường bất động sản thực tế, nơi giá phụ thuộc nhiều vào yếu tố ngầm như vị trí chính xác, tiện ích, quy hoạch, và tâm lý thị trường. Các hình ảnh phân tích được lưu trong thư mục `outputs/` để hỗ trợ giải thích và đánh giá mô hình một cách trực quan.
