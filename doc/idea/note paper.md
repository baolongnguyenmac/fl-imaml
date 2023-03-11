# Energy-Efficient and Federated Meta-Learning via Projected Stochastic Gradient Ascent

## Tóm tắt

- Paper này đề xuất một framework FL-ML. Mục tiêu là cho phép học một meta-model có thể fine-tune một task mới với một vài mẫu trong cài đặt phân tán với năng lượng tính toán và truyền thông thấp.

- Mỗi task là một data owner và một lượng task nhất định được dùng để train meta-model. Mỗi task được train offline trên một data owner. Chúng tôi đề xuất một thuật toán nhẹ bắt nguồn từ local model, sử dụng projected stochastic gradient descent.

- Chú ý một vài cái mà việc xấp xỉ đạo hàm sẽ mang lại: tính toán ma trận hessian, double looping, ma trận nghịch đảo.

- Nó sẽ tốt hơn MAML, iMAML!

## 1. Intro

- ML được đề xuất như 1 framework cho phép agent học nhanh bằng cách tận dụng một khởi tạo tốt được train bởi nhiều task tương tự trước đó. ML có thể được xem như bài toán tối ưu 2 cấp độ, sử dụng 2 vòng lặp:
    - inner loop opt: thể hiện sự thích ứng với 1 task cho trước
    - outer loop opt: train meta-model

- 
