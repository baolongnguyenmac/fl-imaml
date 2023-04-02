# Cải tiến hệ thống FL bằng Meta-learning (iMAML)

## 1. Giới thiệu

### 1.1. Động lực

- Trong thời đại IoT, số lượng thiết bị biên ngày càng tăng cao, đi kèm theo đó là sự gia tăng của dữ liệu sinh ra trên loại thiết bị này.
- Phương pháp truyền thống dùng để khai thác loại dữ liệu này yêu cầu truyền dữ liệu về 1 server để tiến hành huấn luyện mô hình. Điều này không những ảnh hưởng nghiêm trọng đến quyền riêng tư dữ liệu của người dùng mà còn tốn rất nhiều chi phí phần cứng cho server và đường truyền.
- Federated learning (FL) ra đời như một giải pháp thay thế cho phương pháp huấn luyện truyền thống. Về cơ bản, phương pháp học này nhằm huấn luyện các mô hình học trên các tập dữ liệu riêng biệt, phân phối tại các thiết bị biên. Điều này giúp cho hệ thống FL đảm bảo được quyền riêng tư và tiêu tốn ít chi phí phần cứng hơn phương pháp học cũ.

### 1.2. Vấn đề của hệ thống FL

- Dữ liệu phân bố trên thiết bị biên trong cài đặt Horizontal FL thường không tuân theo cùng một phân phối. Người ta gọi đây là kịch bản dữ liệu non-IID.
- **Hệ thống FL với thuật toán gốc `FedAvg` đã được chứng minh là bị giảm hiệu suất nghiêm trọng trên dữ liệu non-IID**.
- Đã có nhiều nghiên cứu đề xuất phương pháp cải thiện hệ thống trên. Một trong số đó là sử dụng các thuật toán Meta-learning trong huấn luyện để cung cấp cho global model khả năng thích ứng nhanh trên tập dữ liệu mới.
- Tuy nhiên, **các thuật toán Meta-learning như `MAML`, `Meta-SGD` đòi hỏi phải tính toán đạo hàm cấp 2, khiến cho hệ thống tiêu tốn rất nhiều chi phí tính toán**.
- Việc xấp xỉ đạo hàm cấp 2 của một số phương pháp như `FOMAML`, `Reptile` thực tế cho kết quả không cao.

### 1.3. Mục tiêu

- Cải thiện hiệu suất của hệ thống FL trong kịch bản dữ liệu non-IID bằng các thuật toán Meta-learning nhưng vẫn đảm bảo chi phí tính toán thấp.
- Việc này được thực hiện bằng cách sử dụng thuật toán `iMAML`. `iMAML` cho phép xấp xỉ đạo hàm bậc 2 của `MAML` một cách hiệu quả (hiệu suất không bị sụt giảm quá nhiều nhưng chi phí tính toán giảm đáng kể).

## 2. Các công trình liên quan

- Tại đây trình bày về các nghiên cứu sử dụng Meta-learning trong việc cải thiện hiệu suất của hệ thống FL.
- Các nghiên cứu áp dụng Meta-learning vào hệ thống FL bằng cách:
    - Coi mỗi client trong hệ thống FL là một task với dữ liệu riêng biệt.
    - Coi global model trong cài đặt FL tương đương với Meta-model trong cài đặt của Meta-learning.
- Cài đặt của hệ thống FL khi áp dụng các thuật toán meta-learning:
    - Tập dữ liệu tại mỗi client được chia thành tập support $D_{support}$ chứa 20% dữ liệu và tập query $D_{query}$ chứa 80% dữ liệu.
    - Trong quá trình huấn luyện, các thuật toán Meta-learning sẽ thực hiện:
        - Tối ưu cấp thấp trên tập $D_{support}$ để thu được tham số tối ưu cục bộ $\phi_i$ cho client $i$.
        - Tối ưu cấp cao trên tập $D_{query}$ để thu được tham số tối ưu toàn cục $\theta$.

<p align="center">
  <img width="700" height="300" src="./img/draft.svg">
</p>

### 2.1. FL + `MAML`

- Sử dụng `MAML` để tối ưu hệ thống FL trong kịch bản dữ liệu non-IID như sau:
    - Tối ưu cấp thấp: Chạy $k$ bước gradient descent
    $$\phi_i \leftarrow \phi_i - \alpha\nabla_{\phi_i}f\left(\phi_i, D_{support}\right)$$
    - Tối ưu cấp cao: Thực hiện trên tập $D_{query}$ của toàn bộ client
    $$
    \begin{aligned}
    \theta &\leftarrow \theta - \beta\sum_{\text{client }i}{\nabla_{\theta}f\left(\phi_i, D_{query}\right)}\\
    &\leftarrow \theta - \beta\sum_{\text{client }i}{\frac{\partial f\left(\phi_i, D_{query}\right)}{\partial \phi_i}\times \frac{\partial \phi_i(\theta)}{\partial \theta}}\\
    \end{aligned}
    $$

    <!-- &\leftarrow \theta - \beta\sum_{\text{client }i}{\frac{\partial f\left(\phi_i, D_{query}\right)}{\partial \phi_i}\times \frac{\partial}{\partial \theta}\left(\phi_i - \alpha\nabla_{\phi_i}f\left(\phi_i, D_{support}\right)\right)} -->

- Đến đây, có 2 thành phần cần phải tính toán là:
    - $\frac{\partial f\left(\phi_i, D_{query}\right)}{\partial \phi_i}$: Tính được rất dễ dàng và chi phí tính toán thấp.
    - $\frac{\partial \phi_i(\theta)}{\partial \theta}$: **Chi phí tính toán và lưu trữ rất cao do xảy ra quá trình đạo hàm bậc 2**.

- Do đó, mặc dù đạt được hiệu suất tính toán tốt, `MAML` thực sự tiêu tốn rất nhiều chi phí để hoạt động được.

### 2.2. FL + `First-order MAML`

- Để tránh việc tốn chi phí tính toán với đạo hàm bậc 2, FOMAML đề xuất một phép xấp xỉ như sau:

$$\nabla_{\theta}f\left(\phi_i, D_{query}\right) \approx \frac{\partial f\left(\phi_i, D_{query}\right)}{\partial \phi_i}$$

- Lúc này, phương trình cập nhật tham số mô hình toàn cục trở thành:

$$\theta \leftarrow \theta - \beta\sum_{\text{client }i}{\frac{\partial f\left(\phi_i, D_{query}\right)}{\partial \phi_i}}$$

- Mặc dù giảm đáng kể được thời gian và bộ nhớ cần dùng để tính toán, việc xấp xỉ quá trình meta-optimization bằng đạo hàm bậc 1 của hàm mục tiêu cho kết quả không tốt vì thuật toán học không nắm bắt được mối quan hệ giữa hàm mục tiêu $f$ và tham số khởi tạo $\theta$.

### 2.3. FL + `Reptile`

- Đây là một phương pháp khác cho phép xấp xỉ đạo hàm trong quá trình tối ưu cấp cao. Cụ thể:

$$\nabla_{\theta}f\left(\phi_i, D_{query}\right) \approx \epsilon \left(\phi_i - \theta\right)$$ Trong đó, $\epsilon$ là siêu tham số học.

- Lúc này, phương trình cập nhật tham số mô hình toàn cục trở thành:

$$\theta \leftarrow \theta - \epsilon\sum_{\text{client }i}{\left(\phi_i - \theta\right)}$$

- Tương tự như `FOMAML`, `Reptile` cho thời gian tính toán nhanh cũng như tiết kiệm được bộ nhớ tính toán nhưng lại gặp khó khăn trong việc nắm bắt mối quan hệ giữa hàm mục tiêu và tham số khởi tạo, dẫn đến việc phương pháp này cũng không đạt được hiệu suất cao trong các cài đặt thực tế.

### 2.4. `iMAML`

<!--  -->
