# 🌾 Rice Disease AI (Nhận diện bệnh lúa)

Dự án này tập trung vào việc phát hiện và phân loại các bệnh trên lá lúa bằng cách sử dụng kỹ thuật Deep Learning. Mô hình được huấn luyện để nhận diện **10 loại phân lớp** trên lá lúa bao gồm: 
`bacterial_leaf_blight`, `brown_spot`, `healthy`, `leaf_blast`, `leaf_scald`, `narrow_brown_spot`, `neck_blast`, `rice_hispa`, `sheath_blight`, và `tungro`.

---

## 💾 Nguồn tài liệu & Bằng chứng Training (Proof of Work)

Để đảm bảo tính minh bạch và có thể tái lập lại kết quả (Reproducibility), toàn bộ dữ liệu và file huấn luyện đã được public trên Google Drive và Google Colab.

- **Bằng chứng Training (Google Colab Notebook):** [Xem Notebook tại đây](https://colab.research.google.com/drive/1S9eEYPjIzA8IFmRRWDORTC2dkGkCGs9k?usp=sharing)
- **Data Training (Google Drive):** [Tải bộ dữ liệu tại đây](https://drive.google.com/drive/folders/1GhVig_8Xp26VKhY90KRAgxvExEH_Hm8G)
- **File Trọng số Tốt nhất (`best.pt`):** [Tải file trọng số AI tại đây](https://drive.google.com/file/d/1o5DKACrJLomF7C9bIWuX7Jz5c9DtKpuz/view?usp=sharing)

---

## ⚙️ Yêu cầu cài đặt (Requirements)

Để chạy dự án trên máy cá nhân, bạn cần cài đặt các thư viện cần thiết. Hãy dùng lệnh dưới đây để cài đặt tự động toàn bộ môi trường từ file `requirements.txt`:

```bash
pip install -r requirements.txt
```

*(Các thư viện chính bao gồm: PyTorch, Torchvision, OpenCV, Scikit-learn, Matplotlib, TensorBoard, Tqdm, Flask, Werkzeug).*

---

## 🚀 Hướng dẫn sử dụng (Inference)

Dự án cung cấp 2 cách để bạn có thể sử dụng mô hình nhận diện: Giao diện Web (kéo/thả) và dòng lệnh (CLI).

### Cách 1: Sử dụng Web App (Khuyên dùng)
Web App cung cấp giao diện Dark-mode cực kỳ trực quan, hỗ trợ hiệu ứng Glassmorphism và kéo thả ảnh.
1. Khởi động Web API Server:
   ```bash
   python app.py
   ```
2. Mở trình duyệt và truy cập vào: [http://localhost:5000](http://localhost:5000)
3. Kéo thả bức ảnh lá lúa cần nhận diện vào vùng sáng (Upload Area) để nhận kết quả phân tích theo thời gian thực.

### Cách 2: Sử dụng dòng lệnh (CLI Inference)
Nếu bạn muốn dự đoán nhanh qua Terminal, sử dụng script `Predict_lua.py`.
```bash
python Predict_lua.py --image_path "duong_dan_anh.jpg" --checkpoint_path "checkpoint/lua/best.pt"
```
Kết quả sẽ hiển thị dạng Log Text trên màn hình và tự động bung một cửa sổ biểu đồ nếu cần.

---

## 🧠 Cấu trúc Chia Việc (Group Work Architecture)

Dự án này tuân thủ nguyên tắc **Separation of Concerns**, được refactor và chia làm 4 Module riêng biệt để các thành viên trong nhóm phát triển đồng thời (Collaborative Development):

1. `dataset_Lua.py`: Data Pipeline, Augmentation (Kỹ sư Dữ liệu).
2. `models.py`: Khai báo mạng AI ResNet50 + CBAM Attention (Kỹ sư Mô hình).
3. `app.py` & `Predict_lua.py`: Xây dựng API và Inference (Backend Developer).
4. `templates/` & `static/`: Thiết kế giao diện (UI/UX Developer).

Để đọc hướng dẫn chi tiết cách chia việc và Push code, hãy xem file `Phan_Chia_Cong_Viec.md`.

---

## 🏗️ Kiến trúc Cốt lõi của AI (Architecture details)

Mô hình được xây dựng dựa trên kiến trúc xương sống **ResNet50** kết hợp với module cơ chế tập trung **CBAM (Convolutional Block Attention Module)** giúp mô hình tự động tập trung (focus) vào các vùng đốm bệnh thực sự trên lá thay vì học các mảng nền thừa. 

- Đầu ra được tinh chỉnh thông qua một **Custom Head** (bao gồm Linear Layers + Dropout) để dự đoán chính xác 10 lớp. 
- **Loss Function:** Cross-Entropy (tích hợp Label Smoothing).
- **Optimizer:** Adam.
- **Scheduler:** CosineAnnealingLR kết hợp Warmup.

---

## 📈 Huấn luyện (Training)

Dự án hỗ trợ 2 phương pháp huấn luyện (Training) tùy thuộc vào tài nguyên phần cứng của bạn:

### Cách 1: Huấn luyện qua Google Colab (Khuyên dùng)
Nếu máy tính cá nhân không có GPU mạnh, bạn nên sử dụng Google Colab để train mô hình nhanh hơn.
- Truy cập vào **[Colab Notebook của dự án](https://colab.research.google.com/drive/1S9eEYPjIzA8IFmRRWDORTC2dkGkCGs9k?usp=sharing)**.
- Kết nối tới GPU (Runtime > Change runtime type > T4 GPU).
- Chạy toàn bộ các ô lệnh (Run all) để tự động tải Data, cài thư viện và bắt đầu train. File `best.pt` sẽ được xuất ra ở bước cuối cùng.

### Cách 2: Huấn luyện Local bằng `Train_Lua.py` (Kèm TensorBoard)
Nếu máy bạn có sẵn Card đồ họa (GPU NVIDIA) hoặc bạn muốn tự test thuật toán dưới Local:

1. Chạy lệnh để bắt đầu huấn luyện:
   ```bash
   python Train_Lua.py --data_path "duong/dan/data" --epochs 50 --batch_size 32
   ```

2. **Theo dõi với TensorBoard:** Trong quá trình script `Train_Lua.py` chạy, các chỉ số như Loss, Accuracy, F1, và **Confusion Matrix** được lưu tự động theo thời gian thực vào thư mục log. Để mở biểu đồ theo dõi, chạy lệnh:
   ```bash
   tensorboard --logdir "TensorBoard/lua"
   ```
   Sau đó truy cập [http://localhost:6006](http://localhost:6006) trên trình duyệt để phân tích hiệu suất mô hình một cách trực quan.
