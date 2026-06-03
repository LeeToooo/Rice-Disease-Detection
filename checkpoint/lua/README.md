# ⚠️ Quan trọng: Tải File Trọng số (best.pt)

Do giới hạn kích thước file của Github (không hỗ trợ lưu trữ tệp tin quá 100MB), file trọng số AI đã được huấn luyện (`best.pt`) không được tải trực tiếp lên kho lưu trữ này.

Để ứng dụng Web hoặc Code dự đoán có thể hoạt động, bạn **bắt buộc** phải tải file trọng số về và đặt vào đúng thư mục hiện tại (`checkpoint/lua/`).

## 📥 Link tải Weights:
- **Google Drive:** [Nhấn vào đây để tải file best.pt](https://drive.google.com/file/d/1o5DKACrJLomF7C9bIWuX7Jz5c9DtKpuz/view?usp=sharing)

## ⚙️ Hướng dẫn cài đặt sau khi tải:
1. Tải file `best.pt` từ đường link phía trên.
2. Đổi tên file (nếu bị Google Drive thêm các ký tự thừa) thành chính xác: `best.pt`.
3. Đặt file đó vào thư mục này. Cấu trúc thư mục cuối cùng phải trông như thế này:
   ```text
   Lúa Detect/
   └── checkpoint/
       └── lua/
           ├── README.md (File bạn đang đọc)
           └── best.pt   (File bạn vừa tải về)
   ```
4. Quay trở lại thư mục gốc và chạy lệnh `python app.py` để tận hưởng thành quả!
