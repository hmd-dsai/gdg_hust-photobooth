# Hướng Dẫn Đóng Gói & Triển Khai GDG HUST Photobooth Đa Nền Tảng

Dự án này đã được đóng gói hoàn chỉnh thành một **Web Application hiện đại** kết hợp với **Docker Container**, giúp chạy mượt mà trên mọi hệ điều hành: **Windows, macOS, Linux**, và người dùng có thể dùng mọi thiết bị có trình duyệt (Laptop, PC, iPad/Tablet, điện thoại) để trải nghiệm mà không lo xung đột môi trường Python.

---

## 🌟 Các Tính Năng Nổi Bật

1. **Đa nền tảng (Cross-platform) & Zero Config**:
   - Chạy trên container chuẩn hóa với **Python 3.10**.
   - Sử dụng phiên bản **PyTorch CPU tối ưu** và MediaPipe headless giúp image gọn nhẹ, khởi động nhanh.
   - **Offline-ready**: Toàn bộ trọng số mô hình ViT (`vit-face-expression` ~336MB) và cử chỉ tay (`gesture_model.pt`) đã được tải sẵn và nhúng trực tiếp vào image khi build.
2. **Giao diện Web Photobooth hiện đại (GDG Branding)**:
   - **Chế độ Thử thách (Challenge Mode)**: Lần lượt hướng dẫn người chơi thực hiện 4 dáng (`Happy -> Angry -> Surprise -> GDG Hand Sign`). Khi giữ đúng biểu cảm đủ 1.0 giây, hệ thống kích hoạt hiệu ứng flash + âm thanh chụp ảnh và tự động chuyển bước.
   - **Chế độ Tự do (Live Mirror)**: So sánh trực quan màn hình webcam và ảnh mẫu tham chiếu tương ứng thời gian thực.
   - **Tự động ghép Photobooth Strip 4x2**: Sau khi hoàn tất, hệ thống tự xuất dải ảnh photobooth mang thương hiệu GDG-HUST kèm ngày giờ và nút bấm tải về máy.
3. **Lưu trữ ảnh đã chụp**:
   - Các ảnh chụp và dải ảnh strip được tự động lưu vào thư mục `output/session_<timestamp>/` trên máy chủ.

---

## 🚀 Cách Khởi Chạy Nhanh Nhất (1 Lệnh / 1 Click)

### Cách 1: Trên Linux / macOS
Mở terminal trong thư mục dự án và chạy:
```bash
./start.sh
```
*(Script sẽ tự động build container Docker và mở trình duyệt tại `http://localhost:8000`)*.

Hoặc nếu bạn muốn dùng lệnh Docker Compose tiêu chuẩn:
```bash
docker compose up --build -d
```
- Mở trình duyệt: `http://localhost:8000`
- Xem log: `docker compose logs -f`
- Dừng ứng dụng: `docker compose down`

---

### Cách 2: Trên Windows
Click đúp chuột vào file:
```cmd
start.bat
```
*(File này sẽ tự động khởi chạy Docker Compose hoặc tự tạo `.venv` nếu máy tính chưa cài Docker).*

---

### Cách 3: Chạy trực tiếp qua Python (Không dùng Docker)
Nếu muốn phát triển hoặc chạy trực tiếp trên máy host:
```bash
# Tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate

# Cài đặt thư viện
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-docker.txt

# Khởi chạy server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Truy cập: `http://localhost:8000`

---

## 📱 Dùng Cho Nhiều Thiết Bị (iPad, Điện Thoại Qua Wi-Fi)

Khi Photobooth Server đang chạy trên laptop/máy tính của ban tổ chức:
1. Xem địa chỉ IP của máy (ví dụ: `192.168.1.100`).
2. Trên iPad, điện thoại hoặc laptop khác kết nối chung mạng Wi-Fi, mở trình duyệt truy cập:
   ```
   http://192.168.1.100:8000
   ```
*(Lưu ý: Để trình duyệt cho phép truy cập camera qua HTTP trên mạng LAN, trên Chrome có thể vào `chrome://flags/#unsafely-treat-insecure-origin-as-secure` và thêm `http://192.168.1.100:8000`)*.
