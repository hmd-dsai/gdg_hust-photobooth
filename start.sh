#!/bin/bash
set -e

echo "=========================================="
echo "   GDG HUST PHOTOBOOTH - QUICK LAUNCHER   "
echo "=========================================="

# Check arguments: if '--docker' or 'docker' is passed, run with Docker
USE_DOCKER=false
if [ "$1" == "--docker" ] || [ "$1" == "docker" ]; then
    USE_DOCKER=true
fi

if [ "$USE_DOCKER" = true ]; then
    if command -v docker &> /dev/null && docker compose version &> /dev/null; then
        echo "[+] Khởi chạy bằng Docker Compose..."
        docker compose up --build -d
        echo ""
        echo "========================================================="
        echo "🎉 Photobooth (Docker) đang chạy tại: http://localhost:8000"
        echo "========================================================="
        if command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:8000 &> /dev/null &
        fi
        echo "Xem log: docker compose logs -f"
        echo "Dừng:    docker compose down"
        exit 0
    else
        echo "[-] Không tìm thấy Docker. Chuyển sang chạy trực tiếp bằng Python..."
    fi
fi

# Chạy trực tiếp bằng Python (.venv) - Khuyên dùng: Siêu nhanh, Hot-reload, không tốn RAM
echo "[+] Chạy trực tiếp với Python local (.venv) [Khuyên dùng]..."
if ! command -v python3 &> /dev/null; then
    echo "[-] Lỗi: Không tìm thấy Python 3 trên máy!"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "[+] Đang tạo môi trường ảo Python .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
    pip install -r requirements-docker.txt
else
    source .venv/bin/activate
fi

# Mở trình duyệt sau 1.5 giây
(sleep 1.5 && (xdg-open http://localhost:8000 || open http://localhost:8000) &> /dev/null) &

echo ""
echo "========================================================="
echo "🎉 Photobooth đang chạy tại: http://localhost:8000"
echo "   (Chế độ Local Hot-Reload: sửa file là cập nhật ngay!)"
echo "   Nếu muốn chạy qua Docker: ./start.sh --docker"
echo "========================================================="
echo ""

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
