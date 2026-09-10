@echo off
title GDG HUST Photobooth Launcher
echo ==========================================
echo    GDG HUST PHOTOBOOTH - QUICK LAUNCHER   
echo ==========================================

where docker >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [+] Docker duoc tim thay!
    echo [+] Dang khoi chay Docker Compose...
    docker compose up --build -d
    echo.
    echo =========================================================
    echo Photobooth da san sang tai: http://localhost:8000
    echo =========================================================
    start http://localhost:8000
    echo.
    echo Lenh xem log: docker compose logs -f
    echo Lenh dung:    docker compose down
    pause
    exit /b 0
)

echo [!] Khong tim thay Docker, dang kiem tra Python...
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [-] Loi: Vui long cai dat Docker Desktop hoac Python 3.10+
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [+] Tao moi truong ao .venv...
    python -m venv .venv
)

call .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pip install -r requirements-docker.txt
start http://localhost:8000
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
