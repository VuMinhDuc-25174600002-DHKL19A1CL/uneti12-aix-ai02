@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
 set "PY=.venv\Scripts\python.exe"
) else if exist "..\.venv\Scripts\python.exe" (
 set "PY=..\.venv\Scripts\python.exe"
) else (
 set "PY=python"
)
if not exist "app.py" (
 echo [LOI] Khong thay app.py. Hay chay BAT nay trong project da cap nhat.
 pause
 exit /b 1
)
"%PY%" -c "import streamlit; import PIL" >nul 2>&1
if errorlevel 1 (
 echo [LOI] Thieu Streamlit hoac Pillow.
 echo Cai: "%PY%" -m pip install -r requirements.txt
 pause
 exit /b 1
)
echo.
echo WEB AI02 V13: http://localhost:8501
echo Camera chi hoat dong tren localhost hoac HTTPS.
echo Da goc: co the TAI 2-6 anh tren web.
echo De dung web: Ctrl+C.
echo.
"%PY%" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
if errorlevel 1 (
 echo [LOI] Khong chay duoc. Tat phien Streamlit cu neu cong 8501 dang ban.
 pause
 exit /b 1
)
