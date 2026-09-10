@echo off
setlocal
set "HEJI_PYTHON=D:\conda\envs\forest-species\python.exe"
set "HEJI_UI=%~dp0code\day69_web_ui.py"

if not exist "%HEJI_PYTHON%" (
  echo [Heji] Python was not found: %HEJI_PYTHON%
  echo Check that the forest-species environment still uses this path.
  pause
  exit /b 1
)

echo [Heji] Starting the local field-vision interface...
echo [Heji] The browser will open http://127.0.0.1:7869
echo [Heji] Close this window when you want to stop the service.
"%HEJI_PYTHON%" -X utf8 "%HEJI_UI%"

if errorlevel 1 (
  echo.
  echo [Heji] Startup failed. Keep this window open and read the error above.
  pause
)
