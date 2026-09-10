@echo off
chcp 65001 >nul
setlocal
set "HEJI_PYTHON=D:\conda\envs\forest-species\python.exe"
set "HEJI_UI=%~dp0code\day69_web_ui.py"

if not exist "%HEJI_PYTHON%" (
  echo [禾迹] 找不到 Python：%HEJI_PYTHON%
  echo 请确认 forest-species 环境仍位于 D:\conda\envs\forest-species。
  pause
  exit /b 1
)

echo [禾迹] 正在启动本地视觉台...
echo [禾迹] 浏览器将打开 http://127.0.0.1:7869
echo [禾迹] 使用完成后关闭本窗口即可停止服务。
"%HEJI_PYTHON%" -X utf8 "%HEJI_UI%"

if errorlevel 1 (
  echo.
  echo [禾迹] 启动失败，请保留本窗口中的错误信息。
  pause
)
