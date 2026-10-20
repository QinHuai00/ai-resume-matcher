@echo off
chcp 65001 >nul
setlocal
rem AI 简历—岗位匹配助手：Windows 一键启动网页版（可选，未装 streamlit 时只给中文提示）
rem 用法：双击本文件，或在项目根目录执行 start-ui.bat
cd /d "%~dp0"

rem 优先使用项目内虚拟环境，没有就回退到系统 python
set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%PY%" goto check_streamlit

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 Python。请安装 Python 3.10+ 并加入 PATH，
  echo        或在项目根目录创建虚拟环境：py -m venv .venv
  pause
  exit /b 1
)
set "PY=python"

:check_streamlit
"%PY%" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
  echo [提示] 当前 Python 未安装 streamlit，网页版不可用；命令行版不受影响：
  echo        "%PY%" -m app.cli analyze --jd data\samples\jd_ai_app.md --resume data\samples\resume_zhang.md
  echo        安装网页依赖： "%PY%" -m pip install streamlit
  pause
  exit /b 0
)

echo 启动简历匹配助手网页版：http://127.0.0.1:8502
echo 关闭本窗口即停止服务。
"%PY%" -m streamlit run ui\app.py --server.port 8502 --browser.gatherUsageStats false
pause
