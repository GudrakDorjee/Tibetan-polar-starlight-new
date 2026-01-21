@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║        🏔️  藏族文化 AI 绘画助手  🏔️                      ║
echo ║           Tibetan Culture AI Art Assistant               ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

:: 获取脚本目录
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"
cd /d "!PROJECT_ROOT!"

:: 检查 Python
echo [INFO] 检查 Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)
echo [OK] Python 已安装

:: 检查虚拟环境
if exist "venv\Scripts\activate.bat" (
    echo [OK] 虚拟环境已存在
    call venv\Scripts\activate.bat
) else (
    echo [INFO] 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] 虚拟环境创建失败
        pause
        exit /b 1
    )
    call venv\Scripts\activate.bat
    echo [INFO] 升级 pip...
    python -m pip install --upgrade pip
    echo [INFO] 安装依赖...
    if exist "requirements.txt" (
        pip install -r requirements.txt
    ) else (
        echo [WARN] requirements.txt 未找到
    )
    echo [OK] 虚拟环境创建完成
)

:: 检查依赖
echo [INFO] 检查依赖...
python -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo [INFO] 安装依赖...
    if exist "requirements.txt" (
        pip install -r requirements.txt
    ) else (
        echo [ERROR] requirements.txt 未找到
        pause
        exit /b 1
    )
) else (
    echo [OK] 依赖检查完成
)

:: 创建必要的目录
echo [INFO] 创建必要的目录...
if not exist "outputs" (
    mkdir outputs
    echo [OK] 创建 outputs 目录
)
if not exist "logs" (
    mkdir logs
    echo [OK] 创建 logs 目录
)
if not exist "data" mkdir data
if not exist "data\chroma_db" (
    mkdir data\chroma_db
    echo [OK] 创建 data\chroma_db 目录
)
if not exist "assets" mkdir assets
if not exist "assets\fonts" (
    mkdir assets\fonts
    echo [OK] 创建 assets\fonts 目录
)

:: 检查服务
echo.
echo [INFO] 检查服务状态...
echo.

:: 检查 SD WebUI
curl -s "http://127.0.0.1:7860/sdapi/v1/options" >nul 2>&1
if errorlevel 1 (
    echo [WARN] SD WebUI: 未运行 - 请确保已启动 SD WebUI --api
) else (
    echo [OK] SD WebUI: 已运行
)

:: 检查 Ollama
curl -s "http://127.0.0.1:11434/api/tags" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Ollama: 未运行 - 请运行 ollama serve
) else (
    echo [OK] Ollama: 已运行
)

:: 启动应用
echo.
echo [INFO] 启动应用...
echo.

python main.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] 应用启动失败，错误代码: !errorlevel!
    echo.
)

pause