@echo off
chcp 65001 >nul
echo ========================================
echo 极地星光汉藏智能图文生成系统
echo GPU 加速模式启动脚本
echo ========================================
echo.

REM 设置 GPU 配置
set OLLAMA_NUM_GPU=1
echo [配置] 使用 GPU 数量: %OLLAMA_NUM_GPU%

REM 检查 .env 文件
if not exist .env (
    echo [警告] 未找到 .env 文件，正在从 .env.example 创建...
    copy .env.example .env
    echo [提示] 请编辑 .env 文件配置您的服务地址
    echo.
)

REM 检查 Ollama 服务
echo [检查] 正在检查 Ollama 服务...
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Ollama 服务未启动！
    echo [提示] 请先启动 Ollama 服务：ollama serve
    echo.
    pause
    exit /b 1
)
echo [成功] Ollama 服务运行正常
echo.

REM 检查 SD WebUI 服务
echo [检查] 正在检查 SD WebUI 服务...
curl -s http://127.0.0.1:7860 >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] SD WebUI 服务未启动！
    echo [提示] 如需使用图像生成功能，请启动 SD WebUI
    echo.
)

REM 启动应用
echo [启动] 正在启动 Streamlit 应用...
echo [提示] 应用将在浏览器中自动打开
echo [提示] 按 Ctrl+C 可停止服务
echo.
echo ========================================
echo.

streamlit run ui/streamlit_app.py --server.port 8501

pause
