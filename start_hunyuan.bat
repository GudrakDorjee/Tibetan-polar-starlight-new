@echo off
chcp 65001 >nul
echo ========================================
echo 腾讯混元 (Hunyuan) llama-server 启动脚本
echo ========================================
echo.

:: llama-server.exe 的完整路径
set SERVER_EXE=D:\Llama\llama-b7601-bin-win-cuda-12.4-x64\llama-server.exe

:: 你的 hunyuan-q4.gguf 模型完整路径
set MODEL_PATH=E:\HuggingFace\tencent-Hunyuan-MT-7B\hunyuan-q4.gguf

:: 服务端口 (避开 Ollama 的 11434)
set PORT=8080

:: 检查 llama-server.exe 是否存在
if not exist "%SERVER_EXE%" (
    echo [错误] 未找到 llama-server.exe: %SERVER_EXE%
    echo 请检查路径是否正确
    pause
    exit /b 1
)

:: 检查模型文件是否存在
if not exist "%MODEL_PATH%" (
    echo [错误] 未找到模型文件: %MODEL_PATH%
    echo 请检查路径是否正确
    pause
    exit /b 1
)

echo [信息] llama-server 路径: %SERVER_EXE%
echo [信息] 模型文件路径: %MODEL_PATH%
echo [信息] 服务端口: %PORT%
echo.
echo [启动] 正在启动 Hunyuan llama-server...
echo.

:: 启动 llama-server
:: 参数说明:
:: -m: 模型路径
:: --port: 服务端口
:: -c: 上下文长度
:: -n: 预测的最大 token 数
:: --host: 监听地址 (0.0.0.0 表示监听所有网络接口)
echo [提示] 服务启动后，可以通过以下地址访问：
echo         - http://localhost:%PORT%
echo         - http://127.0.0.1:%PORT%
echo.
"%SERVER_EXE%" ^
    -m "%MODEL_PATH%" ^
    --port %PORT% ^
    -c 4096 ^
    -n 2048 ^
    --host 0.0.0.0

:: 如果 llama-server 退出，显示错误
if errorlevel 1 (
    echo.
    echo [错误] llama-server 启动失败或已退出
    pause
)
