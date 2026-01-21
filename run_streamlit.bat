@echo off
echo ========================================
echo 启动藏族文化 AI 绘画助手 (Streamlit)
echo ========================================
echo.

cd /d "%~dp0"

echo 正在启动 Streamlit 应用...
streamlit run ui\streamlit_app.py --server.port 7860

pause
