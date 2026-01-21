#!/bin/bash

# 藏族文化 AI 绘画助手 - 安装脚本

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     🏔️  藏族文化 AI 绘画助手 - 安装程序  🏔️              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 检查 Python 版本
echo -e "${BLUE}[1/6] 检查 Python 版本...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ 未找到 Python，请先安装 Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✅ Python 版本: $PYTHON_VERSION${NC}"

# 创建虚拟环境
echo -e "${BLUE}[2/6] 创建虚拟环境...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️ 虚拟环境已存在，跳过创建${NC}"
else
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✅ 虚拟环境创建完成${NC}"
fi

source venv/bin/activate

# 升级 pip
echo -e "${BLUE}[3/6] 升级 pip...${NC}"
pip install --upgrade pip

# 安装依赖
echo -e "${BLUE}[4/6] 安装依赖...${NC}"
echo -e "${YELLOW}选择安装模式:${NC}"
echo "  1) 最小安装 (不含 RAG 功能)"
echo "  2) 完整安装 (包含 RAG 功能)"
read -p "请选择 [1/2]: " choice

case $choice in
    1)
        pip install -r requirements-minimal.txt
        ;;
    2)
        pip install -r requirements.txt
        ;;
    *)
        echo -e "${YELLOW}默认选择完整安装${NC}"
        pip install -r requirements.txt
        ;;
esac
echo -e "${GREEN}✅ 依赖安装完成${NC}"

# 创建目录结构
echo -e "${BLUE}[5/6] 创建目录结构...${NC}"
mkdir -p outputs
mkdir -p logs
mkdir -p data/chroma_db
mkdir -p data/knowledge_base
mkdir -p assets/fonts
echo -e "${GREEN}✅ 目录创建完成${NC}"

# 复制配置文件
echo -e "${BLUE}[6/6] 配置环境...${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ 已创建 .env 配置文件${NC}"
        echo -e "${YELLOW}⚠️ 请编辑 .env 文件配置服务地址${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ .env 文件已存在，跳过${NC}"
fi

# 下载字体提示
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ 安装完成！${NC}"
echo ""
echo -e "${YELLOW}📝 后续步骤:${NC}"
echo ""
echo "  1. 编辑 .env 文件配置服务地址"
echo ""
echo "  2. 下载藏文字体 (可选):"
echo "     - Noto Sans Tibetan: https://fonts.google.com/noto/specimen/Noto+Sans+Tibetan"
echo "     - 将字体文件放入 assets/fonts/ 目录"
echo ""
echo "  3. 确保以下服务已启动:"
echo "     - SD WebUI: python launch.py --api"
echo "     - Ollama: ollama serve && ollama pull qwen2.5:7b"
echo ""
echo "  4. 启动应用:"
echo "     ./scripts/start.sh"
echo "     或"
echo "     source venv/bin/activate && python main.py"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"