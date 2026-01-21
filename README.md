# 极地星光汉藏智能图文生成系统  Tibetan-polar-starlight-new

## 🛠️ 项目团队

| 成员姓名     | 团队角色              |
| :----------- | :-------------------- |
| **格智多杰** | 项目负责人 / 核心开发 |
| **杨秋**     | 核心成员              |
| **杨荣禄**   | 核心成员              |
| **边巴**     | 核心成员              |

基于 **Stable Diffusion + Ollama + RAG** 的藏族文化主题图像生成系统。

支持中文输入，自动翻译优化提示词，内置藏族文化知识库增强生成效果。

<img width="2505" height="1470" alt="1" src="https://github.com/user-attachments/assets/bff6d200-d8a5-4428-83aa-7f95a80a3ca8" />
<img width="2529" height="1413" alt="2" src="https://github.com/user-attachments/assets/626b126c-7520-44c4-a54d-27023bafca4c" />
<img width="2517" height="1377" alt="3" src="https://github.com/user-attachments/assets/a63d2f73-5ba4-48d8-bf38-41070d7d3036" />


## ✨ 功能特点

- 🎨 **文生图**：中文描述自动翻译为英文提示词
- 🖼️ **图生图**：基于参考图片进行风格转换
- ✍️ **文字排版**：支持中文、藏文文字叠加
- 🎭 **海报制作**：一键生成带标题的海报
- 🔍 **图片放大**：支持多种放大算法和面部修复
- 📚 **知识库增强**：内置藏族文化知识，自动优化提示词
- 🤖 **LLM 翻译**：使用 Ollama 本地大模型翻译扩写
- 🎙️ **语音输入**：支持百度、讯飞语音识别

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户界面层                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Gradio Web  │  │ Streamlit UI │  │   CLI 命令行  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                    中间件层                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 提示词工程师  │  │   RAG 引擎   │  │  文本渲染器   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   翻译器     │  │  语音识别     │  │  图像处理    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                    服务层                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ SD WebUI API │  │  Ollama API  │  │ ChromaDB向量库│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 用户界面
https://github.com/GudrakDorjee/Tibetan-polar-starlight-new/blob/main/outputs/1.png

![image-20260121172856086](C:\Users\Dorjee\AppData\Roaming\Typora\typora-user-images\image-20260121172856086.png)

![image-20260121172959259](C:\Users\Dorjee\AppData\Roaming\Typora\typora-user-images\image-20260121172959259.png)

## 📋 目录结构

```
polar_starlight-new/
├── config/                 # 配置模块
│   ├── config.py          # 主配置文件
│   └── settings.py        # 环境设置
├── middleware/            # 中间件模块
│   ├── sd_client.py      # Stable Diffusion 客户端
│   ├── ollama_client.py  # Ollama LLM 客户端
│   ├── prompt_engineer.py # 提示词工程
│   ├── rag_engine.py     # RAG 知识检索
│   ├── text_renderer.py  # 文字渲染
│   ├── translator.py     # 翻译器
│   ├── baidu_asr_client.py   # 百度语音识别
│   ├── xunfei_asr_client.py  # 讯飞语音识别
│   ├── google_client.py  # Google 服务集成
│   └── hunyuan_client.py # 腾讯混元集成
├── ui/                    # 用户界面
│   ├── gradio_app.py     # Gradio Web 界面
│   └── streamlit_app.py  # Streamlit 界面
├── cli/                   # 命令行工具
│   └── commands.py       # CLI 命令
├── data/                  # 数据目录
│   ├── knowledge_base/   # 知识库文档
│   ├── vector_db/        # 向量数据库
│   ├── fonts/            # 字体文件
│   ├── terminology.json  # 藏汉术语对照
│   └── lora_configs.json # LoRA 配置
├── outputs/               # 输出目录
│   ├── images/           # 生成的图片
│   └── posters/          # 生成的海报
├── tests/                 # 测试文件
├── main.py               # 主程序入口
└── requirements.txt      # 依赖列表
```

## 🚀 快速开始

### 1. 环境要求

- **Python**: 3.8+
- **操作系统**: Windows / Linux / macOS
- **硬件推荐**:
  - GPU: NVIDIA RTX 系列（8GB+ 显存）
  - CPU: 多核处理器
  - RAM: 16GB+
  - 存储: 20GB+ 可用空间

### 2. 安装依赖服务

#### 安装 Stable Diffusion WebUI

```bash
# 克隆仓库
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# 启动（首次运行会自动下载模型）
# Windows
webui-user.bat --api --listen

# Linux/Mac
./webui.sh --api --listen
```

访问 http://localhost:7860 确认安装成功。

#### 安装 Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows
# 从 https://ollama.com/download 下载安装包
```

启动 Ollama 并下载模型：

```bash
# 启动服务
ollama serve

# 下载中文大模型（推荐）
ollama pull qwen2:7b

# 下载嵌入模型（用于 RAG）
ollama pull nomic-embed-text
```

### 3. 安装项目

```bash
# 克隆项目
git clone https://github.com/GudrakDorjee/Tibetan-polar-starlight-new.git
cd polar_starlight

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 4. 配置环境

创建 `.env` 文件：

```env
# Stable Diffusion WebUI
SD_WEBUI_URL=http://localhost:7860

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2:7b

# 输出目录
OUTPUT_DIR=outputs
LOG_LEVEL=INFO

# Gradio 服务器
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7788
GRADIO_SHARE=false

# 可选：语音识别 API
BAIDU_APP_ID=your_app_id
BAIDU_API_KEY=your_api_key
BAIDU_SECRET_KEY=your_secret_key
```

### 5. 初始化知识库

```bash
# 自动导入默认藏族文化知识库
python main.py --init-kb
```

### 6. 启动应用

```bash
# 启动 Web 界面（推荐）
python main.py

# 启动命令行界面
python main.py --cli

# 检查服务状态
python main.py --check

# 自定义端口
python main.py --host 0.0.0.0 --port 8080

# 创建公共分享链接
python main.py --share
```

访问 http://localhost:7788 开始使用！

## 📖 使用指南

### 文生图（Text-to-Image）

1. 在输入框输入中文描述，例如："一位身穿传统藏袍的康巴汉子，背景是雪山"
2. 选择风格（唐卡、康巴、建筑等）
3. 调整生成参数
4. 点击"生成图片"

系统会自动：

- 查询知识库，补充文化细节
- 使用 LLM 翻译优化为英文提示词
- 调用 SD WebUI 生成图片

### 图生图（Image-to-Image）

1. 上传参考图片
2. 输入想要的变化描述
3. 调整重绘强度（0.3-0.8）
4. 生成新图片

### 文字排版

1. 选择生成的图片
2. 输入要添加的文字（支持中文、藏文）
3. 选择字体、颜色、位置
4. 预览并保存

### 海报制作

1. 上传或生成底图
2. 输入标题和副标题
3. 选择布局模板
4. 一键生成海报

## 🎨 LoRA 模型配置

系统内置以下 LoRA 配置（需自行下载模型）：

| 名称     | 触发词               | 适用场景           | 推荐权重 |
| -------- | -------------------- | ------------------ | -------- |
| 唐卡风格 | thangka_style        | 宗教艺术、传统绘画 | 0.8      |
| 康巴人物 | khampa_style         | 藏族人物、传统服饰 | 0.7      |
| 藏式建筑 | tibetan_architecture | 寺庙、宫殿、民居   | 0.75     |
| 草原风光 | tibetan_landscape    | 自然风景、高原景观 | 0.6      |
| 藏族人物 | tibetan_portrait     | 人物肖像           | 0.7      |

将 LoRA 模型放置在 SD WebUI 的 `models/Lora/` 目录下。

## 🔧 高级配置

### 自定义知识库

在 `data/knowledge_base/` 目录添加文本文件：

```python
# 添加新知识文档
python main.py --cli
> add_knowledge /path/to/document.txt

# 或使用 API
from middleware.rag_engine import create_rag_engine
rag = create_rag_engine()
rag.add_documents_from_file("custom_knowledge.txt")
```

### 术语对照表

编辑 `data/terminology.json` 添加专业术语：

```json
{
  "藏袍": "chuba (tibetan robe)",
  "氆氇": "pulu (tibetan wool fabric)",
  "新术语": "new term translation"
}
```

### 显存优化

在 [config/config.py](config/config.py) 中调整参数：

```python
# 8GB 显存配置
default_params = {
    "width": 768,
    "height": 1024,
    "batch_size": 1,
    "enable_hr": False
}

# 12GB+ 显存可提高分辨率
default_params = {
    "width": 1024,
    "height": 1536,
    "enable_hr": True,
    "hr_scale": 1.5
}
```

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_sd_client.py

# 查看覆盖率
pytest --cov=middleware --cov-report=html
```

## 🐛 常见问题

### Q: SD WebUI 连接失败

**A**: 确保启动时添加 `--api` 参数：

```bash
./webui.sh --api --listen
```

### Q: Ollama 模型未找到

**A**: 先下载模型：

```bash
ollama pull qwen2:7b
ollama list  # 查看已安装模型
```

### Q: 显存不足（OOM）

**A**: 降低生成分辨率或启用 CPU 卸载：

```python
# 在 config.py 中
default_params["width"] = 512
default_params["height"] = 768
```

### Q: 藏文显示乱码

**A**: 安装藏文字体：

- Windows: 安装 Microsoft Himalaya
- Linux: `sudo apt install fonts-tibetan-machine`
- macOS: 安装 Noto Sans Tibetan

### Q: 生成速度慢

**A**:

1. 确认 GPU 驱动正确安装
2. 启用 xformers 加速
3. 减少生成步数（steps: 20-25）

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

```bash
# 格式化代码
black .
isort .

# 类型检查
mypy middleware/

# 代码质量
flake8 .
```

## 📚 相关资源

- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [Ollama](https://ollama.com/)
- [Gradio](https://www.gradio.app/)
- [ChromaDB](https://www.trychroma.com/)

## 📝 更新日志

### v1.0.0 (2026-01-21)

- ✨ 初始版本发布
- 🎨 支持文生图、图生图
- 📚 集成 RAG 知识库
- 🤖 Ollama 本地 LLM 翻译
- ✍️ 藏文文字渲染
- 🎙️ 语音输入支持

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- Stable Diffusion 社区
- Anthropic Claude 团队
- 所有贡献者

## 📮 联系方式

- 项目主页: https://github.com/GudrakDorjee/Tibetan-polar-starlight
- 问题反馈: https://github.com/GudrakDorjee/Tibetan-polar-starlight/issues
- 邮箱: gudrak_cs@163.com

---

<div align="center">


**⭐ 如果这个项目对您有帮助，请给个 Star！⭐**

Made with ❤️ for Tibetan Culture


</div>



