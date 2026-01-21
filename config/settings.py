"""
应用配置文件
"""

from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class Settings:
    """应用设置"""
    
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # ==================== SD WebUI 配置 ====================
    sd_webui_url: str = os.getenv(
        "SD_WEBUI_URL", 
        "http://127.0.0.1:7860"
    )
    sd_webui_username: Optional[str] = os.getenv("SD_WEBUI_USERNAME")
    sd_webui_password: Optional[str] = os.getenv("SD_WEBUI_PASSWORD")
    
    # ==================== Ollama 配置 ====================
    ollama_url: str = os.getenv(
        "OLLAMA_URL",
        "http://127.0.0.1:11434"
    )
    ollama_model: str = os.getenv(
        "OLLAMA_MODEL",
        "qwen2.5:7b"  # 使用 qwen2.5:7b，也可以改为 "TiLamb-7B"（藏文专用）、"" 等
    )
    ollama_num_gpu: int = int(os.getenv(
        "OLLAMA_NUM_GPU",
        "1"  # 使用的 GPU 数量，0 表示仅使用 CPU
    ))

    # ==================== Hunyuan llama-server 配置 ====================
    hunyuan_url: str = os.getenv(
        "HUNYUAN_URL",
        "http://127.0.0.1:8080"  # llama-server 默认端口 8080
    )
    hunyuan_enabled: bool = os.getenv(
        "HUNYUAN_ENABLED",
        "false"  # 默认禁用，需要手动启动 llama-server
    ).lower() == "true"

    # ==================== 翻译引擎配置 ====================
    # 可选: "ollama", "hunyuan", "google"
    default_translation_engine: str = os.getenv(
        "DEFAULT_TRANSLATION_ENGINE",
        "ollama"  # 默认使用 Ollama
    )

    # ==================== 讯飞语音识别配置 ====================
    xunfei_app_id: str = os.getenv("XUNFEI_APP_ID", "")
    xunfei_api_key: str = os.getenv("XUNFEI_API_KEY", "")
    xunfei_api_secret: str = os.getenv("XUNFEI_API_SECRET", "")
    xunfei_asr_url: str = os.getenv(
        "XUNFEI_ASR_URL",
        "wss://iat.cn-huabei-1.xf-yun.com/v1"  # 统一使用多语种大模型服务
    )

    # ==================== 百度语音识别配置（藏语）====================
    baidu_asr_app_key: str = os.getenv("BAIDU_ASR_APP_KEY", "")
    baidu_asr_secret_key: str = os.getenv("BAIDU_ASR_SECRET_KEY", "")

    # ==================== 藏语配置 ====================
    tibetan_enabled: bool = os.getenv("TIBETAN_ENABLED", "true").lower() == "true"
    tibetan_translate_url: str = os.getenv("TIBETAN_TRANSLATE_URL", "https://fanyi-api.baidu.com/api/trans/vip/translate")
    tibetan_app_key: str = os.getenv("TIBETAN_APP_KEY", "")
    tibetan_secret_key: str = os.getenv("TIBETAN_SECRET_KEY", "")

    # 藏语识别引擎选择: "xunfei" 或 "baidu"
    # 注意: 讯飞标准 IAT API 可能不支持藏语,建议使用百度
    tibetan_asr_engine: str = os.getenv("TIBETAN_ASR_ENGINE", "baidu")

    # ==================== RAG 配置 ====================
    rag_persist_dir: Path = Path(os.getenv(
        "RAG_PERSIST_DIR",
        str(PROJECT_ROOT / "data" / "rag")
    ))
    rag_backend: str = os.getenv(
        "RAG_BACKEND",
        "chroma"  # 可选: "chroma", "milvus", "pinecone"
    )
    rag_embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL",
        "nomic-embed-text"  # Ollama 内置的嵌入模型
    )
    rag_collection_name: str = os.getenv(
        "RAG_COLLECTION_NAME",
        "tibetan_knowledge"
    )
    
    # ==================== 文字渲染配置 ====================
    fonts_dir: Path = Path(os.getenv(
        "FONTS_DIR",
        str(PROJECT_ROOT / "assets" / "fonts")
    ))
    
    # 字体文件
    default_font: str = "NotoSansTibetan-Regular.ttf"
    bold_font: str = "NotoSansTibetan-Bold.ttf"
    
    # ==================== 图生图默认参数 ====================
    default_width: int = int(os.getenv("DEFAULT_WIDTH", "512"))
    default_height: int = int(os.getenv("DEFAULT_HEIGHT", "512"))
    default_steps: int = int(os.getenv("DEFAULT_STEPS", "30"))
    default_cfg_scale: float = float(os.getenv("DEFAULT_CFG_SCALE", "7.5"))
    default_negative_prompt: str = os.getenv(
        "DEFAULT_NEGATIVE_PROMPT",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    )
    
    # ==================== 生成超时配置 ====================
    generation_timeout: int = int(os.getenv("GENERATION_TIMEOUT", "300"))  # 5分钟
    connection_timeout: int = int(os.getenv("CONNECTION_TIMEOUT", "10"))
    
    # ==================== 输出配置 ====================
    output_dir: Path = Path(os.getenv(
        "OUTPUT_DIR",
        str(PROJECT_ROOT / "outputs")
    ))
    
    # ==================== Gradio 配置 ====================
    gradio_host: str = os.getenv("GRADIO_HOST", "0.0.0.0")
    gradio_port: int = int(os.getenv("GRADIO_PORT", "7871"))
    gradio_share: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    
    # ==================== 日志配置 ====================
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: Path = Path(os.getenv(
        "LOG_DIR",
        str(PROJECT_ROOT / "logs")
    ))
    
    # ==================== 样式和质量选项 ====================
    STYLE_OPTIONS = {
        "通用": None,
        "唐卡": "thangka_painting, traditional tibetan art, gold leaf",
        "油画": "oil painting, artistic, masterpiece",
        "水墨": "watercolor, ink painting, minimalist",
        "摄影": "photography, realistic, professional, high quality",
        "3D渲染": "3D rendering, CGI, digital art, modern",
        "民间艺术": "folk art, traditional, ethnic, vibrant colors",
    }
    
    QUALITY_OPTIONS = {
        "低质量": "low quality",
        "普通": "normal quality",
        "高质量": "high quality, masterpiece, best quality",
        "超高质量": "ultra high quality, 8k, masterpiece, best quality, intricate details",
    }
    
    COMPOSITION_OPTIONS = {
        "无": None,
        "特写": "close-up, portrait, detailed face",
        "中景": "medium shot, half body",
        "远景": "wide shot, full body, landscape",
        "俯视": "overhead view, bird's eye view",
        "仰视": "worm's eye view, low angle",
        "对称": "symmetrical, centered composition",
        "三分法": "rule of thirds, balanced composition",
    }
    
    def __init__(self):
        """初始化配置，创建必要的目录"""
        # 创建必要的目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rag_persist_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.fonts_dir.mkdir(parents=True, exist_ok=True)
    
    def get_style_options(self) -> list:
        """获取风格选项列表"""
        return list(self.STYLE_OPTIONS.keys())
    
    def get_quality_options(self) -> list:
        """获取质量选项列表"""
        return list(self.QUALITY_OPTIONS.keys())
    
    def get_composition_options(self) -> list:
        """获取构图选项列表"""
        return list(self.COMPOSITION_OPTIONS.keys())
    
    def get_style_prompt(self, style: Optional[str]) -> Optional[str]:
        """根据风格获取对应的提示词"""
        if not style or style not in self.STYLE_OPTIONS:
            return None
        return self.STYLE_OPTIONS[style]
    
    def get_quality_prompt(self, quality: Optional[str]) -> Optional[str]:
        """根据质量获取对应的提示词"""
        if not quality or quality not in self.QUALITY_OPTIONS:
            return None
        return self.QUALITY_OPTIONS[quality]
    
    def get_composition_prompt(self, composition: Optional[str]) -> Optional[str]:
        """根据构图获取对应的提示词"""
        if not composition or composition not in self.COMPOSITION_OPTIONS:
            return None
        return self.COMPOSITION_OPTIONS[composition]
    
    def __repr__(self):
        return f"""
        ===== 应用配置 =====
        SD WebUI: {self.sd_webui_url}
        Ollama: {self.ollama_url} (模型: {self.ollama_model})
        RAG: {self.rag_persist_dir}
        输出目录: {self.output_dir}
        Gradio: {self.gradio_host}:{self.gradio_port}
        ==================
        """

# 创建全局 settings 实例
settings = Settings()

# 在模块加载时输出配置信息
if __name__ == "__main__":
    print(settings)