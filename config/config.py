"""
极地星光汉藏智能图文生成系统 - 配置文件
针对你的硬件优化：i9-14900HX + RTX 4060 (8GB)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
FONTS_DIR = DATA_DIR / "fonts"
KNOWLEDGE_DIR = DATA_DIR / "knowledge_base"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# 确保目录存在
for dir_path in [DATA_DIR, OUTPUT_DIR, FONTS_DIR, KNOWLEDGE_DIR, VECTOR_DB_DIR,
                 OUTPUT_DIR / "images", OUTPUT_DIR / "posters"]:
    dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class OllamaConfig:
    """Ollama 配置"""
    base_url: str = "http://localhost:11434"
    model: str = "qwen2:7b"  # 推荐：对中文理解极好
    embedding_model: str = "nomic-embed-text"  # 轻量级嵌入模型
    timeout: int = 120
    # 生成参数
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048

@dataclass
class SDWebUIConfig:
    """Stable Diffusion WebUI 配置 - 针对 RTX 4060 8GB 优化"""
    base_url: str = "http://localhost:7860"
    timeout: int = 300
    
    # 默认生成参数 (8GB 显存优化)
    default_params: Dict = field(default_factory=lambda: {
        "steps": 25,
        "cfg_scale": 7.0,
        "width": 768,       # 8GB 显存安全尺寸
        "height": 1024,
        "sampler_name": "DPM++ 2M Karras",
        "batch_size": 1,
        "n_iter": 1,
        "seed": -1,
        "restore_faces": False,
        "enable_hr": False,  # 高分辨率修复（显存紧张时关闭）
    })
    
    # 高清参数 (需要更多显存)
    hires_params: Dict = field(default_factory=lambda: {
        "enable_hr": True,
        "hr_scale": 1.5,
        "hr_upscaler": "Latent",
        "denoising_strength": 0.5,
    })
    
    # 负面提示词
    default_negative: str = (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, worst quality, low quality, "
        "normal quality, jpeg artifacts, signature, watermark, username, blurry, "
        "deformed, mutated, ugly, duplicate, morbid"
    )

@dataclass
class RAGConfig:
    """RAG 知识检索配置"""
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
    collection_name: str = "tibetan_culture"

@dataclass
class RenderConfig:
    """藏文渲染配置"""
    # 默认字体（需要用户自行下载）
    default_font: str = "Microsoft Himalaya"
    fallback_fonts: List[str] = field(default_factory=lambda: [
        "Noto Sans Tibetan",
        "Jomolhari",
        "DDC Uchen",
    ])
    
    # 文字样式
    default_font_size: int = 48
    default_color: str = "#FFFFFF"
    default_stroke_color: str = "#000000"
    default_stroke_width: int = 2
    
    # 排版
    line_spacing: float = 1.5
    margin_ratio: float = 0.05  # 边距占图片的比例

# ==================== LoRA 配置 ====================
LORA_CONFIGS = {
    "唐卡": {
        "trigger": "<lora:thangka_style:0.8>",
        "keywords": ["flat color", "traditional tibetan art", "gold details", "sacred geometry"],
        "negative_add": "3d render, photorealistic"
    },
    "康巴": {
        "trigger": "<lora:khampa_style:0.7>",
        "keywords": ["khampa tibetan", "traditional chuba", "red tassel", "heroic"],
        "negative_add": ""
    },
    "藏式建筑": {
        "trigger": "<lora:tibetan_architecture:0.75>",
        "keywords": ["tibetan monastery", "white walls", "red trim", "golden roof"],
        "negative_add": ""
    },
    "草原风光": {
        "trigger": "<lora:tibetan_landscape:0.6>",
        "keywords": ["tibetan plateau", "grassland", "snow mountain", "blue sky", "yaks"],
        "negative_add": ""
    },
    "藏族人物": {
        "trigger": "<lora:tibetan_portrait:0.7>",
        "keywords": ["tibetan person", "traditional costume", "ethnic features"],
        "negative_add": ""
    }
}

# ==================== 汉藏术语对照 ====================
TERMINOLOGY = {
    # 服饰
    "藏袍": "chuba (tibetan robe)",
    "氆氇": "pulu (tibetan wool fabric)",
    "邦典": "pangden (tibetan apron)",
    "英雄结": "hero knot hairstyle with red tassel",
    
    # 建筑
    "布达拉宫": "Potala Palace",
    "大昭寺": "Jokhang Temple",
    "白塔": "white stupa",
    "经幡": "prayer flags",
    "玛尼堆": "mani stone pile",
    
    # 宗教
    "唐卡": "thangka painting",
    "转经筒": "prayer wheel",
    "哈达": "khata (ceremonial scarf)",
    "酥油灯": "butter lamp",
    "金刚杵": "vajra (dorje)",
    
    # 自然
    "雪山": "snow-capped mountain",
    "草原": "tibetan grassland plateau",
    "青稞": "highland barley",
    "牦牛": "yak",
    "藏羚羊": "tibetan antelope",
    
    # 人物
    "格萨尔王": "King Gesar (epic hero with armor, sword, riding horse, battle flag)",
    "活佛": "living buddha (rinpoche)",
    "喇嘛": "lama (tibetan monk in red robe)",
}

# ==================== 全局配置实例 ====================
ollama_config = OllamaConfig()
sd_config = SDWebUIConfig()
rag_config = RAGConfig()
render_config = RenderConfig()

def get_system_info() -> Dict:
    """获取系统配置信息"""
    return {
        "GPU": "NVIDIA GeForce RTX 4060 Laptop (8GB)",
        "CPU": "Intel Core i9-14900HX",
        "RAM": "DDR5 5600MHz",
        "Storage": "1TB SSD",
        "Display": "3840x2400 @ 240Hz",
        "Optimizations": [
            "SD 默认分辨率 768x1024 (适配 8GB 显存)",
            "启用 xformers 加速",
            "单批次生成避免 OOM",
        ]
    }