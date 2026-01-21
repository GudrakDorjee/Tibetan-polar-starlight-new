"""
Pytest 配置和共享 fixtures
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from PIL import Image
import tempfile

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def project_root_path():
    """项目根目录路径"""
    return project_root

@pytest.fixture
def temp_directory():
    """临时目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_image():
    """创建示例图片"""
    return Image.new('RGB', (512, 512), color='blue')

@pytest.fixture
def sample_image_with_content():
    """创建带内容的示例图片"""
    from PIL import ImageDraw
    
    img = Image.new('RGB', (512, 512), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # 绘制一些简单图形
    draw.rectangle([50, 50, 200, 200], fill='red')
    draw.ellipse([300, 100, 450, 250], fill='green')
    draw.polygon([(256, 300), (150, 450), (362, 450)], fill='yellow')
    
    return img

@pytest.fixture
def mock_sd_client():
    """模拟 SD 客户端"""
    mock = Mock()
    mock.check_connection.return_value = True
    mock.get_models.return_value = ["model1.safetensors", "model2.safetensors"]
    mock.get_loras.return_value = ["lora1", "lora2"]
    mock.get_samplers.return_value = ["Euler a", "DPM++ 2M Karras"]
    
    # 模拟生成结果
    from middleware.sd_client import GenerationResult
    mock_result = GenerationResult(
        images=[Image.new('RGB', (512, 512), color='red')],
        seed=12345,
        generation_time=5.0
    )
    mock.txt2img.return_value = mock_result
    mock.img2img.return_value = mock_result
    
    return mock

@pytest.fixture
def mock_ollama_client():
    """模拟 Ollama 客户端"""
    mock = Mock()
    mock.check_connection.return_value = True
    mock.list_models.return_value = ["qwen2.5:7b", "llama3:8b"]
    mock.generate.return_value = "Generated response"
    mock.translate_to_english.return_value = "Translated English text"
    mock.expand_prompt.return_value = "Expanded prompt with more details"
    mock.get_embeddings.return_value = [0.1] * 384
    
    return mock

@pytest.fixture
def mock_rag_engine():
    """模拟 RAG 引擎"""
    mock = Mock()
    mock.query.return_value = [
        {
            "content": "唐卡是藏族传统绘画艺术",
            "score": 0.95,
            "metadata": {"source": "test"}
        },
        {
            "content": "布达拉宫是西藏著名建筑",
            "score": 0.85,
            "metadata": {"source": "test"}
        }
    ]
    mock.get_stats.return_value = {
        "backend": "simple",
        "collection_name": "test",
        "document_count": 100,
        "embedding_model": "simple"
    }
    mock.add_document.return_value = ["doc_1"]
    
    return mock

@pytest.fixture
def sample_tibetan_text():
    """藏文示例文本"""
    return {
        "tashi_delek": "བཀྲ་ཤིས་བདེ་ལེགས།",  # 扎西德勒
        "tibet": "བོད།",  # 西藏
        "potala": "པོ་ཏ་ལ།",  # 布达拉
        "lhasa": "ལྷ་ས།"  # 拉萨
    }

@pytest.fixture
def sample_prompts():
    """示例提示词"""
    return {
        "chinese": [
            "一位穿着传统藏袍的康巴姑娘",
            "布达拉宫在夕阳下的壮丽景色",
            "格萨尔王骑着白马征战沙场",
            "精美的唐卡画作，描绘观音菩萨"
        ],
        "english": [
            "A Khampa girl wearing traditional Tibetan robe",
            "Magnificent view of Potala Palace at sunset",
            "King Gesar riding a white horse in battle",
            "Exquisite thangka painting depicting Avalokiteshvara"
        ]
    }

@pytest.fixture
def mock_settings():
    """模拟设置"""
    mock = MagicMock()
    mock.sd_webui_url = "http://localhost:7860"
    mock.ollama_url = "http://localhost:11434"
    mock.ollama_model = "qwen2.5:7b"
    mock.default_width = 512
    mock.default_height = 512
    mock.default_steps = 30
    mock.default_cfg_scale = 7.0
    mock.output_dir = Path("/tmp/outputs")
    mock.log_dir = Path("/tmp/logs")
    mock.fonts_dir = Path("/tmp/fonts")
    mock.rag_persist_dir = Path("/tmp/rag")
    mock.default_negative_prompt = "low quality, bad anatomy, blurry"
    mock.generation_timeout = 300
    mock.gradio_host = "127.0.0.1"
    mock.gradio_port = 7861
    mock.gradio_share = False
    
    return mock

# ==================== 测试辅助函数 ====================

def create_test_image(width=512, height=512, color='blue'):
    """创建测试图片的辅助函数"""
    return Image.new('RGB', (width, height), color=color)

def create_gradient_image(width=512, height=512):
    """创建渐变测试图片"""
    from PIL import ImageDraw
    
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    for y in range(height):
        r = int(255 * y / height)
        g = int(255 * (1 - y / height))
        b = 128
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    return img

def assert_image_similar(img1, img2, threshold=0.95):
    """
    断言两张图片相似
    
    Args:
        img1: 第一张图片
        img2: 第二张图片
        threshold: 相似度阈值 (0-1)
    """
    import numpy as np
    
    if img1.size != img2.size:
        return False
    
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # 计算相似度
    diff = np.abs(arr1.astype(float) - arr2.astype(float))
    similarity = 1 - (diff.mean() / 255)
    
    assert similarity >= threshold, f"图片相似度 {similarity:.3f} 低于阈值 {threshold}"

def assert_valid_prompt(prompt):
    """断言提示词有效"""
    assert prompt is not None
    assert isinstance(prompt, str)
    assert len(prompt) > 0# 不应该包含明显的错误标记
    assert "ERROR" not in prompt.upper()
    assert "NONE" not in prompt.upper()