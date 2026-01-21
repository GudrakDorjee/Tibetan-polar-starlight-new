"""
Streamlit Web 界面
提供用户友好的图片生成界面
"""

import streamlit as st
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, List, Dict
import logging
import time
import json

# 导入中间件
import sys
sys.path.append(str(Path(__file__).parent.parent))

from middleware.sd_client import SDClient, get_sd_client, GenerationResult
from middleware.ollama_client import OllamaClient, get_ollama_client
from middleware.hunyuan_client import HunyuanClient, get_hunyuan_client
from middleware.google_client import GoogleTranslateClient, get_google_client
from middleware.prompt_engineer import PromptEngineer, create_prompt_engineer
from middleware.text_renderer import TibetanTextRenderer, create_text_renderer, TextStyle, TextBox, TextPosition
from middleware.rag_engine import RAGEngine, create_rag_engine
from middleware.translator import TibetanChineseTranslator, create_translator, TranslationDirection, TranslationStyle
from middleware.xunfei_asr_client import XunFeiASRClient, get_asr_client
from config.settings import settings

logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="极地星光汉藏智能图文生成系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS - Cyberpunk 深色科技风格
st.markdown("""
<style>
    /* 全局奶白色背景 - 高级感 */
    .main {
        background: linear-gradient(135deg, #FAF8F3 0%, #FFF8F0 50%, #F5F2ED 100%);
        background-attachment: fixed;
        padding-top: 0 !important;
    }

    .stApp {
        background: linear-gradient(135deg, #FAF8F3 0%, #FFF8F0 50%, #F5F2ED 100%);
    }

    /* 顶部居中容器 */
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* 工具栏居中 */
    .stApp header {
        display: flex;
        justify-content: center;
    }

    /* 主内容区域优化 */
    .block-container {
        padding-top: 1rem !important;
        max-width: 1400px;
        margin: 0 auto;
    }

    /* 主标题样式 - 优雅简约 */
    .main-title {
        text-align: center;
        color: #8B7355;
        margin-bottom: 25px;
        font-weight: 600;
        letter-spacing: 1.5px;
    }

    /* 主要按钮 - 优雅玫瑰金渐变 */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        border: none;
        background: linear-gradient(135deg, #D4AF37 0%, #C9A961 100%);
        color: #FFFFFF;
        box-shadow: 0 3px 12px rgba(212, 175, 55, 0.25);
        padding: 0.65rem 1.3rem;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 18px rgba(212, 175, 55, 0.35);
        background: linear-gradient(135deg, #C9A961 0%, #D4AF37 100%);
    }

    /* 优雅卡片样式 */
    .card, .status-box {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(212, 175, 55, 0.15);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        margin: 12px 0;
        transition: all 0.3s ease;
    }

    .card:hover {
        transform: translateY(-3px);
        border: 1px solid rgba(212, 175, 55, 0.3);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }

    /* 侧边栏 - 优雅浅色 */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.6);
        border-right: 1px solid rgba(212, 175, 55, 0.2);
    }

    [data-testid="stSidebar"] .element-container {
        color: #5A4A3A;
    }

    /* 侧边栏选中状态高亮 */
    [data-testid="stSidebar"] .stRadio > label {
        background: transparent;
        padding: 10px;
        border-radius: 8px;
        transition: all 0.3s ease;
        color: #5A4A3A;
    }

    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(212, 175, 55, 0.15);
    }

    /* 输入框优雅样式 */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 8px;
        color: #5A4A3A;
        transition: all 0.3s ease;
    }

    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(212, 175, 55, 0.5);
        box-shadow: 0 0 10px rgba(212, 175, 55, 0.15);
    }

    /* 选择框优雅样式 */
    .stSelectbox>div>div {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 8px;
        color: #5A4A3A;
    }

    /* 滑块样式 */
    .stSlider>div>div>div {
        background: linear-gradient(90deg, #D4AF37 0%, #C9A961 100%);
    }

    /* 标签页优雅样式 - 居中并放大 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 12px;
        padding: 14px 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 16px 32px;
        font-weight: 600;
        font-size: 1.15rem;
        transition: all 0.3s ease;
        border: 1px solid transparent;
        color: #8B7355;
        flex-shrink: 0;
    }

    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-1px);
        border: 1px solid rgba(212, 175, 55, 0.3);
        color: #D4AF37;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #D4AF37 0%, #C9A961 100%);
        color: #FFFFFF;
        box-shadow: 0 3px 12px rgba(212, 175, 55, 0.25);
        transform: scale(1.02);
    }

    /* 展开器优雅样式 */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 8px;
        font-weight: 600;
        color: #5A4A3A;
    }

    /* 提示框优雅样式 */
    .stSuccess, .stError, .stInfo, .stWarning {
        border-radius: 10px;
        padding: 14px;
        font-weight: 500;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(212, 175, 55, 0.2);
    }

    /* 图片容器样式 */
    .stImage {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(212, 175, 55, 0.15);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }

    /* 进度条样式 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #D4AF37 0%, #C9A961 100%);
    }

    /* 分隔线样式 */
    hr {
        margin: 25px 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.3), transparent);
    }

    /* 文件上传器优雅样式 */
    [data-testid="stFileUploader"] {
        border-radius: 10px;
        border: 2px dashed rgba(212, 175, 55, 0.3);
        padding: 20px;
        background: rgba(255, 255, 255, 0.6);
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(212, 175, 55, 0.5);
        background: rgba(255, 255, 255, 0.8);
        box-shadow: 0 0 15px rgba(212, 175, 55, 0.1);
    }

    /* 复选框样式 */
    .stCheckbox {
        font-weight: 500;
        color: #5A4A3A;
    }

    /* 子标题优雅样式 - 适中字体 + 明显底色 + 圆角 */
    h1, h2, h3, h4, h5, h6 {
        color: #5A4A3A;
        font-weight: 600;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    h1 {
        font-size: 1.8rem;
    }

    h2 {
        font-size: 1.05rem;
    }

    h3 {
        font-size: 0.95rem;
    }

    h4 {
        font-size: 0.9rem;
    }

    h5 {
        font-size: 0.88rem;
    }

    /* 文本颜色 - 适中大小 */
    p, span, label, div {
        color: #6B5A4A;
        font-size: 0.95rem;
    }

    /* 输入框标签字体 */
    .stTextInput label, .stTextArea label, .stSelectbox label {
        font-size: 1rem;
        font-weight: 600;
        color: #5A4A3A;
    }

    /* 生成结果占位符 */
    .result-placeholder {
        border: 2px dashed rgba(212, 175, 55, 0.3);
        border-radius: 12px;
        padding: 50px 35px;
        text-align: center;
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
        margin: 18px 0;
        transition: all 0.3s ease;
    }

    .result-placeholder:hover {
        border-color: rgba(212, 175, 55, 0.5);
        background: rgba(255, 255, 255, 0.7);
    }

    .result-placeholder-text {
        color: #8B7355;
        font-size: 1.1rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* 响应式设计 */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 初始化 session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.sd_client = None
    st.session_state.ollama_client = None
    st.session_state.hunyuan_client = None
    st.session_state.google_client = None
    st.session_state.prompt_engineer = None
    st.session_state.text_renderer = None
    st.session_state.rag_engine = None
    st.session_state.translator = None
    st.session_state.generated_image = None
    st.session_state.final_prompt = ""
    st.session_state.final_negative = ""
    st.session_state.translation_result = None
    st.session_state.translation_engine = settings.default_translation_engine

def initialize_clients():
    """初始化所有客户端"""
    if st.session_state.initialized:
        return

    with st.spinner("正在初始化客户端..."):
        try:
            # SD WebUI 客户端
            st.session_state.sd_client = get_sd_client(
                base_url=settings.sd_webui_url,
                timeout=settings.generation_timeout
            )

            # Ollama 客户端（启用 GPU）
            st.session_state.ollama_client = get_ollama_client(
                base_url=settings.ollama_url,
                model=settings.ollama_model,
                num_gpu=settings.ollama_num_gpu
            )

            # RAG 引擎
            st.session_state.rag_engine = create_rag_engine(
                persist_directory=settings.rag_persist_dir,
                ollama_client=st.session_state.ollama_client
            )

            # Prompt 编排器
            st.session_state.prompt_engineer = create_prompt_engineer(
                ollama_client=st.session_state.ollama_client,
                rag_engine=st.session_state.rag_engine
            )

            # 文字渲染器
            st.session_state.text_renderer = create_text_renderer(
                fonts_dir=settings.fonts_dir
            )

            # 翻译器
            st.session_state.translator = create_translator(
                ollama_client=st.session_state.ollama_client
            )

            # Hunyuan 客户端（总是尝试初始化，不检查 enabled 标志）
            try:
                st.session_state.hunyuan_client = get_hunyuan_client(
                    base_url=settings.hunyuan_url
                )
                logger.info("Hunyuan 客户端初始化完成")
            except Exception as e:
                logger.warning(f"Hunyuan 客户端初始化失败: {e}")
                st.session_state.hunyuan_client = None

            # Google Translate 客户端
            try:
                st.session_state.google_client = get_google_client()
                logger.info("Google Translate 客户端初始化完成")
            except Exception as e:
                logger.warning(f"Google Translate 客户端初始化失败: {e}")
                st.session_state.google_client = None

            st.session_state.initialized = True
            logger.info("所有客户端初始化完成")
        except Exception as e:
            st.error(f"初始化失败: {str(e)}")
            logger.exception("初始化客户端时发生错误")

def check_services() -> dict:
    """检查服务状态"""
    status = {
        'sd_webui': False,
        'ollama': False,
        'rag': False,
        'rag_docs': 0
    }

    # 检查 SD WebUI
    if st.session_state.sd_client and st.session_state.sd_client.check_connection():
        status['sd_webui'] = True

    # 检查 Ollama
    if st.session_state.ollama_client and st.session_state.ollama_client.check_connection():
        status['ollama'] = True

    # 检查 RAG
    if st.session_state.rag_engine:
        stats = st.session_state.rag_engine.get_stats()
        status['rag'] = True
        status['rag_docs'] = stats['document_count']

    return status

def generate_image(
    prompt: str,
    style: str,
    quality: str,
    composition: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    use_rag: bool,
    use_llm: bool,
    negative_prompt: str,
    enable_hr: bool,
    hr_scale: float,
    lora_models: Optional[List[Dict[str, float]]] = None
) -> Tuple[Optional[Image.Image], str, str]:
    """生成图片"""
    if not st.session_state.sd_client:
        return None, "", "❌ SD WebUI 未连接"

    try:
        # 语种检测和自动翻译
        processed_prompt = prompt
        detected_language = None
        translation_info = ""

        if st.session_state.translator:
            # 使用翻译器的语言检测功能
            detected_language = st.session_state.translator.detect_language(prompt)
            logger.info(f"检测到语言类型: {detected_language}")

            # 如果检测到藏文，自动翻译为中文
            if detected_language == 'tibetan':
                logger.info("检测到藏文输入，启动藏译中流程")

                # 检查 Hunyuan 客户端是否可用
                if st.session_state.hunyuan_client and st.session_state.hunyuan_client.check_connection():
                    logger.info("使用 Hunyuan 模型进行藏译中")
                    try:
                        # 使用 Hunyuan 进行藏译中
                        processed_prompt = st.session_state.hunyuan_client.translate(
                            text=prompt,
                            direction="bo2zh",
                            style="正式"
                        )
                        translation_info = f"🔄 藏文已翻译为中文: {processed_prompt[:50]}..."
                        logger.info(f"藏译中完成: {processed_prompt}")
                    except Exception as e:
                        logger.error(f"Hunyuan 藏译中失败: {e}")
                        # 降级使用 Ollama 翻译器
                        try:
                            result = st.session_state.translator.translate(
                                text=prompt,
                                direction=TranslationDirection.TIBETAN_TO_CHINESE,
                                style=TranslationStyle.FORMAL
                            )
                            processed_prompt = result.translated_text
                            translation_info = f"🔄 藏文已翻译为中文 (使用 Ollama): {processed_prompt[:50]}..."
                            logger.info(f"藏译中完成 (Ollama 降级): {processed_prompt}")
                        except Exception as e2:
                            logger.error(f"Ollama 藏译中也失败: {e2}")
                            return None, "", f"❌ 藏文翻译失败: {str(e2)}"
                else:
                    # 使用 Ollama 翻译器
                    logger.info("Hunyuan 不可用，使用 Ollama 翻译器")
                    try:
                        result = st.session_state.translator.translate(
                            text=prompt,
                            direction=TranslationDirection.TIBETAN_TO_CHINESE,
                            style=TranslationStyle.FORMAL
                        )
                        processed_prompt = result.translated_text
                        translation_info = f"🔄 藏文已翻译为中文: {processed_prompt[:50]}..."
                        logger.info(f"藏译中完成: {processed_prompt}")
                    except Exception as e:
                        logger.error(f"藏译中失败: {e}")
                        return None, "", f"❌ 藏文翻译失败: {str(e)}"

        # 处理提示词
        if st.session_state.prompt_engineer:
            result = st.session_state.prompt_engineer.process(
                user_input=processed_prompt,
                style=style if style != "通用" else None,
                quality=quality,
                composition=composition if composition != "无" else None,
                use_rag=use_rag,
                use_llm=use_llm,
                custom_negative=negative_prompt if negative_prompt else None
            )
            positive_prompt = result.positive_prompt
            final_negative = result.negative_prompt

            logger.info(f"检测到风格: {result.detected_style}")
            logger.info(f"检测到关键词: {result.detected_keywords}")
        else:
            positive_prompt = processed_prompt
            final_negative = negative_prompt or settings.default_negative_prompt

        # 调用 SD WebUI 生成
        gen_result = st.session_state.sd_client.txt2img(
            prompt=positive_prompt,
            negative_prompt=final_negative,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed if seed != -1 else -1,
            enable_hr=enable_hr,
            hr_scale=hr_scale if enable_hr else 1.0,
            lora_models=lora_models
        )

        if gen_result.images:
            status = f"✅ 生成成功! 耗时: {gen_result.generation_time:.2f}秒, Seed: {gen_result.seed}"
            # 如果有翻译信息，添加到状态消息中
            if translation_info:
                status = f"{translation_info}\n{status}"
            return gen_result.images[0], positive_prompt, status
        else:
            return None, positive_prompt, "❌ 生成失败: 未返回图片"

    except TimeoutError as e:
        return None, "", f"❌ 超时: {str(e)}"
    except ConnectionError as e:
        return None, "", f"❌ 连接错误: {str(e)}"
    except Exception as e:
        logger.exception("生成图片时发生错误")
        return None, "", f"❌ 错误: {str(e)}"

def img2img_generate(
    init_image: Image.Image,
    prompt: str,
    style: str,
    denoising_strength: float,
    steps: int,
    cfg_scale: float,
    seed: int,
    use_rag: bool,
    use_llm: bool,
    negative_prompt: str,
    lora_models: Optional[List[Dict[str, float]]] = None
) -> Tuple[Optional[Image.Image], str, str]:
    """图生图"""
    if not st.session_state.sd_client:
        return None, "", "❌ SD WebUI 未连接"

    if init_image is None:
        return None, "", "❌ 请上传参考图片"

    try:
        # 处理提示词
        if st.session_state.prompt_engineer:
            result = st.session_state.prompt_engineer.process(
                user_input=prompt,
                style=style if style != "通用" else None,
                use_rag=use_rag,
                use_llm=use_llm,
                custom_negative=negative_prompt if negative_prompt else None
            )
            positive_prompt = result.positive_prompt
            final_negative = result.negative_prompt
        else:
            positive_prompt = prompt
            final_negative = negative_prompt or settings.default_negative_prompt

        # 调用图生图
        gen_result = st.session_state.sd_client.img2img(
            init_image=init_image,
            prompt=positive_prompt,
            negative_prompt=final_negative,
            denoising_strength=denoising_strength,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed if seed != -1 else -1,
            lora_models=lora_models
        )

        if gen_result.images:
            status = f"✅ 生成成功! 耗时: {gen_result.generation_time:.2f}秒, Seed: {gen_result.seed}"
            return gen_result.images[0], positive_prompt, status
        else:
            return None, positive_prompt, "❌ 生成失败: 未返回图片"

    except Exception as e:
        logger.exception("图生图时发生错误")
        return None, "", f"❌ 错误: {str(e)}"

def save_image(image: Image.Image, filename: str) -> str:
    """保存图片"""
    if image is None:
        return "❌ 没有图片可保存"

    try:
        output_dir = settings.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"tibetan_art_{timestamp}.png"

        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            filename += '.png'

        filepath = output_dir / filename
        image.save(filepath)

        return f"✅ 已保存到: {filepath}"

    except Exception as e:
        return f"❌ 保存失败: {str(e)}"

def txt2img_tab():
    """文生图标签页 - 优化版"""

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 创作设置")
        prompt = st.text_area(
            "✨ 图像描述 (支持中文/藏文)",
            placeholder="例如：一位穿着传统藏袍的康巴姑娘，站在草原上，远处是雪山...\n\n💡 提示：描述越详细，生成效果越好",
            height=120,
            help="支持中文、藏文输入，系统会自动翻译并优化提示词"
        )

        # 实时语言检测提示
        if prompt and st.session_state.translator:
            detected_lang = st.session_state.translator.detect_language(prompt)
            if detected_lang == 'tibetan':
                st.info("🔍 检测到藏文输入 | 将自动使用 Hunyuan 模型进行藏译中处理")
            elif detected_lang == 'chinese':
                st.success("🔍 检测到中文输入 | 将使用常规生成流程")
            elif detected_lang == 'mixed':
                st.warning("🔍 检测到混合语言输入 | 建议使用单一语言以获得最佳效果")

        # 获取风格选项
        style_options = ["通用"]
        quality_options = ["高质量"]
        composition_options = ["无"]

        if st.session_state.prompt_engineer:
            style_options += st.session_state.prompt_engineer.get_style_options()
            quality_options = st.session_state.prompt_engineer.get_quality_options()
            composition_options += st.session_state.prompt_engineer.get_composition_options()

        st.markdown("#### 风格与质量")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            style = st.selectbox(" 风格预设", style_options, key="t2i_style", help="选择预设风格可自动添加相关提示词")
            quality = st.selectbox("⭐ 质量预设", quality_options, key="t2i_quality", help="控制生成图像的质量等级")
        with col_s2:
            composition = st.selectbox("📐 构图预设", composition_options, key="t2i_composition", help="选择构图方式")

            # 智能增强选项
            col_rag, col_llm = st.columns(2)
            with col_rag:
                use_rag = st.checkbox("🧠 知识库增强", value=True, help="使用 RAG 知识库优化提示词")
            with col_llm:
                use_llm = st.checkbox("🤖 LLM 优化", value=True, help="使用大语言模型翻译和扩写")

        with st.expander("🔧 高级参数设置", expanded=False):
            st.markdown("##### 图像尺寸")
            col_w, col_h = st.columns(2)
            with col_w:
                width = st.slider("🔲 宽度", 256, 1024, settings.default_width, 64,
                                 help="图像宽度，建议使用 512/768/1024")
            with col_h:
                height = st.slider("🔳 高度", 256, 1024, settings.default_height, 64,
                                  help="图像高度，建议使用 512/768/1024")

            st.markdown("##### 生成参数")
            col_st, col_cfg = st.columns(2)
            with col_st:
                steps = st.slider("🔄 采样步数", 10, 100, settings.default_steps, 1,
                                 help="步数越多质量越好，但速度越慢。推荐 25-40")
            with col_cfg:
                cfg_scale = st.slider("🎯 CFG Scale", 1.0, 20.0, settings.default_cfg_scale, 0.5,
                                     help="越高越接近提示词。推荐 7-9")

            seed = st.number_input("🎲 随机种子 (-1 为随机)", value=-1, step=1,
                                  help="固定种子可以复现相同的生成结果")

            st.markdown("##### 负面提示词")
            negative_prompt = st.text_area("不想出现的元素",
                                          placeholder="例如：低质量、模糊、变形...",
                                          height=60,
                                          help="描述不希望在图像中出现的内容")

            st.markdown("##### 高清修复")
            col_hr1, col_hr2 = st.columns(2)
            with col_hr1:
                enable_hr = st.checkbox("✨ 启用高清修复", value=False,
                                       help="生成后自动修复图像，提升细节")
            with col_hr2:
                hr_scale = st.slider("📈修复倍数", 1.0, 2.0, 1.5, 0.1,
                                    help="高清修复的修复倍数")

        st.markdown("---")
        st.markdown("### 操作")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            preview_btn = st.button("👁️ 预览提示词", use_container_width=True,
                                   help="查看 AI 优化后的提示词")
        with col_btn2:
            generate_btn = st.button("🚀 开始生成", type="primary", use_container_width=True,
                                    help="点击开始生成图像")

    with col2:
        st.markdown("###  生成结果")

        if generate_btn and prompt:
            with st.spinner(" AI 正在创作中，请稍候..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("⚙️ 正在处理提示词...")
                progress_bar.progress(20)

                # 获取选中的 LoRA 模型
                lora_models = st.session_state.get('selected_loras', [])

                image, final_prompt, status = generate_image(
                    prompt, style, quality, composition, width, height,
                    steps, cfg_scale, seed, use_rag, use_llm,
                    negative_prompt, enable_hr, hr_scale, lora_models
                )

                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()

                if image:
                    st.session_state.generated_image = image
                    st.session_state.final_prompt = final_prompt
                    st.balloons()  # 成功时显示气球动画

                # 显示状态信息
                if "✅" in status:
                    st.success(status)
                else:
                    st.error(status)

        # 显示生成的图片
        if st.session_state.generated_image:
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 15px;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        backdrop-filter: blur(10px);
                        box-shadow: 0 8px 32px rgba(0,0,0,0.37);'>
            """, unsafe_allow_html=True)

            st.image(st.session_state.generated_image, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # 图片信息
            img = st.session_state.generated_image
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("📐 宽度", f"{img.width}px")
            with col_info2:
                st.metric("📐 高度", f"{img.height}px")
            with col_info3:
                st.metric("📊 比例", f"{img.width/img.height:.2f}:1")

            # 提示词详情
            with st.expander("📝 查看提示词详情", expanded=False):
                st.markdown("**✨ 优化后的正向提示词：**")
                st.code(st.session_state.final_prompt, language="text")

            # 保存功能
            st.markdown("---")
            st.markdown("### 保存图片")
            col_save1, col_save2 = st.columns([3, 1])
            with col_save1:
                filename = st.text_input("📁 文件名", placeholder="留空自动生成时间戳文件名",
                                        label_visibility="collapsed")
            with col_save2:
                if st.button("💾 保存", use_container_width=True, type="primary"):
                    save_status = save_image(st.session_state.generated_image, filename)
                    if "✅" in save_status:
                        st.success(save_status)
                    else:
                        st.error(save_status)
        else:
            # 空状态提示 - Cyberpunk 风格占位符
            st.markdown("""
            <div class='result-placeholder'>
                <h3 class='result-placeholder-text'>⚡ AI 绘画区域</h3>
                <p class='result-placeholder-text' style='font-size: 1rem; margin-top: 10px;'>
                    输入描述后点击"开始生成"<br/>
                    AI 将在此处为您创作
                </p>
            </div>
            """, unsafe_allow_html=True)

        # 预览提示词功能
        if preview_btn and prompt:
            with st.spinner("🔍 正在分析和优化提示词..."):
                if st.session_state.prompt_engineer:
                    result = st.session_state.prompt_engineer.process(
                        user_input=prompt,
                        style=style if style != "通用" else None,
                        quality=quality,
                        composition=composition if composition != "无" else None,
                        use_rag=use_rag,
                        use_llm=use_llm
                    )

                    st.markdown("""
                    <div style='background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px;
                                margin-top: 20px; border: 1px solid rgba(255, 255, 255, 0.1);
                                backdrop-filter: blur(10px);'>
                        <h4 style='color: #fbbf24;'>✨ 提示词预览</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**✅ 正向提示词：**")
                    st.code(result.positive_prompt, language="text")

                    st.markdown("**🚫 负向提示词：**")
                    st.code(result.negative_prompt, language="text")

                    if result.detected_style:
                        st.info(f" 检测到风格: {result.detected_style}")
                    if result.detected_keywords:
                        st.info(f"🔑 关键词: {', '.join(result.detected_keywords)}")

def img2img_tab():
    """图生图标签页 - 优化版"""

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 上传与设置")
        init_image = st.file_uploader("📁 上传参考图片", type=['png', 'jpg', 'jpeg'],
                                     help="支持 PNG、JPG、JPEG 格式")

        uploaded_img = None
        if init_image:
            uploaded_img = Image.open(init_image)
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 10px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);'>
            """, unsafe_allow_html=True)
            st.image(uploaded_img, caption="📷 参考图片", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # 显示图片信息
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.metric("尺寸", f"{uploaded_img.width}×{uploaded_img.height}")
            with col_img2:
                st.metric("格式", uploaded_img.format)

        st.markdown("---")
        st.markdown("#### 重绘描述")
        prompt = st.text_area(
            "描述想要的变化",
            placeholder="例如：将背景改为雪山，增加藏族服饰元素...\n\n💡 提示：描述越具体，重绘效果越精准",
            height=100,
            help="描述您希望对图片进行的修改",
            label_visibility="collapsed"
        )

        # 获取风格选项
        style_options = ["通用"]
        if st.session_state.prompt_engineer:
            style_options += st.session_state.prompt_engineer.get_style_options()

        st.markdown("####  风格与强度")
        style = st.selectbox("🎭 风格预设", style_options, key="i2i_style",
                           help="选择预设风格")

        denoising_strength = st.slider(
            "🎚️ 重绘强度", 0.1, 1.0, 0.75, 0.05, key="i2i_denoising",
            help="0.3-0.5: 轻微变化 | 0.5-0.7: 中等变化 | 0.7-0.9: 大幅变化"
        )

        # 强度提示
        if denoising_strength < 0.5:
            st.info("💡 当前强度较低，会保留原图大部分内容")
        elif denoising_strength < 0.7:
            st.info("💡 当前强度适中，会进行中等程度的修改")
        else:
            st.warning("⚠️ 当前强度较高，会大幅改变原图")

        with st.expander("🔧 高级参数设置", expanded=False):
            st.markdown("##### 生成参数")
            col_st, col_cfg = st.columns(2)
            with col_st:
                steps = st.slider("🔄 采样步数", 10, 100, 30, 1, key="i2i_steps",
                                 help="推荐 20-40 步")
            with col_cfg:
                cfg_scale = st.slider("🎯 CFG Scale", 1.0, 20.0, 7.0, 0.5, key="i2i_cfg",
                                     help="推荐 7-9")

            seed = st.number_input("🎲 随机种子 (-1 为随机)", value=-1, step=1, key="i2i_seed",
                                  help="固定种子可复现结果")

            st.markdown("##### 智能增强")
            col_rag, col_llm = st.columns(2)
            with col_rag:
                use_rag = st.checkbox("🧠 知识库增强", value=True, key="i2i_rag")
            with col_llm:
                use_llm = st.checkbox("🤖 LLM 优化", value=True, key="i2i_llm")

            st.markdown("##### 负面提示词")
            negative_prompt = st.text_area("不想出现的元素", height=60, key="i2i_neg",
                                          placeholder="例如：低质量、模糊、变形...")

        st.markdown("---")
        st.markdown("### 操作")
        generate_btn = st.button("🚀 开始重绘", type="primary", use_container_width=True, key="i2i_gen",
                                help="点击开始图生图")

    with col2:
        st.markdown("###  重绘结果")

        if generate_btn:
            if not uploaded_img:
                st.error("❌ 请先上传参考图片")
            elif not prompt:
                st.error("❌ 请输入重绘描述")
            else:
                with st.spinner(" AI 正在重绘中，请稍候..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("⚙️ 正在处理提示词...")
                    progress_bar.progress(20)

                    # 获取选中的 LoRA 模型
                    lora_models = st.session_state.get('selected_loras', [])

                    image, final_prompt, status = img2img_generate(
                        uploaded_img, prompt, style, denoising_strength,
                        steps, cfg_scale, seed, use_rag, use_llm, negative_prompt,
                        lora_models
                    )

                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()

                    if image:
                        st.session_state.generated_image = image
                        st.session_state.final_prompt = final_prompt
                        st.balloons()

                    # 显示状态
                    if "✅" in status:
                        st.success(status)
                    else:
                        st.error(status)

        if st.session_state.generated_image:
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 15px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
                        box-shadow: 0 8px 32px rgba(0,0,0,0.37);'>
            """, unsafe_allow_html=True)

            st.image(st.session_state.generated_image, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # 图片信息
            img = st.session_state.generated_image
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("📐 尺寸", f"{img.width}×{img.height}")
            with col_info2:
                st.metric("📊 格式", img.format if hasattr(img, 'format') else "PNG")

            # 提示词详情
            with st.expander("📝 查看提示词详情", expanded=False):
                st.markdown("**✨ 优化后的提示词：**")
                st.code(st.session_state.final_prompt, language="text")

            # 保存功能
            st.markdown("---")
            st.markdown("### 保存图片")
            col_save1, col_save2 = st.columns([3, 1])
            with col_save1:
                filename = st.text_input("📁 文件名", placeholder="留空自动生成",
                                        label_visibility="collapsed", key="i2i_filename")
            with col_save2:
                if st.button("💾 保存", use_container_width=True, type="primary", key="i2i_save"):
                    save_status = save_image(st.session_state.generated_image, filename)
                    if "✅" in save_status:
                        st.success(save_status)
                    else:
                        st.error(save_status)
        else:
            # 空状态提示
            st.markdown("""
            <div style='text-align: center; padding: 60px 20px;
                        background: rgba(255,255,255,0.5); border-radius: 15px;
                        border: 2px dashed #ccc;'>
                <h3 style='color: #999;'></h3>
                <p style='color: #999;'>暂无重绘结果</p>
                <p style='color: #bbb; font-size: 0.9rem;'>上传图片并输入描述后点击"开始重绘"</p>
            </div>
            """, unsafe_allow_html=True)

def text_tab():
    """文字排版标签页 - 优化版"""
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px; margin-bottom: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);'>
        <h2 style='color: #667eea; margin: 0;'>✍️ 文字排版 - 智能文字添加</h2>
        <p style='color: #666; margin-top: 5px;'>为图片添加中文、藏文等多语言文字</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 上传与设置")
        text_image = st.file_uploader("📁 上传底图", type=['png', 'jpg', 'jpeg'], key="text_img",
                                     help="支持 PNG、JPG、JPEG 格式")

        if text_image:
            base_img = Image.open(text_image)
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 10px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);'>
            """, unsafe_allow_html=True)
            st.image(base_img, caption="📷 底图预览", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 文字内容")
        text_content = st.text_area(
            "输入文字",
            placeholder="བཀྲ་ཤིས་བདེ་ལེགས། (扎西德勒)\n\n💡 支持中文、藏文、英文等多种语言",
            height=100,
            help="支持多语言文字输入",
            label_visibility="collapsed"
        )

        st.markdown("####  文字样式")
        col_pos, col_size = st.columns(2)
        with col_pos:
            position = st.selectbox(
                "📍 文字位置",
                ["左上", "上中", "右上", "左中", "居中", "右中", "左下", "下中", "右下"],
                index=7,
                help="选择文字在图片中的位置"
            )
        with col_size:
            font_size = st.slider("📏 字体大小", 12, 120, 48, 2,
                                 help="调整文字大小")

        st.markdown("####  颜色设置")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            text_color = st.color_picker(" 文字颜色", "#FFFFFF",
                                        help="选择文字的颜色")
        with col_c2:
            stroke_color = st.color_picker("🖌️ 描边颜色", "#000000",
                                          help="选择文字描边的颜色")

        stroke_width = st.slider("📐 描边宽度", 0, 10, 2, 1,
                                help="描边可以增加文字的可读性")

        st.markdown("#### 背景效果")
        col_bg1, col_bg2 = st.columns(2)
        with col_bg1:
            add_bg = st.checkbox("✨ 添加背景框", value=False,
                               help="为文字添加半透明背景框")
        with col_bg2:
            bg_opacity = st.slider("💫 背景透明度", 0.1, 1.0, 0.5, 0.1,
                                  help="调整背景框的透明度")

        st.markdown("---")
        st.markdown("### 操作")
        add_text_btn = st.button("✍️ 添加文字", type="primary", use_container_width=True,
                                help="点击将文字添加到图片上")

    with col2:
        st.markdown("###  排版结果")

        if add_text_btn and text_image and text_content:
            with st.spinner("✍️ 正在添加文字..."):
                try:
                    position_map = {
                        "左上": TextPosition.TOP_LEFT,
                        "上中": TextPosition.TOP_CENTER,
                        "右上": TextPosition.TOP_RIGHT,
                        "左中": TextPosition.CENTER_LEFT,
                        "居中": TextPosition.CENTER,
                        "右中": TextPosition.CENTER_RIGHT,
                        "左下": TextPosition.BOTTOM_LEFT,
                        "下中": TextPosition.BOTTOM_CENTER,
                        "右下": TextPosition.BOTTOM_RIGHT,
                    }

                    text_position = position_map.get(position, TextPosition.BOTTOM_CENTER)

                    style = TextStyle(
                        font_size=font_size,
                        color=text_color,
                        stroke_width=stroke_width,
                        stroke_color=stroke_color,
                        shadow=True
                    )

                    text_box = TextBox(
                        position=text_position,
                        margin=50,
                        background=add_bg,
                        background_opacity=bg_opacity
                    )

                    result = st.session_state.text_renderer.add_text(base_img, text_content, style, text_box)
                    st.session_state.generated_image = result
                    st.success("✅ 文字添加成功！")
                except Exception as e:
                    st.error(f"❌ 添加失败: {str(e)}")

        if st.session_state.generated_image:
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 15px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
                        box-shadow: 0 8px 32px rgba(0,0,0,0.37);'>
            """, unsafe_allow_html=True)

            st.image(st.session_state.generated_image, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # 保存功能
            st.markdown("---")
            st.markdown("### 保存图片")
            col_save1, col_save2 = st.columns([3, 1])
            with col_save1:
                filename = st.text_input("📁 文件名", placeholder="留空自动生成",
                                        label_visibility="collapsed", key="text_filename")
            with col_save2:
                if st.button("💾 保存", use_container_width=True, type="primary", key="text_save"):
                    save_status = save_image(st.session_state.generated_image, filename)
                    if "✅" in save_status:
                        st.success(save_status)
                    else:
                        st.error(save_status)
        else:
            # 空状态提示
            st.markdown("""
            <div style='text-align: center; padding: 60px 20px;
                        background: rgba(255,255,255,0.5); border-radius: 15px;
                        border: 2px dashed #ccc;'>
                <h3 style='color: #999;'>✍️</h3>
                <p style='color: #999;'>暂无排版结果</p>
                <p style='color: #bbb; font-size: 0.9rem;'>上传图片并输入文字后点击"添加文字"</p>
            </div>
            """, unsafe_allow_html=True)

def poster_tab():
    """海报制作标签页 - 优化版"""
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px; margin-bottom: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);'>
        <h2 style='color: #667eea; margin: 0;'>🎭 海报制作 - 专业海报设计</h2>
        <p style='color: #666; margin-top: 5px;'>快速创建精美的宣传海报</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 上传与设置")
        poster_bg = st.file_uploader("📁 上传背景图片", type=['png', 'jpg', 'jpeg'], key="poster_bg",
                                    help="选择海报的背景图片")

        if poster_bg:
            bg_img = Image.open(poster_bg)
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 10px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);'>
            """, unsafe_allow_html=True)
            st.image(bg_img, caption="📷 背景图片", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 文字内容")
        title = st.text_area("📌 主标题",
                           placeholder="བོད་ཀྱི་རིག་གནས། (藏族文化)\n\n💡 支持中文、藏文等多语言",
                           height=80,
                           help="海报的主标题文字")
        subtitle = st.text_input("📝 副标题", placeholder="可选的副标题...",
                               help="可选的副标题，留空则不显示")
        footer = st.text_input("📄 底部文字", placeholder="可选的底部说明...",
                             help="可选的底部说明文字")

        st.markdown("####  样式设置")
        title_size = st.slider("📏 标题字体大小", 24, 120, 72, 4,
                              help="调整主标题的字体大小")

        st.markdown("#### 装饰效果")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            add_gradient = st.checkbox("✨ 添加渐变遮罩", value=True,
                                      help="为背景添加渐变遮罩，提升文字可读性")
            add_border = st.checkbox(" 添加装饰边框", value=False,
                                   help="为海报添加装饰性边框")
        with col_opt2:
            border_style = st.selectbox(" 边框样式", ["简约", "双线", "藏式"], index=2,
                                       help="选择边框的装饰风格")

        st.markdown("---")
        st.markdown("### 操作")
        create_poster_btn = st.button("🎭 生成海报", type="primary", use_container_width=True,
                                     help="点击生成专业海报")

    with col2:
        st.markdown("###  海报预览")

        if create_poster_btn and poster_bg and title:
            with st.spinner("🎭 正在生成海报..."):
                try:
                    title_style = TextStyle(
                        font_size=title_size,
                        color="#FFFFFF",
                        stroke_width=3,
                        stroke_color="#000000",
                        shadow=True,
                        shadow_offset=(4, 4)
                    )

                    result = st.session_state.text_renderer.create_poster(
                        background=bg_img,
                        title=title,
                        subtitle=subtitle if subtitle else None,
                        footer=footer if footer else None,
                        title_style=title_style,
                        add_gradient_overlay=add_gradient
                    )

                    if add_border:
                        border_style_map = {
                            "简约": "solid",
                            "双线": "double",
                            "藏式": "tibetan"
                        }
                        result = st.session_state.text_renderer.add_decorative_border(
                            result,
                            border_width=20,
                            style=border_style_map.get(border_style, "solid")
                        )

                    st.session_state.generated_image = result
                    st.success("✅ 海报创建成功！")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ 创建失败: {str(e)}")

        if st.session_state.generated_image:
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 15px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
                        box-shadow: 0 8px 32px rgba(0,0,0,0.37);'>
            """, unsafe_allow_html=True)

            st.image(st.session_state.generated_image, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # 保存功能
            st.markdown("---")
            st.markdown("### 保存海报")
            col_save1, col_save2 = st.columns([3, 1])
            with col_save1:
                filename = st.text_input("📁 文件名", placeholder="留空自动生成",
                                        label_visibility="collapsed", key="poster_filename")
            with col_save2:
                if st.button("💾 保存", use_container_width=True, type="primary", key="poster_save"):
                    save_status = save_image(st.session_state.generated_image, filename)
                    if "✅" in save_status:
                        st.success(save_status)
                    else:
                        st.error(save_status)
        else:
            # 空状态提示
            st.markdown("""
            <div style='text-align: center; padding: 60px 20px;
                        background: rgba(255,255,255,0.5); border-radius: 15px;
                        border: 2px dashed #ccc;'>
                <h3 style='color: #999;'>🎭</h3>
                <p style='color: #999;'>暂无海报预览</p>
                <p style='color: #bbb; font-size: 0.9rem;'>上传背景并输入标题后点击"生成海报"</p>
            </div>
            """, unsafe_allow_html=True)

def graphic_design_tab():
    """图文制作标签页 - 合并版"""

    # 选择模式：简单文字 or 海报设计
    mode = st.radio(
        "📋 选择制作模式",
        ["✍️ 简单文字添加", "🎭 专业海报设计"],
        horizontal=True,
        help="简单文字：快速在图片上添加文字 | 海报设计：创建带标题副标题的专业海报"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 上传与设置")
        image_file = st.file_uploader(
            "📁 上传底图/背景图片",
            type=['png', 'jpg', 'jpeg'],
            key="graphic_img",
            help="支持 PNG、JPG、JPEG 格式"
        )

        if image_file:
            base_img = Image.open(image_file)
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 10px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);'>
            """, unsafe_allow_html=True)
            st.image(base_img, caption="📷 原图预览", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # 根据模式显示不同的输入控件
        if mode == "✍️ 简单文字添加":
            st.markdown("#### 文字内容")
            text_content = st.text_area(
                "输入文字",
                placeholder="བཀྲ་ཤིས་བདེ་ལེགས། (扎西德勒)\n\n💡 支持中文、藏文、英文等多种语言",
                height=100,
                help="支持多语言文字输入",
                label_visibility="collapsed",
                key="graphic_text"
            )

            st.markdown("####  文字位置与大小")
            col_pos, col_size = st.columns(2)
            with col_pos:
                position = st.selectbox(
                    "📍 文字位置",
                    ["左上", "上中", "右上", "左中", "居中", "右中", "左下", "下中", "右下"],
                    index=7,
                    help="选择文字在图片中的位置"
                )
            with col_size:
                font_size = st.slider("📏 字体大小", 12, 120, 48, 2,
                                     help="调整文字大小")

        else:  # 海报设计模式
            st.markdown("#### 文字内容")
            text_content = st.text_area(
                "📌 主标题",
                placeholder="བོད་ཀྱི་རིག་གནས། (藏族文化)\n\n💡 支持中文、藏文等多语言",
                height=80,
                help="海报的主标题文字",
                key="graphic_title"
            )
            subtitle = st.text_input("📝 副标题", placeholder="可选的副标题...",
                                   help="可选的副标题，留空则不显示")
            footer = st.text_input("📄 底部文字", placeholder="可选的底部说明...",
                                 help="可选的底部说明文字")

            st.markdown("####  标题样式")
            font_size = st.slider("📏 标题字体大小", 24, 120, 72, 4,
                                  help="调整主标题的字体大小")

        # 字体选择（所有模式共用）
        st.markdown("#### 字体选择")

        # 获取可用的藏文字体
        available_fonts = st.session_state.text_renderer.get_available_fonts()
        font_names = ["系统默认"] + [Path(f).stem for f in available_fonts]

        selected_font_name = st.selectbox(
            "🎯 藏文字体",
            font_names,
            help="选择藏文字体，建议使用支持藏文的字体以正确显示藏文文字"
        )

        # 根据选择获取字体路径
        if selected_font_name == "系统默认":
            font_path = None
        else:
            font_idx = font_names.index(selected_font_name) - 1
            font_path = available_fonts[font_idx] if font_idx < len(available_fonts) else None

        # 如果有可用字体，显示字体预览
        if font_path and st.checkbox("👁️ 预览字体效果", value=False):
            preview_text = "བཀྲ་ཤིས་བདེ་ལེགས།\n扎西德勒"
            try:
                preview_img = st.session_state.text_renderer.preview_text(
                    preview_text,
                    font_path=font_path,
                    font_size=36
                )
                st.image(preview_img, caption="字体预览", use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ 字体预览失败: {e}")

        st.markdown("####  颜色设置")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            text_color = st.color_picker(" 文字颜色", "#FFFFFF",
                                        help="选择文字的颜色")
        with col_c2:
            stroke_color = st.color_picker("🖌️ 描边颜色", "#000000",
                                          help="选择文字描边的颜色")

        stroke_width = st.slider("📐 描边宽度", 0, 10, 2, 1,
                                help="描边可以增加文字的可读性")

        # 背景和装饰效果
        st.markdown("#### 装饰效果")
        col_eff1, col_eff2 = st.columns(2)

        with col_eff1:
            if mode == "✍️ 简单文字添加":
                add_bg = st.checkbox("✨ 添加背景框", value=False,
                                   help="为文字添加半透明背景框")
            else:
                add_gradient = st.checkbox("✨ 添加渐变遮罩", value=True,
                                          help="为背景添加渐变遮罩，提升文字可读性")
                add_border = st.checkbox(" 添加装饰边框", value=False,
                                       help="为海报添加装饰性边框")

        with col_eff2:
            if mode == "✍️ 简单文字添加":
                bg_opacity = st.slider("💫 背景透明度", 0.1, 1.0, 0.5, 0.1,
                                      help="调整背景框的透明度")
            else:
                if 'add_border' in locals() and add_border:
                    border_style = st.selectbox(" 边框样式", ["简约", "双线", "藏式"],
                                               index=2, help="选择边框的装饰风格")

        st.markdown("---")
        st.markdown("### 操作")
        create_btn = st.button(
            " 生成图文" if mode == "✍️ 简单文字添加" else "🎭 生成海报",
            type="primary",
            use_container_width=True,
            help="点击开始制作"
        )

    with col2:
        st.markdown("###  制作结果")

        if create_btn and image_file and text_content:
            with st.spinner(" 正在制作中..."):
                try:
                    if mode == "✍️ 简单文字添加":
                        # 简单文字添加模式
                        position_map = {
                            "左上": TextPosition.TOP_LEFT,
                            "上中": TextPosition.TOP_CENTER,
                            "右上": TextPosition.TOP_RIGHT,
                            "左中": TextPosition.CENTER_LEFT,
                            "居中": TextPosition.CENTER,
                            "右中": TextPosition.CENTER_RIGHT,
                            "左下": TextPosition.BOTTOM_LEFT,
                            "下中": TextPosition.BOTTOM_CENTER,
                            "右下": TextPosition.BOTTOM_RIGHT,
                        }

                        text_position = position_map.get(position, TextPosition.BOTTOM_CENTER)

                        style = TextStyle(
                            font_size=font_size,
                            color=text_color,
                            stroke_width=stroke_width,
                            stroke_color=stroke_color,
                            shadow=True
                        )

                        text_box = TextBox(
                            position=text_position,
                            margin=50,
                            background=add_bg,
                            background_opacity=bg_opacity
                        )

                        result = st.session_state.text_renderer.add_text(
                            base_img, text_content, style, text_box, font_path
                        )
                        st.session_state.generated_image = result
                        st.success("✅ 文字添加成功！")

                    else:
                        # 海报设计模式
                        title_style = TextStyle(
                            font_size=font_size,
                            color=text_color,
                            stroke_width=stroke_width,
                            stroke_color=stroke_color,
                            shadow=True,
                            shadow_offset=(4, 4)
                        )

                        result = st.session_state.text_renderer.create_poster(
                            background=base_img,
                            title=text_content,
                            subtitle=subtitle if subtitle else None,
                            footer=footer if footer else None,
                            title_style=title_style,
                            font_path=font_path,
                            add_gradient_overlay=add_gradient
                        )

                        if add_border:
                            border_style_map = {
                                "简约": "solid",
                                "双线": "double",
                                "藏式": "tibetan"
                            }
                            result = st.session_state.text_renderer.add_decorative_border(
                                result,
                                border_width=20,
                                style=border_style_map.get(border_style, "solid")
                            )

                        st.session_state.generated_image = result
                        st.success("✅ 海报创建成功！")
                        st.balloons()

                except Exception as e:
                    st.error(f"❌ 制作失败: {str(e)}")
                    logger.error(f"图文制作失败: {e}", exc_info=True)

        if st.session_state.generated_image:
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 15px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
                        box-shadow: 0 8px 32px rgba(0,0,0,0.37);'>
            """, unsafe_allow_html=True)

            st.image(st.session_state.generated_image, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # 保存功能
            st.markdown("---")
            st.markdown("### 保存图片")
            col_save1, col_save2 = st.columns([3, 1])
            with col_save1:
                filename = st.text_input("📁 文件名", placeholder="留空自动生成",
                                        label_visibility="collapsed", key="graphic_filename")
            with col_save2:
                if st.button("💾 保存", use_container_width=True, type="primary", key="graphic_save"):
                    save_status = save_image(st.session_state.generated_image, filename)
                    if "✅" in save_status:
                        st.success(save_status)
                    else:
                        st.error(save_status)
        else:
            # 空状态提示
            st.markdown("""
            <div style='text-align: center; padding: 60px 20px;
                        background: rgba(255,255,255,0.5); border-radius: 15px;
                        border: 2px dashed #ccc;'>
                <h3 style='color: #999;'></h3>
                <p style='color: #999;'>暂无制作结果</p>
                <p style='color: #bbb; font-size: 0.9rem;'>上传图片并输入文字后点击"生成"</p>
            </div>
            """, unsafe_allow_html=True)

def upscale_tab():
    """图片修复标签页 - 优化版"""

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 上传与设置")
        upscale_image = st.file_uploader("📁 上传原图", type=['png', 'jpg', 'jpeg'], key="upscale_img",
                                        help="支持 PNG、JPG、JPEG 格式")

        if upscale_image:
            orig_img = Image.open(upscale_image)
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 10px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);'>
            """, unsafe_allow_html=True)
            st.image(orig_img, caption="📷 原图预览", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # 显示原图信息
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.metric("原始尺寸", f"{orig_img.width}×{orig_img.height}")
            with col_img2:
                st.metric("格式", orig_img.format)

        st.markdown("---")
        st.markdown("#### 修复参数")
        scale = st.slider("📈修复倍数", 1.5, 4.0, 2.0, 0.5,
                         help="选择图片修复的倍数")

        # 显示预计尺寸
        if upscale_image:
            new_width = int(orig_img.width * scale)
            new_height = int(orig_img.height * scale)
            st.info(f"💡修复后尺寸: {new_width}×{new_height}")

        upscaler = st.selectbox(
            "🎯修复算法",
            [
                "R-ESRGAN 4x+",
                "R-ESRGAN 4x+ Anime6B",
                "ESRGAN_4x",
                "SwinIR_4x",
                "Lanczos",
                "Nearest"
            ],
            help="R-ESRGAN 4x+: 通用效果最好 | Anime6B: 适合动漫风格"
        )

        st.markdown("#### 面部修复")
        col_face1, col_face2 = st.columns(2)
        with col_face1:
            face_restore = st.checkbox("✨ 启用面部修复", value=False,
                                      help="人像照片建议开启")
        with col_face2:
            face_restore_strength = st.slider("💪 修复强度", 0.1, 1.0, 0.8, 0.1,
                                             help="调整面部修复的强度")

        st.markdown("---")
        st.markdown("### 操作")
        upscale_btn = st.button("🔍 开始修复", type="primary", use_container_width=True,
                               help="点击开始修复图片")

    with col2:
        st.markdown("### 修复结果")

        if upscale_btn and upscale_image:
            with st.spinner("🔍 AI 正在修复图片，请稍候..."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("⚙️ 正在处理图片...")
                    progress_bar.progress(30)

                    codeformer_visibility = face_restore_strength if face_restore else 0.0

                    result = st.session_state.sd_client.upscale(
                        image=orig_img,
                        scale=scale,
                        upscaler=upscaler,
                        codeformer_visibility=codeformer_visibility
                    )

                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()

                    st.session_state.generated_image = result
                    st.success(f"✅修复成功! 新尺寸: {result.width}×{result.height}")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌修复失败: {str(e)}")

        if st.session_state.generated_image:
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 15px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
                        box-shadow: 0 8px 32px rgba(0,0,0,0.37);'>
            """, unsafe_allow_html=True)

            st.image(st.session_state.generated_image, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # 图片信息对比
            img = st.session_state.generated_image
            st.markdown("#### 修复效果对比")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("新尺寸", f"{img.width}×{img.height}")
            with col_info2:
                if upscale_image:
                    ratio = img.width / orig_img.width
                    st.metric("修复倍数", f"{ratio:.1f}x")
            with col_info3:
                st.metric("总像素", f"{img.width * img.height:,}")

            # 保存功能
            st.markdown("---")
            st.markdown("### 保存图片")
            col_save1, col_save2 = st.columns([3, 1])
            with col_save1:
                filename = st.text_input("📁 文件名", placeholder="留空自动生成",
                                        label_visibility="collapsed", key="upscale_filename")
            with col_save2:
                if st.button("💾 保存", use_container_width=True, type="primary", key="upscale_save"):
                    save_status = save_image(st.session_state.generated_image, filename)
                    if "✅" in save_status:
                        st.success(save_status)
                    else:
                        st.error(save_status)
        else:
            # 空状态提示
            st.markdown("""
            <div style='text-align: center; padding: 60px 20px;
                        background: rgba(255,255,255,0.5); border-radius: 15px;
                        border: 2px dashed #ccc;'>
                <h3 style='color: #999;'>🔍</h3>
                <p style='color: #999;'>暂无修复结果</p>
                <p style='color: #bbb; font-size: 0.9rem;'>上传图片后点击"开始修复"</p>
            </div>
            """, unsafe_allow_html=True)

def models_tab():
    """模型管理标签页 - 优化版"""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 模型切换")

        checkpoints = []
        checkpoint_names = []
        loras = []
        samplers = []
        current_model = None

        if st.session_state.sd_client and st.session_state.sd_client.check_connection():
            try:
                with st.spinner("🔍 正在加载模型列表..."):
                    checkpoints = st.session_state.sd_client.get_models()
                    # 提取模型名称用于显示
                    checkpoint_names = [model.get("title", model.get("model_name", "未知模型")) for model in checkpoints]
                    loras = st.session_state.sd_client.get_loras()
                    samplers = st.session_state.sd_client.get_samplers()
                    # 获取当前模型
                    current_model = st.session_state.sd_client.get_current_model()
            except Exception as e:
                st.warning(f"⚠️ 加载模型列表失败: {e}")

        # 显示当前模型
        if current_model:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 15px; border-radius: 10px; margin-bottom: 15px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
                <p style='color: white; margin: 0; font-weight: 600;'>
                    ✅ 当前模型: <strong>{current_model}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("💡 无法获取当前模型信息")

        # 确定默认选中的索引
        default_index = 0
        if current_model and checkpoint_names:
            try:
                default_index = checkpoint_names.index(current_model)
            except ValueError:
                # 如果当前模型不在列表中，尝试部分匹配
                for i, name in enumerate(checkpoint_names):
                    if current_model in name or name in current_model:
                        default_index = i
                        break

        model_name = st.selectbox(
            "🎯 Checkpoint 模型",
            checkpoint_names if checkpoint_names else ["未检测到模型"],
            index=default_index,
            help="选择要使用的主模型"
        )

        if st.button("🔄 切换模型", use_container_width=True, type="primary"):
            if model_name == "未检测到模型":
                st.warning("⚠️ 没有可用的模型可切换")
            else:
                try:
                    with st.spinner("⚙️ 正在切换模型..."):
                        logger.info(f"用户请求切换到模型: {model_name}")
                        success = st.session_state.sd_client.set_model(model_name)
                        if success:
                            st.success(f"✅ 已切换到模型: {model_name}")
                            st.balloons()
                        else:
                            st.error("❌ 切换模型失败，请检查：")
                            st.markdown("""
                            - SD WebUI 服务是否正常运行
                            - 模型是否存在于 models/Stable-diffusion 目录
                            - 查看控制台日志获取详细错误信息
                            """)
                except Exception as e:
                    st.error(f"❌ 错误: {str(e)}")
                    logger.exception("切换模型时发生异常")

        st.markdown("---")
        st.markdown("###  LoRA 模型管理")

        # 添加刷新按钮
        if st.button("🔄 刷新 LoRA 列表", use_container_width=True):
            if st.session_state.sd_client:
                try:
                    with st.spinner("🔄 正在刷新 LoRA 列表..."):
                        success = st.session_state.sd_client.refresh_loras()
                        if success:
                            st.success("✅ LoRA 列表刷新成功")
                            st.rerun()
                        else:
                            st.warning("⚠️ 刷新失败，请检查 SD WebUI 连接")
                except Exception as e:
                    st.error(f"❌ 刷新失败: {e}")

        if loras:
            st.markdown("#### 可用 LoRA 列表")
            # 创建 LoRA 名称列表
            lora_names = [lora.get("name", lora.get("alias", "未知")) for lora in loras]

            # 显示 LoRA 数据表
            st.dataframe(loras, use_container_width=True, height=200)
            st.info(f"💡 共检测到 {len(loras)} 个 LoRA 模型")

            st.markdown("#### 选择要使用的 LoRA")
            st.info("💡 提示：在下方选择 LoRA 并设置权重后，在文生图/图生图时会自动应用")

            # 使用 session_state 保存选中的 LoRAs
            if 'selected_loras' not in st.session_state:
                st.session_state.selected_loras = []

            # 多选 LoRA
            num_loras = st.number_input(
                "选择要使用的 LoRA 数量",
                min_value=0,
                max_value=min(5, len(lora_names)),
                value=min(1, len(lora_names)),
                step=1,
                help="可以同时使用多个 LoRA 模型"
            )

            selected_loras_config = []

            if num_loras > 0:
                for i in range(int(num_loras)):
                    st.markdown(f"**LoRA {i+1}:**")
                    col_lora, col_weight = st.columns([2, 1])

                    with col_lora:
                        selected_lora = st.selectbox(
                            f"LoRA 模型 {i+1}",
                            ["无"] + lora_names,
                            key=f"lora_select_{i}",
                            help="选择要使用的 LoRA 模型"
                        )

                    with col_weight:
                        lora_weight = st.slider(
                            f"权重 {i+1}",
                            min_value=0.0,
                            max_value=2.0,
                            value=1.0,
                            step=0.1,
                            key=f"lora_weight_{i}",
                            help="LoRA 的影响强度，通常在 0.5-1.5 之间"
                        )

                    if selected_lora != "无":
                        selected_loras_config.append({
                            "name": selected_lora,
                            "weight": lora_weight
                        })

                # 保存到 session_state
                st.session_state.selected_loras = selected_loras_config

                # 显示当前配置
                if selected_loras_config:
                    st.markdown("#### ✅ 当前 LoRA 配置")
                    for lora_config in selected_loras_config:
                        st.markdown(f"- **{lora_config['name']}** (权重: {lora_config['weight']})")
                else:
                    st.session_state.selected_loras = []
            else:
                st.session_state.selected_loras = []
        else:
            st.info("📭 未检测到 LoRA 模型")
            st.session_state.selected_loras = []

    with col2:
        st.markdown("### 采样器列表")
        if samplers:
            st.dataframe(samplers, use_container_width=True, height=200)
            st.info(f"💡 共检测到 {len(samplers)} 个采样器")
        else:
            st.info("📭 未检测到采样器")

        st.markdown("---")
        st.markdown("### 📚 推荐配置")

        # 人像写实配置
        with st.expander("👤 人像写实", expanded=True):
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        padding: 15px; border-radius: 10px;'>
                <p><strong> Checkpoint:</strong> ChilloutMix / Realistic Vision</p>
                <p><strong>⚙️ 采样器:</strong> DPM++ 2M Karras</p>
                <p><strong>🔄 Steps:</strong> 30-40</p>
                <p><strong>🎯 CFG:</strong> 7-8</p>
            </div>
            """, unsafe_allow_html=True)

        # 唐卡风格配置
        with st.expander("🏔️ 唐卡风格"):
            st.markdown("""
            <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                        padding: 15px; border-radius: 10px;'>
                <p><strong> Checkpoint:</strong> Deliberate / DreamShaper</p>
                <p><strong>🎭 LoRA:</strong> thangka_style (权重 0.7-0.8)</p>
                <p><strong>⚙️ 采样器:</strong> Euler a</p>
                <p><strong>🔄 Steps:</strong> 25-35</p>
                <p><strong>🎯 CFG:</strong> 7-9</p>
            </div>
            """, unsafe_allow_html=True)

        # 风景配置
        with st.expander("🌄 风景摄影"):
            st.markdown("""
            <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                        padding: 15px; border-radius: 10px;'>
                <p><strong> Checkpoint:</strong> Deliberate / Realistic Vision</p>
                <p><strong>⚙️ 采样器:</strong> DPM++ SDE Karras</p>
                <p><strong>🔄 Steps:</strong> 30-50</p>
                <p><strong>🎯 CFG:</strong> 7-10</p>
            </div>
            """, unsafe_allow_html=True)

def translation_tab():
    """藏汉机器翻译标签页"""

    # 翻译引擎选择
    st.markdown("### 翻译引擎选择")

    engine_options = {
        "Ollama (本地模型)": "ollama",
        "Hunyuan (腾讯混元-本地)": "hunyuan"
    }

    # 反向映射
    engine_names = {v: k for k, v in engine_options.items()}
    current_engine_display = engine_names.get(st.session_state.translation_engine, "Ollama (本地模型)")

    selected_engine_display = st.selectbox(
        "选择翻译引擎",
        list(engine_options.keys()),
        index=list(engine_options.keys()).index(current_engine_display),
        help="选择要使用的翻译引擎",
        key="translation_engine_selector"
    )

    selected_engine = engine_options[selected_engine_display]

    # 更新 session state
    if selected_engine != st.session_state.translation_engine:
        st.session_state.translation_engine = selected_engine
        logger.info(f"切换翻译引擎到: {selected_engine}")

    # 显示引擎状态
    col_status1, col_status2 = st.columns([3, 1])

    with col_status1:
        if selected_engine == "ollama":
            if st.session_state.ollama_client and st.session_state.ollama_client.check_connection():
                st.success("✅ 当前使用：Ollama 本地模型")
            else:
                st.error("❌ Ollama 服务未连接")
        elif selected_engine == "hunyuan":
            if st.session_state.hunyuan_client:
                # 尝试连接检查
                try:
                    if st.session_state.hunyuan_client.check_connection():
                        st.success("✅ 当前使用：腾讯混元本地大模型 (Hunyuan-7B)")
                    else:
                        st.warning("⚠️ Hunyuan llama-server 未连接")
                        st.info("💡 请先启动 start_hunyuan.bat 脚本")
                except Exception as e:
                    st.warning(f"⚠️ Hunyuan 连接失败: {str(e)}")
                    st.info("💡 请检查：\n- 是否已运行 start_hunyuan.bat\n- 端口 8080 是否被占用")
            else:
                st.error("❌ Hunyuan 客户端未初始化")

    with col_status2:
        if st.button("🔄 刷新状态", help="重新检查服务连接状态"):
            st.rerun()

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 输入设置")

        # 仅在使用 Ollama 时显示模型管理
        if selected_engine == "ollama":
            st.markdown("#### 🤖 模型管理")

        # 获取可用模型列表
        available_models = []
        current_model = None

        if st.session_state.ollama_client:
            try:
                available_models = st.session_state.ollama_client.list_models()
                if not available_models:
                    st.warning("⚠️ 未检测到可用的 Ollama 模型")
            except Exception as e:
                st.error(f"❌ 获取模型列表失败: {str(e)}")

        # 显示当前使用的模型
        if st.session_state.translator and hasattr(st.session_state.translator, 'preferred_model'):
            current_model = st.session_state.translator.preferred_model

            # 高亮显示藏文专用模型
            tibetan_models = st.session_state.translator.tibetan_models if hasattr(st.session_state.translator, 'tibetan_models') else []
            if current_model in tibetan_models:
                st.success(f"✅ 当前翻译模型: **{current_model}** (藏文专用)")
            else:
                st.info(f"🤖 当前翻译模型: **{current_model}**")

        # 模型选择和切换
        col_model1, col_model2 = st.columns([3, 1])

        with col_model1:
            # 设置默认选择的模型
            default_index = 0
            if current_model and current_model in available_models:
                default_index = available_models.index(current_model)

            selected_model = st.selectbox(
                "🔄 选择翻译模型",
                available_models if available_models else ["无可用模型"],
                index=default_index,
                help="选择用于翻译的 Ollama 模型。推荐使用藏文专用模型如 monlam-melong、TiLamb 等",
                key="translation_model_selector"
            )

        with col_model2:
            refresh_models_btn = st.button(
                "🔄",
                use_container_width=True,
                help="刷新模型列表",
                key="refresh_translation_models"
            )

            if refresh_models_btn:
                st.rerun()

        # 切换模型按钮
        if available_models and selected_model != current_model:
            if st.button(
                "✅ 切换模型",
                type="primary",
                use_container_width=True,
                help=f"切换到模型: {selected_model}",
                key="switch_translation_model"
            ):
                try:
                    with st.spinner(f"🔄 正在切换到模型 {selected_model}..."):
                        # 更新翻译器的首选模型
                        if st.session_state.translator:
                            st.session_state.translator.preferred_model = selected_model
                            logger.info(f"翻译模型已切换到: {selected_model}")
                            st.success(f"✅ 已成功切换到模型: **{selected_model}**")
                            st.balloons()
                            # 延迟后刷新页面以更新显示
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ 翻译器未初始化")
                except Exception as e:
                    st.error(f"❌ 切换模型失败: {str(e)}")
                    logger.exception("切换翻译模型时发生错误")

        # 显示藏文专用模型提示
        if st.session_state.translator and hasattr(st.session_state.translator, 'tibetan_models'):
            tibetan_models = st.session_state.translator.tibetan_models
            if tibetan_models:
                with st.expander("💡 藏文专用模型列表", expanded=False):
                    st.markdown("以下模型专门针对藏文优化，推荐使用：")
                    for model in tibetan_models:
                        if model == current_model:
                            st.markdown(f"✅ **{model}** (当前使用)")
                        else:
                            st.markdown(f"- {model}")
            else:
                st.info("💡 提示：安装 monlam-melong 或 TiLamb 等藏文专用模型可获得更好的翻译效果")

        st.markdown("---")

        # 翻译方向选择
        direction_map = {
            "自动检测": TranslationDirection.AUTO_DETECT,
            "藏译汉": TranslationDirection.TIBETAN_TO_CHINESE,
            "汉译藏": TranslationDirection.CHINESE_TO_TIBETAN
        }

        direction_choice = st.selectbox(
            "🔄 翻译方向",
            list(direction_map.keys()),
            help="选择翻译方向，或使用自动检测"
        )
        direction = direction_map[direction_choice]

        # 输入文本
        input_text = st.text_area(
            "✏️ 输入文本",
            placeholder="请输入需要翻译的藏文或中文...\n\n示例（藏文）：བཀྲ་ཤིས་བདེ་ལེགས།\n示例（中文）：扎西德勒",
            height=200,
            help="支持藏文和中文输入"
        )

        # 翻译风格
        st.markdown("####  翻译风格")
        style_map = {
            "正式": TranslationStyle.FORMAL,
            "口语": TranslationStyle.COLLOQUIAL,
            "文学": TranslationStyle.LITERARY,
            "专业术语": TranslationStyle.TECHNICAL
        }

        style_choice = st.selectbox(
            "选择翻译风格",
            list(style_map.keys()),
            help="不同风格适用于不同场景"
        )
        style = style_map[style_choice]

        # 高级选项
        with st.expander("🔧 高级选项", expanded=False):
            include_alternatives = st.checkbox(
                "📋 显示备选翻译",
                value=False,
                help="提供多个翻译方案供参考"
            )
            include_explanation = st.checkbox(
                "💡 显示翻译说明",
                value=False,
                help="解释翻译的关键点和文化背景"
            )

        st.markdown("---")
        st.markdown("### 操作")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            translate_btn = st.button(
                "🚀 开始翻译",
                type="primary",
                use_container_width=True,
                help="点击开始翻译"
            )
        with col_btn2:
            clear_btn = st.button(
                "🗑️ 清空",
                use_container_width=True,
                help="清空输入和结果"
            )

        if clear_btn:
            st.session_state.translation_result = None
            st.rerun()

    with col2:
        st.markdown("### 📄 翻译结果")

        if translate_btn and input_text:
            with st.spinner("🔄 正在翻译中，请稍候..."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("⚙️ 正在处理翻译...")
                    progress_bar.progress(20)

                    # 根据选择的引擎进行翻译
                    if selected_engine == "google":
                        # 使用 Google Translate
                        if not st.session_state.google_client:
                            st.error("❌ Google Translate 客户端未初始化")
                        else:
                            # 转换方向格式
                            google_direction = "zh2bo" if direction == TranslationDirection.CHINESE_TO_TIBETAN else "bo2zh"
                            if direction == TranslationDirection.AUTO_DETECT:
                                google_direction = "auto"

                            translated_text = st.session_state.google_client.translate(input_text, google_direction)

                            # 构建简单的结果对象
                            class SimpleResult:
                                def __init__(self, text, dir_str):
                                    self.translated_text = text
                                    self.detected_language = "自动检测"
                                    if dir_str == "zh2bo":
                                        self.direction = TranslationDirection.CHINESE_TO_TIBETAN
                                    elif dir_str == "bo2zh":
                                        self.direction = TranslationDirection.TIBETAN_TO_CHINESE
                                    else:
                                        self.direction = TranslationDirection.AUTO_DETECT
                                    self.alternative_translations = []
                                    self.explanation = None

                            result = SimpleResult(translated_text, google_direction)

                    elif selected_engine == "hunyuan":
                        # 使用 Hunyuan
                        if not st.session_state.hunyuan_client:
                            st.error("❌ Hunyuan 客户端未初始化，请检查 llama-server 是否启动")
                        else:
                            # 转换方向格式
                            hunyuan_direction = "zh2bo" if direction == TranslationDirection.CHINESE_TO_TIBETAN else "bo2zh"

                            translated_text = st.session_state.hunyuan_client.translate(
                                text=input_text,
                                direction=hunyuan_direction,
                                style=style.value if hasattr(style, 'value') else str(style)
                            )

                            # 构建简单的结果对象
                            class SimpleResult:
                                def __init__(self, text, dir_enum):
                                    self.translated_text = text
                                    self.detected_language = "自动检测"
                                    self.direction = dir_enum
                                    self.alternative_translations = []
                                    self.explanation = None

                            result = SimpleResult(translated_text, direction)

                    else:  # ollama
                        # 使用 Ollama (原有逻辑)
                        if not st.session_state.translator:
                            st.error("❌ 翻译器未初始化，请检查 Ollama 服务是否正常")
                        else:
                            result = st.session_state.translator.translate(
                                text=input_text,
                                direction=direction,
                                style=style,
                                include_alternatives=include_alternatives,
                                include_explanation=include_explanation
                            )

                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()

                    st.session_state.translation_result = result
                    st.success("✅ 翻译完成！")

                except Exception as e:
                    st.error(f"❌ 翻译失败: {str(e)}")
                    logger.exception("翻译时发生错误")

        # 显示翻译结果
        if st.session_state.translation_result:
            result = st.session_state.translation_result

            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px;
                        border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
                        box-shadow: 0 8px 32px rgba(0,0,0,0.37);'>
            """, unsafe_allow_html=True)

            # 翻译结果
            st.markdown("#### 翻译结果")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        padding: 20px; border-radius: 10px; font-size: 1.1rem;
                        line-height: 1.8; color: #2c3e50;'>
                {result.translated_text}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # 详细信息
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("检测语言", result.detected_language or "未知")
            with col_info2:
                st.metric("翻译方向", result.direction.value)

            # 备选翻译
            if result.alternative_translations:
                with st.expander("📋 备选翻译方案", expanded=True):
                    for i, alt in enumerate(result.alternative_translations, 1):
                        st.markdown(f"""
                        <div style='background: #f8f9fa; padding: 10px; border-radius: 8px;
                                    margin: 5px 0; border-left: 4px solid #667eea;'>
                            <strong>方案 {i}:</strong> {alt}
                        </div>
                        """, unsafe_allow_html=True)

            # 翻译说明
            if result.explanation:
                with st.expander("💡 翻译说明", expanded=False):
                    st.markdown(f"""
                    <div style='background: #fff3cd; padding: 15px; border-radius: 10px;
                                border-left: 4px solid #ffc107; color: #856404;'>
                        {result.explanation}
                    </div>
                    """, unsafe_allow_html=True)

            # 复制功能
            st.markdown("---")
            st.markdown("### 复制结果")
            st.code(result.translated_text, language="text")

        else:
            # 空状态提示
            st.markdown("""
            <div style='text-align: center; padding: 60px 20px;
                        background: rgba(255,255,255,0.5); border-radius: 15px;
                        border: 2px dashed #ccc;'>
                <h3 style='color: #999;'>🌐</h3>
                <p style='color: #999;'>暂无翻译结果</p>
                <p style='color: #bbb; font-size: 0.9rem;'>输入文本后点击"开始翻译"按钮</p>
            </div>
            """, unsafe_allow_html=True)

def speech_recognition_tab():
    """藏汉语音识别标签页"""
    st.markdown("### 🎤 藏汉语音识别")
    st.info("支持藏语和汉语的实时语音识别")

    # 检查配置
    if not settings.xunfei_app_id or not settings.xunfei_api_key:
        st.error("⚠️ 讯飞语音识别未配置，请在 .env 文件中配置 XUNFEI_APP_ID、XUNFEI_API_KEY 和 XUNFEI_API_SECRET")
        return

    # 语言选择
    col1, col2 = st.columns([1, 3])

    with col1:
        language = st.selectbox(
            "选择识别语言",
            options=["中文", "藏语"],
            help="选择要识别的语言"
        )

        # 根据语言设置参数
        if language == "藏语":
            lang_code = "bo_cn"
            accent = "tibetan"
        else:
            lang_code = "zh_cn"
            accent = "mandarin"

    with col2:
        st.info(f"📌 当前识别语言: **{language}** ")

    st.markdown("---")

    # 录音和上传选项
    tab_record, tab_upload = st.tabs(["🎙️ 实时录音", "📁 上传音频文件"])

    with tab_record:
        st.markdown("#### 🎤 实时录音识别")

        # 美化的录音区域
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                    padding: 30px; border-radius: 15px; margin: 20px 0;
                    border: 2px solid rgba(102, 126, 234, 0.3);
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);'>
            <div style='text-align: center; margin-bottom: 20px;'>
                <div style='font-size: 3rem; margin-bottom: 10px;'>🎙️</div>
                <h3 style='color: #667eea; margin: 10px 0;'>语音录制中心</h3>
                <p style='color: #888; font-size: 0.95rem; margin-bottom: 20px;'>点击下方按钮开始录音，录音完成后自动识别</p>
            </div>
        """, unsafe_allow_html=True)

        # 使用 Streamlit 的音频录制组件（嵌入在美化区域内）
        col_space1, col_audio, col_space2 = st.columns([1, 2, 1])
        with col_audio:
            audio_bytes = st.audio_input("🎤 点击此处开始录音", key="audio_recorder")

        if audio_bytes:
            # 美化的音频播放区域
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(67, 160, 71, 0.1) 100%);
                        padding: 20px; border-radius: 12px; margin: 15px 0;
                        border: 2px solid rgba(76, 175, 80, 0.3);'>
                <div style='text-align: center;'>
                    <p style='color: #4CAF50; font-weight: 600; margin: 0 0 10px 0;'>
                        ✅ 录音完成！请播放确认后点击识别
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.audio(audio_bytes, format="audio/wav")

            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                if st.button("🚀 开始识别", type="primary", use_container_width=True):
                    with st.spinner("正在识别中..."):
                        try:
                            import tempfile

                            # 保存音频到临时文件
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                tmp_file.write(audio_bytes.getvalue())
                                tmp_path = tmp_file.name

                            # 创建 ASR 客户端
                            asr_client = get_asr_client(language=lang_code, accent=accent)

                            # 读取音频文件
                            with open(tmp_path, 'rb') as f:
                                audio_data = f.read()

                            # 识别音频
                            result = asr_client.speech_to_text(audio_data)

                            # 显示结果
                            if result.get("success"):
                                result_text = result.get("text")
                                st.success("✅ 识别完成！")
                                st.markdown("#### 📝 识别结果")
                                st.markdown(f"""
                                <div style='background: rgba(255, 255, 255, 0.8); padding: 20px;
                                            border-radius: 12px; margin: 15px 0;
                                            border-left: 4px solid #D4AF37;'>
                                    <p style='color: #5A4A3A; font-size: 1.1rem; line-height: 1.6;'>{result_text}</p>
                                </div>
                                """, unsafe_allow_html=True)

                                # 保存到 session state
                                st.session_state.asr_result = result_text

                                # 提供复制按钮
                                st.text_area("识别文本（可复制）", result_text, height=100)
                            else:
                                st.warning(f"⚠️ 识别失败: {result.get('error', '未知错误')}")

                            # 清理临时文件
                            import os
                            os.unlink(tmp_path)

                        except Exception as e:
                            st.error(f"❌ 识别失败: {str(e)}")
                            logger.error(f"语音识别错误: {e}", exc_info=True)

            with col_btn2:
                if st.button("🗑️ 清除录音", use_container_width=True):
                    st.rerun()

    with tab_upload:
        st.markdown("#### 📁 上传音频文件识别")
        st.info("支持 WAV、MP3、PCM 格式，最长 60 秒")

        uploaded_file = st.file_uploader(
            "选择音频文件",
            type=["wav", "mp3", "pcm"],
            help="支持 WAV、MP3、PCM 格式，采样率 16kHz 或 8kHz"
        )

        if uploaded_file:
            st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[-1]}")

            if st.button("🚀 开始识别", type="primary", use_container_width=True, key="upload_recognize"):
                with st.spinner("正在识别中..."):
                    try:
                        import tempfile

                        # 保存上传的文件到临时文件
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name

                        # 创建 ASR 客户端
                        asr_client = get_asr_client(language=lang_code, accent=accent)

                        # 读取音频文件
                        with open(tmp_path, 'rb') as f:
                            audio_data = f.read()

                        # 识别音频
                        result = asr_client.speech_to_text(audio_data)

                        # 显示结果
                        if result.get("success"):
                            result_text = result.get("text")
                            st.success("✅ 识别完成！")
                            st.markdown("#### 📝 识别结果")
                            st.markdown(f"""
                            <div style='background: rgba(255, 255, 255, 0.8); padding: 20px;
                                        border-radius: 12px; margin: 15px 0;
                                        border-left: 4px solid #D4AF37;'>
                                <p style='color: #5A4A3A; font-size: 1.1rem; line-height: 1.6;'>{result_text}</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # 保存到 session state
                            st.session_state.asr_result = result_text

                            # 提供复制按钮
                            st.text_area("识别文本（可复制）", result_text, height=100, key="upload_result")
                        else:
                            st.warning(f"⚠️ 识别失败: {result.get('error', '未知错误')}")

                        # 清理临时文件
                        import os
                        os.unlink(tmp_path)

                    except Exception as e:
                        st.error(f"❌ 识别失败: {str(e)}")
                        logger.error(f"语音识别错误: {e}", exc_info=True)

    # 使用说明
    st.markdown("---")
    st.markdown("### 💡 使用说明")

    col_tip1, col_tip2 = st.columns(2)

    with col_tip1:
        st.markdown("#### 🎤 录音要求")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #D4AF37 0%, #C9A961 100%);
                    padding: 20px; border-radius: 10px; color: white;'>
            <ul>
                <li>保持环境安静，减少背景噪音</li>
                <li>说话清晰，语速适中</li>
                <li>录音时长不超过 60 秒</li>
                <li>建议使用外置麦克风以获得更好效果</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_tip2:
        st.markdown("#### 📁 音频格式")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #8B7355 0%, #6B5A4A 100%);
                    padding: 20px; border-radius: 10px; color: white;'>
            <ul>
                <li>支持格式: WAV, MP3, PCM</li>
                <li>采样率: 16kHz (推荐) 或 8kHz</li>
                <li>位深: 16bit</li>
                <li>声道: 单声道</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def help_tab():
    """帮助标签页 - 优化版"""
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px; margin-bottom: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);'>
        <h2 style='color: #667eea; margin: 0;'>📖 使用指南 - 快速上手</h2>
        <p style='color: #666; margin-top: 5px;'>详细的功能说明和使用技巧</p>
    </div>
    """, unsafe_allow_html=True)

    # 文生图指南
    with st.expander(" 文生图 - 详细指南", expanded=True):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 20px; border-radius: 10px; margin: 10px 0;'>
            <h4 style='color: #667eea;'>📝 基本步骤</h4>
            <ol>
                <li><strong>输入描述</strong>：用中文描述你想要生成的图片，系统会自动翻译并优化</li>
                <li><strong>选择风格</strong>：选择预设风格可以自动添加相关的提示词</li>
                <li><strong>调整参数</strong>：根据需要调整生成参数</li>
            </ol>

            <h4 style='color: #667eea; margin-top: 20px;'>⚙️ 参数说明</h4>
            <ul>
                <li><strong>宽度/高度</strong>：图片尺寸，建议使用 512×512 或 768×768</li>
                <li><strong>采样步数</strong>：越高质量越好，但速度越慢，推荐 25-40</li>
                <li><strong>CFG Scale</strong>：越高越接近提示词，推荐 7-9</li>
                <li><strong>种子</strong>：-1 为随机，固定种子可复现结果</li>
            </ul>

            <h4 style='color: #667eea; margin-top: 20px;'>💡 使用技巧</h4>
            <ul>
                <li>描述越详细，生成效果越好</li>
                <li>使用 RAG 知识库可以增强藏族文化相关内容</li>
                <li>开启 LLM 优化可以自动扩写和翻译提示词</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 图生图指南
    with st.expander(" 图生图 - 详细指南"):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                    padding: 20px; border-radius: 10px; margin: 10px 0;'>
            <h4 style='color: #d35400;'>📝 基本步骤</h4>
            <ol>
                <li>上传参考图片</li>
                <li>描述想要的变化</li>
                <li>调整重绘强度</li>
            </ol>

            <h4 style='color: #d35400; margin-top: 20px;'>🎚️ 重绘强度说明</h4>
            <ul>
                <li><strong>0.3-0.5</strong>：轻微变化，保留原图大部分内容</li>
                <li><strong>0.5-0.7</strong>：中等变化，适合风格转换</li>
                <li><strong>0.7-0.9</strong>：大幅变化，只保留基本构图</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 图文制作指南
    with st.expander(" 图文制作 - 详细指南"):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                    padding: 20px; border-radius: 10px; margin: 10px 0;'>
            <h4 style='color: #16a085;'>📝 两种制作模式</h4>
            <ul>
                <li><strong>✍️ 简单文字添加</strong>：快速在图片上添加单个文字内容</li>
                <li><strong>🎭 专业海报设计</strong>：创建带标题、副标题和底部文字的专业海报</li>
            </ul>

            <h4 style='color: #16a085; margin-top: 20px;'>🔤 藏文字体选择</h4>
            <ul>
                <li>支持选择系统中已安装的藏文字体</li>
                <li>推荐字体：Microsoft Himalaya、Noto Sans Tibetan、Jomolhari</li>
                <li>可预览字体效果，确保藏文显示正确</li>
                <li>藏文示例：བཀྲ་ཤིས་བདེ་ལེགས། (扎西德勒)</li>
            </ul>

            <h4 style='color: #16a085; margin-top: 20px;'>✍️ 简单文字添加模式</h4>
            <ul>
                <li>支持中文、藏文、英文等多语言</li>
                <li>可自定义字体大小、颜色、描边</li>
                <li>支持9个位置选择（左上、上中、右上等）</li>
                <li>支持添加半透明背景框</li>
            </ul>

            <h4 style='color: #c0392b; margin-top: 20px;'>🎭 专业海报设计模式</h4>
            <ol>
                <li>先用文生图生成背景图片</li>
                <li>上传背景图片到图文制作</li>
                <li>切换到"专业海报设计"模式</li>
                <li>添加主标题、副标题和底部文字</li>
                <li>选择藏文字体（推荐使用支持藏文的字体）</li>
                <li>选择是否添加渐变遮罩和装饰边框</li>
            </ol>

            <h4 style='color: #16a085; margin-top: 20px;'> 设计建议</h4>
            <ul>
                <li>建议使用描边增加文字可读性</li>
                <li>深色背景使用浅色文字，浅色背景使用深色文字</li>
                <li>渐变遮罩可以提升文字可读性</li>
                <li>藏式边框适合藏族文化主题</li>
                <li>海报标题字体建议 60-80 像素</li>
                <li>简单文字字体建议 40-60 像素</li>
            </ul>

            <h4 style='color: #16a085; margin-top: 20px;'>💡 使用技巧</h4>
            <ul>
                <li>使用字体预览功能确保藏文正确显示</li>
                <li>添加背景框可以让文字更突出</li>
                <li>合理搭配文字颜色和描边颜色</li>
                <li>海报模式下渐变遮罩可显著提升效果</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    #图片修复指南
    with st.expander("🔍图片修复 - 详细指南"):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 10px; margin: 10px 0; color: white;'>
            <h4 style='color: #fff;'>🎯 算法选择</h4>
            <ul>
                <li><strong>R-ESRGAN 4x+</strong>：通用修复，效果最好，适合大多数场景</li>
                <li><strong>R-ESRGAN 4x+ Anime6B</strong>：专为动漫风格优化</li>
                <li><strong>SwinIR_4x</strong>：适合细节丰富的图片</li>
                <li><strong>Lanczos/Nearest</strong>：传统算法，速度快但效果一般</li>
            </ul>

            <h4 style='color: #fff; margin-top: 20px;'>👤 面部修复</h4>
            <ul>
                <li>人像照片建议开启面部修复</li>
                <li>修复强度 0.7-0.9 效果较好</li>
                <li>风景照片不需要开启</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 藏汉翻译指南
    with st.expander("🌐 藏汉翻译 - 详细指南"):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                    padding: 20px; border-radius: 10px; margin: 10px 0; color: white;'>
            <h4 style='color: white;'>🤖 模型选择与切换</h4>
            <ul>
                <li><strong>查看当前模型</strong>：页面顶部显示当前使用的翻译模型</li>
                <li><strong>选择模型</strong>：从下拉列表中选择其他可用的 Ollama 模型</li>
                <li><strong>切换模型</strong>：选择新模型后点击"切换模型"按钮</li>
                <li><strong>刷新模型</strong>：点击 🔄 按钮刷新模型列表</li>
                <li><strong>藏文专用模型</strong>：系统会自动识别并推荐藏文专用模型（如 monlam-melong、TiLamb）</li>
            </ul>

            <h4 style='color: white; margin-top: 20px;'>📝 基本功能</h4>
            <ul>
                <li><strong>自动检测</strong>：系统自动识别输入的语言类型</li>
                <li><strong>双向翻译</strong>：支持藏译汉和汉译藏</li>
                <li><strong>多种风格</strong>：正式、口语、文学、专业术语</li>
                <li><strong>备选方案</strong>：提供多个翻译选项供参考</li>
            </ul>

            <h4 style='color: white; margin-top: 20px;'> 翻译风格说明</h4>
            <ul>
                <li><strong>正式</strong>：适用于公文、新闻、学术文章</li>
                <li><strong>口语</strong>：适用于日常对话、聊天</li>
                <li><strong>文学</strong>：适用于诗歌、散文、文学作品</li>
                <li><strong>专业术语</strong>：适用于佛教、医学等专业领域</li>
            </ul>

            <h4 style='color: white; margin-top: 20px;'>💡 使用技巧</h4>
            <ul>
                <li>输入完整的句子或段落，翻译效果更好</li>
                <li>专有名词（人名、地名）会自动识别并使用标准译名</li>
                <li>开启"备选翻译"可以看到不同的表达方式</li>
                <li>开启"翻译说明"可以了解翻译的关键点</li>
                <li><strong>使用藏文专用模型可获得更准确的翻译结果</strong></li>
            </ul>

            <h4 style='color: white; margin-top: 20px;'>🔧 推荐模型</h4>
            <ul>
                <li><strong>monlam-melong</strong>：Monlam 开发的藏文翻译模型，推荐首选</li>
                <li><strong>TiLamb</strong>：专门针对藏文训练的大语言模型</li>
                <li><strong>qwen2:7b</strong>：通用大模型，支持藏文但效果可能不如专用模型</li>
            </ul>

            <h4 style='color: white; margin-top: 20px;'>📋 常用藏文示例</h4>
            <ul>
                <li>བཀྲ་ཤིས་བདེ་ལེགས། - 扎西德勒（吉祥如意）</li>
                <li>ཐུགས་རྗེ་ཆེ། - 图杰切（谢谢）</li>
                <li>བོད་ཀྱི་རིག་གནས། - 藏族文化</li>
                <li>ཕོ་བྲང་པོ་ཏ་ལ། - 布达拉宫</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 藏族文化关键词
    st.markdown("---")
    st.markdown("## 🏔️ 藏族文化关键词参考")

    col_key1, col_key2, col_key3 = st.columns(3)

    with col_key1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 15px; border-radius: 10px; color: white;'>
            <h4 style='color: white;'>👥 人物</h4>
            <ul>
                <li>格萨尔王</li>
                <li>康巴汉子</li>
                <li>藏族姑娘</li>
                <li>风景、蓝天</li>
            </ul>
            <h4 style='color: white; margin-top: 15px;'>👔 服饰</h4>
            <ul>
                <li>藏袍</li>
                <li>氆氇</li>
                <li>邦典</li>
                <li>英雄结</li>
                <li>哈达</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_key2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    padding: 15px; border-radius: 10px; color: white;'>
            <h4 style='color: white;'>🏛️ 建筑</h4>
            <ul>
                <li>布达拉宫</li>
                <li>大昭寺</li>
                <li>白塔</li>
                <li>玛尼堆</li>
            </ul>
            <h4 style='color: white; margin-top: 15px;'>🙏 宗教</h4>
            <ul>
                <li>唐卡</li>
                <li>转经筒</li>
                <li>经幡</li>
                <li>酥油灯</li>
                <li>金刚杵</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_key3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                    padding: 15px; border-radius: 10px; color: white;'>
            <h4 style='color: white;'>🌄 自然</h4>
            <ul>
                <li>雪山</li>
                <li>草原</li>
                <li>牦牛</li>
                <li>青稞</li>
                <li>纳木错</li>
                <li>珠穆朗玛峰</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 常见问题
    st.markdown("---")
    st.markdown("## ⚠️ 常见问题解答")

    with st.expander("❓ 生成速度很慢怎么办？"):
        st.markdown("""
        <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #667eea;'>
            <p><strong>可能原因：</strong></p>
            <ul>
                <li>GPU 性能不足或未正确配置</li>
                <li>图片尺寸过大</li>
                <li>采样步数设置过高</li>
            </ul>
            <p><strong>解决方案：</strong></p>
            <ul>
                <li>检查 GPU 是否正常工作</li>
                <li>降低图片尺寸（如 512×512）</li>
                <li>减少采样步数（20-30 步）</li>
                <li>关闭高清修复功能</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("❓ 生成的图片质量不好？"):
        st.markdown("""
        <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #667eea;'>
            <p><strong>优化建议：</strong></p>
            <ul>
                <li>增加采样步数到 30-40</li>
                <li>调整 CFG Scale 到 7-9</li>
                <li>使用更详细的描述</li>
                <li>开启 RAG 知识库增强</li>
                <li>开启 LLM 优化功能</li>
                <li>尝试不同的模型和采样器</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("❓ 藏文显示不正常？"):
        st.markdown("""
        <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #667eea;'>
            <p><strong>解决方案：</strong></p>
            <ul>
                <li>确保已安装藏文字体（如 Noto Sans Tibetan）</li>
                <li>检查系统字体配置</li>
                <li>重启应用程序</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("❓ 连接服务失败？"):
        st.markdown("""
        <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #667eea;'>
            <p><strong>检查步骤：</strong></p>
            <ol>
                <li>确认 SD WebUI 服务已启动</li>
                <li>确认 Ollama 服务已启动</li>
                <li>检查服务端口配置是否正确</li>
                <li>检查防火墙设置</li>
                <li>查看侧边栏的服务状态</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # 使用技巧
    st.markdown("---")
    st.markdown("## 💡 使用技巧")

    col_tip1, col_tip2 = st.columns(2)

    with col_tip1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 10px; color: white;'>
            <h4 style='color: white;'> 提示词技巧</h4>
            <ul>
                <li>使用具体的形容词（如"精致的"、"华丽的"）</li>
                <li>描述光线和氛围（如"柔和的光线"、"神秘的氛围"）</li>
                <li>指定视角和构图（如"正面视角"、"特写镜头"）</li>
                <li>添加艺术风格（如"油画风格"、"摄影作品"）</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_tip2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 20px; border-radius: 10px; color: white;'>
            <h4 style='color: white;'>⚡ 效率提升</h4>
            <ul>
                <li>使用预设风格快速生成</li>
                <li>固定种子复现满意的结果</li>
                <li>先用低步数测试，满意后提高质量</li>
                <li>保存常用的提示词模板</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def main():
    """主函数"""
    # 初始化客户端
    initialize_clients()

    # 标题栏 - 长条形式，置于最上方
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, rgba(201, 169, 97, 0.12) 100%);
                padding: 14px 0;
                margin: -60px -100px 25px -100px;
                box-shadow: 0 2px 8px rgba(212, 175, 55, 0.1);
                border-bottom: 2px solid rgba(212, 175, 55, 0.2);
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;'>
        <h2 style='text-align: center; color: #5A4A3A; margin: 0 0 3px 0;
                   font-family: "果洛德昂洒智—艺钦体", "Noto Sans Tibetan", serif;
                   font-size: 2.8rem; font-weight: 600; letter-spacing: 1.5px; line-height: 1.3;'>
            ༄༅།།བྱང་སྐར་རིག་ནུས་པར་ཡིག་ལེགས་འགྲུབ་མ་ལག།
        </h2>
        <h1 style='text-align: center; color: #D4AF37; margin: 0 0 3px 0;
                   text-shadow: 1px 1px 2px rgba(212, 175, 55, 0.15);
                   font-size: 2.4rem; font-weight: 700; letter-spacing: 2px; line-height: 1.3;'>
            极地星光汉藏智能图文生成系统
        </h1>
        <h3 style='text-align: center; color: #C9A961; margin: 0;
                   font-weight: 500; letter-spacing: 1.5px; font-size: 1.5rem; line-height: 1.3;'>
            POLAR STARLIGHT AI GENERATION SYSTEM
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # 侧边栏 - 优化后的服务状态显示
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 16px 0;'>
            <h2 style='color: #D4AF37; margin: 0; font-size: 1.3rem; font-weight: 600;'>⚙️ 系统控制台</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # 刷新按钮
        if st.button("🔄 刷新服务状态", use_container_width=True):
            st.rerun()

        st.markdown("### 服务状态")
        status = check_services()

        # SD WebUI 状态
        sd_status = "✅ 已连接" if status['sd_webui'] else "❌ 未连接"
        sd_color = "#28a745" if status['sd_webui'] else "#dc3545"
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.5); padding: 10px; border-radius: 8px; margin: 5px 0;
                    border-left: 4px solid {sd_color};'>
            <strong style='color: #5A4A3A;'>SD WebUI:</strong> <span style='color: #6B5A4A;'>{sd_status}</span>
        </div>
        """, unsafe_allow_html=True)

        # Ollama 状态
        ollama_status = "✅ 已连接" if status['ollama'] else "❌ 未连接"
        ollama_color = "#28a745" if status['ollama'] else "#dc3545"
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.5); padding: 10px; border-radius: 8px; margin: 5px 0;
                    border-left: 4px solid {ollama_color};'>
            <strong style='color: #5A4A3A;'>Ollama:</strong> <span style='color: #6B5A4A;'>{ollama_status}</span>
        </div>
        """, unsafe_allow_html=True)

        # RAG 状态
        rag_status = f"✅ {status['rag_docs']} 文档" if status['rag'] else "❌ 未初始化"
        rag_color = "#28a745" if status['rag'] else "#dc3545"
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.5); padding: 10px; border-radius: 8px; margin: 5px 0;
                    border-left: 4px solid {rag_color};'>
            <strong style='color: #5A4A3A;'>RAG 知识库:</strong> <span style='color: #6B5A4A;'>{rag_status}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")


        # 系统信息
        st.markdown("### 📈 系统信息")
        if st.session_state.generated_image:
            img = st.session_state.generated_image
            st.metric("最近生成尺寸", f"{img.width}×{img.height}")

        st.markdown("---")

        # 版本信息
        st.markdown("""
        <div style='text-align: center; padding: 10px; color: #999; font-size: 0.8rem;'>
            <p>POLAR STARLIGHT V1.0</p>
            <p>© 2026 By Gudrak Dorjee</p>
        </div>
        """, unsafe_allow_html=True)

    # 主标签页
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        " 文生图",
        " 图生图",
        " 图文制作",
        "🔍图片修复",
        "🌐 藏汉翻译",
        "🎤 语音识别",
        "⚙️ 模型管理",
        "❓ 帮助"
    ])

    # 文生图标签页
    with tab1:
        txt2img_tab()

    # 图生图标签页
    with tab2:
        img2img_tab()

    # 图文制作标签页（合并了文字排版和海报制作）
    with tab3:
        graphic_design_tab()

    # 图片修复标签页
    with tab4:
        upscale_tab()

    # 藏汉翻译标签页
    with tab5:
        translation_tab()

    # 语音识别标签页
    with tab6:
        speech_recognition_tab()

    # 模型管理标签页
    with tab7:
        models_tab()

    # 帮助标签页
    with tab8:
        help_tab()

    # 页脚
    st.markdown("""
    ---
    <center>
     极地星光汉藏智能图文生成系统 | 基于 Stable Diffusion + Ollama + RAG
    </center>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("正在启动极地星光汉藏智能图文生成系统...")
    print("="*60 + "\n")

    main()
