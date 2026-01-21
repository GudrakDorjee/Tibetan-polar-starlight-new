"""
Gradio Web 界面
提供用户友好的图片生成界面
"""

import gradio as gr
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, List
import logging
import time
import json

# 导入中间件
import sys
sys.path.append(str(Path(__file__).parent.parent))

from middleware.sd_client import SDClient, get_sd_client, GenerationResult
from middleware.ollama_client import OllamaClient, get_ollama_client
from middleware.prompt_engineer import PromptEngineer, create_prompt_engineer
from middleware.text_renderer import TibetanTextRenderer, create_text_renderer, TextStyle, TextBox, TextPosition
from middleware.rag_engine import RAGEngine, create_rag_engine
from config.settings import settings

logger = logging.getLogger(__name__)

# 全局客户端实例
sd_client: Optional[SDClient] = None
ollama_client: Optional[OllamaClient] = None
prompt_engineer: Optional[PromptEngineer] = None
text_renderer: Optional[TibetanTextRenderer] = None
rag_engine: Optional[RAGEngine] = None

def initialize_clients():
    """初始化所有客户端"""
    global sd_client, ollama_client, prompt_engineer, text_renderer, rag_engine
    
    # SD WebUI 客户端
    sd_client = get_sd_client(
        base_url=settings.sd_webui_url,
        timeout=settings.generation_timeout
    )
    
    # Ollama 客户端
    ollama_client = get_ollama_client(
        base_url=settings.ollama_url,
        model=settings.ollama_model
    )
    
    # RAG 引擎
    rag_engine = create_rag_engine(
        persist_directory=settings.rag_persist_dir,
        ollama_client=ollama_client
    )
    
    # Prompt 编排器
    prompt_engineer = create_prompt_engineer(
        ollama_client=ollama_client,
        rag_engine=rag_engine
    )
    
    # 文字渲染器
    text_renderer = create_text_renderer(
        fonts_dir=settings.fonts_dir
    )
    
    logger.info("所有客户端初始化完成")

def check_services() -> str:
    """检查服务状态"""
    status = []
    
    # 检查 SD WebUI
    if sd_client and sd_client.check_connection():
        status.append("✅ SD WebUI: 已连接")
    else:
        status.append("❌ SD WebUI: 未连接")
    
    # 检查 Ollama
    if ollama_client and ollama_client.check_connection():
        status.append("✅ Ollama: 已连接")
    else:
        status.append("❌ Ollama: 未连接")
    
    # 检查 RAG
    if rag_engine:
        stats = rag_engine.get_stats()
        status.append(f"✅ RAG: {stats['document_count']} 文档")
    else:
        status.append("❌ RAG: 未初始化")
    
    return "\n".join(status)

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
    progress=gr.Progress()
) -> Tuple[Optional[Image.Image], str, str]:
    """
    生成图片
    
    Returns:
        (生成的图片, 使用的正向提示词, 状态信息)
    """
    if not sd_client:
        return None, "", "❌ SD WebUI 未连接"
    
    try:
        progress(0.1, desc="处理提示词...")
        
        # 处理提示词
        if prompt_engineer:
            result = prompt_engineer.process(
                user_input=prompt,
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
            positive_prompt = prompt
            final_negative = negative_prompt or settings.default_negative_prompt
        
        progress(0.3, desc="生成图片中...")
        
        # 调用 SD WebUI 生成
        gen_result = sd_client.txt2img(
            prompt=positive_prompt,
            negative_prompt=final_negative,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed if seed != -1 else -1,
            enable_hr=enable_hr,
            hr_scale=hr_scale if enable_hr else 1.0
        )
        
        progress(1.0, desc="完成!")
        
        if gen_result.images:
            status = f"✅ 生成成功! 耗时: {gen_result.generation_time:.2f}秒, Seed: {gen_result.seed}"
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
    progress=gr.Progress()
) -> Tuple[Optional[Image.Image], str, str]:
    """图生图"""
    if not sd_client:
        return None, "", "❌ SD WebUI 未连接"
    
    if init_image is None:
        return None, "", "❌ 请上传参考图片"
    
    try:
        progress(0.1, desc="处理提示词...")
        
        # 处理提示词
        if prompt_engineer:
            result = prompt_engineer.process(
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
        
        progress(0.3, desc="生成图片中...")
        
        # 调用图生图
        gen_result = sd_client.img2img(
            init_image=init_image,
            prompt=positive_prompt,
            negative_prompt=final_negative,
            denoising_strength=denoising_strength,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed if seed != -1 else -1
        )
        
        progress(1.0, desc="完成!")
        
        if gen_result.images:
            status = f"✅ 生成成功! 耗时: {gen_result.generation_time:.2f}秒, Seed: {gen_result.seed}"
            return gen_result.images[0], positive_prompt, status
        else:
            return None, positive_prompt, "❌ 生成失败: 未返回图片"
    
    except Exception as e:
        logger.exception("图生图时发生错误")
        return None, "", f"❌ 错误: {str(e)}"

def add_text_to_image(
    image: Image.Image,
    text: str,
    position: str,
    font_size: int,
    text_color: str,
    stroke_width: int,
    stroke_color: str,
    add_background: bool,
    bg_opacity: float
) -> Tuple[Optional[Image.Image], str]:
    """在图片上添加文字"""
    if image is None:
        return None, "❌ 请先生成或上传图片"
    
    if not text:
        return image, "⚠️ 未输入文字"
    
    try:
        # 转换位置
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
        
        # 创建样式
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
            background=add_background,
            background_opacity=bg_opacity
        )
        
        # 渲染文字
        result = text_renderer.add_text(image, text, style, text_box)
        
        return result, "✅ 文字添加成功"
    
    except Exception as e:
        logger.exception("添加文字时发生错误")
        return image, f"❌ 错误: {str(e)}"

def create_poster(
    background: Image.Image,
    title: str,
    subtitle: str,
    footer: str,
    title_size: int,
    add_gradient: bool,
    add_border: bool,
    border_style: str
) -> Tuple[Optional[Image.Image], str]:
    """创建海报"""
    if background is None:
        return None, "❌ 请先生成或上传背景图片"
    
    if not title:
        return background, "⚠️ 请输入标题"
    
    try:
        # 标题样式
        title_style = TextStyle(
            font_size=title_size,
            color="#FFFFFF",
            stroke_width=3,
            stroke_color="#000000",
            shadow=True,
            shadow_offset=(4, 4)
        )
        
        # 创建海报
        result = text_renderer.create_poster(
            background=background,
            title=title,
            subtitle=subtitle if subtitle else None,
            footer=footer if footer else None,
            title_style=title_style,
            add_gradient_overlay=add_gradient
        )
        
        # 添加边框
        if add_border:
            border_style_map = {
                "简约": "solid",
                "双线": "double",
                "藏式": "tibetan"
            }
            result = text_renderer.add_decorative_border(
                result,
                border_width=20,
                style=border_style_map.get(border_style, "solid")
            )
        
        return result, "✅ 海报创建成功"
    
    except Exception as e:
        logger.exception("创建海报时发生错误")
        return background, f"❌ 错误: {str(e)}"

def upscale_image(
    image: Image.Image,
    scale: float,
    upscaler: str,
    face_restore: bool,
    face_restore_strength: float
) -> Tuple[Optional[Image.Image], str]:
    """放大图片"""
    if image is None:
        return None, "❌ 请先生成或上传图片"
    
    if not sd_client:
        return image, "❌ SD WebUI 未连接"
    
    try:
        codeformer_visibility = face_restore_strength if face_restore else 0.0
        
        result = sd_client.upscale(
            image=image,
            scale=scale,
            upscaler=upscaler,
            codeformer_visibility=codeformer_visibility
        )
        
        return result, f"✅ 放大成功! 新尺寸: {result.width}x{result.height}"
    
    except Exception as e:
        logger.exception("放大图片时发生错误")
        return image, f"❌ 错误: {str(e)}"

def preview_prompt(
    prompt: str,
    style: str,
    quality: str,
    composition: str,
    use_rag: bool,
    use_llm: bool
) -> Tuple[str, str]:
    """预览处理后的提示词"""
    if not prompt:
        return "", ""
    
    try:
        if prompt_engineer:
            result = prompt_engineer.process(
                user_input=prompt,
                style=style if style != "通用" else None,
                quality=quality,
                composition=composition if composition != "无" else None,
                use_rag=use_rag,
                use_llm=use_llm
            )
            return result.positive_prompt, result.negative_prompt
        else:
            return prompt, settings.default_negative_prompt
    
    except Exception as e:
        logger.exception("预览提示词时发生错误")
        return f"错误: {str(e)}", ""

def load_models_list() -> Tuple[List[str], List[str], List[str]]:
    """加载可用的模型列表"""
    checkpoints = []
    loras = []
    samplers = []
    
    if sd_client and sd_client.check_connection():
        try:
            checkpoints = sd_client.get_models()
            loras = sd_client.get_loras()
            samplers = sd_client.get_samplers()
        except Exception as e:
            logger.warning(f"加载模型列表失败: {e}")
    
    return checkpoints, loras, samplers

def change_model(model_name: str) -> str:
    """切换模型"""
    if not sd_client:
        return "❌ SD WebUI 未连接"
    
    try:
        success = sd_client.set_model(model_name)
        if success:
            return f"✅ 已切换到模型: {model_name}"
        else:
            return f"❌ 切换模型失败"
    except Exception as e:
        return f"❌ 错误: {str(e)}"

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

def create_ui() -> gr.Blocks:
    """创建 Gradio 界面"""

    # 初始化客户端
    initialize_clients()

    # 获取模型列表
    checkpoints, loras, samplers = load_models_list()

    # 风格和质量选项
    style_options = ["通用"] + (prompt_engineer.get_style_options() if prompt_engineer else [])
    quality_options = prompt_engineer.get_quality_options() if prompt_engineer else ["高质量"]
    composition_options = ["无"] + (prompt_engineer.get_composition_options() if prompt_engineer else [])

    # CSS 样式
    css = """
    .main-title {
        text-align: center;
        color: #D4AF37;
        margin-bottom: 20px;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        background: #f0f0f0;
    }
    .generate-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
    }
    """

    with gr.Blocks(css=css, title="藏族文化 AI 绘画助手", theme=gr.themes.Soft()) as app:
        
        # 标题
        gr.Markdown(
            """
            # 🏔️ 藏族文化 AI 绘画助手
            ### 基于 Stable Diffusion 的藏族文化主题图像生成系统
            """,
            elem_classes=["main-title"]
        )
        
        # 服务状态
        with gr.Row():
            status_text = gr.Textbox(
                label="服务状态",
                value=check_services(),
                interactive=False,lines=4,
                elem_classes=["status-box"]
            )
            refresh_btn = gr.Button("🔄 刷新状态", scale=0)
            refresh_btn.click(fn=check_services, outputs=status_text)
        
        # 主要标签页
        with gr.Tabs():
            
            # ==================== 文生图标签页 ====================
            with gr.TabItem("🎨 文生图", id="txt2img"):
                with gr.Row():
                    # 左侧：输入区域
                    with gr.Column(scale=1):
                        prompt_input = gr.Textbox(
                            label="描述 (支持中文)",
                            placeholder="例如：一位穿着传统藏袍的康巴姑娘，站在草原上，远处是雪山...",
                            lines=4
                        )
                        
                        with gr.Row():
                            style_dropdown = gr.Dropdown(
                                choices=style_options,
                                value="通用",
                                label="风格预设"
                            )
                            quality_dropdown = gr.Dropdown(
                                choices=quality_options,
                                value="高质量",
                                label="质量预设"
                            )
                        
                        with gr.Row():
                            composition_dropdown = gr.Dropdown(
                                choices=composition_options,
                                value="无",
                                label="构图预设"
                            )
                        
                        with gr.Accordion("🔧 高级设置", open=False):
                            with gr.Row():
                                width_slider = gr.Slider(
                                    minimum=256, maximum=1024, step=64,
                                    value=settings.default_width,
                                    label="宽度"
                                )
                                height_slider = gr.Slider(
                                    minimum=256, maximum=1024, step=64,
                                    value=settings.default_height,
                                    label="高度"
                                )
                            
                            with gr.Row():
                                steps_slider = gr.Slider(
                                    minimum=10, maximum=100, step=1,
                                    value=settings.default_steps,
                                    label="采样步数"
                                )
                                cfg_slider = gr.Slider(
                                    minimum=1, maximum=20, step=0.5,
                                    value=settings.default_cfg_scale,
                                    label="CFG Scale"
                                )
                            
                            seed_input = gr.Number(
                                value=-1,
                                label="种子 (-1 为随机)",
                                precision=0
                            )
                            
                            with gr.Row():
                                use_rag_checkbox = gr.Checkbox(
                                    value=True,
                                    label="使用知识库增强"
                                )
                                use_llm_checkbox = gr.Checkbox(
                                    value=True,
                                    label="使用 LLM 翻译扩写"
                                )
                            
                            negative_input = gr.Textbox(
                                label="自定义负面提示词",
                                placeholder="不想出现的元素...",
                                lines=2
                            )
                            
                            with gr.Row():
                                enable_hr = gr.Checkbox(
                                    value=False,
                                    label="高清修复"
                                )
                                hr_scale = gr.Slider(
                                    minimum=1.0, maximum=2.0, step=0.1,
                                    value=1.5,
                                    label="高清放大倍数"
                                )
                        
                        with gr.Row():
                            preview_btn = gr.Button("👁️ 预览提示词")
                            generate_btn = gr.Button(
                                "🚀 生成图片",
                                variant="primary",
                                elem_classes=["generate-btn"]
                            )
                    
                    # 右侧：输出区域
                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="生成结果",
                            type="pil",
                            interactive=False
                        )
                        
                        with gr.Accordion("📝 提示词详情", open=False):
                            final_prompt_display = gr.Textbox(
                                label="实际使用的正向提示词",
                                lines=4,
                                interactive=False
                            )
                            final_negative_display = gr.Textbox(
                                label="实际使用的负向提示词",
                                lines=3,
                                interactive=False
                            )
                        
                        generation_status = gr.Textbox(
                            label="状态",
                            interactive=False
                        )
                        
                        with gr.Row():
                            save_filename = gr.Textbox(
                                label="文件名",
                                placeholder="留空自动生成",
                                scale=2
                            )
                            save_btn = gr.Button("💾 保存", scale=1)
                            save_status = gr.Textbox(
                                label="保存状态",
                                interactive=False,
                                scale=2
                            )
                
                # 绑定事件
                preview_btn.click(
                    fn=preview_prompt,
                    inputs=[
                        prompt_input, style_dropdown, quality_dropdown,
                        composition_dropdown, use_rag_checkbox, use_llm_checkbox
                    ],
                    outputs=[final_prompt_display, final_negative_display]
                )
                
                generate_btn.click(
                    fn=generate_image,
                    inputs=[
                        prompt_input, style_dropdown, quality_dropdown,
                        composition_dropdown, width_slider, height_slider,
                        steps_slider, cfg_slider, seed_input,
                        use_rag_checkbox, use_llm_checkbox, negative_input,
                        enable_hr, hr_scale
                    ],
                    outputs=[output_image, final_prompt_display, generation_status]
                )
                
                save_btn.click(
                    fn=save_image,
                    inputs=[output_image, save_filename],
                    outputs=save_status
                )
            
            # ==================== 图生图标签页 ====================
            with gr.TabItem("🖼️ 图生图", id="img2img"):
                with gr.Row():
                    with gr.Column(scale=1):
                        init_image_input = gr.Image(
                            label="参考图片",
                            type="pil"
                        )
                        i2i_prompt_input = gr.Textbox(
                            label="描述 (支持中文)",
                            placeholder="描述你想要的变化...",
                            lines=3
                        )
                        
                        i2i_style_dropdown = gr.Dropdown(
                            choices=style_options,
                            value="通用",
                            label="风格预设"
                        )
                        
                        denoising_slider = gr.Slider(
                            minimum=0.1, maximum=1.0, step=0.05,
                            value=0.75,
                            label="重绘强度 (越高变化越大)"
                        )
                        
                        with gr.Accordion("🔧 高级设置", open=False):
                            i2i_steps_slider = gr.Slider(
                                minimum=10, maximum=100, step=1,
                                value=30,
                                label="采样步数"
                            )
                            i2i_cfg_slider = gr.Slider(
                                minimum=1, maximum=20, step=0.5,
                                value=7.0,
                                label="CFG Scale"
                            )
                            i2i_seed_input = gr.Number(
                                value=-1,
                                label="种子 (-1 为随机)",
                                precision=0
                            )
                            with gr.Row():
                                i2i_use_rag = gr.Checkbox(value=True, label="使用知识库")
                                i2i_use_llm = gr.Checkbox(value=True, label="使用 LLM")
                            i2i_negative_input = gr.Textbox(
                                label="自定义负面提示词",
                                lines=2
                            )
                        
                        i2i_generate_btn = gr.Button(
                            "🚀 生成图片",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=1):
                        i2i_output_image = gr.Image(
                            label="生成结果",
                            type="pil",
                            interactive=False
                        )
                        i2i_final_prompt = gr.Textbox(
                            label="实际使用的提示词",
                            lines=3,
                            interactive=False
                        )
                        i2i_status = gr.Textbox(
                            label="状态",
                            interactive=False
                        )
                
                # 绑定事件
                i2i_generate_btn.click(
                    fn=img2img_generate,
                    inputs=[
                        init_image_input, i2i_prompt_input, i2i_style_dropdown,
                        denoising_slider, i2i_steps_slider, i2i_cfg_slider,
                        i2i_seed_input, i2i_use_rag, i2i_use_llm, i2i_negative_input
                    ],
                    outputs=[i2i_output_image, i2i_final_prompt, i2i_status]
                )
            
            # ==================== 文字排版标签页 ====================
            with gr.TabItem("✍️ 文字排版", id="text"):
                with gr.Row():
                    with gr.Column(scale=1):
                        text_image_input = gr.Image(
                            label="底图 (可从文生图/图生图获取)",
                            type="pil"
                        )
                        
                        text_input = gr.Textbox(
                            label="文字内容 (支持藏文)",
                            placeholder="བཀྲ་ཤིས་བདེ་ལེགས། (扎西德勒)",
                            lines=2
                        )
                        
                        with gr.Row():
                            text_position = gr.Dropdown(
                                choices=["左上", "上中", "右上", "左中", "居中", "右中", "左下", "下中", "右下"],
                                value="下中",
                                label="位置"
                            )
                            text_font_size = gr.Slider(
                                minimum=12, maximum=120, step=2,
                                value=48,
                                label="字体大小"
                            )
                        with gr.Row():
                            text_color = gr.ColorPicker(
                                label="文字颜色",
                                value="#FFFFFF"
                            )
                            stroke_color = gr.ColorPicker(
                                label="描边颜色",
                                value="#000000"
                            )
                        
                        stroke_width = gr.Slider(
                            minimum=0, maximum=10, step=1,
                            value=2,
                            label="描边宽度"
                        )
                        
                        with gr.Row():
                            add_bg = gr.Checkbox(
                                value=False,
                                label="添加背景框"
                            )
                            bg_opacity = gr.Slider(
                                minimum=0.1, maximum=1.0, step=0.1,
                                value=0.5,
                                label="背景透明度"
                            )
                        
                        add_text_btn = gr.Button("✍️ 添加文字", variant="primary")
                    
                    with gr.Column(scale=1):
                        text_output_image = gr.Image(
                            label="结果",
                            type="pil",
                            interactive=False
                        )
                        text_status = gr.Textbox(
                            label="状态",
                            interactive=False
                        )
                        
                        with gr.Row():
                            text_save_filename = gr.Textbox(
                                label="文件名",
                                placeholder="留空自动生成",
                                scale=2
                            )
                            text_save_btn = gr.Button("💾 保存", scale=1)
                            text_save_status = gr.Textbox(
                                label="保存状态",
                                interactive=False,
                                scale=2
                            )
                
                # 绑定事件
                add_text_btn.click(
                    fn=add_text_to_image,
                    inputs=[
                        text_image_input, text_input, text_position,
                        text_font_size, text_color, stroke_width, stroke_color,
                        add_bg, bg_opacity
                    ],
                    outputs=[text_output_image, text_status]
                )
                
                text_save_btn.click(
                    fn=save_image,
                    inputs=[text_output_image, text_save_filename],
                    outputs=text_save_status
                )
            
            # ==================== 海报制作标签页 ====================
            with gr.TabItem("🎭 海报制作", id="poster"):
                with gr.Row():
                    with gr.Column(scale=1):
                        poster_bg_input = gr.Image(
                            label="背景图片",
                            type="pil"
                        )
                        
                        poster_title = gr.Textbox(
                            label="主标题 (支持藏文)",
                            placeholder="བོད་ཀྱི་རིག་གནས། (藏族文化)",
                            lines=2
                        )
                        
                        poster_subtitle = gr.Textbox(
                            label="副标题",
                            placeholder="可选的副标题...",
                            lines=1
                        )
                        
                        poster_footer = gr.Textbox(
                            label="底部文字",
                            placeholder="可选的底部说明...",
                            lines=1
                        )
                        
                        poster_title_size = gr.Slider(
                            minimum=24, maximum=120, step=4,
                            value=72,
                            label="标题字体大小"
                        )
                        
                        with gr.Row():
                            poster_add_gradient = gr.Checkbox(
                                value=True,
                                label="添加渐变遮罩"
                            )
                            poster_add_border = gr.Checkbox(
                                value=False,
                                label="添加装饰边框"
                            )
                        
                        poster_border_style = gr.Dropdown(
                            choices=["简约", "双线", "藏式"],
                            value="藏式",
                            label="边框样式"
                        )
                        
                        create_poster_btn = gr.Button(
                            "🎭 生成海报",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=1):
                        poster_output = gr.Image(
                            label="海报预览",
                            type="pil",
                            interactive=False
                        )
                        poster_status = gr.Textbox(
                            label="状态",
                            interactive=False
                        )
                        
                        with gr.Row():
                            poster_save_filename = gr.Textbox(
                                label="文件名",
                                placeholder="留空自动生成",
                                scale=2
                            )
                            poster_save_btn = gr.Button("💾 保存", scale=1)
                            poster_save_status = gr.Textbox(
                                label="保存状态",
                                interactive=False,
                                scale=2
                            )
                
                # 绑定事件
                create_poster_btn.click(
                    fn=create_poster,
                    inputs=[
                        poster_bg_input, poster_title, poster_subtitle,
                        poster_footer, poster_title_size, poster_add_gradient,
                        poster_add_border, poster_border_style
                    ],
                    outputs=[poster_output, poster_status]
                )
                
                poster_save_btn.click(
                    fn=save_image,
                    inputs=[poster_output, poster_save_filename],
                    outputs=poster_save_status
                )
            
            # ==================== 图片放大标签页 ====================
            with gr.TabItem("🔍 图片放大", id="upscale"):
                with gr.Row():
                    with gr.Column(scale=1):
                        upscale_input = gr.Image(
                            label="原图",
                            type="pil"
                        )
                        
                        upscale_scale = gr.Slider(
                            minimum=1.5, maximum=4.0, step=0.5,
                            value=2.0,
                            label="放大倍数"
                        )
                        
                        upscaler_dropdown = gr.Dropdown(
                            choices=[
                                "R-ESRGAN 4x+",
                                "R-ESRGAN 4x+ Anime6B",
                                "ESRGAN_4x",
                                "SwinIR_4x",
                                "Lanczos",
                                "Nearest"
                            ],
                            value="R-ESRGAN 4x+",
                            label="放大算法"
                        )
                        
                        with gr.Row():
                            face_restore = gr.Checkbox(
                                value=False,
                                label="面部修复"
                            )
                            face_restore_strength = gr.Slider(
                                minimum=0.1, maximum=1.0, step=0.1,
                                value=0.8,
                                label="修复强度"
                            )
                        
                        upscale_btn = gr.Button(
                            "🔍 放大图片",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=1):
                        upscale_output = gr.Image(
                            label="放大结果",
                            type="pil",
                            interactive=False
                        )
                        upscale_status = gr.Textbox(
                            label="状态",
                            interactive=False
                        )
                        
                        with gr.Row():
                            upscale_save_filename = gr.Textbox(
                                label="文件名",
                                placeholder="留空自动生成",
                                scale=2
                            )
                            upscale_save_btn = gr.Button("💾 保存", scale=1)
                            upscale_save_status = gr.Textbox(
                                label="保存状态",
                                interactive=False,
                                scale=2
                            )
                
                # 绑定事件
                upscale_btn.click(
                    fn=upscale_image,
                    inputs=[
                        upscale_input, upscale_scale, upscaler_dropdown,
                        face_restore, face_restore_strength
                    ],
                    outputs=[upscale_output, upscale_status]
                )
                upscale_save_btn.click(
                    fn=save_image,
                    inputs=[upscale_output, upscale_save_filename],
                    outputs=upscale_save_status
                )
            
            # ==================== 模型管理标签页 ====================
            with gr.TabItem("⚙️ 模型管理", id="models"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 模型切换")
                        
                        model_dropdown = gr.Dropdown(
                            choices=checkpoints if checkpoints else ["未检测到模型"],
                            label="Checkpoint 模型",
                            value=checkpoints[0] if checkpoints else None
                        )
                        
                        change_model_btn = gr.Button("🔄 切换模型")
                        model_status = gr.Textbox(
                            label="状态",
                            interactive=False
                        )
                        
                        gr.Markdown("### 可用 LoRA")
                        lora_list = gr.Dataframe(
                            headers=["LoRA 名称"],
                            value=[[lora] for lora in loras] if loras else [["未检测到 LoRA"]],
                            interactive=False
                        )

                        refresh_models_btn = gr.Button("🔄 刷新模型列表")
                    
                    with gr.Column():
                        gr.Markdown("### 采样器列表")
                        sampler_list = gr.Dataframe(
                            headers=["采样器名称"],
                            value=[[s] for s in samplers] if samplers else [["未检测到采样器"]],
                            interactive=False
                        )
                        
                        gr.Markdown("### 推荐配置")
                        gr.Markdown("""
                        **人像写实：**
                        - Checkpoint: ChilloutMix / Realistic Vision
                        - 采样器: DPM++ 2M Karras
                        - Steps: 30-40
                        - CFG: 7-8
                        
                        **唐卡风格：**
                        - Checkpoint: Deliberate / DreamShaper
                        - LoRA: thangka_style (权重 0.7-0.8)
                        - 采样器: Euler a
                        - Steps: 25-35
                        - CFG: 7-9
                        
                        **风景：**
                        - Checkpoint: Deliberate / Realistic Vision
                        - 采样器: DPM++ SDE Karras
                        - Steps: 30-50
                        - CFG: 7-10
                        """)
                # 绑定事件
                change_model_btn.click(
                    fn=change_model,
                    inputs=[model_dropdown],
                    outputs=[model_status]
                )
                
                def refresh_models():
                    ckpts, lrs, smps = load_models_list()
                    return (
                        gr.Dropdown(choices=ckpts if ckpts else ["未检测到模型"]),
                        [[lora] for lora in lrs] if lrs else [["未检测到 LoRA"]],
                        [[s] for s in smps] if smps else [["未检测到采样器"]]
                    )
                
                refresh_models_btn.click(
                    fn=refresh_models,
                    outputs=[model_dropdown, lora_list, sampler_list]
                )
            # ==================== 知识库管理标签页 ====================
            with gr.TabItem("📚 知识库", id="knowledge"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 知识库状态")
                        
                        def get_rag_stats():
                            if rag_engine:
                                stats = rag_engine.get_stats()
                                return f"""
                                **后端**: {stats['backend']}
                                **集合名称**: {stats['collection_name']}
                                **嵌入模型**: {stats['embedding_model']}
                                **文档数量**: {stats['document_count']}
                                """
                            return "RAG 引擎未初始化"
                        
                        rag_stats_display = gr.Markdown(get_rag_stats())
                        refresh_rag_btn = gr.Button("🔄 刷新状态")
                        
                        gr.Markdown("### 添加文档")
                        doc_content = gr.Textbox(
                            label="文档内容",
                            placeholder="输入要添加到知识库的文本...",
                            lines=5
                        )
                        doc_source = gr.Textbox(
                            label="来源标注",
                            placeholder="例如：藏族服饰介绍"
                        )
                        
                        add_doc_btn = gr.Button("➕ 添加到知识库")
                        add_doc_status = gr.Textbox(
                            label="状态",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 知识检索测试")
                        
                        query_input = gr.Textbox(
                            label="查询内容",
                            placeholder="输入要查询的关键词或问题..."
                        )
                        
                        query_btn = gr.Button("🔍 检索")
                        
                        query_results = gr.Textbox(
                            label="检索结果",
                            lines=10,
                            interactive=False
                        )
                        
                        gr.Markdown("### 导入/导出")
                        
                        with gr.Row():
                            export_btn = gr.Button("📤 导出知识库")
                            export_status = gr.Textbox(
                                label="导出状态",
                                interactive=False
                            )
                        
                        import_file = gr.File(
                            label="导入知识库文件 (JSON)",
                            file_types=[".json"]
                        )
                        import_btn = gr.Button("📥 导入")
                        import_status = gr.Textbox(
                            label="导入状态",
                            interactive=False
                        )
                
                # 知识库相关函数
                def add_document(content, source):
                    if not content:
                        return "❌ 请输入文档内容"
                    if not rag_engine:
                        return "❌ RAG 引擎未初始化"
                    
                    try:
                        metadata = {"source": source} if source else {}
                        doc_ids = rag_engine.add_document(content, metadata)
                        return f"✅ 成功添加 {len(doc_ids)} 个文档块"
                    except Exception as e:
                        return f"❌ 添加失败: {str(e)}"
                
                def query_knowledge(query):
                    if not query:
                        return "请输入查询内容"
                    if not rag_engine:
                        return "RAG 引擎未初始化"
                    
                    try:
                        results = rag_engine.query(query, top_k=3)
                        if not results:
                            return "未找到相关内容"
                        output = []
                        for i, r in enumerate(results, 1):
                            output.append(f"【结果 {i}】相似度: {r['score']:.3f}")
                            output.append(f"来源: {r['metadata'].get('source', '未知')}")
                            output.append(f"内容: {r['content'][:200]}...")
                            output.append("-" * 50)
                        
                        return "\n".join(output)
                    except Exception as e:
                        return f"查询失败: {str(e)}"
                
                def export_knowledge():
                    if not rag_engine:
                        return "❌ RAG 引擎未初始化"
                    
                    try:
                        output_path = settings.output_dir / "knowledge_export.json"
                        success = rag_engine.export_knowledge_base(output_path)
                        if success:
                            return f"✅ 已导出到: {output_path}"
                        return "❌ 导出失败"
                    except Exception as e:
                        return f"❌ 导出失败: {str(e)}"
                
                def import_knowledge(file):
                    if not file:
                        return "❌ 请选择文件"
                    if not rag_engine:
                        return "❌ RAG 引擎未初始化"
                    
                    try:
                        success = rag_engine.import_knowledge_base(Path(file.name))
                        if success:
                            return "✅ 导入成功"
                        return "❌ 导入失败"
                    except Exception as e:
                        return f"❌ 导入失败: {str(e)}"
                
                # 绑定事件
                refresh_rag_btn.click(
                    fn=lambda: get_rag_stats(),
                    outputs=[rag_stats_display]
                )
                
                add_doc_btn.click(
                    fn=add_document,
                    inputs=[doc_content, doc_source],
                    outputs=[add_doc_status]
                )
                
                query_btn.click(
                    fn=query_knowledge,
                    inputs=[query_input],
                    outputs=[query_results]
                )
                
                export_btn.click(
                    fn=export_knowledge,
                    outputs=[export_status]
                )
                
                import_btn.click(
                    fn=import_knowledge,
                    inputs=[import_file],
                    outputs=[import_status]
                )
            
            # ==================== 帮助标签页 ====================
            with gr.TabItem("❓ 帮助", id="help"):
                gr.Markdown("""
                # 📖 使用指南
                
                ## 🎨 文生图
                
                1. **输入描述**：用中文描述你想要生成的图片，系统会自动翻译并优化
                2. **选择风格**：选择预设风格可以自动添加相关的提示词
                3. **调整参数**：
                   - **宽度/高度**：图片尺寸，建议使用 512x512 或 768x768
                   - **采样步数**：越高质量越好，但速度越慢，推荐 25-40
                   - **CFG Scale**：越高越接近提示词，推荐 7-9
                   - **种子**：-1 为随机，固定种子可复现结果
                
                ## 🖼️ 图生图
                
                1. 上传参考图片
                2. 描述想要的变化
                3. 调整**重绘强度**：
                   - 0.3-0.5：轻微变化，保留原图大部分内容
                   - 0.5-0.7：中等变化
                   - 0.7-0.9：大幅变化，只保留构图
                
                ## ✍️ 文字排版
                
                - 支持中文、藏文、英文
                - 藏文示例：བཀྲ་ཤིས་བདེ་ལེགས། (扎西德勒)
                - 建议使用描边增加可读性
                
                ## 🎭 海报制作
                
                1. 先用文生图生成背景
                2. 添加标题和副标题
                3. 选择是否添加渐变遮罩和装饰边框
                
                ## 🔍 图片放大
                
                - **R-ESRGAN 4x+**：通用放大，效果最好
                - **R-ESRGAN 4x+ Anime6B**：适合动漫风格
                - **面部修复**：人像照片建议开启
                
                ---
                
                ## 🏔️ 藏族文化关键词参考
                
                ### 人物
                - 格萨尔王、康巴汉子、藏族姑娘、喇嘛、活佛
                
                ### 服饰
                - 藏袍、氆氇、邦典、英雄结、哈达
                
                ### 建筑
                - 布达拉宫、大昭寺、白塔、玛尼堆
                
                ### 宗教
                - 唐卡、转经筒、经幡、酥油灯、金刚杵
                
                ### 自然
                - 雪山、草原、牦牛、青稞、纳木错
                
                ---
                
                ## ⚠️ 常见问题
                
                **Q: 生成速度很慢？**
                A: 检查 GPU 是否正常工作，降低图片尺寸和采样步数
                
                **Q: 生成的图片质量不好？**
                A: 尝试增加采样步数，调整 CFG Scale，使用更详细的描述
                
                **Q: 藏文显示不正常？**
                A: 确保已安装藏文字体（如 Noto Sans Tibetan）
                
                **Q: 连接失败？**
                A: 检查 SD WebUI 和 Ollama 服务是否正常运行
                """)
        
        # 页脚
        gr.Markdown(
            """
            ---
            <center>
            🏔️ 藏族文化 AI 绘画助手 | 基于 Stable Diffusion + Ollama + RAG
            </center>
            """,
            elem_classes=["footer"]
        )
    
    return app

def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建并启动应用
    app = create_ui()

    # 启用队列
    app.queue()

    print("\n" + "="*60)
    print("正在启动藏族文化 AI 绘画助手...")
    print("="*60 + "\n")

    # 使用更简单的配置，不进行启动检查
    app.launch(
        server_name="0.0.0.0",
        server_port=settings.gradio_port,
        share=False,
        inbrowser=False,  # 不自动打开浏览器
        show_api=False,
        prevent_thread_lock=False
    )

    print(f"\n应用已启动！请访问: http://localhost:{settings.gradio_port}\n")

if __name__ == "__main__":
    main()