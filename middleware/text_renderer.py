"""
藏文排版渲染引擎
负责：在图片上添加藏文文字、水印、海报排版
"""

import io
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging
import platform

logger = logging.getLogger(__name__)

class TextPosition(Enum):
    """文字位置枚举"""
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"

@dataclass
class TextStyle:
    """文字样式配置"""
    font_size: int = 48
    color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: int = 2
    opacity: float = 1.0
    shadow: bool = True
    shadow_offset: Tuple[int, int] = (3, 3)
    shadow_color: str = "#000000"
    shadow_blur: int = 5
    line_spacing: float = 1.5
    letter_spacing: int = 0

@dataclass
class TextBox:
    """文字框配置"""
    position: TextPosition = TextPosition.BOTTOM_CENTER
    margin: int = 50
    padding: int = 20
    background: bool = False
    background_color: str = "#000000"
    background_opacity: float = 0.5
    border_radius: int = 10
    max_width_ratio: float = 0.8  # 最大宽度占图片宽度的比例

class TibetanTextRenderer:
    """藏文文字渲染器"""
    
    # 常用藏文字体列表
    TIBETAN_FONTS = [
        "Microsoft Himalaya",
        "Noto Sans Tibetan",
        "Jomolhari",
        "DDC Uchen",
        "Tibetan Machine Uni",
        "Yagpo",
        "Himalaya",
    ]
    
    # 系统字体路径
    FONT_PATHS = {
        "Windows": [
            "C:/Windows/Fonts",
            Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
        ],
        "Darwin": [  # macOS
            "/System/Library/Fonts",
            "/Library/Fonts",
            Path.home() / "Library/Fonts",
        ],
        "Linux": [
            "/usr/share/fonts",
            "/usr/local/share/fonts",Path.home() / ".fonts",
            Path.home() / ".local/share/fonts",
        ],
    }
    
    def __init__(self, fonts_dir: Optional[Path] = None):
        """
        初始化渲染器
        
        Args:
            fonts_dir: 自定义字体目录
        """
        self.fonts_dir = fonts_dir
        self.font_cache: Dict[str, ImageFont.FreeTypeFont] = {}
        self.available_fonts = self._scan_fonts()
        
        if not self.available_fonts:
            logger.warning("未找到藏文字体，文字渲染可能显示为方块")
    
    def _scan_fonts(self) -> List[str]:
        """扫描可用的藏文字体"""
        available = []
        system = platform.system()

        # 优先从 tests/fonts 目录加载所有字体（不过滤）
        tests_fonts_dir = Path(__file__).parent.parent / "tests" / "fonts"
        if tests_fonts_dir.exists():
            logger.info(f"扫描字体目录: {tests_fonts_dir}")
            for font_file in tests_fonts_dir.glob("*.ttf"):
                available.append(str(font_file))
                logger.info(f"找到字体: {font_file.stem}")
            for font_file in tests_fonts_dir.glob("*.otf"):
                available.append(str(font_file))
                logger.info(f"找到字体: {font_file.stem}")

        # 添加自定义字体目录（如果指定）
        if self.fonts_dir and self.fonts_dir.exists() and self.fonts_dir != tests_fonts_dir:
            for font_file in self.fonts_dir.glob("*.ttf"):
                available.append(str(font_file))
                logger.info(f"找到字体: {font_file.stem}")
            for font_file in self.fonts_dir.glob("*.otf"):
                available.append(str(font_file))
                logger.info(f"找到字体: {font_file.stem}")

        # 最后扫描系统字体路径（仅限包含藏文关键词的字体）
        search_paths = self.FONT_PATHS.get(system, [])
        for font_path in search_paths:
            font_path = Path(font_path)
            if not font_path.exists():
                continue

            for font_file in font_path.rglob("*.ttf"):
                font_name = font_file.stem
                if any(tf.lower() in font_name.lower() for tf in self.TIBETAN_FONTS):
                    available.append(str(font_file))
                    logger.info(f"找到系统藏文字体: {font_name}")

            for font_file in font_path.rglob("*.otf"):
                font_name = font_file.stem
                if any(tf.lower() in font_name.lower() for tf in self.TIBETAN_FONTS):
                    available.append(str(font_file))
                    logger.info(f"找到系统藏文字体: {font_name}")

        return available
    
    def _get_font(
        self, 
        font_path: Optional[str] = None, 
        size: int = 48
    ) -> ImageFont.FreeTypeFont:
        """获取字体对象"""
        cache_key = f"{font_path}_{size}"
        
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        font = None
        
        # 尝试使用指定字体
        if font_path:
            try:
                font = ImageFont.truetype(font_path, size)
                self.font_cache[cache_key] = font
                return font
            except Exception as e:
                logger.warning(f"加载字体失败 {font_path}: {e}")
        
        # 尝试使用扫描到的藏文字体
        for available_font in self.available_fonts:
            try:
                font = ImageFont.truetype(available_font, size)
                self.font_cache[cache_key] = font
                logger.info(f"使用字体: {available_font}")
                return font
            except Exception:
                continue
        
        # 尝试系统默认藏文字体
        for font_name in self.TIBETAN_FONTS:
            try:
                font = ImageFont.truetype(font_name, size)
                self.font_cache[cache_key] = font
                return font
            except Exception:
                continue
        
        # 最后使用默认字体
        logger.warning("未找到藏文字体，使用默认字体")
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except Exception:
            font = ImageFont.load_default()
        
        self.font_cache[cache_key] = font
        return font
    
    def _hex_to_rgba(self, hex_color: str, opacity: float = 1.0) -> Tuple[int, int, int, int]:
        """将十六进制颜色转换为 RGBA"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = int(opacity * 255)
        return (r, g, b, a)
    
    def _calculate_text_position(
        self,
        image_size: Tuple[int, int],
        text_size: Tuple[int, int],
        position: TextPosition,
        margin: int
    ) -> Tuple[int, int]:
        """计算文字位置"""
        img_width, img_height = image_size
        text_width, text_height = text_size
        
        positions = {
            TextPosition.TOP_LEFT: (margin, margin),
            TextPosition.TOP_CENTER: ((img_width - text_width) // 2, margin),
            TextPosition.TOP_RIGHT: (img_width - text_width - margin, margin),
            TextPosition.CENTER_LEFT: (margin, (img_height - text_height) // 2),
            TextPosition.CENTER: ((img_width - text_width) // 2, (img_height - text_height) // 2),
            TextPosition.CENTER_RIGHT: (img_width - text_width - margin, (img_height - text_height) // 2),
            TextPosition.BOTTOM_LEFT: (margin, img_height - text_height - margin),
            TextPosition.BOTTOM_CENTER: ((img_width - text_width) // 2, img_height - text_height - margin),
            TextPosition.BOTTOM_RIGHT: (img_width - text_width - margin, img_height - text_height - margin),
        }
        
        return positions.get(position, positions[TextPosition.BOTTOM_CENTER])
    
    def _wrap_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int
    ) -> List[str]:
        """文字自动换行"""
        lines = []
        current_line = ""
        
        for char in text:
            test_line = current_line + char
            bbox = font.getbbox(test_line)
            line_width = bbox[2] - bbox[0]
            
            if line_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _get_text_dimensions(
        self,
        lines: List[str],
        font: ImageFont.FreeTypeFont,
        line_spacing: float
    ) -> Tuple[int, int]:
        """计算多行文字的总尺寸"""
        if not lines:
            return (0, 0)
        
        max_width = 0
        total_height = 0
        
        for i, line in enumerate(lines):
            bbox = font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
            
            max_width = max(max_width, line_width)
            total_height += line_height
            
            if i < len(lines) - 1:
                total_height += int(line_height * (line_spacing - 1))
        
        return (max_width, total_height)
    
    def add_text(
        self,
        image: Image.Image,
        text: str,
        style: Optional[TextStyle] = None,
        text_box: Optional[TextBox] = None,
        font_path: Optional[str] = None,
        custom_position: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        在图片上添加文字
        
        Args:
            image: 输入图片
            text: 要添加的文字
            style: 文字样式
            text_box: 文字框配置
            font_path: 自定义字体路径
            custom_position: 自定义位置 (x, y)
            
        Returns:
            添加文字后的图片
        """
        if not text:
            return image
        
        # 使用默认配置
        if style is None:
            style = TextStyle()
        if text_box is None:
            text_box = TextBox()
        
        # 转换为 RGBA 模式
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # 创建透明图层
        txt_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)
        
        # 获取字体
        font = self._get_font(font_path, style.font_size)
        
        # 计算最大宽度并换行
        max_width = int(image.width * text_box.max_width_ratio)
        lines = self._wrap_text(text, font, max_width)
        
        # 计算文字尺寸
        text_width, text_height = self._get_text_dimensions(
            lines, font, style.line_spacing
        )
        
        # 计算位置
        if custom_position:
            x, y = custom_position
        else:
            x, y = self._calculate_text_position(
                image.size,
                (text_width + text_box.padding * 2, text_height + text_box.padding * 2),
                text_box.position,
                text_box.margin
            )
        
        # 绘制背景框
        if text_box.background:
            bg_color = self._hex_to_rgba(
                text_box.background_color, 
                text_box.background_opacity
            )
            bg_rect = [
                x - text_box.padding,
                y - text_box.padding,
                x + text_width + text_box.padding,
                y + text_height + text_box.padding
            ]
            draw.rounded_rectangle(
                bg_rect,
                radius=text_box.border_radius,
                fill=bg_color
            )
        
        # 绘制文字
        current_y = y
        text_color = self._hex_to_rgba(style.color, style.opacity)
        stroke_color = self._hex_to_rgba(style.stroke_color, style.opacity)
        
        for line in lines:
            bbox = font.getbbox(line)
            line_height = bbox[3] - bbox[1]
            
            # 绘制阴影
            if style.shadow:
                shadow_color = self._hex_to_rgba(style.shadow_color, 0.5)
                shadow_x = x + style.shadow_offset[0]
                shadow_y = current_y + style.shadow_offset[1]
                draw.text(
                    (shadow_x, shadow_y),
                    line,
                    font=font,
                    fill=shadow_color
                )
            
            # 绘制描边
            if style.stroke_width > 0:
                draw.text(
                    (x, current_y),
                    line,
                    font=font,
                    fill=text_color,
                    stroke_width=style.stroke_width,
                    stroke_fill=stroke_color
                )
            else:
                draw.text(
                    (x, current_y),
                    line,
                    font=font,
                    fill=text_color
                )
            
            current_y += int(line_height * style.line_spacing)
        
        # 合并图层
        result = Image.alpha_composite(image, txt_layer)
        
        return result
    
    def add_watermark(
        self,
        image: Image.Image,
        text: str,
        opacity: float = 0.3,
        font_size: int = 24,
        position: TextPosition = TextPosition.BOTTOM_RIGHT,
        font_path: Optional[str] = None
    ) -> Image.Image:
        """
        添加水印
        
        Args:
            image: 输入图片
            text: 水印文字
            opacity: 透明度
            font_size: 字体大小
            position: 位置
            font_path: 字体路径
            
        Returns:
            添加水印后的图片
        """
        style = TextStyle(
            font_size=font_size,
            color="#FFFFFF",
            opacity=opacity,
            stroke_width=1,
            stroke_color="#000000",
            shadow=False
        )
        
        text_box = TextBox(
            position=position,
            margin=20,
            background=False
        )
        
        return self.add_text(image, text, style, text_box, font_path)
    
    def create_poster(
        self,
        background: Image.Image,
        title: str,
        subtitle: Optional[str] = None,
        footer: Optional[str] = None,
        title_style: Optional[TextStyle] = None,
        subtitle_style: Optional[TextStyle] = None,
        footer_style: Optional[TextStyle] = None,
        font_path: Optional[str] = None,
        add_gradient_overlay: bool = True
    ) -> Image.Image:
        """
        创建海报
        
        Args:
            background: 背景图片
            title: 主标题（藏文）
            subtitle: 副标题
            footer: 底部文字
            title_style: 标题样式
            subtitle_style: 副标题样式
            footer_style: 底部文字样式
            font_path: 字体路径
            add_gradient_overlay: 是否添加渐变遮罩
            
        Returns:
            海报图片
        """
        # 转换为 RGBA
        if background.mode != 'RGBA':
            background = background.convert('RGBA')
        
        result = background.copy()
        
        # 添加渐变遮罩（增强文字可读性）
        if add_gradient_overlay:
            result = self._add_gradient_overlay(result)
        
        # 默认样式
        if title_style is None:
            title_style = TextStyle(
                font_size=72,
                color="#FFFFFF",
                stroke_width=3,
                stroke_color="#000000",
                shadow=True,
                shadow_offset=(4, 4)
            )
        
        if subtitle_style is None:
            subtitle_style = TextStyle(
                font_size=36,
                color="#F0F0F0",
                stroke_width=1,
                stroke_color="#333333",
                shadow=True
            )
        
        if footer_style is None:
            footer_style = TextStyle(
                font_size=24,
                color="#CCCCCC",
                stroke_width=0,
                opacity=0.8,
                shadow=False
            )
        
        # 添加标题（居中偏上）
        title_box = TextBox(
            position=TextPosition.CENTER,
            margin=50,
            max_width_ratio=0.85
        )
        result = self.add_text(result, title, title_style, title_box, font_path)
        
        # 添加副标题（标题下方）
        if subtitle:
            # 计算副标题位置
            subtitle_y = int(result.height * 0.6)
            result = self.add_text(
                result, 
                subtitle, 
                subtitle_style,
                custom_position=(result.width // 2 - len(subtitle) * subtitle_style.font_size // 4, subtitle_y),
                font_path=font_path
            )
        
        # 添加底部文字
        if footer:
            footer_box = TextBox(
                position=TextPosition.BOTTOM_CENTER,
                margin=30
            )
            result = self.add_text(result, footer, footer_style, footer_box, font_path)
        
        return result
    
    def _add_gradient_overlay(
        self,
        image: Image.Image,
        direction: str = "bottom",
        color: str = "#000000",
        start_opacity: float = 0.0,
        end_opacity: float = 0.6
    ) -> Image.Image:
        """
        添加渐变遮罩
        
        Args:
            image: 输入图片
            direction: 渐变方向 (top, bottom, left, right)
            color: 渐变颜色
            start_opacity: 起始透明度
            end_opacity: 结束透明度
            
        Returns:
            添加渐变后的图片
        """
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # 创建渐变图层
        gradient = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(gradient)
        
        r, g, b = self._hex_to_rgba(color)[:3]
        
        if direction == "bottom":
            for y in range(image.height):
                progress = y / image.height
                alpha = int((start_opacity + (end_opacity - start_opacity) * progress) * 255)
                draw.line([(0, y), (image.width, y)], fill=(r, g, b, alpha))
        
        elif direction == "top":
            for y in range(image.height):
                progress = 1 - (y / image.height)
                alpha = int((start_opacity + (end_opacity - start_opacity) * progress) * 255)
                draw.line([(0, y), (image.width, y)], fill=(r, g, b, alpha))
        
        elif direction == "left":
            for x in range(image.width):
                progress = 1 - (x / image.width)
                alpha = int((start_opacity + (end_opacity - start_opacity) * progress) * 255)
                draw.line([(x, 0), (x, image.height)], fill=(r, g, b, alpha))
        
        elif direction == "right":
            for x in range(image.width):
                progress = x / image.width
                alpha = int((start_opacity + (end_opacity - start_opacity) * progress) * 255)
                draw.line([(x, 0), (x, image.height)], fill=(r, g, b, alpha))
        
        return Image.alpha_composite(image, gradient)
    
    def add_decorative_border(
        self,
        image: Image.Image,
        border_width: int = 20,
        border_color: str = "#D4AF37",  # 金色
        style: str = "solid",  # solid, double, tibetan
        inner_padding: int = 10
    ) -> Image.Image:
        """
        添加装饰边框
        
        Args:
            image: 输入图片
            border_width: 边框宽度
            border_color: 边框颜色
            style: 边框样式
            inner_padding: 内边距
            
        Returns:
            添加边框后的图片
        """
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # 创建新画布
        new_width = image.width + (border_width + inner_padding) * 2
        new_height = image.height + (border_width + inner_padding) * 2
        
        result = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(result)
        
        color = self._hex_to_rgba(border_color)
        
        if style == "solid":
            # 外边框
            draw.rectangle(
                [0, 0, new_width - 1, new_height - 1],
                outline=color,
                width=border_width
            )
        
        elif style == "double":
            # 双线边框
            draw.rectangle(
                [0, 0, new_width - 1, new_height - 1],
                outline=color,
                width=border_width // 2
            )
            inner_offset = border_width // 2 + 5
            draw.rectangle(
                [inner_offset, inner_offset, new_width - 1 - inner_offset, new_height - 1 - inner_offset],
                outline=color,
                width=2
            )
        
        elif style == "tibetan":
            # 藏式装饰边框
            draw.rectangle(
                [0, 0, new_width - 1, new_height - 1],
                outline=color,
                width=border_width
            )
            # 添加角落装饰
            corner_size = border_width * 2
            # 四个角落的装饰图案
            for corner in [(0, 0), (new_width - corner_size, 0), 
                          (0, new_height - corner_size), 
                          (new_width - corner_size, new_height - corner_size)]:
                draw.rectangle(
                    [corner[0], corner[1], 
                     corner[0] + corner_size, corner[1] + corner_size],
                    fill=color
                )
                # 内部小方块
                inner_margin = corner_size // 4
                draw.rectangle(
                    [corner[0] + inner_margin, corner[1] + inner_margin,
                     corner[0] + corner_size - inner_margin, 
                     corner[1] + corner_size - inner_margin],
                    fill=self._hex_to_rgba("#FFFFFF", 0.3)
                )
        
        # 将原图粘贴到中心
        paste_x = border_width + inner_padding
        paste_y = border_width + inner_padding
        result.paste(image, (paste_x, paste_y), image)
        
        return result
    
    def apply_tibetan_style_filter(
        self,
        image: Image.Image,
        style: str = "thangka"
    ) -> Image.Image:
        """
        应用藏式风格滤镜
        
        Args:
            image: 输入图片
            style: 风格类型 (thangka, vintage, sacred)
            
        Returns:
            处理后的图片
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if style == "thangka":
            # 唐卡风格：增强金色调，提高对比度
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.3)
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # 添加暖色调
            r, g, b = image.split()
            r = r.point(lambda x: min(255, int(x * 1.1)))
            image = Image.merge('RGB', (r, g, b))
        
        elif style == "vintage":
            # 复古风格：降低饱和度，添加褐色调
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.7)
            
            # 添加褐色调
            r, g, b = image.split()
            r = r.point(lambda x: min(255, int(x * 1.1)))
            b = b.point(lambda x: int(x * 0.9))
            image = Image.merge('RGB', (r, g, b))
        
        elif style == "sacred":
            # 神圣风格：柔和光晕效果
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            # 添加柔和效果
            image = image.filter(ImageFilter.SMOOTH)
        
        return image
    
    def get_available_fonts(self) -> List[str]:
        """获取可用的字体列表"""
        return self.available_fonts.copy()
    
    def preview_text(
        self,
        text: str,
        font_path: Optional[str] = None,
        font_size: int = 48,
        width: int = 800,
        height: int = 200
    ) -> Image.Image:
        """
        预览文字效果
        
        Args:
            text: 预览文字
            font_path: 字体路径
            font_size: 字体大小
            width: 预览图宽度
            height: 预览图高度
            
        Returns:
            预览图片
        """
        # 创建预览背景
        preview = Image.new('RGBA', (width, height), (50, 50, 50, 255))
        
        style = TextStyle(
            font_size=font_size,
            color="#FFFFFF",
            stroke_width=2,
            stroke_color="#000000"
        )
        
        text_box = TextBox(
            position=TextPosition.CENTER,
            margin=20
        )
        
        return self.add_text(preview, text, style, text_box, font_path)

# 便捷函数
def create_text_renderer(fonts_dir: Optional[Path] = None) -> TibetanTextRenderer:
    """创建文字渲染器实例"""
    return TibetanTextRenderer(fonts_dir=fonts_dir)
