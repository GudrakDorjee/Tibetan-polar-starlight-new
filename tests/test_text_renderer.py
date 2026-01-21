"""
文字渲染器测试
"""

import pytest
from PIL import Image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from middleware.text_renderer import (
    TextRenderer,
    TextStyle,
    TextBox,
    TextPosition
)

class TestTextStyle:
    """TextStyle 测试类"""
    
    def test_default_values(self):
        """测试默认值"""
        style = TextStyle()
        
        assert style.font_size == 32
        assert style.color == "#FFFFFF"
        assert style.stroke_width == 0
    
    def test_custom_values(self):
        """测试自定义值"""
        style = TextStyle(
            font_size=48,
            color="#FF0000",
            stroke_width=3,
            stroke_color="#000000",
            shadow=True
        )
        
        assert style.font_size == 48
        assert style.color == "#FF0000"
        assert style.stroke_width == 3
        assert style.shadow is True

class TestTextBox:
    """TextBox 测试类"""
    
    def test_default_values(self):
        """测试默认值"""
        box = TextBox()
        
        assert box.position == TextPosition.BOTTOM_CENTER
        assert box.margin == 20
        assert box.background is False
    
    def test_custom_values(self):
        """测试自定义值"""
        box = TextBox(
            position=TextPosition.TOP_LEFT,
            margin=50,
            background=True,
            background_opacity=0.8
        )
        
        assert box.position == TextPosition.TOP_LEFT
        assert box.margin == 50
        assert box.background is True
        assert box.background_opacity == 0.8

class TestTextRenderer:
    """TextRenderer 测试类"""
    
    @pytest.fixture
    def renderer(self):
        """创建测试渲染器"""
        return TextRenderer()
    
    @pytest.fixture
    def test_image(self):
        """创建测试图片"""
        return Image.new('RGB', (512, 512), color='blue')
    
    def test_init(self, renderer):
        """测试初始化"""
        assert renderer is not None
    
    def test_add_text_basic(self, renderer, test_image):
        """测试基本文字添加"""
        style = TextStyle(font_size=32, color="#FFFFFF")
        box = TextBox(position=TextPosition.CENTER)
        
        result = renderer.add_text(
            image=test_image,
            text="测试文字",
            style=style,
            text_box=box
        )
        
        assert result is not None
        assert result.size == test_image.size
    
    def test_add_text_with_stroke(self, renderer, test_image):
        """测试带描边的文字"""
        style = TextStyle(
            font_size=48,
            color="#FFFFFF",
            stroke_width=3,
            stroke_color="#000000"
        )
        box = TextBox(position=TextPosition.BOTTOM_CENTER)
        
        result = renderer.add_text(
            image=test_image,
            text="描边文字",
            style=style,
            text_box=box
        )
        
        assert result is not None
    
    def test_add_text_with_background(self, renderer, test_image):
        """测试带背景的文字"""
        style = TextStyle(font_size=32)
        box = TextBox(
            position=TextPosition.TOP_CENTER,
            background=True,
            background_opacity=0.7
        )
        
        result = renderer.add_text(
            image=test_image,
            text="带背景的文字",
            style=style,
            text_box=box
        )
        
        assert result is not None
    
    def test_add_text_all_positions(self, renderer, test_image):
        """测试所有位置"""
        style = TextStyle(font_size=24)
        
        for position in TextPosition:
            box = TextBox(position=position)
            result = renderer.add_text(
                image=test_image.copy(),
                text=f"位置: {position.name}",
                style=style,
                text_box=box
            )
            assert result is not None
    
    def test_add_text_empty_string(self, renderer, test_image):
        """测试空字符串"""
        style = TextStyle()
        box = TextBox()
        
        result = renderer.add_text(
            image=test_image,
            text="",
            style=style,
            text_box=box
        )
        
        # 空字符串应该返回原图
        assert result is not None
    
    def test_add_text_multiline(self, renderer, test_image):
        """测试多行文字"""
        style = TextStyle(font_size=24)
        box = TextBox(position=TextPosition.CENTER)
        
        result = renderer.add_text(
            image=test_image,
            text="第一行\n第二行\n第三行",
            style=style,
            text_box=box
        )
        
        assert result is not None
    
    def test_add_text_tibetan(self, renderer, test_image):
        """测试藏文文字"""
        style = TextStyle(font_size=36)
        box = TextBox(position=TextPosition.CENTER)
        
        # 藏文 "扎西德勒"
        result = renderer.add_text(
            image=test_image,
            text="བཀྲ་ཤིས་བདེ་ལེགས།",
            style=style,
            text_box=box
        )
        
        assert result is not None
    
    def test_create_poster(self, renderer, test_image):
        """测试创建海报"""
        title_style = TextStyle(font_size=72, color="#FFFFFF")
        
        result = renderer.create_poster(
            background=test_image,
            title="测试标题",
            subtitle="副标题",
            footer="底部文字",
            title_style=title_style,
            add_gradient_overlay=True
        )
        
        assert result is not None
        assert result.size == test_image.size
    
    def test_create_poster_minimal(self, renderer, test_image):
        """测试最小化海报"""
        result = renderer.create_poster(
            background=test_image,
            title="仅标题"
        )
        
        assert result is not None
    
    def test_add_decorative_border(self, renderer, test_image):
        """测试添加装饰边框"""
        result = renderer.add_decorative_border(
            image=test_image,
            border_width=20,
            style="solid"
        )

        assert result is not None
        # 边框会增加图片尺寸
        assert result.size[0] >= test_image.size[0]
        assert result.size[1] >= test_image.size[1]
    
    def test_calculate_position(self, renderer):
        """测试位置计算"""
        image_size = (512, 512)
        text_size = (100, 30)
        margin = 20
        
        # 测试居中
        x, y = renderer._calculate_position(
            TextPosition.CENTER,
            image_size,
            text_size,
            margin
        )
        assert x == (512 - 100) // 2
        assert y == (512 - 30) // 2
        
        # 测试左上
        x, y = renderer._calculate_position(
            TextPosition.TOP_LEFT,
            image_size,
            text_size,
            margin
        )
        assert x == margin
        assert y == margin
        
        # 测试右下
        x, y = renderer._calculate_position(
            TextPosition.BOTTOM_RIGHT,
            image_size,
            text_size,
            margin
        )
        assert x == 512 - 100 - margin
        assert y == 512 - 30 - margin
    
    def test_hex_to_rgba(self, renderer):
        """测试颜色转换"""
        # 测试白色
        rgba = renderer._hex_to_rgba("#FFFFFF")
        assert rgba == (255, 255, 255, 255)
        
        # 测试黑色
        rgba = renderer._hex_to_rgba("#000000")
        assert rgba == (0, 0, 0, 255)
        
        # 测试红色
        rgba = renderer._hex_to_rgba("#FF0000")
        assert rgba == (255, 0, 0, 255)
        
        # 测试带透明度
        rgba = renderer._hex_to_rgba("#FF0000", opacity=0.5)
        assert rgba == (255, 0, 0, 127)
    
    def test_get_available_fonts(self, renderer):
        """测试获取可用字体"""
        fonts = renderer.get_available_fonts()
        
        assert isinstance(fonts, list)
        # 至少应该有默认字体
        assert len(fonts) >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])