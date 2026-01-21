"""
Prompt 编排器测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from middleware.prompt_engineer import PromptEngineer, PromptResult

class TestPromptEngineer:
    """Prompt 编排器测试类"""
    
    @pytest.fixture
    def mock_ollama(self):
        """创建模拟的 Ollama 客户端"""
        mock = Mock()
        mock.translate_to_english.return_value = "translated text"
        mock.expand_prompt.return_value = "expanded prompt with details"
        mock.check_connection.return_value = True
        return mock
    
    @pytest.fixture
    def mock_rag(self):
        """创建模拟的 RAG 引擎"""
        mock = Mock()
        mock.query.return_value = [
            {
                "content": "唐卡是藏族传统绘画艺术",
                "score": 0.9,
                "metadata": {"source": "test"}
            }
        ]
        return mock
    
    @pytest.fixture
    def engineer(self, mock_ollama, mock_rag):
        """创建测试用的编排器"""
        return PromptEngineer(
            ollama_client=mock_ollama,
            rag_engine=mock_rag
        )
    
    def test_init(self, engineer):
        """测试初始化"""
        assert engineer is not None
        assert engineer.ollama_client is not None
    
    def test_detect_style_thangka(self, engineer):
        """测试风格检测 - 唐卡"""
        style = engineer._detect_style("一幅精美的唐卡画作")
        assert style == "唐卡风格"
    
    def test_detect_style_portrait(self, engineer):
        """测试风格检测 - 人像"""
        style = engineer._detect_style("康巴汉子的肖像")
        assert style in ["康巴风格", "藏族人像"]
    
    def test_detect_style_landscape(self, engineer):
        """测试风格检测 - 风景"""
        style = engineer._detect_style("青藏高原的草原风光")
        assert style == "草原风光"
    
    def test_detect_style_none(self, engineer):
        """测试风格检测 - 无匹配"""
        style = engineer._detect_style("一只猫")
        assert style is None
    
    def test_extract_keywords(self, engineer):
        """测试关键词提取"""
        keywords = engineer._extract_keywords("布达拉宫前的转经筒和经幡")
        
        assert "布达拉宫" in keywords
        assert "转经筒" in keywords
        assert "经幡" in keywords
    
    def test_process_basic(self, engineer):
        """测试基本处理流程"""
        result = engineer.process(
            user_input="一位藏族姑娘",
            use_rag=False,
            use_llm=False
        )
        
        assert isinstance(result, PromptResult)
        assert result.positive_prompt is not None
        assert result.negative_prompt is not None
    
    def test_process_with_style(self, engineer):
        """测试带风格的处理"""
        result = engineer.process(
            user_input="佛像",
            style="唐卡风格",
            use_rag=False,
            use_llm=False
        )
        
        assert "thangka" in result.positive_prompt.lower() or "唐卡" in result.positive_prompt
    
    def test_process_with_rag(self, engineer, mock_rag):
        """测试带 RAG 的处理"""
        result = engineer.process(
            user_input="唐卡艺术",
            use_rag=True,
            use_llm=False
        )
        
        # 验证 RAG 被调用
        mock_rag.query.assert_called()
    
    def test_process_with_llm(self, engineer, mock_ollama):
        """测试带 LLM 的处理"""
        result = engineer.process(
            user_input="藏族姑娘",
            use_rag=False,
            use_llm=True
        )
        
        # 验证 LLM 被调用
        assert mock_ollama.translate_to_english.called or mock_ollama.expand_prompt.called
    
    def test_process_with_quality(self, engineer):
        """测试质量预设"""
        result = engineer.process(
            user_input="测试",
            quality="高质量",
            use_rag=False,
            use_llm=False
        )
        
        # 高质量应该包含质量相关词汇
        quality_keywords = ["detailed", "quality", "8k", "masterpiece"]
        has_quality = any(kw in result.positive_prompt.lower() for kw in quality_keywords)
        assert has_quality
    
    def test_process_with_composition(self, engineer):
        """测试构图预设"""
        result = engineer.process(
            user_input="测试",
            composition="特写",
            use_rag=False,
            use_llm=False
        )
        
        # 应该包含构图相关词汇
        assert "close" in result.positive_prompt.lower() or "特写" in result.positive_prompt
    
    def test_process_custom_negative(self, engineer):
        """测试自定义负面提示词"""
        custom_negative = "no cats, no dogs"
        result = engineer.process(
            user_input="测试",
            custom_negative=custom_negative,
            use_rag=False,
            use_llm=False
        )
        
        assert custom_negative in result.negative_prompt
    
    def test_get_style_options(self, engineer):
        """测试获取风格选项"""
        options = engineer.get_style_options()
        
        assert isinstance(options, list)
        assert len(options) > 0
        assert "唐卡风格" in options
    
    def test_get_quality_options(self, engineer):
        """测试获取质量选项"""
        options = engineer.get_quality_options()
        
        assert isinstance(options, list)
        assert "高质量" in options
    
    def test_get_composition_options(self, engineer):
        """测试获取构图选项"""
        options = engineer.get_composition_options()
        
        assert isinstance(options, list)
        assert len(options) > 0

class TestPromptResult:
    """PromptResult 测试类"""
    
    def test_creation(self):
        """测试创建"""
        result = PromptResult(
            positive_prompt="test positive",
            negative_prompt="test negative",
            detected_style="唐卡风格",
            detected_keywords=["唐卡", "佛像"],
            rag_context="some context"
        )
        
        assert result.positive_prompt == "test positive"
        assert result.negative_prompt == "test negative"
        assert result.detected_style == "唐卡风格"
        assert len(result.detected_keywords) == 2
    
    def test_default_values(self):
        """测试默认值"""
        result = PromptResult(
            positive_prompt="test",
            negative_prompt="bad"
        )
        
        assert result.detected_style is None
        assert result.detected_keywords == []
        assert result.rag_context is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])