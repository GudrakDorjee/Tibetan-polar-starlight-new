"""
藏汉机器翻译引擎
基于 Ollama 本地模型实现藏汉双向翻译
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TranslationDirection(Enum):
    """翻译方向枚举"""
    TIBETAN_TO_CHINESE = "藏译汉"
    CHINESE_TO_TIBETAN = "汉译藏"
    AUTO_DETECT = "自动检测"


class TranslationStyle(Enum):
    """翻译风格枚举"""
    FORMAL = "正式"
    COLLOQUIAL = "口语"
    LITERARY = "文学"
    TECHNICAL = "专业术语"


@dataclass
class TranslationResult:
    """翻译结果数据类"""
    source_text: str
    translated_text: str
    direction: TranslationDirection
    detected_language: Optional[str] = None
    confidence: Optional[float] = None
    alternative_translations: Optional[List[str]] = None
    explanation: Optional[str] = None


class TibetanChineseTranslator:
    """藏汉机器翻译器"""

    def __init__(self, ollama_client):
        """
        初始化翻译器

        Args:
            ollama_client: Ollama 客户端实例
        """
        self.ollama_client = ollama_client

        # 藏文字符范围（用于语言检测）
        self.tibetan_unicode_ranges = [
            (0x0F00, 0x0FFF),  # 藏文基本区
            (0x1000, 0x104F),  # 缅甸文（部分藏文扩展）
        ]

        # 检测可用的藏文专用模型
        self.tibetan_models = self._detect_tibetan_models()
        self.preferred_model = self._select_preferred_model()
        logger.info(f"翻译器初始化完成，使用模型: {self.preferred_model}")

    def _detect_tibetan_models(self) -> List[str]:
        """检测系统中可用的藏文专用模型"""
        try:
            available_models = self.ollama_client.list_models()
            # 藏文专用模型列表（按优先级排序）
            tibetan_model_keywords = [
                'tilamb', 'monlam', 'melong', 'tibetan', 'bod'
            ]

            detected = []
            for model in available_models:
                model_lower = model.lower()
                if any(keyword in model_lower for keyword in tibetan_model_keywords):
                    detected.append(model)

            logger.info(f"检测到藏文专用模型: {detected}")
            return detected
        except Exception as e:
            logger.warning(f"检测藏文模型失败: {e}")
            return []

    def _select_preferred_model(self) -> str:
        """选择首选模型"""
        # 优先使用藏文专用模型
        if self.tibetan_models:
            # 优先级: monlam-melong > TiLamb > 其他
            # 注意：TiLamb 模型可能较大，加载较慢，所以优先使用 monlam-melong
            for model in self.tibetan_models:
                if 'monlam' in model.lower() or 'melong' in model.lower():
                    return model
            for model in self.tibetan_models:
                if 'tilamb' in model.lower():
                    return model
            return self.tibetan_models[0]

        # 如果没有藏文专用模型，使用默认模型
        logger.warning("未检测到藏文专用模型，将使用默认模型")
        return self.ollama_client.model

    def detect_language(self, text: str) -> str:
        """
        检测文本语言

        Args:
            text: 输入文本

        Returns:
            检测到的语言：'tibetan', 'chinese', 'mixed', 'unknown'
        """
        if not text or not text.strip():
            return 'unknown'

        tibetan_count = 0
        chinese_count = 0
        total_chars = 0

        for char in text:
            code_point = ord(char)

            # 检查是否为藏文
            for start, end in self.tibetan_unicode_ranges:
                if start <= code_point <= end:
                    tibetan_count += 1
                    total_chars += 1
                    break

            # 检查是否为中文
            if 0x4E00 <= code_point <= 0x9FFF:
                chinese_count += 1
                total_chars += 1

        if total_chars == 0:
            return 'unknown'

        tibetan_ratio = tibetan_count / total_chars
        chinese_ratio = chinese_count / total_chars

        if tibetan_ratio > 0.3 and chinese_ratio > 0.3:
            return 'mixed'
        elif tibetan_ratio > 0.3:
            return 'tibetan'
        elif chinese_ratio > 0.3:
            return 'chinese'
        else:
            return 'unknown'

    def translate(
        self,
        text: str,
        direction: TranslationDirection = TranslationDirection.AUTO_DETECT,
        style: TranslationStyle = TranslationStyle.FORMAL,
        include_alternatives: bool = False,
        include_explanation: bool = False
    ) -> TranslationResult:
        """
        执行翻译

        Args:
            text: 待翻译文本
            direction: 翻译方向
            style: 翻译风格
            include_alternatives: 是否包含备选翻译
            include_explanation: 是否包含解释说明

        Returns:
            TranslationResult: 翻译结果
        """
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        # 自动检测语言
        detected_lang = self.detect_language(text)
        logger.info(f"检测到语言: {detected_lang}")

        # 确定翻译方向
        if direction == TranslationDirection.AUTO_DETECT:
            if detected_lang == 'tibetan':
                direction = TranslationDirection.TIBETAN_TO_CHINESE
            elif detected_lang == 'chinese':
                direction = TranslationDirection.CHINESE_TO_TIBETAN
            else:
                raise ValueError(f"无法自动检测语言类型，请手动指定翻译方向。检测结果: {detected_lang}")

        # 执行翻译
        if direction == TranslationDirection.TIBETAN_TO_CHINESE:
            translated = self._translate_tibetan_to_chinese(text, style)
        else:
            translated = self._translate_chinese_to_tibetan(text, style)

        # 获取备选翻译
        alternatives = None
        if include_alternatives:
            alternatives = self._get_alternative_translations(text, direction, style)

        # 获取解释说明
        explanation = None
        if include_explanation:
            explanation = self._get_translation_explanation(text, translated, direction)

        return TranslationResult(
            source_text=text,
            translated_text=translated,
            direction=direction,
            detected_language=detected_lang,
            alternative_translations=alternatives,
            explanation=explanation
        )

    def _translate_tibetan_to_chinese(
        self,
        tibetan_text: str,
        style: TranslationStyle
    ) -> str:
        """
        藏文翻译为中文

        Args:
            tibetan_text: 藏文文本
            style: 翻译风格

        Returns:
            中文翻译结果
        """
        style_instructions = {
            TranslationStyle.FORMAL: "使用正式、书面的中文表达",
            TranslationStyle.COLLOQUIAL: "使用口语化、通俗易懂的中文表达",
            TranslationStyle.LITERARY: "使用优美、文学化的中文表达",
            TranslationStyle.TECHNICAL: "使用专业术语和学术化的中文表达"
        }

        style_instruction = style_instructions.get(style, style_instructions[TranslationStyle.FORMAL])

        system_prompt = f"""你是一位精通藏汉双语的专业翻译专家，具有深厚的藏族文化和语言学背景。

翻译要求：
1. 准确传达原文含义，不遗漏、不添加内容
2. {style_instruction}
3. 保持藏文原有的语气和情感
4. 对于专有名词（人名、地名、佛教术语等），使用标准的汉译名称
5. 注意藏文的敬语和谦语，在中文中适当体现
6. 只输出翻译结果，不要添加任何解释或说明

翻译风格：{style.value}"""

        prompt = f"请将以下藏文翻译为中文：\n\n{tibetan_text}"

        # 临时切换到首选模型
        original_model = self.ollama_client.model
        try:
            logger.info(f"切换模型: {original_model} -> {self.preferred_model}")
            self.ollama_client.model = self.preferred_model

            translation = self.ollama_client.chat(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=2048
            )

            return translation.strip()
        except Exception as e:
            logger.error(f"藏译汉失败 (模型: {self.preferred_model}): {e}")
            raise
        finally:
            # 确保恢复原始模型
            self.ollama_client.model = original_model
            logger.info(f"恢复模型: {self.preferred_model} -> {original_model}")

    def _translate_chinese_to_tibetan(
        self,
        chinese_text: str,
        style: TranslationStyle
    ) -> str:
        """
        中文翻译为藏文

        Args:
            chinese_text: 中文文本
            style: 翻译风格

        Returns:
            藏文翻译结果
        """
        style_instructions = {
            TranslationStyle.FORMAL: "使用正式、书面的藏文表达",
            TranslationStyle.COLLOQUIAL: "使用口语化、日常的藏文表达",
            TranslationStyle.LITERARY: "使用优美、文学化的藏文表达",
            TranslationStyle.TECHNICAL: "使用专业术语和学术化的藏文表达"
        }

        style_instruction = style_instructions.get(style, style_instructions[TranslationStyle.FORMAL])

        system_prompt = f"""你是一位精通藏汉双语的专业翻译专家，具有深厚的藏族文化和语言学背景。

翻译要求：
1. 准确传达原文含义，不遗漏、不添加内容
2. {style_instruction}
3. 使用标准的藏文书写规范（包括正确的音节分隔符 ་ 和句子结束符 །）
4. 对于专有名词，使用标准的藏文音译或意译
5. 注意藏文的敬语体系，根据语境适当使用
6. 只输出翻译结果，不要添加任何解释或说明

翻译风格：{style.value}"""

        prompt = f"请将以下中文翻译为藏文：\n\n{chinese_text}"

        # 临时切换到首选模型
        original_model = self.ollama_client.model
        try:
            logger.info(f"切换模型: {original_model} -> {self.preferred_model}")
            self.ollama_client.model = self.preferred_model

            translation = self.ollama_client.chat(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=2048
            )

            return translation.strip()
        except Exception as e:
            logger.error(f"汉译藏失败 (模型: {self.preferred_model}): {e}")
            raise
        finally:
            # 确保恢复原始模型
            self.ollama_client.model = original_model
            logger.info(f"恢复模型: {self.preferred_model} -> {original_model}")

    def _get_alternative_translations(
        self,
        text: str,
        direction: TranslationDirection,
        style: TranslationStyle,
        num_alternatives: int = 3
    ) -> List[str]:
        """
        获取备选翻译

        Args:
            text: 原文
            direction: 翻译方向
            style: 翻译风格
            num_alternatives: 备选翻译数量

        Returns:
            备选翻译列表
        """
        if direction == TranslationDirection.TIBETAN_TO_CHINESE:
            lang_pair = "藏文翻译为中文"
        else:
            lang_pair = "中文翻译为藏文"

        system_prompt = f"""你是一位精通藏汉双语的专业翻译专家。

请为以下文本提供 {num_alternatives} 种不同的翻译方案，每种方案应该：
1. 准确传达原文含义
2. 在表达方式上有所不同
3. 都是合理且地道的翻译

输出格式：
每行一个翻译方案，不要编号，不要添加任何解释。"""

        prompt = f"请为以下文本提供 {num_alternatives} 种{lang_pair}的备选方案：\n\n{text}"

        original_model = self.ollama_client.model
        try:
            self.ollama_client.model = self.preferred_model

            response = self.ollama_client.chat(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2048
            )

            alternatives = [line.strip() for line in response.strip().split('\n') if line.strip()]
            return alternatives[:num_alternatives]
        except Exception as e:
            logger.error(f"获取备选翻译失败: {e}")
            return []
        finally:
            self.ollama_client.model = original_model

    def _get_translation_explanation(
        self,
        source_text: str,
        translated_text: str,
        direction: TranslationDirection
    ) -> str:
        """
        获取翻译解释说明

        Args:
            source_text: 原文
            translated_text: 译文
            direction: 翻译方向

        Returns:
            翻译解释
        """
        if direction == TranslationDirection.TIBETAN_TO_CHINESE:
            lang_info = "藏文原文和中文译文"
        else:
            lang_info = "中文原文和藏文译文"

        system_prompt = f"""你是一位精通藏汉双语的语言学专家。

请对这次翻译进行简要说明，包括：
1. 关键词汇的翻译选择
2. 语法结构的转换
3. 文化背景的考虑
4. 特殊表达的处理

说明应简洁明了，不超过200字。"""

        prompt = f"""请解释以下翻译：

原文：{source_text}

译文：{translated_text}

翻译方向：{direction.value}"""

        original_model = self.ollama_client.model
        try:
            self.ollama_client.model = self.preferred_model

            explanation = self.ollama_client.chat(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=512
            )

            return explanation.strip()
        except Exception as e:
            logger.error(f"获取翻译解释失败: {e}")
            return ""
        finally:
            self.ollama_client.model = original_model

    def batch_translate(
        self,
        texts: List[str],
        direction: TranslationDirection = TranslationDirection.AUTO_DETECT,
        style: TranslationStyle = TranslationStyle.FORMAL
    ) -> List[TranslationResult]:
        """
        批量翻译

        Args:
            texts: 待翻译文本列表
            direction: 翻译方向
            style: 翻译风格

        Returns:
            翻译结果列表
        """
        results = []
        for text in texts:
            try:
                result = self.translate(text, direction, style)
                results.append(result)
            except Exception as e:
                logger.error(f"批量翻译失败 (文本: {text[:50]}...): {e}")
                # 添加错误结果
                results.append(TranslationResult(
                    source_text=text,
                    translated_text=f"[翻译失败: {str(e)}]",
                    direction=direction,
                    detected_language='unknown'
                ))
        return results


# 单例模式
_translator: Optional[TibetanChineseTranslator] = None


def create_translator(ollama_client) -> TibetanChineseTranslator:
    """
    创建翻译器实例

    Args:
        ollama_client: Ollama 客户端

    Returns:
        TibetanChineseTranslator: 翻译器实例
    """
    global _translator
    if _translator is None:
        _translator = TibetanChineseTranslator(ollama_client)
    return _translator


def get_translator() -> Optional[TibetanChineseTranslator]:
    """获取翻译器实例（如果已创建）"""
    return _translator

