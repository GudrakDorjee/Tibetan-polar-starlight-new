"""
Prompt 编排器
负责：汉藏互译、RAG 增强、SD Prompt 转换
"""

import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# 汉藏术语对照表
TERMINOLOGY = {
    # 服饰
    "藏袍": "chuba (tibetan robe)",
    "氆氇": "pulu (tibetan wool fabric)",
    "邦典": "pangden (tibetan apron)",
    "英雄结": "hero knot hairstyle with red tassel",
    "康巴": "khampa tibetan",
    
    # 建筑
    "布达拉宫": "Potala Palace, magnificent tibetan palace",
    "大昭寺": "Jokhang Temple",
    "白塔": "white stupa",
    "经幡": "colorful prayer flags",
    "玛尼堆": "mani stone pile with carved mantras",
    "藏式建筑": "traditional tibetan architecture, white walls, red trim, golden roof",
    
    # 宗教
    "唐卡": "thangka painting, traditional tibetan buddhist art",
    "转经筒": "prayer wheel",
    "哈达": "khata ceremonial white scarf",
    "酥油灯": "butter lamp with warm glow",
    "金刚杵": "vajra dorje",
    "佛像": "buddha statue, golden, serene",
    
    # 自然
    "雪山": "snow-capped mountain, majestic peaks",
    "草原": "tibetan grassland plateau, vast green meadow",
    "青稞": "highland barley field",
    "牦牛": "yak, long-haired bovine",
    "藏羚羊": "tibetan antelope, graceful",
    "纳木错": "Namtso Lake, sacred turquoise lake",
    "珠穆朗玛": "Mount Everest, highest peak",
    
    # 人物
    "格萨尔王": "King Gesar, epic hero wearing golden armor, riding white horse, holding sword and battle flag, heroic pose",
    "活佛": "living buddha rinpoche, wearing red and yellow robes",
    "喇嘛": "tibetan lama monk in maroon robe",
    "藏族姑娘": "tibetan girl, beautiful, traditional costume, coral and turquoise jewelry",
    "藏族男子": "tibetan man, strong, traditional chuba robe",
    "牧民": "tibetan nomad herder",
}

# LoRA 配置
LORA_CONFIGS = {
    "唐卡": {
        "trigger": "<lora:thangka_style:0.8>",
        "keywords": ["flat color", "traditional tibetan art", "gold details", 
                    "sacred geometry", "intricate patterns", "buddhist iconography"],
        "negative_add": "3d render, photorealistic, modern",
        "quality_boost": "masterpiece, best quality, highly detailed, 8k"
    },
    "康巴风格": {
        "trigger": "<lora:khampa_style:0.7>",
        "keywords": ["khampa tibetan", "traditional chuba", "red tassel hair", 
                    "heroic", "strongfeatures", "ethnic jewelry"],"negative_add": "",
        "quality_boost": "masterpiece, best quality, detailed face, sharp focus"
    },
    "藏式建筑": {
        "trigger": "<lora:tibetan_architecture:0.75>",
        "keywords": ["tibetan monastery", "white walls", "red trim", "golden roof",
                    "traditional architecture", "mountain backdrop"],
        "negative_add": "modern building, skyscraper",
        "quality_boost": "masterpiece, best quality, architectural photography, dramatic lighting"
    },
    "草原风光": {
        "trigger": "<lora:tibetan_landscape:0.6>",
        "keywords": ["tibetan plateau", "vast grassland", "snow mountain distance",
                    "blue sky", "white clouds", "yaks grazing"],
        "negative_add": "urban, city, buildings",
        "quality_boost": "masterpiece, best quality, landscape photography, golden hour, panoramic"
    },
    "藏族人像": {
        "trigger": "<lora:tibetan_portrait:0.7>",
        "keywords": ["tibetan person", "traditional costume", "ethnic features",
                    "coral turquoise jewelry", "warm smile"],
        "negative_add": "western clothing, modern fashion",
        "quality_boost": "masterpiece, best quality, portrait photography, detailed face, sharp eyes"
    },
    "藏传佛教": {
        "trigger": "<lora:tibetan_buddhism:0.75>",
        "keywords": ["tibetan buddhism", "sacred", "spiritual", "monastery interior",
                    "butter lamps", "prayer wheels", "monks"],
        "negative_add": "",
        "quality_boost": "masterpiece, best quality, atmospheric lighting, sacred atmosphere"
    }
}

# 质量提升词
QUALITY_PROMPTS = {
    "高质量": "masterpiece, best quality, highly detailed, sharp focus, 8k uhd",
    "电影感": "cinematic lighting, dramatic atmosphere, movie still, film grain",
    "艺术风": "artistic, painterly, beautiful composition, award winning",
    "写实": "photorealistic, hyperrealistic, professional photography, DSLR",
    "插画": "illustration, digital art, vibrant colors, detailed illustration",
}

# 构图提示词
COMPOSITION_PROMPTS = {
    "特写": "close-up shot, face focus, detailed features",
    "半身": "upper body, medium shot, waist up",
    "全身": "full body, full length shot",
    "远景": "wide shot, landscape, panoramic view, establishing shot",
    "俯视": "bird's eye view, top-down perspective, aerial view",
    "仰视": "low angle shot, looking up, dramatic perspective",
}

@dataclass
class PromptResult:
    """Prompt 处理结果"""
    positive_prompt: str
    negative_prompt: str
    detected_style: str
    detected_keywords: List[str]
    lora_triggers: List[str]
    original_input: str

class PromptEngineer:
    """Prompt 编排器"""
    
    def __init__(self, ollama_client=None, rag_engine=None):
        """
        初始化 Prompt 编排器
        
        Args:
            ollama_client: Ollama 客户端实例
            rag_engine: RAG 引擎实例
        """
        self.ollama = ollama_client
        self.rag = rag_engine
        self.terminology = TERMINOLOGY
        self.lora_configs = LORA_CONFIGS
    
    def process(
        self,
        user_input: str,
        style: Optional[str] = None,
        quality: str = "高质量",
        composition: Optional[str] = None,
        use_rag: bool = True,
        use_llm: bool = True,
        custom_negative: Optional[str] = None
    ) -> PromptResult:
        """
        处理用户输入，生成完整的 SD Prompt
        
        Args:
            user_input: 用户原始输入（中文）
            style: 指定风格（唐卡、康巴风格等）
            quality: 质量预设
            composition: 构图预设
            use_rag: 是否使用 RAG 增强
            use_llm: 是否使用 LLM 翻译扩写
            custom_negative: 自定义负面提示词
            
        Returns:
            PromptResult 对象
        """
        detected_keywords = []
        lora_triggers = []
        detected_style = style or "通用"
        
        # Step 1: 检测关键词并替换术语
        processed_text = user_input
        for cn_term, en_term in self.terminology.items():
            if cn_term in processed_text:
                detected_keywords.append(cn_term)
                processed_text = processed_text.replace(cn_term, en_term)
        # Step 2: 自动检测风格（如果未指定）
        if style is None:
            detected_style = self._detect_style(user_input)
        
        # Step 3: RAG 知识增强
        rag_context = ""
        if use_rag and self.rag:
            try:
                rag_results = self.rag.query(user_input, top_k=2)
                if rag_results:
                    rag_context = " ".join([r["content"] for r in rag_results])
                    logger.info(f"RAG 检索到 {len(rag_results)} 条相关知识")
            except Exception as e:
                logger.warning(f"RAG 检索失败: {e}")
        
        # Step 4: LLM 翻译与扩写
        if use_llm and self.ollama:
            try:
                # 构建 LLM 提示
                llm_prompt = self._build_llm_prompt(
                    user_input, 
                    detected_style, rag_context
                )
                english_prompt = self.ollama.chat(
                    llm_prompt,
                    system_prompt=self._get_translation_system_prompt(),
                    temperature=0.3
                )
                # 清理 LLM 输出
                english_prompt = self._clean_llm_output(english_prompt)
            except Exception as e:
                logger.warning(f"LLM 处理失败，使用基础翻译: {e}")
                english_prompt = processed_text
        else:
            english_prompt = processed_text
        
        # Step 5: 添加风格 LoRA 和关键词
        style_config = self.lora_configs.get(detected_style, {})
        if style_config:
            lora_triggers.append(style_config.get("trigger", ""))
            style_keywords = style_config.get("keywords", [])
            english_prompt = f"{english_prompt}, {', '.join(style_keywords)}"
        
        # Step 6: 添加质量和构图提示词
        quality_prompt = QUALITY_PROMPTS.get(quality, QUALITY_PROMPTS["高质量"])
        
        if composition and composition in COMPOSITION_PROMPTS:
            composition_prompt = COMPOSITION_PROMPTS[composition]
            english_prompt = f"{composition_prompt}, {english_prompt}"
        
        # Step 7: 组装最终 Prompt
        final_positive = self._assemble_positive_prompt(
            english_prompt,
            quality_prompt,
            lora_triggers
        )
        
        # Step 8: 组装负面提示词
        final_negative = self._assemble_negative_prompt(
            custom_negative,
            style_config.get("negative_add", "")
        )
        
        return PromptResult(
            positive_prompt=final_positive,
            negative_prompt=final_negative,
            detected_style=detected_style,
            detected_keywords=detected_keywords,
            lora_triggers=lora_triggers,
            original_input=user_input
        )
    
    def _detect_style(self, text: str) -> str:
        """自动检测文本中的风格"""
        style_keywords = {
            "唐卡": ["唐卡", "佛像", "菩萨", "度母", "金刚"],
            "康巴风格": ["康巴", "英雄结", "康定", "甘孜"],
            "藏式建筑": ["寺庙", "宫殿", "布达拉", "大昭寺", "白塔", "建筑"],
            "草原风光": ["草原", "草地", "牧场", "放牧", "牦牛群"],
            "藏族人像": ["姑娘", "男子", "老人", "孩子", "人物", "肖像"],
            "藏传佛教": ["喇嘛", "僧人", "活佛", "诵经", "转经"],
        }
        
        for style, keywords in style_keywords.items():
            if any(kw in text for kw in keywords):
                return style
        
        return "通用"
    
    def _build_llm_prompt(
        self, 
        user_input: str, 
        style: str, 
        rag_context: str
    ) -> str:
        """构建 LLM 翻译提示"""
        prompt = f"用户描述：{user_input}\n"
        prompt += f"风格：{style}\n"
        
        if rag_context:
            prompt += f"参考知识：{rag_context}\n"
        
        prompt += "\n请将上述内容转换为 Stable Diffusion 英文提示词。"
        
        return prompt
    def _get_translation_system_prompt(self) -> str:
        """获取翻译系统提示词"""
        return """你是一个专业的 Stable Diffusion 提示词工程师，专注于藏族文化主题。

你的任务是将中文描述转换为高质量的英文提示词。

规则：
1. 准确翻译文化元素，使用正确的英文术语
2. 添加适当的细节描述（材质、颜色、光影等）
3. 使用 SD 常用的描述格式
4. 保持简洁，用逗号分隔各元素
5. 不要添加质量词（如 masterpiece），这些会单独添加
6. 不要输出解释，只输出英文提示词

示例输入：一个穿着康巴服饰的男子在草原骑马
示例输出：a khampa tibetan man, wearing traditional chuba robe, red tassel hero knot hairstyle, riding a horse, vast grassland, snow mountains in background, dynamic pose, wind blowing clothes"""

    def _clean_llm_output(self, text: str) -> str:
        """清理 LLM 输出"""
        # 移除可能的解释性文字
        lines = text.strip().split('\n')
        
        # 取最长的一行（通常是实际的 prompt）
        if len(lines) > 1:
            text = max(lines, key=len)
        else:
            text = lines[0]
        
        # 移除引号
        text = text.strip('"\'')
        
        # 移除开头的常见前缀
        prefixes_to_remove = [
            "English prompt:",
            "Prompt:",
            "Output:",
            "Translation:",
            "Here is",
            "The prompt is:",]
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        return text.strip()
    
    def _assemble_positive_prompt(
        self,
        main_prompt: str,
        quality_prompt: str,
        lora_triggers: List[str]
    ) -> str:
        """组装正向提示词"""
        parts = []
        
        # 质量词放在最前面
        parts.append(quality_prompt)
        
        # 主要描述
        parts.append(main_prompt)
        
        # LoRA 触发词放在最后
        for trigger in lora_triggers:
            if trigger:
                parts.append(trigger)
        
        # 合并并清理
        full_prompt = ", ".join(parts)
        
        # 清理多余的逗号和空格
        full_prompt = re.sub(r',\s*,', ',', full_prompt)
        full_prompt = re.sub(r'\s+', ' ', full_prompt)
        
        return full_prompt.strip()
    
    def _assemble_negative_prompt(
        self,
        custom_negative: Optional[str],
        style_negative: str
    ) -> str:
        """组装负向提示词"""
        base_negative = (
            "lowres, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, fewer digits, cropped, worst quality, low quality, "
            "normal quality, jpeg artifacts, signature, watermark, username, "
            "blurry, deformed, mutated, ugly, duplicate, morbid, mutilated, "
            "poorly drawn hands, poorly drawn face, mutation, extra limbs, "
            "extra legs, extra arms, disfigured, malformed limbs, "
            "fused fingers, too many fingers, long neck"
        )
        
        parts = [base_negative]
        
        if style_negative:
            parts.append(style_negative)
        
        if custom_negative:
            parts.append(custom_negative)
        
        return ", ".join(parts)
    
    def quick_translate(self, chinese_text: str) -> str:
        """快速翻译（仅术语替换，不使用 LLM）"""
        result = chinese_text
        for cn_term, en_term in self.terminology.items():
            result = result.replace(cn_term, en_term)
        return result
    
    def get_style_options(self) -> List[str]:
        """获取可用的风格选项"""
        return ["通用"] + list(self.lora_configs.keys())
    
    def get_quality_options(self) -> List[str]:
        """获取可用的质量选项"""
        return list(QUALITY_PROMPTS.keys())
    
    def get_composition_options(self) -> List[str]:
        """获取可用的构图选项"""
        return list(COMPOSITION_PROMPTS.keys())

# 便捷函数
def create_prompt_engineer(ollama_client=None, rag_engine=None) -> PromptEngineer:
    """创建 Prompt 编排器实例"""
    return PromptEngineer(ollama_client=ollama_client, rag_engine=rag_engine)