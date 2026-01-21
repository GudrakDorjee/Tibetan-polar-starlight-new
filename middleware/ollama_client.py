"""
Ollama API 客户端
提供与 Ollama 服务交互的接口
"""
import json
import logging
import requests
from typing import List, Optional, Generator

logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama API 客户端，用于与本地 Ollama 服务交互"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2:7b",
        timeout: int = 600,
        num_gpu: int = 1
    ):
        """
        初始化 Ollama 客户端

        Args:
            base_url: Ollama 服务地址
            model: 默认使用的模型名称
            timeout: 请求超时时间（秒），默认 600 秒（10分钟）
            num_gpu: 使用的 GPU 数量，默认 1（0 表示仅使用 CPU）
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.num_gpu = num_gpu
        self.session = requests.Session()

        # 预加载模型到 GPU
        if num_gpu > 0:
            self._preload_model()

    def _preload_model(self):
        """预加载模型到 GPU 内存"""
        try:
            logger.info(f"正在预加载模型 {self.model} 到 GPU...")
            payload = {
                "model": self.model,
                "keep_alive": -1  # 保持模型常驻内存
            }
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                logger.info(f"模型 {self.model} 已加载到 GPU")
        except Exception as e:
            logger.warning(f"预加载模型失败: {e}，将在首次使用时加载")

    def check_connection(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama 连接失败: {e}")
            return False

    def list_models(self) -> List[str]:
        """列出可用模型"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> str:
        """
        发送聊天请求

        Args:
            prompt: 用户输入
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大生成长度
            stream: 是否流式输出

        Returns:
            模型回复文本
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_gpu": self.num_gpu  # 指定使用的 GPU 数量
            }
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            if stream:
                return self._handle_stream_response(response)
            else:
                result = response.json()
                return result.get("message", {}).get("content", "")

        except requests.exceptions.Timeout:
            logger.error("Ollama 请求超时")
            raise TimeoutError("LLM 请求超时，请稍后重试")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama 请求失败: {e}")
            raise ConnectionError(f"LLM 服务连接失败: {e}")

    def _handle_stream_response(self, response) -> Generator[str, None, None]:
        """处理流式响应"""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

    def generate_embedding(
        self,
        text: str,
        model: str = "nomic-embed-text"
    ) -> List[float]:
        """
        生成文本嵌入向量

        Args:
            text: 输入文本
            model: 嵌入模型名称

        Returns:
            嵌入向量
        """
        payload = {
            "model": model,
            "prompt": text
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])

        except Exception as e:
            logger.error(f"生成嵌入失败: {e}")
            raise

    def translate_to_english(self, chinese_text: str) -> str:
        """将中文翻译为英文（用于 SD Prompt）"""
        system_prompt = """你是一个专业的翻译助手，专门将中文描述翻译为 Stable Diffusion 可用的英文提示词。

规则：
1. 翻译要准确、简洁
2. 使用 SD 常用的描述词汇
3. 保留专有名词的英文表达
4. 输出纯英文，不要解释
5. 使用逗号分隔不同元素"""

        prompt = f"将以下中文描述翻译为 Stable Diffusion 英文提示词：\n\n{chinese_text}"

        return self.chat(prompt, system_prompt=system_prompt, temperature=0.3)

    def expand_prompt(self, basic_prompt: str, style: str = "通用") -> str:
        """扩写和增强 Prompt"""
        system_prompt = f"""你是一个 Stable Diffusion 提示词专家，专注于藏族文化主题的图像生成。

当前风格：{style}

你的任务是扩写用户的基础描述，添加：
1. 画面质量词（masterpiece, best quality, highly detailed 等）
2. 光影描述（lighting, atmosphere）
3. 构图建议（composition, angle）
4. 风格关键词

输出格式：直接输出英文提示词，用逗号分隔，不要解释。"""

        return self.chat(basic_prompt, system_prompt=system_prompt, temperature=0.7)

    def generate_tibetan_caption(self, description: str, style: str = "诗意") -> str:
        """生成藏文配文"""
        style_prompts = {
            "诗意": "请用优美的藏语诗歌风格",
            "祝福": "请用藏族传统祝福语风格",
            "简洁": "请用简洁的藏语表达",
            "安多方言": "请用安多方言口语风格",
            "康巴方言": "请用康巴方言口语风格"
        }

        style_instruction = style_prompts.get(style, style_prompts["诗意"])

        system_prompt = f"""你是一个精通藏语的文化专家。
{style_instruction}为图片生成配文。

要求：
1. 内容与描述相关
2. 语言优美、有文化内涵
3. 长度适中（10-30个藏文字符）
4. 只输出藏文，不要翻译或解释"""

        prompt = f"为以下图片内容生成藏文配文：\n\n{description}"

        return self.chat(prompt, system_prompt=system_prompt, temperature=0.8)


# 单例模式
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client(
    base_url: str = "http://localhost:11434",
    model: str = "qwen2:7b",
    num_gpu: int = 1
) -> OllamaClient:
    """获取 Ollama 客户端单例"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient(base_url=base_url, model=model, num_gpu=num_gpu)
    return _ollama_client
