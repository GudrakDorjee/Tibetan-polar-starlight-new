"""
Hunyuan llama-server 客户端
通过 OpenAI 兼容的 API 接口与 llama-server 交互
"""

import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class HunyuanClient:
    """
    腾讯混元 (Hunyuan) 模型客户端
    通过 llama-server.exe 提供的 OpenAI 兼容 API 进行交互
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: int = 600
    ):
        """
        初始化 Hunyuan 客户端

        Args:
            base_url: llama-server 服务地址，默认 http://localhost:8080
            timeout: 请求超时时间（秒），默认 600 秒（10分钟）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        logger.info(f"Hunyuan 客户端初始化: {self.base_url}")

    def check_connection(self) -> bool:
        """
        检查 llama-server 服务是否可用

        Returns:
            True 如果服务正常，False 如果连接失败
        """
        try:
            # 尝试访问 /v1/models 端点检查服务状态
            response = self.session.get(
                f"{self.base_url}/v1/models",
                timeout=10
            )
            if response.status_code == 200:
                logger.info("Hunyuan llama-server 连接成功")
                return True
            else:
                logger.warning(f"Hunyuan llama-server 返回状态码: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Hunyuan llama-server 连接失败: {e}")
            return False

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        发送聊天请求到 Hunyuan 模型

        Args:
            prompt: 用户输入
            system_prompt: 系统提示词
            temperature: 温度参数（0-1），控制随机性
            max_tokens: 最大生成长度

        Returns:
            模型回复文本

        Raises:
            TimeoutError: 请求超时
            ConnectionError: 连接错误
        """
        messages = []

        # 添加系统提示词
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 添加用户消息
        messages.append({"role": "user", "content": prompt})

        # 构建 OpenAI 兼容的请求负载
        payload = {
            "model": "hunyuan",  # 模型名称（llama-server 通常忽略此字段）
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            logger.info(f"发送请求到 Hunyuan: {prompt[:100]}...")
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            # 解析响应
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            logger.info(f"Hunyuan 响应: {content[:100]}...")
            return content

        except requests.exceptions.Timeout:
            logger.error("Hunyuan 请求超时")
            raise TimeoutError("Hunyuan 模型请求超时，请稍后重试")
        except requests.exceptions.RequestException as e:
            logger.error(f"Hunyuan 请求失败: {e}")
            raise ConnectionError(f"Hunyuan 服务连接失败: {e}")

    def translate(
        self,
        text: str,
        direction: str,
        style: str = "正式"
    ) -> str:
        """
        使用 Hunyuan 进行翻译

        Args:
            text: 要翻译的文本
            direction: 翻译方向 "zh2bo" (汉译藏) 或 "bo2zh" (藏译汉)
            style: 翻译风格，如"正式"、"口语"等

        Returns:
            翻译后的文本
        """
        # 根据方向设置系统提示词
        if direction == "zh2bo":
            system_prompt = f"""你是一位专业的汉藏翻译专家。请将用户提供的中文翻译成藏文。
翻译风格：{style}
要求：
1. 准确传达原文含义
2. 符合藏语表达习惯
3. 只输出藏文翻译结果，不要添加任何解释"""

            prompt = f"请将以下中文翻译成藏文：\n\n{text}"

        elif direction == "bo2zh":
            system_prompt = f"""你是一位专业的藏汉翻译专家。请将用户提供的藏文翻译成中文。
翻译风格：{style}
要求：
1. 准确传达原文含义
2. 符合汉语表达习惯
3. 只输出中文翻译结果，不要添加任何解释"""

            prompt = f"请将以下藏文翻译成中文：\n\n{text}"

        else:
            raise ValueError(f"不支持的翻译方向: {direction}")

        return self.chat(prompt, system_prompt=system_prompt, temperature=0.3)


# 单例模式
_hunyuan_client: Optional[HunyuanClient] = None


def get_hunyuan_client(base_url: str = "http://localhost:8080") -> HunyuanClient:
    """获取 Hunyuan 客户端单例"""
    global _hunyuan_client
    if _hunyuan_client is None:
        _hunyuan_client = HunyuanClient(base_url=base_url)
    return _hunyuan_client
