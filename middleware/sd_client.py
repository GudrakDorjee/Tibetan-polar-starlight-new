"""
Stable Diffusion WebUI API 客户端
针对 RTX 4060 8GB 显存优化
"""

import requests
import base64
import io
import json
from PIL import Image
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    """生成结果"""
    images: List[Image.Image]
    parameters: Dict
    info: str
    seed: int
    generation_time: float

class SDClient:
    """Stable Diffusion WebUI API 客户端"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: int = 300
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        
        # RTX 4060 8GB 优化的默认参数
        self.default_params = {
            "steps": 25,
            "cfg_scale": 7.0,
            "width": 768,
            "height": 1024,
            "sampler_name": "DPM++ 2M Karras",
            "batch_size": 1,
            "n_iter": 1,
            "seed": -1,
            "restore_faces": False,
        }
        
        self.default_negative = (
            "lowres, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, fewer digits, cropped, worst quality, low quality, "
            "normal quality, jpeg artifacts, signature, watermark, username, "
            "blurry, deformed, mutated, ugly, duplicate, morbid, mutilated, "
            "poorly drawn hands, poorly drawn face, mutation, extra limbs, "
            "extra legs, extra arms, disfigured, malformed limbs"
        )
    
    def check_connection(self) -> bool:
        """检查 SD WebUI 服务是否可用"""
        try:
            response = self.session.get(
                f"{self.base_url}/sdapi/v1/sd-models",
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"SD WebUI 连接失败: {e}")
            return False
    
    def get_models(self) -> List[Dict]:
        """获取可用的模型列表"""
        try:
            response = self.session.get(
                f"{self.base_url}/sdapi/v1/sd-models",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []
    
    def get_samplers(self) -> List[str]:
        """获取可用的采样器列表"""
        try:
            response = self.session.get(
                f"{self.base_url}/sdapi/v1/samplers",
                timeout=10
            )
            response.raise_for_status()
            return [s["name"] for s in response.json()]
        except Exception as e:
            logger.error(f"获取采样器列表失败: {e}")
            return ["DPM++ 2M Karras", "Euler a", "DDIM"]
    
    def get_loras(self) -> List[Dict]:
        """获取可用的 LoRA 列表"""
        try:
            response = self.session.get(
                f"{self.base_url}/sdapi/v1/loras",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取 LoRA 列表失败: {e}")
            return []

    def refresh_loras(self) -> bool:
        """
        刷新 LoRA 列表

        Returns:
            是否刷新成功
        """
        try:
            response = self.session.post(
                f"{self.base_url}/sdapi/v1/refresh-loras",
                timeout=30
            )
            response.raise_for_status()
            logger.info("LoRA 列表刷新成功")
            return True
        except Exception as e:
            logger.error(f"刷新 LoRA 列表失败: {e}")
            return False

    def set_model(self, model_name: str) -> bool:
        """
        切换 Stable Diffusion 模型

        Args:
            model_name: 模型名称（可以是完整名称或标题）

        Returns:
            是否切换成功
        """
        try:
            logger.info(f"开始切换模型: {model_name}")

            # 获取所有可用模型
            models = self.get_models()

            if not models:
                logger.error("无法获取模型列表")
                return False

            logger.info(f"获取到 {len(models)} 个可用模型")

            # 查找匹配的模型
            target_model = None
            for model in models:
                title = model.get("title", "")
                model_name_field = model.get("model_name", "")

                # 支持通过 title 或 model_name 匹配
                if title == model_name or model_name_field == model_name:
                    target_model = model
                    logger.info(f"找到匹配的模型: title={title}, model_name={model_name_field}")
                    break

            if not target_model:
                logger.error(f"未找到模型: {model_name}")
                logger.error(f"可用模型列表: {[m.get('title', m.get('model_name', '')) for m in models]}")
                return False

            # 使用 SD WebUI API 切换模型
            checkpoint_name = target_model.get("title") or target_model.get("model_name")
            payload = {
                "sd_model_checkpoint": checkpoint_name
            }

            logger.info(f"正在切换模型到: {checkpoint_name}")
            response = self.session.post(
                f"{self.base_url}/sdapi/v1/options",
                json=payload,
                timeout=60  # 切换模型可能需要较长时间
            )
            response.raise_for_status()

            logger.info(f"模型切换成功: {checkpoint_name}")
            return True

        except requests.exceptions.Timeout:
            logger.error(f"切换模型超时: {model_name}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"切换模型请求失败: {e}")
            return False
        except Exception as e:
            logger.error(f"切换模型失败: {e}", exc_info=True)
            return False

    def get_current_model(self) -> Optional[str]:
        """
        获取当前正在使用的模型名称

        Returns:
            当前模型名称，失败返回 None
        """
        try:
            response = self.session.get(
                f"{self.base_url}/sdapi/v1/options",
                timeout=10
            )
            response.raise_for_status()
            options = response.json()
            return options.get("sd_model_checkpoint")
        except Exception as e:
            logger.error(f"获取当前模型失败: {e}")
            return None
    
    def _encode_image(self, image: Image.Image) -> str:
        """将 PIL Image 编码为 base64"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _decode_image(self, base64_str: str) -> Image.Image:
        """将 base64 解码为 PIL Image"""
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    
    def txt2img(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 768,
        height: int = 1024,
        steps: int = 25,
        cfg_scale: float = 7.0,
        sampler_name: str = "DPM++ 2M Karras",
        seed: int = -1,
        batch_size: int = 1,
        enable_hr: bool = False,
        hr_scale: float = 1.5,
        hr_upscaler: str = "Latent",
        denoising_strength: float = 0.5,
        controlnet_args: Optional[Dict] = None,
        lora_models: Optional[List[Dict[str, float]]] = None,
        **kwargs
    ) -> GenerationResult:
        """
        文生图

        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词
            width: 图片宽度
            height: 图片高度
            steps: 采样步数
            cfg_scale: CFG 强度
            sampler_name: 采样器名称
            seed: 随机种子 (-1 为随机)
            batch_size: 批次大小
            enable_hr: 是否启用高清修复
            hr_scale: 高清放大倍数
            hr_upscaler: 高清放大器
            denoising_strength: 重绘强度
            controlnet_args: ControlNet 参数
            lora_models: LoRA 模型列表，格式为 [{"name": "lora_name", "weight": 0.8}]

        Returns:
            GenerationResult 对象
        """
        if negative_prompt is None:
            negative_prompt = self.default_negative

        # 处理 LoRA 模型
        final_prompt = prompt
        if lora_models:
            lora_tags = []
            for lora in lora_models:
                lora_name = lora.get("name", "")
                lora_weight = lora.get("weight", 1.0)
                if lora_name:
                    lora_tags.append(f"<lora:{lora_name}:{lora_weight}>")
            if lora_tags:
                final_prompt = prompt + " " + " ".join(lora_tags)
                logger.info(f"添加 LoRA 标签: {' '.join(lora_tags)}")

        payload = {
            "prompt": final_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler_name,
            "seed": seed,
            "batch_size": batch_size,
            "n_iter": 1,
        }
        
        # 高清修复参数 (注意显存限制)
        if enable_hr:
            # RTX 4060 8GB 限制高清尺寸
            max_hr_pixels = 1536 * 1536
            target_pixels = (width * hr_scale) * (height * hr_scale)
            
            if target_pixels > max_hr_pixels:
                hr_scale = (max_hr_pixels / (width * height)) ** 0.5
                logger.warning(f"高清倍数已调整为 {hr_scale:.2f} 以适配显存")
            
            payload.update({
                "enable_hr": True,
                "hr_scale": hr_scale,
                "hr_upscaler": hr_upscaler,
                "denoising_strength": denoising_strength,
            })
        
        # ControlNet 参数
        if controlnet_args:
            payload["alwayson_scripts"] = {
                "controlnet": {
                    "args": [controlnet_args]
                }
            }
        
        # 合并额外参数
        payload.update(kwargs)
        
        logger.info(f"开始生成图片: {width}x{height}, steps={steps}")
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            generation_time = time.time() - start_time
            logger.info(f"图片生成完成，耗时: {generation_time:.2f}秒")
            
            # 解码图片
            images = [
                self._decode_image(img_base64)
                for img_base64 in result.get("images", [])
            ]
            
            # 解析信息
            info = json.loads(result.get("info", "{}"))
            
            return GenerationResult(
                images=images,
                parameters=payload,
                info=result.get("info", ""),
                seed=info.get("seed", -1),
                generation_time=generation_time
            )
            
        except requests.exceptions.Timeout:
            logger.error("SD WebUI 请求超时")
            raise TimeoutError("图片生成超时，请尝试降低分辨率或步数")
        except requests.exceptions.RequestException as e:
            logger.error(f"SD WebUI 请求失败: {e}")
            raise ConnectionError(f"SD WebUI 服务连接失败: {e}")
    
    def img2img(
        self,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        denoising_strength: float = 0.75,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: int = 25,
        cfg_scale: float = 7.0,
        sampler_name: str = "DPM++ 2M Karras",
        seed: int = -1,
        mask_image: Optional[Image.Image] = None,
        inpainting_fill: int = 1,  # 0=fill, 1=original, 2=latent noise, 3=latent nothing
        inpaint_full_res: bool = True,
        lora_models: Optional[List[Dict[str, float]]] = None,
        **kwargs
    ) -> GenerationResult:
        """
        图生图 / 局部重绘

        Args:
            init_image: 初始图片
            prompt: 正向提示词
            negative_prompt: 负向提示词
            denoising_strength: 重绘强度 (0-1)
            width: 输出宽度 (None 则使用原图尺寸)
            height: 输出高度
            mask_image: 蒙版图片 (用于局部重绘，白色区域会被重绘)
            inpainting_fill: 蒙版区域填充方式
            inpaint_full_res: 是否在全分辨率下重绘
            lora_models: LoRA 模型列表，格式为 [{"name": "lora_name", "weight": 0.8}]

        Returns:
            GenerationResult 对象
        """
        if negative_prompt is None:
            negative_prompt = self.default_negative

        # 处理 LoRA 模型
        final_prompt = prompt
        if lora_models:
            lora_tags = []
            for lora in lora_models:
                lora_name = lora.get("name", "")
                lora_weight = lora.get("weight", 1.0)
                if lora_name:
                    lora_tags.append(f"<lora:{lora_name}:{lora_weight}>")
            if lora_tags:
                final_prompt = prompt + " " + " ".join(lora_tags)
                logger.info(f"添加 LoRA 标签: {' '.join(lora_tags)}")

        # 使用原图尺寸
        if width is None:
            width = init_image.width
        if height is None:
            height = init_image.height

        # 限制尺寸以适配显存
        max_pixels = 1024 * 1024
        current_pixels = width * height
        if current_pixels > max_pixels:
            scale = (max_pixels / current_pixels) ** 0.5
            width = int(width * scale)
            height = int(height * scale)
            logger.warning(f"图片尺寸已调整为 {width}x{height} 以适配显存")

        payload = {
            "init_images": [self._encode_image(init_image)],
            "prompt": final_prompt,
            "negative_prompt": negative_prompt,
            "denoising_strength": denoising_strength,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler_name,
            "seed": seed,
        }
        
        # 局部重绘参数
        if mask_image is not None:
            payload["mask"] = self._encode_image(mask_image)
            payload["inpainting_fill"] = inpainting_fill
            payload["inpaint_full_res"] = inpaint_full_res
            payload["inpaint_full_res_padding"] = 32
        
        payload.update(kwargs)
        
        logger.info(f"开始图生图: {width}x{height}, denoising={denoising_strength}")
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/sdapi/v1/img2img",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            generation_time = time.time() - start_time
            logger.info(f"图生图完成，耗时: {generation_time:.2f}秒")
            
            images = [
                self._decode_image(img_base64)
                for img_base64 in result.get("images", [])
            ]
            
            info = json.loads(result.get("info", "{}"))
            
            return GenerationResult(
                images=images,
                parameters=payload,
                info=result.get("info", ""),
                seed=info.get("seed", -1),
                generation_time=generation_time
            )
            
        except requests.exceptions.Timeout:
            raise TimeoutError("图生图超时，请尝试降低分辨率")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"SD WebUI 服务连接失败: {e}")
    
    def upscale(
        self,
        image: Image.Image,
        upscaler: str = "R-ESRGAN 4x+",
        scale: float = 2.0,
        codeformer_visibility: float = 0.0,
        codeformer_weight: float = 0.5,
        gfpgan_visibility: float = 0.0
    ) -> Image.Image:
        """
        图片放大与修复
        
        Args:
            image: 输入图片
            upscaler: 放大器名称
            scale: 放大倍数
            codeformer_visibility: CodeFormer 人脸修复强度
            codeformer_weight: CodeFormer 权重
            gfpgan_visibility: GFPGAN 人脸修复强度
            
        Returns:
            放大后的图片
        """
        payload = {
            "image": self._encode_image(image),
            "upscaler_1": upscaler,
            "upscaling_resize": scale,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
            "gfpgan_visibility": gfpgan_visibility,
        }
        
        logger.info(f"开始放大图片: scale={scale}, upscaler={upscaler}")
        try:
            response = self.session.post(
                f"{self.base_url}/sdapi/v1/extra-single-image",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            return self._decode_image(result["image"])
            
        except Exception as e:
            logger.error(f"图片放大失败: {e}")
            raise
    
    def get_controlnet_models(self) -> List[str]:
        """获取可用的 ControlNet 模型"""
        try:
            response = self.session.get(
                f"{self.base_url}/controlnet/model_list",
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("model_list", [])
        except Exception as e:
            logger.warning(f"获取 ControlNet 模型失败: {e}")
            return []
    
    def create_controlnet_args(
        self,
        image: Image.Image,
        module: str = "canny",
        model: str = "control_v11p_sd15_canny",
        weight: float = 1.0,
        guidance_start: float = 0.0,
        guidance_end: float = 1.0,
        processor_res: int = 512,
        threshold_a: float = 100,
        threshold_b: float = 200,
    ) -> Dict:
        """
        创建 ControlNet 参数
        
        Args:
            image: 控制图片
            module: 预处理器 (canny, depth, openpose, etc.)
            model: ControlNet 模型
            weight: 控制权重
            guidance_start: 引导开始点
            guidance_end: 引导结束点
            
        Returns:
            ControlNet 参数字典
        """
        return {
            "input_image": self._encode_image(image),
            "module": module,
            "model": model,
            "weight": weight,
            "guidance_start": guidance_start,
            "guidance_end": guidance_end,
            "processor_res": processor_res,
            "threshold_a": threshold_a,
            "threshold_b": threshold_b,
            "resize_mode": 1,  # Scale to Fit
        }
    
    def interrupt(self) -> bool:
        """中断当前生成"""
        try:
            response = self.session.post(
                f"{self.base_url}/sdapi/v1/interrupt",
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_progress(self) -> Dict:
        """获取当前生成进度"""
        try:
            response = self.session.get(
                f"{self.base_url}/sdapi/v1/progress",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return {"progress": 0, "eta_relative": 0}

# 单例模式
_sd_client: Optional[SDClient] = None

def get_sd_client(
    base_url: str = "http://localhost:7860",
    timeout: int = 300
) -> SDClient:
    """获取 SD 客户端单例"""
    global _sd_client
    if _sd_client is None:
        _sd_client = SDClient(base_url=base_url, timeout=timeout)
    return _sd_client