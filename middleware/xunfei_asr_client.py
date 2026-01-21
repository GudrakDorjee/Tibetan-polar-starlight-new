"""
讯飞语音识别客户端
支持藏汉语音识别，使用 WebSocket 协议
参照官方demo实现，使用流式传输
"""

import base64
import hashlib
import hmac
import json
import ssl
import time
from datetime import datetime
from time import mktime
from typing import Optional, Dict, Any
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import logging

import websocket
import _thread as thread
from pydub import AudioSegment
import io

logger = logging.getLogger(__name__)


class XunFeiASRClient:
    """讯飞语音识别客户端 - 使用官方demo的流式传输方式"""

    def __init__(self, app_id: str, api_key: str, api_secret: str, asr_url: str = None):
        """
        初始化客户端

        Args:
            app_id: 应用 ID
            api_key: API Key
            api_secret: API Secret
            asr_url: ASR WebSocket URL (默认使用多语种服务)
        """
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.asr_url = asr_url or "wss://iat.cn-huabei-1.xf-yun.com/v1"

        # 状态常量
        self.STATUS_FIRST_FRAME = 0  # 第一帧的标识
        self.STATUS_CONTINUE_FRAME = 1  # 中间帧标识
        self.STATUS_LAST_FRAME = 2  # 最后一帧的标识

        # 识别结果
        self.final_text = ""
        self.error_code = None
        self.error_message = None
        self.is_closed = False

    def _get_auth_url(self) -> str:
        """生成鉴权URL - 严格按照官方demo实现"""
        try:
            # 使用讯飞官方demo中的时间格式
            now = datetime.now()
            date = format_date_time(mktime(now.timetuple()))

            logger.info(f"生成鉴权URL - 时间: {date}")
            logger.debug(f"API Key: {self.api_key[:8]}...")
            logger.debug(f"API Secret: {self.api_secret[:8]}...")

            # 拼接字符串
            signature_origin = "host: " + "iat.cn-huabei-1.xf-yun.com" + "\n"
            signature_origin += "date: " + date + "\n"
            signature_origin += "GET " + "/v1 " + "HTTP/1.1"

            logger.debug(f"签名原始字符串:\n{signature_origin}")

            # 进行hmac-sha256进行加密
            signature_sha = hmac.new(
                self.api_secret.encode('utf-8'),
                signature_origin.encode('utf-8'),
                digestmod=hashlib.sha256
            ).digest()

            signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

            authorization_origin = 'api_key="%s", algorithm="%s", headers="%s", signature="%s"' % (
                self.api_key, "hmac-sha256", "host date request-line", signature_sha)

            authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

            # 将请求的鉴权参数组合为字典
            v = {
                "authorization": authorization,
                "date": date,
                "host": "iat.cn-huabei-1.xf-yun.com"
            }

            # 拼接鉴权参数，生成url
            url = 'wss://iat.cn-huabei-1.xf-yun.com/v1' + '?' + urlencode(v)

            logger.info(f"生成的URL长度: {len(url)} 字符")
            logger.debug(f"URL: {url[:200]}...")

            return url

        except Exception as e:
            logger.error(f"生成鉴权URL失败: {str(e)}")
            raise

    def _convert_audio_to_wav(self, audio_data: bytes) -> bytes:
        """
        将音频转换为讯飞要求的格式: 16kHz, 单声道, 16bit, PCM WAV
        """
        try:
            # 将bytes转换为AudioSegment
            audio = AudioSegment.from_file(io.BytesIO(audio_data))

            logger.info(
                f"原始音频信息 - 采样率: {audio.frame_rate}Hz, 声道: {audio.channels}, 时长: {len(audio) / 1000:.2f}秒")

            # 转换为单声道
            if audio.channels > 1:
                audio = audio.set_channels(1)
                logger.info("转换为单声道")

            # 转换为16kHz采样率
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
                logger.info(f"转换采样率到 16000Hz")

            # 转换为16bit
            audio = audio.set_sample_width(2)  # 2字节 = 16bit

            # 导出为WAV格式
            output = io.BytesIO()
            audio.export(output, format="wav")
            converted_audio = output.getvalue()

            logger.info(f"音频转换完成 - 大小: {len(converted_audio)} bytes")
            return converted_audio

        except Exception as e:
            logger.error(f"音频转换失败: {str(e)}")
            # 如果转换失败，返回原始数据
            return audio_data

    def _prepare_audio_for_streaming(self, audio_data: bytes) -> list:
        """
        将音频数据分割为帧，准备流式传输
        每帧1280字节（对应40ms的音频）
        """
        frame_size = 1280  # 每一帧的音频大小
        frames = []

        for i in range(0, len(audio_data), frame_size):
            frame = audio_data[i:i + frame_size]
            if frame:  # 只添加非空帧
                frames.append(frame)

        logger.info(f"音频分割为 {len(frames)} 帧，每帧 {frame_size} 字节")
        return frames

    # WebSocket回调函数
    def _on_message(self, ws, message):
        """收到websocket消息的处理 - 按官方demo实现"""
        try:
            message_dict = json.loads(message)
            code = message_dict["header"]["code"]
            status = message_dict["header"]["status"]
            message_text = message_dict["header"].get("message", "")

            logger.info(f"收到响应 - 状态码: {code}, 状态: {status}, 消息: {message_text}")

            if code != 0:
                logger.error(f"请求错误：{code} - {message_text}")
                logger.error(f"完整响应: {json.dumps(message_dict, indent=2, ensure_ascii=False)}")
                self.error_code = code
                self.error_message = message_text
                ws.close()
            else:
                payload = message_dict.get("payload")
                if payload:
                    text = payload["result"]["text"]
                    text = json.loads(str(base64.b64decode(text), "utf8"))
                    text_ws = text['ws']
                    result = ''
                    for i in text_ws:
                        for j in i["cw"]:
                            w = j["w"]
                            result += w

                    if result:
                        self.final_text += result
                        logger.info(f"识别文本片段: {result}")

                if status == 2:
                    logger.info(f"识别完成，最终文本: {self.final_text}")
                    ws.close()

        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            self.error_message = f"处理消息失败: {str(e)}"

    def _on_error(self, ws, error):
        """收到websocket错误的处理"""
        logger.error(f"WebSocket错误: {error}")
        import traceback
        traceback.print_exc()
        self.error_message = f"WebSocket错误: {str(error)}"

    def _on_close(self, ws, close_status_code, close_msg):
        """收到websocket关闭的处理"""
        logger.info(f"WebSocket连接关闭 - 状态码: {close_status_code}, 消息: {close_msg}")
        self.is_closed = True

    def _on_open(self, ws, frames):
        """收到websocket连接建立的处理"""
        logger.info("### WebSocket连接已建立，开始发送音频数据...")

        def run(*args):
            try:
                frameSize = 1280  # 每一帧的音频大小
                intervel = 0.04  # 发送音频间隔(单位:s)
                status = self.STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧

                # 发送所有帧
                for i, frame in enumerate(frames):
                    audio = str(base64.b64encode(frame), 'utf-8')

                    # 最后一帧处理
                    if i == len(frames) - 1:
                        status = self.STATUS_LAST_FRAME

                    # 第一帧处理
                    if status == self.STATUS_FIRST_FRAME:
                        d = {
                            "header": {
                                "status": 0,
                                "app_id": self.app_id
                            },
                            "parameter": {
                                "iat": {
                                    "domain": "slm",
                                    "language": "mul_cn",
                                    "accent": "mandarin",
                                    "result": {
                                        "encoding": "utf8",
                                        "compress": "raw",
                                        "format": "json"
                                    }
                                }
                            },
                            "payload": {
                                "audio": {
                                    "audio": audio,
                                    "sample_rate": 16000,
                                    "encoding": "raw"
                                }
                            }
                        }
                        ws.send(json.dumps(d))
                        status = self.STATUS_CONTINUE_FRAME

                    # 中间帧处理
                    elif status == self.STATUS_CONTINUE_FRAME:
                        d = {
                            "header": {
                                "status": 1,
                                "app_id": self.app_id
                            },
                            "payload": {
                                "audio": {
                                    "audio": audio,
                                    "sample_rate": 16000,
                                    "encoding": "raw"
                                }
                            }
                        }
                        ws.send(json.dumps(d))

                    # 最后一帧处理
                    elif status == self.STATUS_LAST_FRAME:
                        d = {
                            "header": {
                                "status": 2,
                                "app_id": self.app_id
                            },
                            "payload": {
                                "audio": {
                                    "audio": audio,
                                    "sample_rate": 16000,
                                    "encoding": "raw"
                                }
                            }
                        }
                        ws.send(json.dumps(d))
                        break

                    # 模拟音频采样间隔
                    time.sleep(intervel)

                    logger.debug(f"发送第 {i + 1}/{len(frames)} 帧，状态: {status}")

                logger.info("音频数据发送完成")

            except Exception as e:
                logger.error(f"发送音频数据失败: {e}")
                self.error_message = f"发送音频数据失败: {str(e)}"
                ws.close()

        # 在新线程中运行音频发送
        thread.start_new_thread(run, ())

    def speech_to_text(self, audio_data: bytes) -> Dict[str, Any]:
        """
        语音转文字 - 使用流式传输

        Args:
            audio_data: 音频数据

        Returns:
            识别结果
        """
        try:
            # 重置状态
            self.final_text = ""
            self.error_code = None
            self.error_message = None
            self.is_closed = False

            logger.info("=" * 60)
            logger.info("开始语音识别")
            logger.info(f"原始音频大小: {len(audio_data)} bytes")

            # 1. 转换音频格式
            start_time = time.time()
            converted_audio = self._convert_audio_to_wav(audio_data)
            conversion_time = time.time() - start_time
            logger.info(f"音频转换完成 - 耗时: {conversion_time:.3f}s")

            # 2. 准备音频帧
            frames = self._prepare_audio_for_streaming(converted_audio)

            # 检查音频时长
            audio_duration = len(converted_audio) / 32000  # 16k采样率，16bit，单声道
            if audio_duration > 60:
                logger.warning(f"音频过长: {audio_duration:.1f}秒，将截断前60秒")
                frames = frames[:int(60 / 0.04)]  # 保留前60秒的帧

            logger.info(f"准备识别 {len(frames)} 帧音频，约 {len(frames) * 0.04:.1f}秒")

            # 3. 获取鉴权URL
            auth_url = self._get_auth_url()

            # 4. 创建WebSocket连接
            logger.info("创建WebSocket连接...")

            # 创建WebSocketApp实例
            ws = websocket.WebSocketApp(
                auth_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )

            # 设置on_open回调，传入音频帧
            def on_open_wrapper(ws):
                self._on_open(ws, frames)

            ws.on_open = on_open_wrapper

            # 5. 运行WebSocket连接（阻塞直到连接关闭）
            logger.info("开始WebSocket通信...")
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

            # 6. 处理结果
            processing_time = time.time() - start_time

            if self.error_code is not None:
                logger.error(f"识别失败: {self.error_code} - {self.error_message}")
                return {
                    "success": False,
                    "error": f"识别失败 [{self.error_code}]: {self.error_message}",
                    "code": self.error_code
                }

            if not self.final_text.strip():
                logger.warning("未识别到有效文本")
                return {
                    "success": False,
                    "error": "未识别到有效文本",
                    "processing_time": processing_time
                }

            # 7. 返回成功结果
            is_tibetan = self._contains_tibetan_chars(self.final_text)

            logger.info("=" * 60)
            logger.info(f"语音识别成功!")
            logger.info(f"识别文本: {self.final_text}")
            logger.info(f"是否藏语: {is_tibetan}")
            logger.info(f"总处理时间: {processing_time:.3f}s")
            logger.info("=" * 60)

            return {
                "success": True,
                "text": self.final_text.strip(),
                "is_tibetan": is_tibetan,
                "processing_time": processing_time,
                "audio_duration": audio_duration,
                "frame_count": len(frames)
            }

        except Exception as e:
            logger.error(f"语音识别失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            return {
                "success": False,
                "error": f"语音识别失败: {str(e)}"
            }

    def _contains_tibetan_chars(self, text: str) -> bool:
        """检查文本是否包含藏文字符"""
        # 藏文字符范围: U+0F00 到 U+0FFF
        import re
        tibetan_range = re.compile(r'[\u0F00-\u0FFF]')
        return bool(tibetan_range.search(text))


def get_asr_client(
    app_id: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    asr_url: Optional[str] = None,
    language: str = "zh_cn",
    accent: str = "mandarin"
) -> XunFeiASRClient:
    """
    获取 ASR 客户端实例

    Args:
        app_id: 应用 ID (可选，从环境变量读取)
        api_key: API Key (可选，从环境变量读取)
        api_secret: API Secret (可选，从环境变量读取)
        asr_url: ASR WebSocket URL (可选，从环境变量读取)
        language: 语言 (暂时未使用，保留接口兼容性)
        accent: 方言 (暂时未使用，保留接口兼容性)

    Returns:
        ASR 客户端实例
    """
    from config.settings import settings

    return XunFeiASRClient(
        app_id=app_id or settings.xunfei_app_id,
        api_key=api_key or settings.xunfei_api_key,
        api_secret=api_secret or settings.xunfei_api_secret,
        asr_url=asr_url or settings.xunfei_asr_url
    )
