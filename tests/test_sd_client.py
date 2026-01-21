"""
SD WebUI 客户端测试
"""

import pytest
from unittest.mock import Mock, patch
from PIL import Image
import io
import base64

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from middleware.sd_client import SDClient, GenerationResult

class TestSDClient:
    """SD 客户端测试类"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return SDClient(base_url="http://localhost:7860")
    
    @pytest.fixture
    def mock_image_b64(self):
        """创建模拟的 base64 图片"""
        img = Image.new('RGB', (64, 64), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def test_init(self, client):
        """测试初始化"""
        assert client.base_url == "http://localhost:7860"
        assert client.timeout == 300
    @patch('requests.get')
    def test_check_connection_success(self, mock_get, client):
        """测试连接检查 - 成功"""
        mock_get.return_value.status_code = 200
        assert client.check_connection() is True
    
    @patch('requests.get')
    def test_check_connection_failure(self, mock_get, client):
        """测试连接检查 - 失败"""
        mock_get.side_effect = Exception("Connection refused")
        assert client.check_connection() is False
    
    @patch('requests.post')
    def test_txt2img_success(self, mock_post, client, mock_image_b64):
        """测试文生图 - 成功"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "images": [mock_image_b64],
            "info": '{"seed": 12345}'
        }
        mock_post.return_value = mock_response
        
        result = client.txt2img(
            prompt="test prompt",
            negative_prompt="bad quality",
            width=512,
            height=512
        )
        
        assert isinstance(result, GenerationResult)
        assert len(result.images) == 1
        assert result.seed == 12345
    
    @patch('requests.post')
    def test_txt2img_timeout(self, mock_post, client):
        """测试文生图 - 超时"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(TimeoutError):
            client.txt2img(prompt="test")
    
    def test_build_payload(self, client):
        """测试构建请求参数"""
        payload = client._build_payload(
            prompt="test prompt",
            negative_prompt="bad",
            width=768,
            height=512,
            steps=25,
            cfg_scale=8.0,
            seed=42
        )
        
        assert payload["prompt"] == "test prompt"
        assert payload["negative_prompt"] == "bad"
        assert payload["width"] == 768
        assert payload["height"] == 512
        assert payload["steps"] == 25
        assert payload["cfg_scale"] == 8.0
        assert payload["seed"] == 42

class TestGenerationResult:
    """生成结果测试类"""
    
    def test_creation(self):
        """测试创建"""
        img = Image.new('RGB', (64, 64))
        result = GenerationResult(
            images=[img],
            seed=12345,
            generation_time=5.5
        )
        
        assert len(result.images) == 1
        assert result.seed == 12345
        assert result.generation_time == 5.5
    
    def test_empty_images(self):
        """测试空图片列表"""
        result = GenerationResult(images=[], seed=-1)
        assert len(result.images) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])