"""
Ollama 客户端测试
"""

import pytest
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from middleware.ollama_client import OllamaClient

class TestOllamaClient:
    """Ollama 客户端测试类"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return OllamaClient(
            base_url="http://localhost:11434",
            model="qwen2.5:7b"
        )
    
    def test_init(self, client):
        """测试初始化"""
        assert client.base_url == "http://localhost:11434"
        assert client.model == "qwen2.5:7b"
    
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
    
    @patch('requests.get')
    def test_list_models(self, mock_get, client):
        """测试列出模型"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen2.5:7b"},
                {"name": "llama3:8b"},
                {"name": "mistral:7b"}
            ]
        }
        mock_get.return_value = mock_response
        
        models = client.list_models()
        
        assert len(models) == 3
        assert "qwen2.5:7b" in models
    
    @patch('requests.post')
    def test_generate_success(self, mock_post, client):
        """测试生成 - 成功"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "This is a test response"
        }
        mock_post.return_value = mock_response
        
        result = client.generate("Hello, world!")
        
        assert result == "This is a test response"
    
    @patch('requests.post')
    def test_generate_with_system_prompt(self, mock_post, client):
        """测试带系统提示的生成"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Translated text"
        }
        mock_post.return_value = mock_response
        
        result = client.generate(
            prompt="翻译这段话",
            system_prompt="你是一个翻译助手"
        )
        
        assert result == "Translated text"
        
        # 验证请求参数
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        assert request_data['system'] == "你是一个翻译助手"
    
    @patch('requests.post')
    def test_generate_failure(self, mock_post, client):
        """测试生成 - 失败"""
        mock_post.side_effect = Exception("API Error")
        
        result = client.generate("test")
        
        assert result is None
    
    @patch('requests.post')
    def test_translate_to_english(self, mock_post, client):
        """测试翻译到英文"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "A Tibetan girl in traditional costume"
        }
        mock_post.return_value = mock_response
        
        result = client.translate_to_english("一位穿着传统服饰的藏族姑娘")
        
        assert "Tibetan" in result or "girl" in result
    
    @patch('requests.post')
    def test_expand_prompt(self, mock_post, client):
        """测试扩展提示词"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Potala Palace, majestic architecture, golden roofs, white walls, blue sky, dramatic lighting, highly detailed"
        }
        mock_post.return_value = mock_response
        
        result = client.expand_prompt("布达拉宫")
        
        assert len(result) > 10  # 应该有扩展内容
    
    @patch('requests.post')
    def test_get_embeddings(self, mock_post, client):
        """测试获取嵌入向量"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        mock_post.return_value = mock_response
        
        result = client.get_embeddings("test text")
        
        assert result is not None
        assert len(result) == 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])