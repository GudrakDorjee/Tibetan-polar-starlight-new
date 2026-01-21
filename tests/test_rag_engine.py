"""
RAG 引擎测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from middleware.rag_engine import RAGEngine, SimpleRAGEngine

class TestSimpleRAGEngine:
    """简单 RAG 引擎测试类"""
    
    @pytest.fixture
    def engine(self):
        """创建测试引擎"""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SimpleRAGEngine(
                persist_directory=Path(tmpdir),
                collection_name="test_collection"
            )
            yield engine
    
    def test_init(self, engine):
        """测试初始化"""
        assert engine is not None
        assert engine.collection_name == "test_collection"
    
    def test_add_document(self, engine):
        """测试添加文档"""
        doc_ids = engine.add_document(
            content="唐卡是藏族传统绘画艺术的瑰宝",
            metadata={"source": "test"}
        )
        
        assert len(doc_ids) > 0
    
    def test_add_multiple_documents(self, engine):
        """测试添加多个文档"""
        contents = [
            "布达拉宫是西藏最著名的建筑",
            "格萨尔王是藏族史诗中的英雄",
            "酥油茶是藏族传统饮品"
        ]
        
        for content in contents:
            engine.add_document(content)
        
        stats = engine.get_stats()
        assert stats['document_count'] >= 3
    
    def test_query_basic(self, engine):
        """测试基本查询"""
        # 添加测试数据
        engine.add_document("唐卡是藏族传统绘画艺术")
        engine.add_document("布达拉宫位于拉萨")
        engine.add_document("格萨尔王是藏族英雄")
        
        # 查询
        results = engine.query("唐卡艺术", top_k=2)
        
        assert len(results) <= 2
        if results:
            assert 'content' in results[0]
            assert 'score' in results[0]
    
    def test_query_empty_database(self, engine):
        """测试空数据库查询"""
        results = engine.query("测试查询")
        
        assert results == []
    
    def test_query_with_metadata(self, engine):
        """测试带元数据的查询"""
        engine.add_document(
            "唐卡绘画技艺",
            metadata={"source": "art_book", "category": "painting"}
        )
        
        results = engine.query("唐卡", top_k=1)
        
        if results:
            assert 'metadata' in results[0]
            assert results[0]['metadata'].get('source') == "art_book"
    
    def test_get_stats(self, engine):
        """测试获取统计信息"""
        engine.add_document("测试文档1")
        engine.add_document("测试文档2")
        
        stats = engine.get_stats()
        
        assert 'backend' in stats
        assert 'collection_name' in stats
        assert 'document_count' in stats
        assert stats['document_count'] >= 2
    
    def test_clear(self, engine):
        """测试清空数据库"""
        engine.add_document("测试文档")
        engine.clear()
        
        stats = engine.get_stats()
        assert stats['document_count'] == 0
    
    def test_export_import(self, engine):
        """测试导出和导入"""
        # 添加数据
        engine.add_document("文档1", metadata={"id": 1})
        engine.add_document("文档2", metadata={"id": 2})
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            export_path = Path(f.name)
        
        try:
            # 导出
            success = engine.export_knowledge_base(export_path)
            assert success
            assert export_path.exists()
            
            # 验证导出内容
            with open(export_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            assert 'documents' in data
            assert len(data['documents']) >= 2
            
            # 清空并导入
            engine.clear()
            assert engine.get_stats()['document_count'] == 0
            
            success = engine.import_knowledge_base(export_path)
            assert success
            assert engine.get_stats()['document_count'] >= 2
        
        finally:
            if export_path.exists():
                export_path.unlink()
    
    def test_add_documents_from_file(self, engine):
        """测试从文件添加文档"""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write("第一段内容。\n\n")
            f.write("第二段内容。\n\n")
            f.write("第三段内容。")
            file_path = Path(f.name)
        
        try:
            doc_ids = engine.add_documents_from_file(file_path)
            assert len(doc_ids) >= 1
        finally:
            if file_path.exists():
                file_path.unlink()
    
    def test_text_chunking(self, engine):
        """测试文本分块"""
        long_text = "这是一段很长的文本。" * 100
        
        chunks = engine._chunk_text(long_text, chunk_size=200, overlap=50)
        
        assert len(chunks) > 1
        # 验证分块大小
        for chunk in chunks:
            assert len(chunk) <= 250  # chunk_size + some buffer


class TestRAGEngineWithChroma:
    """ChromaDB RAG 引擎测试类"""
    
    @pytest.fixture
    def mock_chroma(self):
        """模拟 ChromaDB"""
        with patch('middleware.rag_engine.CHROMA_AVAILABLE', True):
            with patch('middleware.rag_engine.chromadb') as mock_chromadb:
                mock_client = MagicMock()
                mock_collection = MagicMock()
                mock_collection.count.return_value = 0
                mock_client.get_or_create_collection.return_value = mock_collection
                mock_chromadb.PersistentClient.return_value = mock_client
                
                yield mock_chromadb, mock_client, mock_collection
    
    def test_chroma_init(self, mock_chroma):
        """测试 ChromaDB 初始化"""
        mock_chromadb, mock_client, mock_collection = mock_chroma
        
        with tempfile.TemporaryDirectory() as tmpdir:
            from middleware.rag_engine import create_rag_engine
            
            engine = create_rag_engine(
                persist_directory=Path(tmpdir),
                use_chroma=True
            )
            
            # 验证引擎创建成功
            assert engine is not None


class TestRAGEngineIntegration:
    """RAG 引擎集成测试"""
    
    @pytest.fixture
    def engine_with_data(self):
        """创建带有测试数据的引擎"""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SimpleRAGEngine(
                persist_directory=Path(tmpdir),
                collection_name="integration_test"
            )
            
            # 添加藏族文化相关测试数据
            test_documents = [
                {
                    "content": "唐卡是藏族文化中一种独特的绘画艺术形式，用彩缎装裱后悬挂供奉。唐卡题材广泛，包括佛像、菩萨、护法神、坛城等宗教内容。",
                    "metadata": {"source": "tibetan_art", "category": "painting"}
                },
                {
                    "content": "布达拉宫坐落于中国西藏自治区拉萨市区西北玛布日山上，是世界上海拔最高的宫殿。布达拉宫最初为吐蕃王朝赞普松赞干布为迎娶尺尊公主和文成公主而兴建。",
                    "metadata": {"source": "tibetan_architecture", "category": "building"}
                },
                {
                    "content": "格萨尔王是藏族人民心目中的英雄，《格萨尔王传》是世界上最长的史诗。格萨尔王降妖除魔、抑强扶弱、造福百姓的故事在藏区广为流传。",
                    "metadata": {"source": "tibetan_epic", "category": "literature"}
                },
                {
                    "content": "藏袍是藏族人民的传统服饰，宽大保暖，适应高原气候。康巴地区的藏袍以华丽著称，常配以金银饰品和珊瑚、绿松石等宝石。",
                    "metadata": {"source": "tibetan_costume", "category": "clothing"}
                },
                {
                    "content": "酥油茶是藏族人民日常生活中不可缺少的饮品，由砖茶、酥油、盐等原料制成。酥油茶能够补充热量，适应高原寒冷气候。",
                    "metadata": {"source": "tibetan_food", "category": "cuisine"}
                }
            ]
            
            for doc in test_documents:
                engine.add_document(doc["content"], doc["metadata"])
            
            yield engine
    
    def test_query_relevance(self, engine_with_data):
        """测试查询相关性"""
        # 查询唐卡相关内容
        results = engine_with_data.query("唐卡绘画艺术", top_k=3)
        
        assert len(results) > 0
        # 最相关的结果应该包含唐卡
        assert "唐卡" in results[0]['content']
    
    def test_query_architecture(self, engine_with_data):
        """测试建筑查询"""
        results = engine_with_data.query("西藏宫殿建筑", top_k=2)
        
        assert len(results) > 0
        # 应该返回布达拉宫相关内容
        top_contents = [r['content'] for r in results]
        assert any("布达拉宫" in c for c in top_contents)
    
    def test_query_costume(self, engine_with_data):
        """测试服饰查询"""
        results = engine_with_data.query("康巴服饰", top_k=2)
        
        assert len(results) > 0
        top_contents = [r['content'] for r in results]
        assert any("藏袍" in c or "康巴" in c for c in top_contents)
    
    def test_query_no_match(self, engine_with_data):
        """测试无匹配查询"""
        results = engine_with_data.query("日本料理寿司", top_k=3)
        
        # 应该返回结果，但相关性较低
        # 或者返回空列表（取决于实现）
        if results:
            # 相关性分数应该较低
            assert results[0]['score'] < 0.8
    
    def test_metadata_filtering(self, engine_with_data):
        """测试元数据"""
        results = engine_with_data.query("藏族文化", top_k=5)
        
        # 验证所有结果都有元数据
        for result in results:
            assert 'metadata' in result
            assert 'source' in result['metadata']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])