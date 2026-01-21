"""
RAG 知识检索引擎
负责：藏族文化知识库管理、向量检索、幻觉抑制
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import hashlib

logger = logging.getLogger(__name__)

# 尝试导入向量数据库相关库
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB 未安装，RAG 功能将受限")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers 未安装，将使用 Ollama 嵌入")

@dataclass
class Document:
    """文档数据类"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class SearchResult:
    """搜索结果数据类"""
    content: str
    metadata: Dict[str, Any]
    score: float
    document_id: str

class RAGEngine:
    """RAG 知识检索引擎"""
    
    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: str = "tibetan_culture",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        ollama_client=None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        初始化 RAG 引擎
        
        Args:
            persist_directory: 向量数据库持久化目录
            collection_name: 集合名称
            embedding_model: 嵌入模型名称
            ollama_client: Ollama 客户端（用于生成嵌入）
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_client = ollama_client
        
        # 初始化嵌入模型
        self.embedding_model = None
        self.use_ollama_embedding = False
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"加载嵌入模型: {embedding_model}")
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info("嵌入模型加载成功")
            except Exception as e:
                logger.warning(f"加载嵌入模型失败: {e}")
                self.use_ollama_embedding = True
        else:
            self.use_ollama_embedding = True
        # 初始化向量数据库
        self.client = None
        self.collection = None
        if CHROMADB_AVAILABLE:
            self._init_chromadb()
        else:
            logger.warning("ChromaDB 不可用，使用内存存储")
            self.documents: List[Document] = []
    
    def _init_chromadb(self):
        """初始化 ChromaDB"""
        try:
            if self.persist_directory:
                self.persist_directory = Path(self.persist_directory)
                self.persist_directory.mkdir(parents=True, exist_ok=True)
                
                self.client = chromadb.PersistentClient(
                    path=str(self.persist_directory),
                    settings=Settings(anonymized_telemetry=False)
                )
            else:
                self.client = chromadb.Client(
                    settings=Settings(anonymized_telemetry=False)
                )
            
            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "藏族文化知识库"}
            )
            
            logger.info(f"ChromaDB 初始化成功，集合: {self.collection_name}")
            logger.info(f"当前文档数量: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"ChromaDB 初始化失败: {e}")
            self.client = None
            self.collection = None
    
    def _generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        if self.embedding_model is not None:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        elif self.use_ollama_embedding and self.ollama_client:
            return self.ollama_client.generate_embedding(text)
        else:
            raise RuntimeError("没有可用的嵌入模型")
    
    def _generate_doc_id(self, content: str) -> str:
        """生成文档 ID"""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        将文本分块
        
        Args:
            text: 原始文本
            
        Returns:
            文本块列表
        """
        chunks = []
        
        # 按段落分割
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 如果当前块加上新段落不超过限制
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果单个段落超过限制，按句子分割
                if len(para) > self.chunk_size:
                    sentences = para.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
                    current_chunk = ""
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        if len(current_chunk) + len(sent) <= self.chunk_size:
                            current_chunk += sent
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sent
                else:
                    current_chunk = para
        
        # 添加最后一块
        if current_chunk:
            chunks.append(current_chunk)
        
        # 添加重叠
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # 从前一块取重叠部分
                    overlap = chunks[i-1][-self.chunk_overlap:]
                    chunk = overlap + " " + chunk
                overlapped_chunks.append(chunk)
            chunks = overlapped_chunks
        
        return chunks
    
    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> List[str]:
        """
        添加文档到知识库
        
        Args:
            content: 文档内容
            metadata: 元数据
            chunk: 是否分块
            
        Returns:
            添加的文档 ID 列表
        """
        if metadata is None:
            metadata = {}

        doc_ids = []
        
        # 分块处理
        if chunk:
            chunks = self._chunk_text(content)
        else:
            chunks = [content]
        
        for i, chunk_content in enumerate(chunks):
            doc_id = self._generate_doc_id(chunk_content + str(i))
            
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            
            if self.collection is not None:
                try:
                    # 生成嵌入
                    embedding = self._generate_embedding(chunk_content)
                    
                    # 添加到 ChromaDB
                    self.collection.add(
                        ids=[doc_id],
                        embeddings=[embedding],
                        documents=[chunk_content],
                        metadatas=[chunk_metadata]
                    )
                    doc_ids.append(doc_id)
                    
                except Exception as e:
                    logger.error(f"添加文档失败: {e}")
            else:
                # 使用内存存储
                doc = Document(
                    id=doc_id,
                    content=chunk_content,
                    metadata=chunk_metadata,
                    embedding=self._generate_embedding(chunk_content) if self.embedding_model else None
                )
                self.documents.append(doc)
                doc_ids.append(doc_id)
        
        logger.info(f"添加了 {len(doc_ids)} 个文档块")
        return doc_ids
    
    def add_documents_from_file(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        从文件添加文档
        
        Args:
            file_path: 文件路径
            metadata: 额外元数据
            
        Returns:
            添加的文档 ID 列表
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 读取文件内容
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    content = "\n\n".join([str(item) for item in data])
                elif isinstance(data, dict):
                    content = json.dumps(data, ensure_ascii=False, indent=2)
                else:
                    content = str(data)
        elif suffix == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
        
        # 构建元数据
        file_metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": suffix
        }
        if metadata:
            file_metadata.update(metadata)
        
        return self.add_document(content, file_metadata)
    
    def query(
        self,
        query_text: str,
        top_k: int = 3,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        查询知识库
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            min_score: 最小相似度分数

        Returns:
            搜索结果列表
        """
        results = []
        
        if self.collection is not None:
            try:
                # 生成查询嵌入
                query_embedding = self._generate_embedding(query_text)
                
                # 查询 ChromaDB
                search_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                
                # 处理结果
                if search_results and search_results['documents']:
                    for i, doc in enumerate(search_results['documents'][0]):
                        # ChromaDB 返回的是距离，转换为相似度分数
                        distance = search_results['distances'][0][i]
                        score = 1 / (1 + distance)  # 转换为 0-1 的相似度
                        
                        if score >= min_score:
                            results.append({
                                "content": doc,
                                "metadata": search_results['metadatas'][0][i] if search_results['metadatas'] else {},
                                "score": score,
                                "document_id": search_results['ids'][0][i] if search_results['ids'] else ""
                            })
                
            except Exception as e:
                logger.error(f"查询失败: {e}")
        
        else:
            # 使用内存存储的简单搜索
            if self.documents and self.embedding_model:
                query_embedding = self._generate_embedding(query_text)
                
                for doc in self.documents:
                    if doc.embedding:
                        # 计算余弦相似度
                        score = self._cosine_similarity(query_embedding, doc.embedding)
                        if score >= min_score:
                            results.append({
                                "content": doc.content,
                                "metadata": doc.metadata,
                                "score": score,
                                "document_id": doc.id
                            })
                # 按分数排序
                results.sort(key=lambda x: x["score"], reverse=True)
                results = results[:top_k]

        logger.info(f"查询 '{query_text[:50]}...' 返回 {len(results)} 条结果")
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def get_context_for_prompt(
        self,
        query: str,
        top_k: int = 3,
        max_length: int = 1000
    ) -> str:
        """
        获取用于 Prompt 增强的上下文
        
        Args:
            query: 查询文本
            top_k: 检索数量
            max_length: 最大上下文长度
            
        Returns:
            格式化的上下文字符串
        """
        results = self.query(query, top_k=top_k)
        
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result["content"]
            
            if current_length + len(content) > max_length:
                # 截断以适应长度限制
                remaining = max_length - current_length
                if remaining > 100:
                    content = content[:remaining] + "..."
                    context_parts.append(content)
                break
            
            context_parts.append(content)
            current_length += len(content)
        
        return "\n\n".join(context_parts)
    
    def get_cultural_features(self, keyword: str) -> Dict[str, Any]:
        """
        获取文化元素的特征描述（用于幻觉抑制）
        
        Args:
            keyword: 文化元素关键词
            
        Returns:
            特征字典
        """
        # 预定义的文化特征（作为 RAG 的补充）
        cultural_features = {
            "格萨尔王": {
                "visual_features": [
                    "golden armor with intricate patterns",
                    "riding a white horse",
                    "holding a sword and battle flag",
                    "wearing a warrior helmet with feathers",
                    "heroic and majestic pose",
                    "red cape flowing in wind"
                ],
                "context": "藏族史诗英雄，岭国国王，降妖除魔的战神",
                "colors": ["gold", "red", "white", "blue"],
                "must_include": ["armor", "horse", "weapon"],
                "must_avoid": ["modern clothing", "casual pose"]
            },
            "唐卡": {
                "visual_features": [
                    "flat traditional painting style",
                    "gold leaf details",
                    "sacred geometry patterns",
                    "vibrant mineral pigments",
                    "intricate border decorations",
                    "buddhist iconography"
                ],
                "context": "藏传佛教卷轴画，用于宗教修行和供奉",
                "colors": ["gold", "red", "blue", "green", "white"],
                "must_include": ["flat style", "decorative border"],
                "must_avoid": ["3d render", "photorealistic", "modern art"]
            },
            "布达拉宫": {
                "visual_features": [
                    "massive white and red palace",
                    "built on Red Mountain",
                    "golden roofs",
                    "thousands of windows",
                    "white walls with red upper section",
                    "traditional tibetan architecture"
                ],
                "context": "位于拉萨，是藏传佛教圣地和历代达赖喇嘛的冬宫",
                "colors": ["white", "red", "gold"],
                "must_include": ["mountain location", "palace structure"],
                "must_avoid": ["modern buildings nearby", "incorrect colors"]
            },
            "康巴汉子": {
                "visual_features": [
                    "tall and strong build",
                    "hero knot hairstyle with red tassel",
                    "traditional chuba robe",
                    "coral and turquoise jewelry",
                    "confident and heroic expression",
                    "leather boots"
                ],
                "context": "康巴地区的藏族男子，以勇武著称",
                "colors": ["red", "black", "brown", "turquoise"],
                "must_include": ["traditional costume", "distinctive hairstyle"],
                "must_avoid": ["modern clothing", "short hair"]
            },
            "藏族姑娘": {
                "visual_features": [
                    "beautiful ethnic features",
                    "long braided hair with ornaments",
                    "colorful traditional dress",
                    "coral and turquoise jewelry",
                    "silver waist ornaments",
                    "warm genuine smile"
                ],
                "context": "藏族女性，以美丽和勤劳著称",
                "colors": ["colorful", "red", "pink", "turquoise"],
                "must_include": ["traditional dress", "ethnic jewelry"],
                "must_avoid": ["modern fashion", "western style"]
            },
            "经幡": {
                "visual_features": [
                    "five colored rectangular flags",
                    "blue, white, red, green, yellow colors",
                    "printed with prayers and mantras",
                    "fluttering in mountain wind",
                    "strung on ropes between poles"
                ],
                "context": "藏传佛教祈福用品，五色代表五行",
                "colors": ["blue", "white", "red", "green", "yellow"],
                "must_include": ["five colors", "rectangular shape"],
                "must_avoid": ["wrong colors", "wrong order"]
            }
        }
        
        # 首先检查预定义特征
        if keyword in cultural_features:
            features = cultural_features[keyword]
        else:
            # 从 RAG 检索
            results = self.query(keyword, top_k=2)
            features = {
                "visual_features": [],
                "context": "",
                "colors": [],
                "must_include": [],
                "must_avoid": []
            }
            
            if results:
                features["context"] = results[0]["content"]
        
        return features
    
    def enhance_prompt_with_features(
        self,
        base_prompt: str,
        keywords: List[str]
    ) -> Tuple[str, str]:
        """
        使用文化特征增强 Prompt（幻觉抑制）
        
        Args:
            base_prompt: 基础提示词
            keywords: 检测到的关键词
            
        Returns:
            (增强后的正向提示词, 增强后的负向提示词)
        """
        additional_positive = []
        additional_negative = []
        
        for keyword in keywords:
            features = self.get_cultural_features(keyword)
            
            if features.get("visual_features"):
                additional_positive.extend(features["visual_features"][:3])
            
            if features.get("must_avoid"):
                additional_negative.extend(features["must_avoid"])
        
        # 去重
        additional_positive = list(set(additional_positive))
        additional_negative = list(set(additional_negative))
        
        enhanced_positive = base_prompt
        if additional_positive:
            enhanced_positive = f"{base_prompt}, {', '.join(additional_positive)}"
        enhanced_negative = ", ".join(additional_negative) if additional_negative else ""
        
        return enhanced_positive, enhanced_negative
    
    def clear_collection(self):
        """清空知识库"""
        if self.collection is not None:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "藏族文化知识库"}
                )
                logger.info("知识库已清空")
            except Exception as e:
                logger.error(f"清空知识库失败: {e}")
        else:
            self.documents = []
            logger.info("内存知识库已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        stats = {
            "backend": "chromadb" if self.collection else "memory",
            "collection_name": self.collection_name,
            "embedding_model": "sentence-transformers" if self.embedding_model else "ollama",
        }
        
        if self.collection is not None:
            stats["document_count"] = self.collection.count()
        else:
            stats["document_count"] = len(self.documents)
        
        return stats
    
    def export_knowledge_base(self, output_path: Path) -> bool:
        """
        导出知识库
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            是否成功
        """
        try:
            export_data = {
                "collection_name": self.collection_name,
                "documents": []
            }
            
            if self.collection is not None:
                # 从 ChromaDB 导出
                all_data = self.collection.get(include=["documents", "metadatas"])
                
                for i, doc_id in enumerate(all_data["ids"]):
                    export_data["documents"].append({
                        "id": doc_id,
                        "content": all_data["documents"][i],
                        "metadata": all_data["metadatas"][i] if all_data["metadatas"] else {}
                    })
            else:
                # 从内存导出
                for doc in self.documents:
                    export_data["documents"].append({
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata
                    })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"知识库已导出到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出知识库失败: {e}")
            return False
    
    def import_knowledge_base(self, input_path: Path) -> bool:
        """
        导入知识库
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            是否成功
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 清空现有数据
            self.clear_collection()
            
            # 导入文档
            for doc_data in import_data.get("documents", []):
                self.add_document(
                    content=doc_data["content"],
                    metadata=doc_data.get("metadata", {}),
                    chunk=False  # 已经是分块的数据
                )
            
            logger.info(f"知识库已从 {input_path} 导入")
            return True
            
        except Exception as e:
            logger.error(f"导入知识库失败: {e}")
            return False

# 便捷函数
def create_rag_engine(
    persist_directory: Optional[Path] = None,
    collection_name: str = "tibetan_culture",
    ollama_client=None
) -> RAGEngine:
    """创建 RAG 引擎实例"""
    return RAGEngine(
        persist_directory=persist_directory,
        collection_name=collection_name,
        ollama_client=ollama_client
    )