"""
语义分块和概念提取模块
"""

import json
import logging
import hashlib
import pickle
import os
import tiktoken  # 添加tiktoken用于token计数
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """Embedding缓存管理器，避免重复调用API"""
    
    def __init__(self, cache_dir: str, expiry_days: int = 30):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            expiry_days: 缓存过期天数
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiry_days = expiry_days
        self.cache_file = self.cache_dir / "embedding_cache.pkl"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # 加载缓存
        self.cache = self._load_cache()
        self.metadata = self._load_metadata()
    
    def get_text_hash(self, text: str) -> str:
        """获取文本的哈希值作为缓存键"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def is_cached(self, text: str) -> bool:
        """检查文本是否已缓存且未过期"""
        text_hash = self.get_text_hash(text)
        
        if text_hash not in self.cache:
            return False
        
        # 检查是否过期
        cached_time = self.metadata.get(text_hash, {}).get('cached_at')
        if not cached_time:
            return False
        
        cached_date = datetime.fromisoformat(cached_time)
        if datetime.now() - cached_date > timedelta(days=self.expiry_days):
            # 过期，删除缓存
            del self.cache[text_hash]
            del self.metadata[text_hash]
            return False
        
        return True
    
    def get_embedding(self, text: str) -> List[float]:
        """获取缓存的embedding"""
        text_hash = self.get_text_hash(text)
        return self.cache.get(text_hash)
    
    def cache_embedding(self, text: str, embedding: List[float]):
        """缓存embedding"""
        text_hash = self.get_text_hash(text)
        self.cache[text_hash] = embedding
        self.metadata[text_hash] = {
            'cached_at': datetime.now().isoformat(),
            'text_length': len(text),
            'text_preview': text[:100]
        }
        
        # 定期保存（每10个新缓存保存一次）
        if len(self.cache) % 10 == 0:
            self.save_cache()
    
    def save_cache(self):
        """保存缓存到文件"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"💾 embedding缓存已保存: {len(self.cache)}条记录")
        except Exception as e:
            logger.warning(f"保存embedding缓存失败: {e}")
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """加载缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"📂 加载embedding缓存: {len(cache)}条记录")
                return cache
            except Exception as e:
                logger.warning(f"加载embedding缓存失败: {e}")
        
        return {}
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """加载元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存元数据失败: {e}")
        
        return {}
    
    def clear_expired(self):
        """清理过期缓存"""
        expired_keys = []
        now = datetime.now()
        
        for text_hash, meta in self.metadata.items():
            cached_time = meta.get('cached_at')
            if cached_time:
                cached_date = datetime.fromisoformat(cached_time)
                if now - cached_date > timedelta(days=self.expiry_days):
                    expired_keys.append(text_hash)
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.metadata.pop(key, None)
        
        if expired_keys:
            logger.info(f"🗑️ 清理过期embedding缓存: {len(expired_keys)}条")
            self.save_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_size = sum(len(emb) * 8 for emb in self.cache.values())  # 估算字节数
        
        return {
            "total_entries": len(self.cache),
            "estimated_size_mb": total_size / (1024 * 1024),
            "cache_directory": str(self.cache_dir),
            "expiry_days": self.expiry_days
        }

class SemanticChunker:
    """语义分块器 - 使用 LlamaIndex 的 SemanticSplitterNodeParser"""
    
    def __init__(self, config):
        """
        初始化语义分块器
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 🔑 新增：token限制配置 - 更新配置读取路径
        self.max_tokens_per_chunk = config.get("chunking.max_tokens_per_chunk", 6000)  # 保留一些余量
        self.max_char_per_chunk = config.get("chunking.max_char_per_chunk", 24000)
        self.min_chunk_length = config.get("chunking.min_chunk_length", 10)
        self.skip_oversized = config.get("chunking.skip_oversized_chunks", False)
        self.enable_validation = config.get("chunking.enable_token_validation", True)
        
        self.embedding_model_name = config.get("api.embedding_model", "text-embedding-ada-002")
        
        # 初始化token编码器
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")  # 使用通用的编码器
            logger.info(f"✅ Token编码器初始化成功，最大tokens: {self.max_tokens_per_chunk}")
        except Exception as e:
            logger.warning(f"Token编码器初始化失败: {e}，将使用字符长度估算")
            self.tokenizer = None
        
        # 🔑 新增：确保LlamaIndex设置正确初始化
        self._setup_llamaindex_settings()
        
        # 🆕 初始化embedding缓存
        if config.get("vector_store.enable_embedding_cache", True):
            cache_dir = config.get("vector_store.embedding_cache_dir", "./embedding_cache")
            expiry_days = config.get("vector_store.cache_expiry_days", 30)
            self.embedding_cache = EmbeddingCache(cache_dir, expiry_days)
            logger.info(f"✅ Embedding缓存已启用: {cache_dir}")
        else:
            self.embedding_cache = None
            logger.info("❌ Embedding缓存已禁用")
        
        # 初始化 LlamaIndex 的语义分块器
        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=config.get("chunking.buffer_size", 1),
            breakpoint_percentile_threshold=config.get("chunking.breakpoint_percentile_threshold", 95),
            embed_model=Settings.embed_model
        )
        
        # 🆕 添加备用分割器，用于处理超长chunk
        self.fallback_splitter = SentenceSplitter(
            chunk_size=self.max_tokens_per_chunk,
            chunk_overlap=config.get("chunking.fallback_chunk_overlap", 200)
        )
        
        self.chunk_nodes: List[TextNode] = []
        self.chunk_index: VectorStoreIndex = None
    
    def _setup_llamaindex_settings(self):
        """🔑 新增：初始化LlamaIndex设置"""
        try:
            # 配置LLM
            api_key = self.config.get('api.openai_key')
            base_url = self.config.get('api.base_url')
            model_name = self.config.get('api.model', 'gpt-4o-mini')
            
            if not api_key:
                raise ValueError("缺少API密钥")
            
            logger.info(f"🔧 初始化LLM: {model_name}")
            logger.info(f"🔧 API地址: {base_url}")
            
            llm_kwargs = {
                "model": model_name,
                "api_key": api_key,
                "temperature": self.config.get('api.temperature', 0.1)
            }
            
            if base_url:
                llm_kwargs["api_base"] = base_url
            
            Settings.llm = OpenAI(**llm_kwargs)
            logger.info(f"✅ LLM初始化成功: {model_name}")
            
            # 配置嵌入模型
            embed_kwargs = {
                "model": self.config.get('api.embedding_model', 'text-embedding-ada-002'),
                "api_key": api_key
            }
            
            if base_url:
                embed_kwargs["api_base"] = base_url
            
            Settings.embed_model = OpenAIEmbedding(**embed_kwargs)
            logger.info("✅ 嵌入模型初始化成功")
            
            # 🔗 测试LLM连接
            test_response = Settings.llm.complete("测试")
            logger.info(f"✅ LLM连接测试成功: {len(test_response.text)} 字符")
            
        except Exception as e:
            logger.error(f"❌ LlamaIndex设置失败: {e}")
            logger.warning("⚠️ 将使用简单关键词提取作为备选方案")
            Settings.llm = None  # 确保后续能正确检测失败

    def _count_tokens(self, text: str) -> int:
        """
        🆕 计算文本的token数量
        
        Args:
            text: 输入文本
            
        Returns:
            int: token数量
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Token计数失败: {e}，使用字符长度估算")
        
        # 使用字符长度估算（1 token ≈ 4 字符）
        return len(text) // 4
    
    def _split_oversized_chunk(self, node: TextNode) -> List[TextNode]:
        """
        🆕 分割超大的chunk
        
        Args:
            node: 超大的文本节点
            
        Returns:
            List[TextNode]: 分割后的节点列表
        """
        token_count = self._count_tokens(node.text)
        if token_count <= self.max_tokens_per_chunk:
            return [node]
        
        logger.warning(f"⚠️ Chunk过大 ({token_count} tokens)，进行分割...")
        
        # 使用句子分割器进一步分割
        doc = Document(text=node.text, metadata=node.metadata.copy())
        sub_nodes = self.fallback_splitter.get_nodes_from_documents([doc])
        
        # 更新metadata
        for i, sub_node in enumerate(sub_nodes):
            original_chunk_id = node.metadata.get("chunk_id", "unknown")
            sub_node.metadata.update(node.metadata)
            sub_node.metadata["chunk_id"] = f"{original_chunk_id}_sub_{i}"
            sub_node.metadata["is_split"] = True
            sub_node.metadata["original_chunk_id"] = original_chunk_id
            sub_node.metadata["sub_chunk_index"] = i
            sub_node.metadata["total_sub_chunks"] = len(sub_nodes)
        
        logger.info(f"   📝 分割为 {len(sub_nodes)} 个子chunk")
        return sub_nodes
    
    def _validate_chunk_sizes(self, nodes: List[TextNode]) -> List[TextNode]:
        """
        🆕 验证并修复chunk大小
        
        Args:
            nodes: 节点列表
            
        Returns:
            List[TextNode]: 验证和修复后的节点列表
        """
        if not self.enable_validation:
            logger.info("⚠️ Token验证已禁用，跳过chunk大小检查")
            return nodes
        
        valid_nodes = []
        oversized_count = 0
        skipped_count = 0
        
        for node in nodes:
            # 检查最小长度
            if len(node.text.strip()) < self.min_chunk_length:
                skipped_count += 1
                logger.debug(f"跳过过短chunk: {len(node.text)} 字符")
                continue
            
            token_count = self._count_tokens(node.text)
            char_count = len(node.text)
            
            # 检查是否超过限制
            if token_count > self.max_tokens_per_chunk or char_count > self.max_char_per_chunk:
                oversized_count += 1
                
                if self.skip_oversized:
                    logger.warning(f"⚠️ 跳过超大chunk: {token_count} tokens, {char_count} 字符")
                    skipped_count += 1
                    continue
                else:
                    # 分割超大chunk
                    split_nodes = self._split_oversized_chunk(node)
                    valid_nodes.extend(split_nodes)
            else:
                valid_nodes.append(node)
        
        if oversized_count > 0:
            action = "跳过" if self.skip_oversized else "分割"
            logger.info(f"🔧 {action}了 {oversized_count} 个超大chunk")
        
        if skipped_count > 0:
            logger.info(f"⚠️ 跳过了 {skipped_count} 个不符合要求的chunk")
        
        logger.info(f"✅ 最终得到 {len(valid_nodes)} 个有效chunk")
        
        # 最终验证
        if self.enable_validation:
            failed_nodes = 0
            for node in valid_nodes:
                token_count = self._count_tokens(node.text)
                if token_count > self.max_tokens_per_chunk:
                    logger.error(f"❌ 仍有chunk超过限制: {token_count} tokens")
                    failed_nodes += 1
            
            if failed_nodes == 0:
                logger.info(f"✅ 所有chunk都在token限制内")
        
        return valid_nodes

    def _preprocess_large_documents(self, documents: List[Document]) -> List[Document]:
        """
        🆕 预处理超大文档，确保输入到语义分块器的文档大小合适
        
        Args:
            documents: 原始文档列表
            
        Returns:
            List[Document]: 预处理后的安全文档列表
        """
        safe_docs = []
        split_count = 0
        
        for doc_idx, doc in enumerate(documents):
            char_count = len(doc.text)
            token_count = self._count_tokens(doc.text)
            
            # 检查文档是否过大（使用更保守的限制）
            safe_token_limit = self.max_tokens_per_chunk // 2  # 3000 tokens，为语义分块留余量
            safe_char_limit = self.max_char_per_chunk // 2     # 12000 字符
            
            if token_count > safe_token_limit or char_count > safe_char_limit:
                logger.warning(f"⚠️ 文档 {doc_idx} 过大 ({token_count} tokens, {char_count} 字符)，进行预分块...")
                split_count += 1
                
                # 使用简单分割器进行预分块
                temp_doc = Document(text=doc.text, metadata=doc.metadata.copy())
                pre_nodes = self.fallback_splitter.get_nodes_from_documents([temp_doc])
                
                # 转换回Document对象
                for i, node in enumerate(pre_nodes):
                    pre_doc = Document(
                        text=node.text,
                        metadata={
                            **doc.metadata,
                            "pre_split": True,
                            "original_doc_index": doc_idx,
                            "pre_split_index": i,
                            "total_pre_splits": len(pre_nodes)
                        }
                    )
                    safe_docs.append(pre_doc)
            else:
                # 文档大小安全，直接使用
                safe_docs.append(doc)
        
        logger.info(f"📄 预处理完成: {len(documents)} → {len(safe_docs)} 个文档")
        if split_count > 0:
            logger.info(f"🔧 分割了 {split_count} 个超大文档")
        
        return safe_docs

    def chunk_and_extract_concepts(self, documents: List[Document]) -> List[TextNode]:
        """
        语义分块并提取概念
        
        Args:
            documents: 文档列表
            
        Returns:
            List[TextNode]: 包含概念的文本节点列表
        """
        logger.info("开始语义分块和概念提取")
        
        # 验证输入
        if not documents:
            logger.warning("文档列表为空")
            return []
        
        # 🆕 检查是否已有相同文档的向量数据
        if self._should_skip_processing(documents):
            logger.info("🎯 检测到相同文档已处理，尝试加载现有向量数据")
            return self.chunk_nodes
        
        # 🔑 关键修改：预处理超大文档
        logger.info("🔄 预处理超大文档...")
        safe_documents = self._preprocess_large_documents(documents)
        
        # 使用 LlamaIndex 的语义分块（现在输入是安全的）
        logger.info("🔄 执行语义分块...")
        nodes = self.semantic_splitter.get_nodes_from_documents(safe_documents)  # 这里用safe_documents
        
        if not nodes:
            logger.warning("语义分块未产生任何节点")
            return []
        
        logger.info(f"📄 初始分块完成: {len(nodes)} 个chunk")
        
        # 🆕 验证和修复chunk大小
        logger.info("🔍 验证chunk大小...")
        nodes = self._validate_chunk_sizes(nodes)
        
        # 🆕 在创建向量索引前进行最后检查
        logger.info("🔐 创建向量索引前的安全检查...")
        safe_nodes = []
        skipped_nodes = 0
        
        for i, node in enumerate(nodes):
            token_count = self._count_tokens(node.text)
            
            if token_count > self.max_tokens_per_chunk:
                logger.warning(f"⚠️ 跳过超大chunk {i}: {token_count} tokens")
                skipped_nodes += 1
                continue
            
            if len(node.text.strip()) < 10:
                logger.warning(f"⚠️ 跳过过短chunk {i}: {len(node.text)} 字符")
                skipped_nodes += 1
                continue
            
            safe_nodes.append(node)
        
        if skipped_nodes > 0:
            logger.warning(f"⚠️ 跳过了 {skipped_nodes} 个不安全的chunk")
        
        logger.info(f"✅ 安全检查完成: {len(safe_nodes)} 个chunk准备创建索引")
        
        # 🔐 安全地创建向量索引
        try:
            logger.info("🚀 开始创建向量索引...")
            if safe_nodes:
                self.chunk_index = VectorStoreIndex(safe_nodes)
                logger.info("✅ 向量索引创建成功")
            else:
                logger.warning("❌ 没有安全的chunk可以创建索引")
                return []
        except Exception as e:
            logger.error(f"❌ 向量索引创建失败: {e}")
            # 如果还是失败，尝试逐个添加节点
            logger.info("🔄 尝试逐个处理chunk...")
            return self._create_index_safely(safe_nodes)
        
        # 为每个 node 提取概念
        for i, node in enumerate(safe_nodes):
            logger.info(f"正在处理 chunk {i+1}/{len(safe_nodes)}")
            logger.info(f"chunk 内容: {node.text}")
            
            # 提取概念并存储到 metadata 中
            concepts = self._extract_chunk_concepts(node.text)
            
            # 🔑 修复Chroma兼容性：将concepts列表转换为JSON字符串存储
            node.metadata["concepts"] = json.dumps(concepts) if concepts else "[]"
            node.metadata["chunk_id"] = f"chunk_{i}"
            node.metadata["chunk_length"] = len(node.text)
            node.metadata["concept_count"] = len(concepts)
            node.metadata["token_count"] = self._count_tokens(node.text)
        
        self.chunk_nodes = safe_nodes
        logger.info(f"完成语义分块: 共处理 {len(self.chunk_nodes)} 个 chunks")
        
        # 🆕 保存embedding缓存
        if self.embedding_cache:
            self.embedding_cache.save_cache()
            cache_stats = self.embedding_cache.get_cache_stats()
            logger.info(f"💾 缓存统计: {cache_stats['total_entries']}条记录, "
                       f"{cache_stats['estimated_size_mb']:.1f}MB")
        
        return self.chunk_nodes
    
    def _create_index_safely(self, nodes: List[TextNode]) -> List[TextNode]:
        """
        🆕 安全地创建索引，逐个处理节点
        
        Args:
            nodes: 节点列表
            
        Returns:
            List[TextNode]: 成功处理的节点列表
        """
        successful_nodes = []
        
        for i, node in enumerate(nodes):
            try:
                # 尝试为单个节点创建小索引以验证
                test_index = VectorStoreIndex([node])
                successful_nodes.append(node)
                logger.debug(f"✅ 节点 {i} 处理成功")
            except Exception as e:
                logger.warning(f"⚠️ 节点 {i} 处理失败: {e}")
                continue
        
        if successful_nodes:
            try:
                self.chunk_index = VectorStoreIndex(successful_nodes)
                logger.info(f"✅ 安全索引创建成功: {len(successful_nodes)} 个节点")
            except Exception as e:
                logger.error(f"❌ 即使安全处理后仍然失败: {e}")
                return []
        
        return successful_nodes
    
    def _should_skip_processing(self, documents: List[Document]) -> bool:
        """
        检查是否应该跳过处理（文档已存在于向量数据库中）
        
        Args:
            documents: 文档列表
            
        Returns:
            bool: 是否跳过处理
        """
        # 这里可以添加更复杂的检查逻辑
        # 例如：根据文档哈希值或元数据来判断
        
        if not documents:
            return False
        
        # 简单检查：如果当前已有chunk_nodes且数量合理，可能是重复处理
        if self.chunk_nodes and len(self.chunk_nodes) > 0:
            logger.debug("当前已有chunk数据，可能是重复处理")
            return True
        
        # 可以在这里添加更多检查逻辑：
        # 1. 检查向量数据库中是否已有相同文档的embedding
        # 2. 检查文档内容的哈希值
        # 3. 检查文档的元数据标识
        
        return False
    
    def get_document_hash(self, documents: List[Document]) -> str:
        """
        🆕 获取文档集合的哈希值，用于识别重复文档
        
        Args:
            documents: 文档列表
            
        Returns:
            str: 文档集合的哈希值
        """
        import hashlib
        
        # 将所有文档文本合并
        combined_text = ""
        for doc in documents:
            combined_text += doc.text
        
        # 生成哈希值
        return hashlib.sha256(combined_text.encode('utf-8')).hexdigest()[:16]
    
    def _extract_chunk_concepts(self, chunk_text: str) -> List[str]:
        """
        🚀 增强版概念提取 - 使用高质量提示词和多层质量控制
        
        Args:
            chunk_text: chunk 文本
            
        Returns:
            List[str]: 高质量概念列表
        """
        if not chunk_text or len(chunk_text.strip()) < 10:
            logger.debug("文本太短，跳过概念提取")
            return []
        
        # 🔧 修复配置路径
        num_concepts = self.config.get("concept_extraction.concepts_per_chunk", 5)
        
        # 📊 首先分析文本领域，以便使用合适的提取策略
        domain_hint = self._detect_text_domain(chunk_text)
        
        # 🔍 检查LLM是否可用
        if Settings.llm is None:
            logger.warning("LLM未初始化，使用增强关键词提取")
            return self._enhanced_keyword_extraction(chunk_text, num_concepts, domain_hint)
        
        try:
            # 🎯 使用配置文件中的高质量提示词模板
            prompt_template = self.config.get("prompts.concept_extraction", self._get_default_concept_prompt())
            
            # 🔧 增强提示词，添加领域上下文
            enhanced_prompt = self._enhance_prompt_with_domain(prompt_template, domain_hint)
            
            prompt = enhanced_prompt.format(
                num_concepts=num_concepts,
                text=chunk_text[:3000]  # 限制长度避免token超限
            )
            
            logger.debug(f"🤖 使用{domain_hint}领域提示词调用LLM...")
            response = Settings.llm.complete(prompt)
            logger.debug(f"🤖 LLM响应: {response.text[:150]}...")
            
            # 🧹 解析和清理结果
            concepts = self._parse_and_validate_concepts(response.text, num_concepts)
            
            if concepts and len(concepts) >= num_concepts // 2:  # 至少得到一半的概念
                logger.debug(f"✅ LLM概念提取成功: {len(concepts)}个高质量概念")
                return concepts
            else:
                logger.warning(f"⚠️ LLM概念质量不足，使用增强关键词提取作为备选")
                return self._enhanced_keyword_extraction(chunk_text, num_concepts, domain_hint)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}, 使用增强关键词提取")
            return self._enhanced_keyword_extraction(chunk_text, num_concepts, domain_hint)
        except Exception as e:
            logger.warning(f"概念提取失败: {e}, 使用增强关键词提取")
            return self._enhanced_keyword_extraction(chunk_text, num_concepts, domain_hint)

    def _detect_text_domain(self, text: str) -> str:
        """
        🔍 检测文本所属的学术/专业领域
        
        Args:
            text: 输入文本
            
        Returns:
            str: 检测到的领域标识
        """
        text_lower = text.lower()
        
        # 定义领域关键词
        domain_keywords = {
            "computer_science": ["algorithm", "neural network", "machine learning", "artificial intelligence", 
                               "transformer", "attention", "deep learning", "model", "training", "computer", "software"],
            "medicine": ["patient", "clinical", "medical", "diagnosis", "treatment", "therapy", "disease", 
                        "symptom", "health", "medicine", "hospital", "doctor"],
            "physics": ["quantum", "energy", "particle", "wave", "mechanics", "electromagnetic", "relativity", 
                       "physics", "force", "momentum", "mass"],
            "biology": ["cell", "gene", "protein", "organism", "evolution", "dna", "rna", "species", 
                       "biology", "molecular", "cellular"],
            "economics": ["market", "economy", "price", "economic", "financial", "trade", "investment", 
                         "business", "profit", "cost"],
            "law": ["legal", "court", "law", "rights", "constitution", "justice", "judge", "case", 
                   "legislation", "attorney"],
            "psychology": ["behavior", "cognitive", "mental", "psychological", "brain", "mind", 
                          "emotion", "learning", "memory"],
            "chemistry": ["chemical", "molecule", "reaction", "compound", "element", "bond", "chemistry", 
                         "atomic", "synthesis"],
            "mathematics": ["equation", "function", "proof", "theorem", "mathematical", "algebra", 
                           "calculus", "geometry", "statistics"]
        }
        
        # 计算每个领域的匹配分数
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            # 返回匹配度最高的领域
            best_domain = max(domain_scores, key=domain_scores.get)
            logger.debug(f"🎯 检测到文本领域: {best_domain} (匹配度: {domain_scores[best_domain]})")
            return best_domain
        else:
            return "general"

    def _enhance_prompt_with_domain(self, base_prompt: str, domain: str) -> str:
        """
        🎯 根据检测到的领域增强提示词
        
        Args:
            base_prompt: 基础提示词
            domain: 检测到的领域
            
        Returns:
            str: 增强后的提示词
        """
        domain_examples = {
            "computer_science": """
            For computer science/AI content, prioritize concepts like:
            - "attention mechanism", "transformer architecture", "neural networks"
            - "machine learning algorithms", "deep learning models", "artificial intelligence"
            - "computational complexity", "algorithm optimization", "model training"
            """,
            "medicine": """
            For medical content, prioritize concepts like:
            - "clinical diagnosis", "therapeutic intervention", "patient outcomes"
            - "medical procedures", "disease pathology", "treatment protocols"
            - "healthcare systems", "clinical trials", "medical research"
            """,
            "physics": """
            For physics content, prioritize concepts like:
            - "quantum mechanics", "electromagnetic fields", "particle physics"
            - "conservation laws", "wave functions", "energy systems"
            - "relativity theory", "fundamental forces", "physical phenomena"
            """,
            "general": """
            Focus on identifying the most important domain-specific terminology and key concepts.
            """
        }
        
        domain_guidance = domain_examples.get(domain, domain_examples["general"])
        
        return base_prompt + "\n\nDOMAIN-SPECIFIC GUIDANCE:" + domain_guidance

    def _parse_and_validate_concepts(self, llm_response: str, target_count: int) -> List[str]:
        """
        🧹 解析LLM响应并验证概念质量
        
        Args:
            llm_response: LLM的原始响应
            target_count: 目标概念数量
            
        Returns:
            List[str]: 验证后的高质量概念列表
        """
        try:
            # 尝试解析JSON
            result = json.loads(llm_response.strip())
            raw_concepts = result.get("concepts", [])
            
            if not isinstance(raw_concepts, list):
                logger.warning("概念不是列表格式")
                return []
            
            # 🔍 概念质量验证和清理
            validated_concepts = []
            
            for concept in raw_concepts:
                if not isinstance(concept, str):
                    continue
                    
                cleaned_concept = concept.strip()
                
                # 质量检查
                if self._is_high_quality_concept(cleaned_concept):
                    validated_concepts.append(cleaned_concept)
                else:
                    logger.debug(f"过滤低质量概念: {cleaned_concept}")
            
            # 去重（保持顺序）
            seen = set()
            unique_concepts = []
            for concept in validated_concepts:
                concept_lower = concept.lower()
                if concept_lower not in seen:
                    seen.add(concept_lower)
                    unique_concepts.append(concept)
            
            logger.debug(f"📊 概念验证结果: {len(raw_concepts)} → {len(unique_concepts)} 个高质量概念")
            
            return unique_concepts[:target_count]
            
        except json.JSONDecodeError:
            logger.warning("LLM响应不是有效JSON格式")
            return []

    def _is_high_quality_concept(self, concept: str) -> bool:
        """
        🎯 判断概念是否为高质量
        
        Args:
            concept: 待验证的概念
            
        Returns:
            bool: 是否为高质量概念
        """
        if not concept or len(concept.strip()) < 2:
            return False
        
        concept = concept.strip().lower()
        
        # 长度检查：2-6个词
        word_count = len(concept.split())
        if word_count < 1 or word_count > 6:
            return False
        
        # 过滤低质量模式
        low_quality_patterns = [
            # 单个通用词
            r'^(method|system|process|study|analysis|research|data|information|knowledge|content)$',
            # 通用形容词
            r'^(good|bad|new|old|important|significant|main|key|major|minor)$', 
            # 模糊术语
            r'^(thing|aspect|factor|element|issue|item|part|component)s?$',
            # 文档元引用
            r'^(paper|article|document|text|work|study|research|publication)$',
            # 过于简单的组合
            r'^(the |a |an )',
            # 数字或标点开头
            r'^[\d\W]',
        ]
        
        import re
        for pattern in low_quality_patterns:
            if re.search(pattern, concept):
                return False
        
        # 检查是否包含有意义的内容（非纯停用词）
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        concept_words = set(concept.split())
        meaningful_words = concept_words - stop_words
        
        if len(meaningful_words) == 0:
            return False
        
        return True

    def _enhanced_keyword_extraction(self, text: str, num_concepts: int, domain: str) -> List[str]:
        """
        🚀 增强版关键词提取（用作LLM失败时的高质量备选方案）
        
        Args:
            text: 输入文本
            num_concepts: 目标概念数量
            domain: 文本领域
            
        Returns:
            List[str]: 高质量关键词列表
        """
        if not text:
            return []
        
        # 🔧 基于领域的关键词提取策略
        if domain != "general":
            domain_concepts = self._extract_domain_specific_terms(text, domain)
            if len(domain_concepts) >= num_concepts // 2:
                logger.debug(f"✅ 领域特定术语提取成功: {len(domain_concepts)}个概念")
                return domain_concepts[:num_concepts]
        
        # 🔍 通用高质量关键词提取
        concepts = []
        
        # 1. 提取命名实体和专有名词
        named_entities = self._extract_named_entities(text)
        concepts.extend(named_entities)
        
        # 2. 提取复合术语
        compound_terms = self._extract_compound_terms(text)
        concepts.extend(compound_terms)
        
        # 3. 基于TF-IDF的关键词（如果前面的不够）
        if len(concepts) < num_concepts:
            tfidf_keywords = self._extract_tfidf_keywords(text, num_concepts - len(concepts))
            concepts.extend(tfidf_keywords)
        
        # 🧹 清理和验证
        validated_concepts = []
        for concept in concepts:
            if self._is_high_quality_concept(concept):
                validated_concepts.append(concept)
        
        # 去重
        seen = set()
        unique_concepts = []
        for concept in validated_concepts:
            concept_lower = concept.lower()
            if concept_lower not in seen:
                seen.add(concept_lower)
                unique_concepts.append(concept)
        
        logger.debug(f"📊 增强关键词提取: {len(unique_concepts)}个高质量概念")
        return unique_concepts[:num_concepts]

    def _extract_domain_specific_terms(self, text: str, domain: str) -> List[str]:
        """
        🎯 提取领域特定术语
        
        Args:
            text: 输入文本
            domain: 领域标识
            
        Returns:
            List[str]: 领域特定术语列表
        """
        domain_patterns = {
            "computer_science": [
                r'\b(?:deep|machine|artificial)\s+(?:learning|intelligence|network)\b',
                r'\b(?:neural|transformer|attention)\s+(?:network|mechanism|model)\b',
                r'\b(?:algorithm|model|training|optimization)\b',
                r'\b(?:convolutional|recurrent|generative)\s+\w+\b'
            ],
            "medicine": [
                r'\b(?:clinical|medical|therapeutic)\s+(?:trial|diagnosis|intervention)\b',
                r'\b(?:patient|disease|treatment|therapy)\s+\w+\b',
                r'\b(?:cardiovascular|neurological|oncological)\s+\w+\b'
            ],
            "physics": [
                r'\b(?:quantum|electromagnetic|gravitational)\s+(?:field|force|effect)\b',
                r'\b(?:particle|wave|energy)\s+(?:physics|mechanics|dynamics)\b',
                r'\b(?:conservation|uncertainty|relativity)\s+\w+\b'
            ]
        }
        
        patterns = domain_patterns.get(domain, [])
        concepts = []
        
        import re
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend([match.strip() for match in matches])
        
        return concepts[:10]  # 限制数量

    def _extract_named_entities(self, text: str) -> List[str]:
        """
        🏷️ 提取命名实体和专有名词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 命名实体列表
        """
        import re
        
        # 简单的命名实体模式
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Algorithm|Model|Network|Method|Approach)\b',
            r'\b(?:the\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Theorem|Principle|Law|Effect)\b',
            r'\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)*\b',  # 缩写词
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b'  # 专有名词
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend([match.strip() for match in matches])
        
        return entities[:5]

    def _extract_compound_terms(self, text: str) -> List[str]:
        """
        🔗 提取复合术语
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 复合术语列表
        """
        import re
        
        # 寻找形容词+名词、名词+名词的组合
        patterns = [
            r'\b(?:deep|machine|artificial|neural|quantum|clinical|medical)\s+\w+\b',
            r'\b\w+(?:ing|ed|er|al|ic|ous|ive)\s+\w+\b',  # 形容词+名词
            r'\b\w+\s+(?:analysis|method|approach|technique|system|model|theory)\b'
        ]
        
        compounds = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            compounds.extend([match.strip() for match in matches])
        
        return compounds[:8]

    def _extract_tfidf_keywords(self, text: str, num_keywords: int) -> List[str]:
        """
        📊 基于TF-IDF的关键词提取
        
        Args:
            text: 输入文本
            num_keywords: 关键词数量
            
        Returns:
            List[str]: TF-IDF关键词列表
        """
        words = text.lower().split()
        
        # 过滤停用词和短词
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'they', 'their',
            'there', 'where', 'when', 'why', 'how', 'what', 'which', 'who', 'whom'
        }
        
        filtered_words = [
            word.strip('.,!?;:"()[]{}') for word in words 
            if len(word) > 3 and word.lower() not in stop_words and word.isalpha()
        ]
        
        # 计算词频
        from collections import Counter
        word_freq = Counter(filtered_words)
        
        # 选择高频且有意义的词
        candidates = []
        for word, freq in word_freq.most_common(num_keywords * 2):
            if freq > 1 and len(word) > 3:  # 至少出现2次
                candidates.append(word)
        
        return candidates[:num_keywords]

    def _get_default_concept_prompt(self) -> str:
        """
        📝 获取默认的概念提取提示词
        
        Returns:
            str: 默认提示词模板
        """
        return """
        You are an expert in extracting key concepts from academic and professional texts.
        
        Extract {num_concepts} important concepts from the following text that capture the main ideas and themes.
        
        Text:
        {text}
        
        REQUIREMENTS:
        - Each concept should be 2-6 words
        - Focus on domain-specific terminology and key ideas
        - Avoid generic words like "method", "system", "process"
        - Prefer compound terms that capture precise meaning
        
        Return as JSON: {{"concepts": ["concept1", "concept2", ...]}}
        """

    def get_chunk_index(self) -> VectorStoreIndex:
        """获取 chunk 索引"""
        return self.chunk_index
    
    def get_chunking_statistics(self) -> Dict[str, Any]:
        """
        获取分块统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.chunk_nodes:
            return {"total_chunks": 0}
        
        chunk_lengths = [len(node.text) for node in self.chunk_nodes]
        concept_counts = [len(node.metadata.get("concepts", [])) for node in self.chunk_nodes]
        
        stats = {
            "total_chunks": len(self.chunk_nodes),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "total_concepts": sum(concept_counts),
            "avg_concepts_per_chunk": sum(concept_counts) / len(concept_counts) if concept_counts else 0,
            "min_concepts_per_chunk": min(concept_counts) if concept_counts else 0,
            "max_concepts_per_chunk": max(concept_counts) if concept_counts else 0
        }
        
        return stats
    
    def reset(self):
        """重置分块器状态"""
        # 🆕 保存embedding缓存
        if self.embedding_cache:
            self.embedding_cache.save_cache()
            logger.info("💾 Embedding缓存已保存")
        
        self.chunk_nodes = []
        self.chunk_index = None
        logger.info("语义分块器状态已重置")
    
    def get_chunks_by_concept_count(self, min_concepts: int = 1) -> List[TextNode]:
        """
        根据概念数量过滤 chunks
        
        Args:
            min_concepts: 最小概念数量
            
        Returns:
            List[TextNode]: 过滤后的 chunks
        """
        filtered_chunks = []
        for node in self.chunk_nodes:
            concept_count = len(node.metadata.get("concepts", []))
            if concept_count >= min_concepts:
                filtered_chunks.append(node)
        
        logger.info(f"过滤后保留 {len(filtered_chunks)} 个 chunks (最少 {min_concepts} 个概念)")
        return filtered_chunks
    
    def export_chunks_with_concepts(self, output_path: str):
        """
        导出 chunks 和概念到文件
        
        Args:
            output_path: 输出文件路径
        """
        export_data = []
        
        for node in self.chunk_nodes:
            chunk_data = {
                "chunk_id": node.metadata.get("chunk_id"),
                "text": node.text,
                "concepts": node.metadata.get("concepts", []),
                "chunk_length": len(node.text),
                "concept_count": len(node.metadata.get("concepts", []))
            }
            export_data.append(chunk_data)
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已导出 {len(export_data)} 个 chunks 到: {output_path}")
    
    def get_concepts_from_node(self, node) -> List[str]:
        """
        🆕 从节点的metadata中恢复concepts列表
        
        Args:
            node: 文本节点
            
        Returns:
            List[str]: 概念列表
        """
        concepts_data = node.metadata.get("concepts", "[]")
        
        # 如果已经是列表（向后兼容）
        if isinstance(concepts_data, list):
            return concepts_data
        
        # 如果是字符串，尝试解析JSON
        if isinstance(concepts_data, str):
            try:
                return json.loads(concepts_data)
            except json.JSONDecodeError:
                logger.warning(f"无法解析概念字符串: {concepts_data[:50]}...")
                return []
        
        return [] 