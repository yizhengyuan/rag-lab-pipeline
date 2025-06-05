"""
重构后的 ConceptBasedPipeline - 模块化设计，保持向后兼容
"""

import logging
from typing import List, Dict, Any, Optional
from llama_index.core import Document

# 导入配置模块
from config import LlamaIndexConfig, load_config_from_yaml

# 导入核心模块
from core import (
    ConceptNode, 
    EvidenceNode,
    SemanticChunker,
    ConceptMerger,
    ConceptRetriever,
    EvidenceExtractor,
    VectorStoreManager
)

# 导入工具模块
from utils import ConceptValidator

logger = logging.getLogger(__name__)

class ImprovedConceptBasedPipeline:
    """
    改进版基于概念的问题生成 Pipeline - 模块化设计
    
    保持向后兼容的接口，内部使用模块化实现
    """
    
    def __init__(self, 
                 openai_api_key: str = None,
                 base_url: str = None,
                 model_name: str = None,
                 embedding_model: str = "text-embedding-ada-002",
                 config_path: str = None):
        """
        初始化 Pipeline
        
        Args:
            openai_api_key: OpenAI API 密钥（向后兼容）
            base_url: API 基础 URL（向后兼容）
            model_name: 模型名称（向后兼容）
            embedding_model: 嵌入模型名称（向后兼容）
            config_path: 配置文件路径（新增）
        """
        logger.info("初始化 ImprovedConceptBasedPipeline")
        
        # 加载配置
        if config_path:
            self.config = load_config_from_yaml(config_path)
        else:
            # 为了向后兼容，如果提供了参数则覆盖默认配置
            config_dict = {}
            if openai_api_key or base_url or model_name:
                config_dict = {
                    "api": {
                        "openai_key": openai_api_key or "sk-zk2884399e3bbb43b998bd31be7b517f82f67bb0e95df2a1",
                        "base_url": base_url or "https://api.zhizengzeng.com/v1/",
                        "model": model_name or "gpt-4o-mini",
                        "embedding_model": embedding_model
                    }
                }
            self.config = LlamaIndexConfig(config_dict)
        
        # 初始化各个模块
        self.chunker = SemanticChunker(self.config)
        self.concept_merger = ConceptMerger(self.config)
        self.retriever = ConceptRetriever(self.config)
        self.evidence_extractor = EvidenceExtractor(self.config)
        self.vector_store_manager = VectorStoreManager(self.config)
        
        # 存储中间结果（向后兼容）
        self.chunk_nodes: List = []
        self.concept_nodes: List[ConceptNode] = []
        self.evidence_nodes: List[EvidenceNode] = []
        
        # 索引系统（向后兼容）
        self.chunk_index: Optional = None
        self.concept_index: Optional = None
        
        logger.info("Pipeline 初始化完成")
    
    def step1_semantic_chunking_and_concept_extraction(self, documents: List[Document]) -> List:
        """
        步骤1: Semantic Chunking + 每个 chunk 提取 Chunk-Level-Concepts
        
        保持向后兼容的接口
        """
        logger.info("开始步骤1: Semantic Chunking 和 Concept 提取")
        
        # 验证输入数据
        if not ConceptValidator.validate_documents(documents):
            raise ValueError("文档验证失败")
        
        # 使用模块化的分块器
        self.chunk_nodes = self.chunker.chunk_and_extract_concepts(documents)
        self.chunk_index = self.chunker.get_chunk_index()
        
        logger.info(f"完成步骤1: 共处理 {len(self.chunk_nodes)} 个 chunks")
        return self.chunk_nodes
    
    def step2_merge_document_concepts(self, 
                                    similarity_threshold: float = None, 
                                    max_concepts: int = None) -> List[ConceptNode]:
        """
        步骤2: 使用 LlamaIndex 的嵌入系统进行概念合并
        
        保持向后兼容的接口
        """
        logger.info("开始步骤2: 使用 LlamaIndex 系统合并 Document-Level-Concept")
        
        if not self.chunk_nodes:
            raise ValueError("请先执行步骤1")
        
        # 如果提供了参数，临时更新配置（向后兼容）
        original_config = {}
        if similarity_threshold is not None:
            original_config["similarity_threshold"] = self.config.get("concepts.similarity_threshold")
            self.config.config["concepts"]["similarity_threshold"] = similarity_threshold
        
        if max_concepts is not None:
            original_config["max_concepts"] = self.config.get("concepts.max_concepts")
            self.config.config["concepts"]["max_concepts"] = max_concepts
        
        try:
            # 使用模块化的概念合并器
            self.concept_nodes = self.concept_merger.merge_document_concepts(self.chunk_nodes)
            self.concept_index = self.concept_merger.get_concept_index()
            
            logger.info(f"完成步骤2: 合并得到 {len(self.concept_nodes)} 个 document concepts")
            return self.concept_nodes
            
        finally:
            # 恢复原始配置
            for key, value in original_config.items():
                if key == "similarity_threshold":
                    self.config.config["concepts"]["similarity_threshold"] = value
                elif key == "max_concepts":
                    self.config.config["concepts"]["max_concepts"] = value
    
    def step3_concept_based_retrieval(self, top_k: int = None) -> Dict[str, List]:
        """
        步骤3: 使用 LlamaIndex 的检索系统进行概念检索
        
        保持向后兼容的接口
        """
        logger.info("开始步骤3: 基于 Doc-Concept 的检索")
        
        if not self.concept_nodes or not self.chunk_index:
            raise ValueError("请先执行步骤1和步骤2")
        
        # 如果提供了参数，临时更新配置（向后兼容）
        original_top_k = None
        if top_k is not None:
            original_top_k = self.config.get("retrieval.top_k")
            self.config.config["retrieval"]["top_k"] = top_k
        
        try:
            # 使用模块化的检索器
            concept_to_chunks = self.retriever.retrieve_chunks_for_concepts(
                self.concept_nodes, 
                self.chunk_index
            )
            
            logger.info(f"完成步骤3: 为 {len(self.concept_nodes)} 个概念检索了相关 chunks")
            return concept_to_chunks
            
        finally:
            # 恢复原始配置
            if original_top_k is not None:
                self.config.config["retrieval"]["top_k"] = original_top_k
    
    def step4_extract_evidence(self, concept_to_chunks: Dict[str, List]) -> List[EvidenceNode]:
        """
        步骤4: 使用 LlamaIndex 的查询引擎提取证据
        
        保持向后兼容的接口
        """
        logger.info("开始步骤4: Evidence 提取")
        
        # 使用模块化的证据提取器
        self.evidence_nodes = self.evidence_extractor.extract_evidence_for_concepts(
            self.concept_nodes,
            concept_to_chunks
        )
        
        logger.info(f"完成步骤4: 提取了 {len(self.evidence_nodes)} 个 evidence")
        return self.evidence_nodes
    
    def run_pipeline(self, documents: List[Document]) -> Dict[str, Any]:
        """
        运行完整的前4步 pipeline - 全面使用 LlamaIndex
        
        保持向后兼容的接口
        """
        logger.info("开始运行改进版 Pipeline (模块化实现)")
        
        # 步骤1: Semantic Chunking + Concept 提取
        chunk_nodes = self.step1_semantic_chunking_and_concept_extraction(documents)
        
        
        logger.info("完成步骤1: Semantic Chunking + Concept 提取")

        # 步骤2: Document-Level-Concept 合并
        concept_nodes = self.step2_merge_document_concepts()
        
        # 步骤3: 基于概念的检索
        concept_to_chunks = self.step3_concept_based_retrieval()
        
        # 步骤4: Evidence 提取
        evidence_nodes = self.step4_extract_evidence(concept_to_chunks)
        
        results = {
            "chunk_nodes": chunk_nodes,
            "concept_nodes": concept_nodes,
            "concept_to_chunks": concept_to_chunks,
            "evidence_nodes": evidence_nodes,
            "indexes": {
                "chunk_index": self.chunk_index,
                "concept_index": self.concept_index
            }
        }
        
        logger.info("改进版 Pipeline 执行完成")
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        保存结果 - 兼容 LlamaIndex Node 系统
        
        保持向后兼容的接口
        """
        serializable_results = {
            "chunk_nodes": [
                {
                    "node_id": node.node_id,
                    "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    "concepts": node.metadata.get("concepts", []),
                    "metadata": node.metadata
                }
                for node in results["chunk_nodes"]
            ],
            "concept_nodes": [
                {
                    "node_id": node.node_id,
                    "concept_text": node.text,
                    "source_chunks": node.source_chunks,
                    "confidence_score": node.confidence_score,
                    "metadata": node.metadata
                }
                for node in results["concept_nodes"]
            ],
            "evidence_nodes": [
                {
                    "node_id": node.node_id,
                    "evidence_text": node.text,
                    "concept_id": node.concept_id,
                    "relevance_score": node.relevance_score,
                    "metadata": node.metadata
                }
                for node in results["evidence_nodes"]
            ]
        }
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"改进版结果已保存到: {output_path}")
    
    # 新增的便捷方法
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """获取 Pipeline 统计信息"""
        return {
            "chunk_count": len(self.chunk_nodes),
            "concept_count": len(self.concept_nodes),
            "evidence_count": len(self.evidence_nodes),
            "config_summary": {
                "model": self.config.model_name,
                "embedding_model": self.config.embedding_model,
                "similarity_threshold": self.config.get("concepts.similarity_threshold"),
                "max_concepts": self.config.get("concepts.max_concepts")
            }
        }
    
    def reset_pipeline(self):
        """重置 Pipeline 状态"""
        self.chunk_nodes = []
        self.concept_nodes = []
        self.evidence_nodes = []
        self.chunk_index = None
        self.concept_index = None
        logger.info("Pipeline 状态已重置")

# 使用示例
if __name__ == "__main__":
    # 向后兼容的使用方式
    pipeline = ImprovedConceptBasedPipeline(
        openai_api_key="your-api-key-here"
    )
    
    # 新的配置文件使用方式
    # pipeline = ImprovedConceptBasedPipeline(config_path="config/config.yml")
    
    from llama_index.core import SimpleDirectoryReader
    
    # 加载文档
    reader = SimpleDirectoryReader(input_files=["attention is all you need.pdf"])
    documents = reader.load_data()
    
    # 运行 Pipeline
    results = pipeline.run_pipeline(documents)
    pipeline.save_results(results, "modular_results.json")
    
    # 获取统计信息
    stats = pipeline.get_pipeline_statistics()
    print("Pipeline 统计信息:", stats) 