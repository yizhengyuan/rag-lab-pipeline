"""
证据提取模块
"""

import logging
import numpy as np
from typing import List, Dict
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import ResponseMode
from .nodes import ConceptNode, EvidenceNode

logger = logging.getLogger(__name__)

class EvidenceExtractor:
    """证据提取器 - 使用 LlamaIndex 的查询引擎"""
    
    def __init__(self, config):
        """
        初始化证据提取器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.min_length = config.get("evidence.min_length", 20)
        self.max_length = config.get("evidence.max_length", 200)
        self.evidence_nodes: List[EvidenceNode] = []
    
    def extract_evidence_for_concepts(self, 
                                    concept_nodes: List[ConceptNode],
                                    concept_to_chunks: Dict[str, List[NodeWithScore]]) -> List[EvidenceNode]:
        """
        为概念提取证据
        
        Args:
            concept_nodes: 概念节点列表
            concept_to_chunks: 概念到chunks的映射
            
        Returns:
            List[EvidenceNode]: 证据节点列表
        """
        logger.info("开始证据提取")
        
        self.evidence_nodes = []
        
        for concept_node in concept_nodes:
            concept_id = concept_node.node_id
            chunks = concept_to_chunks.get(concept_id, [])
            
            if not chunks:
                logger.debug(f"概念 '{concept_node.text}' 没有相关chunks，跳过证据提取")
                continue
                
            logger.info(f"正在为概念 '{concept_node.text}' 提取证据")
            
            try:
                evidence = self._extract_evidence_for_single_concept(concept_node, chunks)
                if evidence:
                    self.evidence_nodes.append(evidence)
            except Exception as e:
                logger.warning(f"为概念 '{concept_node.text}' 提取证据时出错: {e}")
                continue
        
        logger.info(f"完成证据提取: 提取了 {len(self.evidence_nodes)} 个证据")
        return self.evidence_nodes
    
    def _extract_evidence_for_single_concept(self, 
                                           concept_node: ConceptNode, 
                                           chunks: List[NodeWithScore]) -> EvidenceNode:
        """
        为单个概念提取证据
        
        Args:
            concept_node: 概念节点
            chunks: 相关的文档块
            
        Returns:
            EvidenceNode: 证据节点，如果提取失败则返回None
        """
        logger.info(f"正在为概念 '{concept_node.text}' 提取证据")
        if not chunks:
            logger.debug(f"概念 '{concept_node.text}' 没有相关chunks，跳过证据提取")
            return None
        
        # 创建临时索引用于证据提取
        temp_nodes = [chunk.node for chunk in chunks]
        temp_index = VectorStoreIndex(temp_nodes)
        
        # 使用 LlamaIndex 的查询引擎
        query_engine = temp_index.as_query_engine(
            response_mode=ResponseMode.COMPACT,
            similarity_top_k=3
        )
        
        # 构建证据提取查询
        evidence_query = self._build_evidence_query(concept_node.text)
        
        try:
            response = query_engine.query(evidence_query)
            evidence_text = str(response).strip()
            
            # 验证证据质量
            if self._validate_evidence_quality(evidence_text):
                # 计算相关性分数
                relevance_score = np.mean([chunk.score for chunk in chunks]) if chunks else 0.0
                
                evidence_node = EvidenceNode(
                    evidence_text=evidence_text,
                    concept_id=concept_node.node_id,
                    relevance_score=float(relevance_score),
                    id_=f"evidence_{len(self.evidence_nodes)}"
                )
                evidence_node.metadata.update({
                    "concept_text": concept_node.text,
                    "source_chunks": [chunk.node.node_id for chunk in chunks],
                    "extraction_method": "llamaindex_query_engine",
                    "evidence_length": len(evidence_text)
                })
                
                return evidence_node
            else:
                logger.debug(f"证据质量验证失败: {evidence_text[:50]}...")
                return None
                
        except Exception as e:
            logger.warning(f"证据提取失败: {e}")
            return None
    
    def _build_evidence_query(self, concept: str) -> str:
        """
        构建证据提取查询
        
        Args:
            concept: 概念文本
            
        Returns:
            str: 查询字符串
        """
        return f"""
        请从给定的文本中提取与概念"{concept}"最相关的证据片段。
        要求：
        1. 提取的证据应该直接支持或解释该概念
        2. 去除无关信息和噪声
        3. 保持证据的完整性和准确性
        4. 证据长度控制在{self.min_length}-{self.max_length}字
        请直接返回提取的证据文本，不需要额外说明。
        """
    
    def _validate_evidence_quality(self, evidence_text: str) -> bool:
        """
        验证证据质量
        
        Args:
            evidence_text: 证据文本
            
        Returns:
            bool: 是否通过质量验证
        """
        if not evidence_text:
            logger.debug(f"证据为空")
            return False
        
        # 检查长度
        if len(evidence_text) < self.min_length:
            logger.debug(f"证据太短: {len(evidence_text)} < {self.min_length}")
            return False
        
        if len(evidence_text) > self.max_length * 3:  # 放宽长度限制
            logger.debug(f"证据太长: {len(evidence_text)} > {self.max_length * 3}")
            return False
        
        # 检查是否包含有意义的内容
        meaningless_phrases = ["", "无", "没有", "不知道", "无法确定", "无相关内容"]
        if evidence_text.strip() in meaningless_phrases:
            logger.debug(f"证据无意义: {evidence_text}")
            return False
        
        # 🔧 修复：只检查明确的API错误响应，不拒绝法律术语
        api_error_indicators = ["抱歉，我无法", "API调用失败", "请求超时", "服务不可用"]
        if any(indicator in evidence_text for indicator in api_error_indicators):
            logger.debug(f"检测到API错误响应: {evidence_text[:50]}...")
            return False
        
        # 🆕 添加：输出通过验证的证据用于调试
        logger.debug(f"证据通过验证: {evidence_text[:100]}...")
        return True
    
    def get_evidence_statistics(self) -> Dict[str, any]:
        """
        获取证据提取统计信息
        
        Returns:
            Dict[str, any]: 统计信息
        """
        if not self.evidence_nodes:
            return {"total_evidence": 0}
        
        evidence_lengths = [len(node.text) for node in self.evidence_nodes]
        relevance_scores = [node.relevance_score for node in self.evidence_nodes]
        
        return {
            "total_evidence": len(self.evidence_nodes),
            "avg_evidence_length": np.mean(evidence_lengths),
            "min_evidence_length": min(evidence_lengths),
            "max_evidence_length": max(evidence_lengths),
            "avg_relevance_score": np.mean(relevance_scores),
            "min_relevance_score": min(relevance_scores),
            "max_relevance_score": max(relevance_scores)
        } 