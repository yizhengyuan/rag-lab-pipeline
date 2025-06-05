"""
自定义节点类模块 - 扩展 LlamaIndex 的 TextNode
"""

from typing import List
from llama_index.core.schema import TextNode

class ConceptNode(TextNode):
    """扩展 LlamaIndex 的 TextNode 来表示概念"""
    
    def __init__(self, 
                 concept_text: str, 
                 source_chunks: List[str] = None, 
                 confidence_score: float = 0.0, 
                 **kwargs):
        """
        初始化概念节点
        
        Args:
            concept_text: 概念文本
            source_chunks: 来源 chunk 列表
            confidence_score: 置信度分数
            **kwargs: 其他参数传递给父类
        """
        super().__init__(text=concept_text, **kwargs)
        self.source_chunks = source_chunks or []
        self.confidence_score = confidence_score
        self.node_type = "concept"

class EvidenceNode(TextNode):
    """扩展 LlamaIndex 的 TextNode 来表示证据"""
    
    def __init__(self, 
                 evidence_text: str, 
                 concept_id: str, 
                 relevance_score: float = 0.0, 
                 **kwargs):
        """
        初始化证据节点
        
        Args:
            evidence_text: 证据文本
            concept_id: 关联的概念ID
            relevance_score: 相关性分数
            **kwargs: 其他参数传递给父类
        """
        super().__init__(text=evidence_text, **kwargs)
        self.concept_id = concept_id
        self.relevance_score = relevance_score
        self.node_type = "evidence" 