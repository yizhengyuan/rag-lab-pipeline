"""
自定义节点类模块 - 扩展 LlamaIndex 的 TextNode
"""

import json
from typing import List, Dict, Any, Optional
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

class ConceptNode(TextNode):
    """扩展 LlamaIndex 的 TextNode 来表示概念"""
    
    def __init__(self, 
                 concept_text: str, 
                 concept_name: str = None,
                 definition: str = None,
                 source_chunks: List[str] = None, 
                 confidence_score: float = 0.0,
                 keywords: List[str] = None,
                 category: str = None,
                 **kwargs):
        """
        初始化概念节点
        
        Args:
            concept_text: 概念文本
            concept_name: 概念名称
            definition: 概念定义
            source_chunks: 来源 chunk 列表
            confidence_score: 置信度分数
            keywords: 关键词列表
            category: 概念类别
            **kwargs: 其他参数传递给父类
        """
        metadata = kwargs.get('metadata', {})
        metadata.update({
            "concept_name": concept_name or concept_text[:50],
            "definition": definition,
            "source_chunks": source_chunks or [],
            "confidence_score": confidence_score,
            "keywords": keywords or [],
            "category": category,
            "node_type": "concept"
        })
        kwargs['metadata'] = metadata
        
        super().__init__(text=concept_text, **kwargs)
    
    @property
    def concept_name(self) -> str:
        return self.metadata.get("concept_name", "")
    
    @property
    def definition(self) -> str:
        return self.metadata.get("definition", "")
    
    @property
    def source_chunks(self) -> List[str]:
        return self.metadata.get("source_chunks", [])
    
    @property
    def confidence_score(self) -> float:
        return self.metadata.get("confidence_score", 0.0)
    
    @property
    def keywords(self) -> List[str]:
        return self.metadata.get("keywords", [])
    
    @property
    def category(self) -> str:
        return self.metadata.get("category", "")
    
    def add_source_chunk(self, chunk: str) -> None:
        """添加来源chunk"""
        source_chunks = self.metadata.get("source_chunks", [])
        if chunk not in source_chunks:
            source_chunks.append(chunk)
            self.metadata["source_chunks"] = source_chunks
    
    def add_keyword(self, keyword: str) -> None:
        """添加关键词"""
        keywords = self.metadata.get("keywords", [])
        if keyword not in keywords:
            keywords.append(keyword)
            self.metadata["keywords"] = keywords
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.node_id,
            "concept_name": self.concept_name,
            "definition": self.definition,
            "text": self.text,
            "source_chunks": self.source_chunks,
            "confidence_score": self.confidence_score,
            "keywords": self.keywords,
            "category": self.category,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptNode':
        """从字典创建概念节点"""
        return cls(
            concept_text=data.get("text", ""),
            concept_name=data.get("concept_name"),
            definition=data.get("definition"),
            source_chunks=data.get("source_chunks", []),
            confidence_score=data.get("confidence_score", 0.0),
            keywords=data.get("keywords", []),
            category=data.get("category"),
            node_id=data.get("id")
        )

class EvidenceNode(TextNode):
    """扩展 LlamaIndex 的 TextNode 来表示证据"""
    
    def __init__(self, 
                 evidence_text: str, 
                 concept_id: str, 
                 concept_name: str = None,
                 relevance_score: float = 0.0,
                 evidence_type: str = "supporting",
                 source_document: str = None,
                 page_number: int = None,
                 **kwargs):
        """
        初始化证据节点
        
        Args:
            evidence_text: 证据文本
            concept_id: 关联的概念ID
            concept_name: 关联的概念名称
            relevance_score: 相关性分数
            evidence_type: 证据类型 (supporting, contradicting, neutral)
            source_document: 来源文档
            page_number: 页码
            **kwargs: 其他参数传递给父类
        """
        metadata = kwargs.get('metadata', {})
        metadata.update({
            "concept_id": concept_id,
            "concept_name": concept_name,
            "relevance_score": relevance_score,
            "evidence_type": evidence_type,
            "source_document": source_document,
            "page_number": page_number,
            "node_type": "evidence"
        })
        kwargs['metadata'] = metadata
        
        super().__init__(text=evidence_text, **kwargs)
        
        # 建立与概念节点的关系
        if concept_id:
            self.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=concept_id,
                metadata={"relationship_type": "supports_concept"}
            )
    
    @property
    def concept_id(self) -> str:
        return self.metadata.get("concept_id", "")
    
    @property
    def concept_name(self) -> str:
        return self.metadata.get("concept_name", "")
    
    @property
    def relevance_score(self) -> float:
        return self.metadata.get("relevance_score", 0.0)
    
    @property
    def evidence_type(self) -> str:
        return self.metadata.get("evidence_type", "supporting")
    
    @property
    def source_document(self) -> str:
        return self.metadata.get("source_document", "")
    
    @property
    def page_number(self) -> int:
        return self.metadata.get("page_number", None)
    
    def update_relevance_score(self, score: float) -> None:
        """更新相关性分数"""
        score = max(0.0, min(1.0, score))
        self.metadata["relevance_score"] = score
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.node_id,
            "text": self.text,
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "relevance_score": self.relevance_score,
            "evidence_type": self.evidence_type,
            "source_document": self.source_document,
            "page_number": self.page_number,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvidenceNode':
        """从字典创建证据节点"""
        return cls(
            evidence_text=data.get("text", ""),
            concept_id=data.get("concept_id", ""),
            concept_name=data.get("concept_name"),
            relevance_score=data.get("relevance_score", 0.0),
            evidence_type=data.get("evidence_type", "supporting"),
            source_document=data.get("source_document"),
            page_number=data.get("page_number"),
            node_id=data.get("id")
        )

class QANode(TextNode):
    """扩展 LlamaIndex 的 TextNode 来表示问答对"""
    
    def __init__(self,
                 question: str,
                 answer: str,
                 concept_id: str = None,
                 concept_name: str = None,
                 difficulty_level: str = "medium",
                 question_type: str = "factual",
                 evidence_ids: List[str] = None,
                 **kwargs):
        """
        初始化问答节点
        
        Args:
            question: 问题文本
            answer: 答案文本
            concept_id: 关联的概念ID
            concept_name: 关联的概念名称
            difficulty_level: 难度级别 (easy, medium, hard)
            question_type: 问题类型 (factual, analytical, creative)
            evidence_ids: 支持证据的ID列表
            **kwargs: 其他参数传递给父类
        """
        qa_text = f"Q: {question}\nA: {answer}"
        
        # 🔧 修复：将自定义属性存储在metadata中
        metadata = kwargs.get('metadata', {})
        metadata.update({
            "question": question,
            "answer": answer,
            "concept_id": concept_id,
            "concept_name": concept_name,
            "difficulty_level": difficulty_level,
            "question_type": question_type,
            "evidence_ids": evidence_ids or [],
            "node_type": "qa"
        })
        kwargs['metadata'] = metadata
        
        super().__init__(text=qa_text, **kwargs)
        
        # 问答特有属性
        self.question = question
        self.answer = answer
        self.concept_id = concept_id
        self.concept_name = concept_name
        self.difficulty_level = difficulty_level
        self.question_type = question_type
        self.evidence_ids = evidence_ids or []
    
    def add_evidence_id(self, evidence_id: str) -> None:
        """添加支持证据ID"""
        if evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence_id)
            self.metadata["evidence_ids"] = self.evidence_ids
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.node_id,
            "question": self.question,
            "answer": self.answer,
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "difficulty_level": self.difficulty_level,
            "question_type": self.question_type,
            "evidence_ids": self.evidence_ids,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QANode':
        """从字典创建问答节点"""
        return cls(
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            concept_id=data.get("concept_id"),
            concept_name=data.get("concept_name"),
            difficulty_level=data.get("difficulty_level", "medium"),
            question_type=data.get("question_type", "factual"),
            evidence_ids=data.get("evidence_ids", []),
            node_id=data.get("id")
        )

# 节点工厂类
class NodeFactory:
    """节点工厂类，用于创建不同类型的节点"""
    
    @staticmethod
    def create_concept_node(concept_data: Dict[str, Any]) -> ConceptNode:
        """创建概念节点"""
        return ConceptNode.from_dict(concept_data)
    
    @staticmethod
    def create_evidence_node(evidence_data: Dict[str, Any]) -> EvidenceNode:
        """创建证据节点"""
        return EvidenceNode.from_dict(evidence_data)
    
    @staticmethod
    def create_qa_node(qa_data: Dict[str, Any]) -> QANode:
        """创建问答节点"""
        return QANode.from_dict(qa_data)
    
    @staticmethod
    def create_node_from_type(node_type: str, data: Dict[str, Any]) -> TextNode:
        """根据类型创建节点"""
        if node_type == "concept":
            return NodeFactory.create_concept_node(data)
        elif node_type == "evidence":
            return NodeFactory.create_evidence_node(data)
        elif node_type == "qa":
            return NodeFactory.create_qa_node(data)
        else:
            raise ValueError(f"不支持的节点类型: {node_type}") 