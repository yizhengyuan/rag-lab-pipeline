"""
概念合并模块
"""

import json
import logging
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict, Counter
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from .nodes import ConceptNode
from utils.validators import ConceptValidator
from utils.helpers import TextProcessor

logger = logging.getLogger(__name__)

class ConceptMerger:
    """概念合并器 - 使用 LlamaIndex 的嵌入和检索系统"""
    
    def __init__(self, config):
        """
        初始化概念合并器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.concept_nodes: List[ConceptNode] = []
        self.concept_index: Optional[VectorStoreIndex] = None
        self.similarity_threshold = config.get("concepts.similarity_threshold", 0.7)
        self.max_concepts = config.get("concepts.max_concepts", 10)
        ##self.min_concept_length = config.get("concepts.min_concept_length", 1)
        ##self.max_concept_length = config.get("concepts.max_concept_length", 5000)
        self.min_concept_length = 1
        self.max_concept_length = 5000
    
    def merge_document_concepts(self, chunk_nodes: List) -> List[ConceptNode]:
        """
        合并文档级别的概念
        
        Args:
            chunk_nodes: chunk 节点列表
            
        Returns:
            List[ConceptNode]: 合并后的概念节点列表
        """
        logger.info("开始合并文档级别概念")
        
        if not chunk_nodes:
            raise ValueError("chunk_nodes 不能为空")
        
        # 收集所有概念
        all_concepts, concept_to_chunks = self._collect_concepts_from_chunks(chunk_nodes)
        
        if not all_concepts:
            logger.warning("没有找到任何概念")
            return []
        
        # 预处理概念
        processed_concepts = self._preprocess_concepts(all_concepts)
        
        # 创建概念文档并使用 LlamaIndex 的嵌入系统
        concept_documents = self._create_concept_documents(processed_concepts)
        
        # 使用 LlamaIndex 创建概念索引
        concept_index = VectorStoreIndex.from_documents(concept_documents)
        
        # 聚类相似概念
        merged_concepts = self._cluster_concepts_with_llamaindex(
            processed_concepts, 
            concept_index,
            self.similarity_threshold
        )
        
        # 创建 ConceptNode 对象
        self.concept_nodes = self._create_concept_nodes(
            merged_concepts, 
            concept_to_chunks, 
            all_concepts
        )
        
        # 创建最终概念索引
        if self.concept_nodes:
            self.concept_index = VectorStoreIndex(self.concept_nodes)
        
        # 后处理：排序、过滤和限制数量
        self.concept_nodes = self._postprocess_concepts(self.concept_nodes)
        
        logger.info(f"完成概念合并: 得到 {len(self.concept_nodes)} 个文档概念")
        return self.concept_nodes
    
    def _collect_concepts_from_chunks(self, chunk_nodes: List) -> Tuple[List[str], Dict[str, List[str]]]:
        """从chunk节点中收集概念"""
        all_concepts = []
        concept_to_chunks = defaultdict(list)
        
        for node in chunk_nodes:
            concepts = node.metadata.get("concepts", [])
            chunk_id = node.metadata.get("chunk_id", node.node_id)
            logger.info(f"chunk_id: {chunk_id}")
            logger.info(f"concepts: {concepts}")
            
            logger.debug(f"从chunk {chunk_id} 获取到 {len(concepts)} 个概念")
            
            for concept in concepts:
                logger.info(f"concept: {concept}")
                if isinstance(concept, dict):
                    concept_text = concept.get("name", concept.get("text", str(concept)))
                else:
                    concept_text = str(concept)
                
                # 验证概念长度
                if self.min_concept_length <= len(concept_text) <= self.max_concept_length:
                    all_concepts.append(concept_text)
                    concept_to_chunks[concept_text].append(chunk_id)
        
        return all_concepts, dict(concept_to_chunks)
    
    def _preprocess_concepts(self, concepts: List[str]) -> List[str]:
        """预处理概念"""
        processed = []
        for concept in concepts:
            # 清理文本
            cleaned = TextProcessor.clean_text(concept)
            if cleaned and len(cleaned) >= self.min_concept_length:
                processed.append(cleaned)
        
        # 去重但保持顺序
        seen = set()
        unique_concepts = []
        for concept in processed:
            if concept.lower() not in seen:
                seen.add(concept.lower())
                unique_concepts.append(concept)
        
        return unique_concepts
    
    def _create_concept_documents(self, concepts: List[str]) -> List[Document]:
        """创建概念文档"""
        documents = []
        for i, concept in enumerate(concepts):
            # 提取关键词
            keywords = TextProcessor.extract_keywords(concept, top_k=5)
            
            doc = Document(
                text=concept,
                metadata={
                    "original_concept": concept,
                    "concept_id": f"concept_{i}",
                    "keywords": keywords,
                    "length": len(concept)
                }
            )
            documents.append(doc)
        
        return documents
    
    def _cluster_concepts_with_llamaindex(self, concepts: List[str], concept_index: VectorStoreIndex, threshold: float) -> List[Tuple[str, List[str]]]:
        """
        使用 LlamaIndex 的检索系统进行概念聚类
        
        Args:
            concepts: 概念列表
            concept_index: 概念索引
            threshold: 相似性阈值
            
        Returns:
            List[Tuple[str, List[str]]]: 合并后的概念和源概念列表
        """
        retriever = VectorIndexRetriever(
            index=concept_index,
            similarity_top_k=min(len(concepts), 20)  # 限制检索数量
        )
        
        merged_concepts = []
        used_concepts = set()
        
        # 按概念频率排序，优先处理高频概念
        concept_counts = Counter(concepts)
        sorted_concepts = sorted(set(concepts), key=lambda x: concept_counts[x], reverse=True)
        
        for concept in sorted_concepts:
            if concept in used_concepts:
                continue
            
            # 使用 LlamaIndex 检索相似概念
            try:
                similar_nodes = retriever.retrieve(concept)
                
                # 找到相似度超过阈值的概念
                similar_concepts = [concept]  # 包含自己
                
                for node_with_score in similar_nodes:
                    original_concept = node_with_score.node.metadata.get("original_concept")
                    if (original_concept and 
                        original_concept not in used_concepts and 
                        original_concept != concept and 
                        node_with_score.score and
                        node_with_score.score >= threshold):
                        similar_concepts.append(original_concept)
                
                # 合并概念文本
                merged_concept = self._merge_concept_texts_with_llm(similar_concepts)
                if merged_concept:
                    merged_concepts.append((merged_concept, similar_concepts))
                    # 标记为已使用
                    used_concepts.update(similar_concepts)
                
            except Exception as e:
                logger.warning(f"处理概念 '{concept}' 时出错: {e}")
                # 如果出错，至少保留原概念
                if concept not in used_concepts:
                    merged_concepts.append((concept, [concept]))
                    used_concepts.add(concept)
        
        return merged_concepts
    
    def _merge_concept_texts_with_llm(self, concepts: List[str]) -> str:
        """
        使用 LlamaIndex 的 LLM 接口合并概念
        
        Args:
            concepts: 概念列表
            
        Returns:
            str: 合并后的概念
        """
        if len(concepts) == 1:
            return concepts[0]
        
        # 如果概念很相似，选择最短的
        if len(concepts) <= 3:
            similarities = []
            for i, c1 in enumerate(concepts):
                for j, c2 in enumerate(concepts[i+1:], i+1):
                    sim = TextProcessor.calculate_similarity(c1, c2)
                    similarities.append(sim)
            
            if similarities and sum(similarities) / len(similarities) > 0.8:
                return min(concepts, key=len)
        
        prompt = f"""
        请将以下相似的概念合并成一个更通用、更简洁的概念表述：
        
        概念列表：
        {chr(10).join(f"- {concept}" for concept in concepts)}
        
        要求：
        1. 返回一个合并后的概念（不超过20个字）
        2. 保持概念的核心含义
        3. 使用最通用的表述
        4. 只返回合并后的概念，不要其他解释
        
        合并后的概念：
        """
        
        try:
            response = Settings.llm.complete(prompt)
            merged_concept = response.text.strip()
            
            # 验证合并结果
            if (merged_concept and 
                len(merged_concept) <= self.max_concept_length and
                len(merged_concept) >= self.min_concept_length):
                return merged_concept
            else:
                logger.warning(f"LLM合并结果不符合要求: {merged_concept}")
                return concepts[0]  # 返回第一个概念作为备选
                
        except Exception as e:
            logger.warning(f"概念合并失败: {e}")
            return concepts[0]
    
    def _create_concept_nodes(self, merged_concepts: List[Tuple[str, List[str]]], 
                            concept_to_chunks: Dict[str, List[str]], 
                            all_concepts: List[str]) -> List[ConceptNode]:
        """创建概念节点"""
        concept_nodes = []
        
        for i, (merged_concept, source_concepts) in enumerate(merged_concepts):
            # 收集相关的 chunks
            source_chunks = set()
            for concept in source_concepts:
                if concept in concept_to_chunks:
                    source_chunks.update(concept_to_chunks[concept])
            
            # 计算置信度分数（基于出现频率和chunk覆盖度）
            frequency_score = len(source_concepts) / len(all_concepts)
            coverage_score = len(source_chunks) / max(1, len(set().union(*concept_to_chunks.values())))
            confidence_score = (frequency_score + coverage_score) / 2
            
            # 提取关键词
            keywords = TextProcessor.extract_keywords(merged_concept, top_k=5)
            
            # 确定概念类别（简单分类）
            category = self._classify_concept(merged_concept)
            
            concept_node = ConceptNode(
                concept_text=merged_concept,
                concept_name=merged_concept[:50],  # 限制名称长度
                definition=merged_concept,  # 可以后续用LLM生成更详细的定义
                source_chunks=list(source_chunks),
                confidence_score=confidence_score,
                keywords=keywords,
                category=category,
                node_id=f"doc_concept_{i}"
            )
            
            concept_node.metadata.update({
                "concept_type": "document_level",
                "source_concepts": source_concepts,
                "confidence_score": confidence_score,
                "frequency": len(source_concepts),
                "chunk_coverage": len(source_chunks)
            })
            
            concept_nodes.append(concept_node)
        
        return concept_nodes
    
    def _classify_concept(self, concept: str) -> str:
        """简单的概念分类"""
        concept_lower = concept.lower()
        
        # 定义一些简单的分类规则
        if any(word in concept_lower for word in ['方法', '技术', '算法', '策略']):
            return "方法技术"
        elif any(word in concept_lower for word in ['理论', '原理', '定律', '公式']):
            return "理论原理"
        elif any(word in concept_lower for word in ['系统', '平台', '工具', '软件']):
            return "系统工具"
        elif any(word in concept_lower for word in ['数据', '信息', '知识']):
            return "数据信息"
        elif any(word in concept_lower for word in ['问题', '挑战', '困难']):
            return "问题挑战"
        else:
            return "其他"
    
    def _postprocess_concepts(self, concept_nodes: List[ConceptNode]) -> List[ConceptNode]:
        """后处理概念节点"""
        # 按置信度排序
        concept_nodes.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # 过滤低质量概念
        min_confidence = 0.1  # 最低置信度阈值
        filtered_nodes = [node for node in concept_nodes if node.confidence_score >= min_confidence]
        
        # 限制数量
        final_nodes = filtered_nodes[:self.max_concepts]
        
        logger.info(f"后处理完成: {len(concept_nodes)} -> {len(filtered_nodes)} -> {len(final_nodes)}")
        return final_nodes
    
    def get_concept_index(self) -> Optional[VectorStoreIndex]:
        """获取概念索引"""
        return self.concept_index
    
    def get_concept_by_id(self, concept_id: str) -> Optional[ConceptNode]:
        """根据ID获取概念节点"""
        for node in self.concept_nodes:
            if node.node_id == concept_id:
                return node
        return None
    
    def get_concepts_by_category(self, category: str) -> List[ConceptNode]:
        """根据类别获取概念节点"""
        return [node for node in self.concept_nodes if node.category == category]
    
    def get_top_concepts(self, top_k: int = 5) -> List[ConceptNode]:
        """获取置信度最高的概念"""
        return sorted(self.concept_nodes, key=lambda x: x.confidence_score, reverse=True)[:top_k]
    
    def export_concepts(self) -> List[Dict[str, Any]]:
        """导出概念为字典列表"""
        return [node.to_dict() for node in self.concept_nodes]
    
    def get_concept_statistics(self) -> Dict[str, Any]:
        """获取概念统计信息"""
        if not self.concept_nodes:
            return {}
        
        categories = [node.category for node in self.concept_nodes]
        category_counts = Counter(categories)
        
        confidence_scores = [node.confidence_score for node in self.concept_nodes]
        
        return {
            "total_concepts": len(self.concept_nodes),
            "categories": dict(category_counts),
            "avg_confidence": sum(confidence_scores) / len(confidence_scores),
            "max_confidence": max(confidence_scores),
            "min_confidence": min(confidence_scores)
        } 