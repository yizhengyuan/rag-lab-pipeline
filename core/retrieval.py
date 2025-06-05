"""
概念检索模块
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryType
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from .nodes import ConceptNode, EvidenceNode
from utils.helpers import TextProcessor

logger = logging.getLogger(__name__)

class ConceptRetriever:
    """概念检索器 - 使用 LlamaIndex 的检索系统"""
    
    def __init__(self, config):
        """
        初始化概念检索器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.top_k = config.get("retrieval.top_k", 5)
        self.similarity_cutoff = config.get("retrieval.similarity_cutoff", 0.7)
        self.use_rerank = config.get("retrieval.rerank", True)
        self.alpha = config.get("retrieval.alpha", 0.5)  # 混合检索权重
        
        # 初始化后处理器
        self.similarity_postprocessor = SimilarityPostprocessor(
            similarity_cutoff=self.similarity_cutoff
        )
        
        # 如果启用重排序，初始化LLM重排序器
        if self.use_rerank:
            try:
                self.llm_rerank = LLMRerank(top_n=self.top_k)
            except Exception as e:
                logger.warning(f"LLM重排序器初始化失败: {e}")
                self.llm_rerank = None
        else:
            self.llm_rerank = None
    
    def retrieve_chunks_for_concepts(self, 
                                   concept_nodes: List[ConceptNode], 
                                   chunk_index: VectorStoreIndex) -> Dict[str, List[NodeWithScore]]:
        """
        为每个概念检索相关的文档块
        
        Args:
            concept_nodes: 概念节点列表
            chunk_index: chunk 向量索引
            
        Returns:
            Dict[str, List[NodeWithScore]]: 概念ID到相关chunks的映射
        """
        logger.info("开始基于概念的文档检索")
        
        if not concept_nodes:
            logger.warning("概念节点列表为空")
            return {}
        
        if not chunk_index:
            raise ValueError("chunk_index 不能为空")
        
        # 使用 LlamaIndex 的检索器
        retriever = VectorIndexRetriever(
            index=chunk_index,
            similarity_top_k=self.top_k * 2  # 检索更多候选，后续过滤
        )
        
        concept_to_chunks = {}
        
        for concept_node in concept_nodes:
            logger.info(f"正在检索概念: {concept_node.concept_name}")
            
            try:
                # 构建增强查询
                enhanced_query = self._build_enhanced_query(concept_node)
                
                # 使用概念进行检索
                retrieved_nodes = retriever.retrieve(enhanced_query)
                
                # 应用后处理器链
                filtered_nodes = self._apply_postprocessors(
                    retrieved_nodes, 
                    concept_node.text
                )
                
                # 计算概念特定的相关性分数
                scored_nodes = self._calculate_concept_relevance(
                    filtered_nodes, 
                    concept_node
                )
                
                concept_to_chunks[concept_node.node_id] = scored_nodes
                
                logger.debug(f"概念 '{concept_node.concept_name}' 检索到 {len(scored_nodes)} 个相关chunks")
                
            except Exception as e:
                logger.warning(f"检索概念 '{concept_node.concept_name}' 时出错: {e}")
                concept_to_chunks[concept_node.node_id] = []
        
        total_retrieved = sum(len(chunks) for chunks in concept_to_chunks.values())
        logger.info(f"完成概念检索: 为 {len(concept_nodes)} 个概念检索了 {total_retrieved} 个相关chunks")
        
        return concept_to_chunks
    
    def _build_enhanced_query(self, concept_node: ConceptNode) -> str:
        """构建增强查询"""
        query_parts = [concept_node.text]
        
        # 添加关键词
        if concept_node.keywords:
            query_parts.extend(concept_node.keywords[:3])  # 只取前3个关键词
        
        # 添加定义（如果有）
        if concept_node.definition and concept_node.definition != concept_node.text:
            query_parts.append(concept_node.definition[:100])  # 限制长度
        
        return " ".join(query_parts)
    
    def _apply_postprocessors(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        """应用后处理器链"""
        # 1. 相似度过滤
        filtered_nodes = self.similarity_postprocessor.postprocess_nodes(nodes)
        
        # 2. LLM重排序（如果启用）
        if self.llm_rerank and filtered_nodes:
            try:
                query_bundle = QueryBundle(query_str=query)
                reranked_nodes = self.llm_rerank.postprocess_nodes(
                    filtered_nodes, 
                    query_bundle
                )
                return reranked_nodes
            except Exception as e:
                logger.warning(f"LLM重排序失败: {e}")
        
        return filtered_nodes[:self.top_k]
    
    def _calculate_concept_relevance(self, nodes: List[NodeWithScore], concept_node: ConceptNode) -> List[NodeWithScore]:
        """计算概念特定的相关性分数"""
        enhanced_nodes = []
        
        for node in nodes:
            # 原始相似度分数
            original_score = node.score if node.score else 0.0
            
            # 计算关键词匹配分数
            keyword_score = self._calculate_keyword_match_score(
                node.node.text, 
                concept_node.keywords
            )
            
            # 计算文本相似度分数
            text_similarity = TextProcessor.calculate_similarity(
                node.node.text, 
                concept_node.text
            )
            
            # 综合分数
            final_score = (
                self.alpha * original_score + 
                (1 - self.alpha) * (keyword_score + text_similarity) / 2
            )
            
            # 更新分数
            node.score = final_score
            enhanced_nodes.append(node)
        
        # 按新分数排序
        enhanced_nodes.sort(key=lambda x: x.score, reverse=True)
        return enhanced_nodes
    
    def _calculate_keyword_match_score(self, text: str, keywords: List[str]) -> float:
        """计算关键词匹配分数"""
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return matches / len(keywords)
    
    def retrieve_similar_concepts(self, 
                                query_concept: str, 
                                concept_index: VectorStoreIndex, 
                                top_k: int = None,
                                exclude_ids: List[str] = None) -> List[NodeWithScore]:
        """
        检索与查询概念相似的概念
        
        Args:
            query_concept: 查询概念
            concept_index: 概念向量索引
            top_k: 返回的相似概念数量
            exclude_ids: 要排除的概念ID列表
            
        Returns:
            List[NodeWithScore]: 相似概念列表
        """
        if top_k is None:
            top_k = self.top_k
        
        retriever = VectorIndexRetriever(
            index=concept_index,
            similarity_top_k=top_k * 2  # 检索更多以便过滤
        )
        
        try:
            similar_concepts = retriever.retrieve(query_concept)
            
            # 过滤排除的ID
            if exclude_ids:
                similar_concepts = [
                    node for node in similar_concepts 
                    if node.node.node_id not in exclude_ids
                ]
            
            # 限制返回数量
            similar_concepts = similar_concepts[:top_k]
            
            logger.debug(f"为概念 '{query_concept}' 找到 {len(similar_concepts)} 个相似概念")
            return similar_concepts
            
        except Exception as e:
            logger.warning(f"检索相似概念时出错: {e}")
            return []
    
    def hybrid_retrieve(self, 
                       query: str, 
                       vector_index: VectorStoreIndex,
                       keyword_boost: Dict[str, float] = None) -> List[NodeWithScore]:
        """
        混合检索：结合向量检索和关键词匹配
        
        Args:
            query: 查询字符串
            vector_index: 向量索引
            keyword_boost: 关键词权重提升
            
        Returns:
            List[NodeWithScore]: 检索结果
        """
        # 向量检索
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self.top_k * 2
        )
        
        vector_results = vector_retriever.retrieve(query)
        
        # 如果有关键词提升，应用权重调整
        if keyword_boost:
            for node in vector_results:
                boost_factor = 1.0
                node_text_lower = node.node.text.lower()
                
                for keyword, weight in keyword_boost.items():
                    if keyword.lower() in node_text_lower:
                        boost_factor += weight
                
                if node.score:
                    node.score *= boost_factor
        
        # 重新排序并限制数量
        vector_results.sort(key=lambda x: x.score or 0, reverse=True)
        return vector_results[:self.top_k]
    
    def batch_retrieve(self, 
                      queries: List[str], 
                      index: VectorStoreIndex,
                      parallel: bool = False) -> Dict[str, List[NodeWithScore]]:
        """
        批量检索
        
        Args:
            queries: 查询列表
            index: 向量索引
            parallel: 是否并行处理
            
        Returns:
            Dict[str, List[NodeWithScore]]: 查询到结果的映射
        """
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.top_k
        )
        
        results = {}
        
        if parallel:
            # 简单的并行处理（可以用ThreadPoolExecutor优化）
            import concurrent.futures
            
            def retrieve_single(query):
                try:
                    return query, retriever.retrieve(query)
                except Exception as e:
                    logger.warning(f"批量检索查询 '{query}' 时出错: {e}")
                    return query, []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_query = {executor.submit(retrieve_single, query): query for query in queries}
                for future in concurrent.futures.as_completed(future_to_query):
                    query, result = future.result()
                    results[query] = result
        else:
            # 串行处理
            for query in queries:
                try:
                    results[query] = retriever.retrieve(query)
                except Exception as e:
                    logger.warning(f"批量检索查询 '{query}' 时出错: {e}")
                    results[query] = []
        
        return results
    
    def retrieve_with_filters(self, 
                            query: str, 
                            index: VectorStoreIndex,
                            metadata_filters: Dict[str, Any] = None,
                            score_threshold: float = None) -> List[NodeWithScore]:
        """
        带过滤条件的检索
        
        Args:
            query: 查询字符串
            index: 向量索引
            metadata_filters: 元数据过滤条件
            score_threshold: 分数阈值
            
        Returns:
            List[NodeWithScore]: 过滤后的检索结果
        """
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.top_k * 2
        )
        
        results = retriever.retrieve(query)
        
        # 应用元数据过滤
        if metadata_filters:
            filtered_results = []
            for node in results:
                metadata = node.node.metadata or {}
                match = True
                
                for key, value in metadata_filters.items():
                    if key not in metadata or metadata[key] != value:
                        match = False
                        break
                
                if match:
                    filtered_results.append(node)
            
            results = filtered_results
        
        # 应用分数阈值
        if score_threshold is not None:
            results = [node for node in results if (node.score or 0) >= score_threshold]
        
        return results[:self.top_k]
    
    def create_query_engine(self, index: VectorStoreIndex) -> RetrieverQueryEngine:
        """
        创建查询引擎
        
        Args:
            index: 向量索引
            
        Returns:
            RetrieverQueryEngine: 查询引擎
        """
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.top_k
        )
        
        # 配置后处理器
        postprocessors = [self.similarity_postprocessor]
        if self.llm_rerank:
            postprocessors.append(self.llm_rerank)
        
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=postprocessors
        )
        
        return query_engine
    
    def get_retrieval_statistics(self, concept_to_chunks: Dict[str, List[NodeWithScore]]) -> Dict[str, Any]:
        """
        获取检索统计信息
        
        Args:
            concept_to_chunks: 概念到chunks的映射
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not concept_to_chunks:
            return {"total_concepts": 0, "total_chunks": 0}
        
        chunk_counts = [len(chunks) for chunks in concept_to_chunks.values()]
        scores = []
        for chunks in concept_to_chunks.values():
            scores.extend([chunk.score for chunk in chunks if chunk.score is not None])
        
        stats = {
            "total_concepts": len(concept_to_chunks),
            "total_chunks": sum(chunk_counts),
            "avg_chunks_per_concept": sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0,
            "min_chunks_per_concept": min(chunk_counts) if chunk_counts else 0,
            "max_chunks_per_concept": max(chunk_counts) if chunk_counts else 0,
            "concepts_with_results": len([c for c in chunk_counts if c > 0]),
            "concepts_without_results": len([c for c in chunk_counts if c == 0])
        }
        
        if scores:
            stats.update({
                "avg_similarity_score": np.mean(scores),
                "median_similarity_score": np.median(scores),
                "min_similarity_score": min(scores),
                "max_similarity_score": max(scores),
                "std_similarity_score": np.std(scores)
            })
        
        return stats
    
    def analyze_retrieval_quality(self, concept_to_chunks: Dict[str, List[NodeWithScore]]) -> Dict[str, Any]:
        """
        分析检索质量
        
        Args:
            concept_to_chunks: 概念到chunks的映射
            
        Returns:
            Dict[str, Any]: 质量分析结果
        """
        stats = self.get_retrieval_statistics(concept_to_chunks)
        
        # 计算质量指标
        quality_metrics = {
            "coverage_rate": stats.get("concepts_with_results", 0) / max(1, stats.get("total_concepts", 1)),
            "avg_results_per_concept": stats.get("avg_chunks_per_concept", 0),
            "score_distribution": {}
        }
        
        # 分析分数分布
        all_scores = []
        for chunks in concept_to_chunks.values():
            all_scores.extend([chunk.score for chunk in chunks if chunk.score is not None])
        
        if all_scores:
            score_ranges = {
                "high (>0.8)": len([s for s in all_scores if s > 0.8]),
                "medium (0.5-0.8)": len([s for s in all_scores if 0.5 <= s <= 0.8]),
                "low (<0.5)": len([s for s in all_scores if s < 0.5])
            }
            quality_metrics["score_distribution"] = score_ranges
        
        return {**stats, **quality_metrics}
    
    def export_retrieval_results(self, concept_to_chunks: Dict[str, List[NodeWithScore]]) -> List[Dict[str, Any]]:
        """
        导出检索结果
        
        Args:
            concept_to_chunks: 概念到chunks的映射
            
        Returns:
            List[Dict[str, Any]]: 导出的结果列表
        """
        results = []
        
        for concept_id, chunks in concept_to_chunks.items():
            for i, chunk in enumerate(chunks):
                result = {
                    "concept_id": concept_id,
                    "chunk_rank": i + 1,
                    "chunk_id": chunk.node.node_id,
                    "chunk_text": chunk.node.text,
                    "similarity_score": chunk.score,
                    "chunk_metadata": chunk.node.metadata
                }
                results.append(result)
        
        return results 