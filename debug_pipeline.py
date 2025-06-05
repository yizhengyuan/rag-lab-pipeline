"""
Pipeline 调试工具 - 展示所有中间步骤的详细结果
=======================================================

功能：
1. 详细记录每个步骤的输入输出
2. 保存中间结果到可读格式
3. 生成可视化报告
4. 提供质量评估指标
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback
from pathlib import Path

# 导入主要的 Pipeline 类
from integrated_pipeline_new import ModularIntegratedPipeline
from llama_index.core import Document
from debug_visualizer import PipelineVisualizer

logger = logging.getLogger(__name__)

class PipelineDebugger:
    """Pipeline 调试器 - 详细记录每个步骤"""
    
    def __init__(self, 
                 config_path: str = "config/config.yml",
                 debug_output_dir: str = "./debug_output"):
        """
        初始化调试器
        
        Args:
            config_path: 配置文件路径
            debug_output_dir: 调试输出目录
        """
        self.debug_output_dir = debug_output_dir
        self.config_path = config_path
        
        # 创建调试输出目录
        os.makedirs(debug_output_dir, exist_ok=True)
        
        # 初始化主 Pipeline
        self.pipeline = ModularIntegratedPipeline(
            config_path=config_path,
            output_dir=debug_output_dir
        )
        
        # 初始化可视化器
        self.visualizer = PipelineVisualizer(debug_output_dir)
        
        # 调试信息存储
        self.debug_info = {
            "steps": [],
            "timing": {},
            "quality_metrics": {},
            "errors": []
        }
        
        logger.info(f"🔍 Pipeline 调试器已初始化")
        logger.info(f"📁 调试输出目录: {debug_output_dir}")
    
    def debug_full_pipeline(self, file_path: str) -> Dict[str, Any]:
        """
        调试完整的 Pipeline 流程
        
        Args:
            file_path: 输入文件路径
            
        Returns:
            包含所有调试信息的字典
        """
        start_time = datetime.now()
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        
        logger.info(f"🚀 开始调试 Pipeline: {file_path}")
        logger.info("=" * 80)
        
        try:
            # 步骤1: 文档加载调试
            step1_result = self._debug_document_loading(file_path)
            
            # 步骤2: 文档分块调试
            step2_result = self._debug_chunking(step1_result["documents"])
            
            # 步骤3: Embedding 调试
            step3_result = self._debug_embedding(step2_result["chunks"])
            
            # 步骤4: Vector Store 调试
            step4_result = self._debug_vector_store(step3_result["embeddings"], step2_result["chunks"])
            
            # 步骤5: 概念提取调试
            step5_result = self._debug_concept_extraction(step2_result["chunks"])
            
            # 步骤6: 概念合并调试
            step6_result = self._debug_concept_merging(step5_result["concepts"])
            
            # 步骤7: 证据提取调试
            step7_result = self._debug_evidence_extraction(step6_result["merged_concepts"], step2_result["chunks"])
            
            # 步骤8: 检索调试
            step8_result = self._debug_retrieval(step6_result["merged_concepts"], step4_result["vector_store"])
            
            # 步骤9: 问答生成调试
            step9_result = self._debug_qa_generation(step1_result["full_text"])
            
            # 汇总所有结果
            debug_results = {
                "file_info": {
                    "file_path": file_path,
                    "file_name": file_name,
                    "base_name": base_name,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                },
                "step_results": {
                    "document_loading": step1_result,
                    "chunking": step2_result,
                    "embedding": step3_result,
                    "vector_store": step4_result,
                    "concept_extraction": step5_result,
                    "concept_merging": step6_result,
                    "evidence_extraction": step7_result,
                    "retrieval": step8_result,
                    "qa_generation": step9_result
                },
                "debug_info": self.debug_info,
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存调试结果
            self._save_debug_results(debug_results, base_name)
            
            # 生成调试报告
            report_path, html_path = self._generate_debug_report(debug_results, base_name)
            
            logger.info(f"✅ Pipeline 调试完成: {file_path}")
            return debug_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline 调试失败: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.debug_info["errors"].append({
                "step": "full_pipeline",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            })
            return {"error": str(e), "debug_info": self.debug_info}
    
    def _debug_document_loading(self, file_path: str) -> Dict[str, Any]:
        """调试文档加载步骤"""
        step_start = datetime.now()
        logger.info("📄 步骤1: 调试文档加载...")
        
        try:
            # 加载文档
            documents = self.pipeline._load_document(file_path)
            full_text = "\n\n".join([doc.text for doc in documents])
            
            result = {
                "success": True,
                "documents_count": len(documents),
                "total_characters": len(full_text),
                "documents": documents,
                "full_text": full_text,
                "document_details": [
                    {
                        "index": i,
                        "text_length": len(doc.text),
                        "text_preview": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
                        "metadata": doc.metadata
                    }
                    for i, doc in enumerate(documents)
                ],
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            
            # 保存文档内容
            self._save_step_result("01_document_loading", result, exclude_keys=["documents", "full_text"])
            
            logger.info(f"   ✅ 文档加载完成: {len(documents)} 个文档, {len(full_text)} 字符")
            return result
            
        except Exception as e:
            logger.error(f"   ❌ 文档加载失败: {str(e)}")
            result = {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            self._save_step_result("01_document_loading", result)
            return result
    
    def _debug_chunking(self, documents: List[Document]) -> Dict[str, Any]:
        """调试文档分块步骤"""
        step_start = datetime.now()
        logger.info("✂️ 步骤2: 调试文档分块...")
        
        try:
            # 使用 Pipeline 的分块器
            chunker = self.pipeline.concept_pipeline.chunker
            chunks = []
            
            for doc in documents:
                doc_chunks = chunker.chunk_document(doc)
                chunks.extend(doc_chunks)
            
            result = {
                "success": True,
                "chunks_count": len(chunks),
                "chunks": chunks,
                "chunk_details": [
                    {
                        "index": i,
                        "node_id": chunk.node_id,
                        "text_length": len(chunk.text),
                        "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                        "metadata": chunk.metadata
                    }
                    for i, chunk in enumerate(chunks)
                ],
                "chunking_stats": {
                    "avg_chunk_length": sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0,
                    "min_chunk_length": min(len(chunk.text) for chunk in chunks) if chunks else 0,
                    "max_chunk_length": max(len(chunk.text) for chunk in chunks) if chunks else 0
                },
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            
            self._save_step_result("02_chunking", result, exclude_keys=["chunks"])
            
            logger.info(f"   ✅ 文档分块完成: {len(chunks)} 个块")
            return result
            
        except Exception as e:
            logger.error(f"   ❌ 文档分块失败: {str(e)}")
            result = {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            self._save_step_result("02_chunking", result)
            return result
    
    def _debug_embedding(self, chunks: List) -> Dict[str, Any]:
        """调试 Embedding 生成步骤"""
        step_start = datetime.now()
        logger.info("🔢 步骤3: 调试 Embedding 生成...")
        
        try:
            # 使用 Pipeline 的 embedding 模型
            embed_model = self.pipeline.concept_pipeline.vector_store_manager.embed_model
            
            embeddings = []
            embedding_details = []
            
            for i, chunk in enumerate(chunks[:5]):  # 只处理前5个块作为示例
                embedding = embed_model.get_text_embedding(chunk.text)
                embeddings.append(embedding)
                
                embedding_details.append({
                    "chunk_index": i,
                    "chunk_id": chunk.node_id,
                    "embedding_dimension": len(embedding),
                    "embedding_preview": embedding[:10],  # 前10个维度
                    "embedding_norm": sum(x*x for x in embedding) ** 0.5
                })
            
            result = {
                "success": True,
                "embeddings_count": len(embeddings),
                "embeddings": embeddings,
                "embedding_details": embedding_details,
                "embedding_stats": {
                    "dimension": len(embeddings[0]) if embeddings else 0,
                    "avg_norm": sum(sum(x*x for x in emb) ** 0.5 for emb in embeddings) / len(embeddings) if embeddings else 0
                },
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            
            self._save_step_result("03_embedding", result, exclude_keys=["embeddings"])
            
            logger.info(f"   ✅ Embedding 生成完成: {len(embeddings)} 个向量")
            return result
            
        except Exception as e:
            logger.error(f"   ❌ Embedding 生成失败: {str(e)}")
            result = {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            self._save_step_result("03_embedding", result)
            return result
    
    def _debug_vector_store(self, embeddings: List, chunks: List) -> Dict[str, Any]:
        """调试 Vector Store 构建步骤"""
        step_start = datetime.now()
        logger.info("🗄️ 步骤4: 调试 Vector Store 构建...")
        
        try:
            # 使用 Pipeline 的 vector store manager
            vector_store_manager = self.pipeline.concept_pipeline.vector_store_manager
            
            # 构建向量存储
            vector_store = vector_store_manager.create_vector_store(chunks)
            
            result = {
                "success": True,
                "vector_store_type": type(vector_store).__name__,
                "stored_nodes_count": len(chunks),
                "vector_store": vector_store,
                "vector_store_stats": {
                    "total_vectors": len(chunks),
                    "vector_dimension": len(embeddings[0]) if embeddings else 0
                },
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            
            self._save_step_result("04_vector_store", result, exclude_keys=["vector_store"])
            
            logger.info(f"   ✅ Vector Store 构建完成: {len(chunks)} 个向量")
            return result
            
        except Exception as e:
            logger.error(f"   ❌ Vector Store 构建失败: {str(e)}")
            result = {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            self._save_step_result("04_vector_store", result)
            return result
    
    def _debug_concept_extraction(self, chunks: List) -> Dict[str, Any]:
        """调试概念提取步骤"""
        step_start = datetime.now()
        logger.info("🧠 步骤5: 调试概念提取...")
        
        try:
            # 使用 Pipeline 的概念提取器
            concept_extractor = self.pipeline.concept_pipeline.concept_extractor
            
            concepts = []
            concept_details = []
            
            for i, chunk in enumerate(chunks):
                chunk_concepts = concept_extractor.extract_concepts_from_chunk(chunk)
                concepts.extend(chunk_concepts)
                
                concept_details.append({
                    "chunk_index": i,
                    "chunk_id": chunk.node_id,
                    "concepts_count": len(chunk_concepts),
                    "concepts": [
                        {
                            "concept_id": concept.node_id,
                            "concept_text": concept.text[:100] + "..." if len(concept.text) > 100 else concept.text,
                            "confidence": getattr(concept, 'confidence_score', 0.0)
                        }
                        for concept in chunk_concepts
                    ]
                })
            
            result = {
                "success": True,
                "concepts_count": len(concepts),
                "concepts": concepts,
                "concept_details": concept_details,
                "concept_stats": {
                    "avg_concepts_per_chunk": len(concepts) / len(chunks) if chunks else 0,
                    "total_concepts": len(concepts)
                },
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            
            self._save_step_result("05_concept_extraction", result, exclude_keys=["concepts"])
            
            logger.info(f"   ✅ 概念提取完成: {len(concepts)} 个概念")
            return result
            
        except Exception as e:
            logger.error(f"   ❌ 概念提取失败: {str(e)}")
            result = {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            self._save_step_result("05_concept_extraction", result)
            return result
    
    def _debug_concept_merging(self, concepts: List) -> Dict[str, Any]:
        """调试概念合并步骤"""
        step_start = datetime.now()
        logger.info("🔗 步骤6: 调试概念合并...")
        
        try:
            # 使用 Pipeline 的概念合并器
            concept_merger = self.pipeline.concept_pipeline.concept_merger
            
            merged_concepts = concept_merger.merge_similar_concepts(concepts)
            
            result = {
                "success": True,
                "original_concepts_count": len(concepts),
                "merged_concepts_count": len(merged_concepts),
                "merged_concepts": merged_concepts,
                "merge_stats": {
                    "reduction_ratio": (len(concepts) - len(merged_concepts)) / len(concepts) if concepts else 0,
                    "concepts_before": len(concepts),
                    "concepts_after": len(merged_concepts)
                },
                "merge_details": [
                    {
                        "concept_id": concept.node_id,
                        "concept_text": concept.text[:100] + "..." if len(concept.text) > 100 else concept.text,
                        "source_chunks": getattr(concept, 'source_chunks', []),
                        "confidence": getattr(concept, 'confidence_score', 0.0)
                    }
                    for concept in merged_concepts
                ],
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            
            self._save_step_result("06_concept_merging", result, exclude_keys=["merged_concepts"])
            
            logger.info(f"   ✅ 概念合并完成: {len(concepts)} -> {len(merged_concepts)} 个概念")
            return result
            
        except Exception as e:
            logger.error(f"   ❌ 概念合并失败: {str(e)}")
            result = {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            self._save_step_result("06_concept_merging", result)
            return result
    
    def _debug_evidence_extraction(self, concepts: List, chunks: List) -> Dict[str, Any]:
        """调试证据提取步骤"""
        step_start = datetime.now()
        logger.info("🔍 步骤7: 调试证据提取...")
        
        try:
            # 使用 Pipeline 的证据提取器
            evidence_extractor = self.pipeline.concept_pipeline.evidence_extractor
            
            evidence_nodes = []
            evidence_details = []
            
            for concept in concepts:
                concept_evidence = evidence_extractor.extract_evidence_for_concept(concept, chunks)
                evidence_nodes.extend(concept_evidence)
                
                evidence_details.append({
                    "concept_id": concept.node_id,
                    "concept_text": concept.text[:100] + "..." if len(concept.text) > 100 else concept.text,
                    "evidence_count": len(concept_evidence),
                    "evidence": [
                        {
                            "evidence_id": evidence.node_id,
                            "evidence_text": evidence.text[:100] + "..." if len(evidence.text) > 100 else evidence.text,
                            "relevance_score": getattr(evidence, 'relevance_score', 0.0)
                        }
                        for evidence in concept_evidence
                    ]
                })
            
            result = {
                "success": True,
                "evidence_count": len(evidence_nodes),
                "evidence_nodes": evidence_nodes,
                "evidence_details": evidence_details,
                "evidence_stats": {
                    "avg_evidence_per_concept": len(evidence_nodes) / len(concepts) if concepts else 0,
                    "total_evidence": len(evidence_nodes)
                },
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            
            self._save_step_result("07_evidence_extraction", result, exclude_keys=["evidence_nodes"])
            
            logger.info(f"   ✅ 证据提取完成: {len(evidence_nodes)} 个证据")
            return result
            
        except Exception as e:
            logger.error(f"   ❌ 证据提取失败: {str(e)}")
            result = {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            self._save_step_result("07_evidence_extraction", result)
            return result
    
    def _debug_retrieval(self, concepts: List, vector_store) -> Dict[str, Any]:
        """调试检索步骤"""
        step_start = datetime.now()
        logger.info("🎯 步骤8: 调试检索...")
        
        try:
            # 使用 Pipeline 的检索器
            retriever = self.pipeline.concept_pipeline.retriever
            
            retrieval_results = []
            
            # 为每个概念执行检索
            for concept in concepts[:3]:  # 只测试前3个概念
                query_results = retriever.retrieve_for_concept(concept, vector_store, top_k=5)
                
                retrieval_results.append({
                    "concept_id": concept.node_id,
                    "concept_text": concept.text[:100] + "..." if len(concept.text) > 100 else concept.text,
                    "retrieved_count": len(query_results),
                    "retrieved_chunks": [
                        {
                            "chunk_id": result.node.node_id,
                            "chunk_text": result.node.text[:100] + "..." if len(result.node.text) > 100 else result.node.text,
                            "similarity_score": result.score
                        }
                        for result in query_results
                    ]
                })
            
            result = {
                "success": True,
                "retrieval_results": retrieval_results,
                "retrieval_stats": {
                    "concepts_tested": len(retrieval_results),
                    "avg_retrieved_per_concept": sum(len(r["retrieved_chunks"]) for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0
                },
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            
            self._save_step_result("08_retrieval", result)
            
            logger.info(f"   ✅ 检索测试完成: 测试了 {len(retrieval_results)} 个概念")
            return result
            
        except Exception as e:
            logger.error(f"   ❌ 检索测试失败: {str(e)}")
            result = {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            self._save_step_result("08_retrieval", result)
            return result
    
    def _debug_qa_generation(self, full_text: str) -> Dict[str, Any]:
        """调试问答生成步骤"""
        step_start = datetime.now()
        logger.info("❓ 步骤9: 调试问答生成...")
        
        try:
            # 使用 Pipeline 的问答生成器
            qa_generator = self.pipeline.qa_generator
            
            qa_pairs = qa_generator.generate_qa_pairs_from_text(full_text)
            
            result = {
                "success": True,
                "qa_pairs_count": len(qa_pairs),
                "qa_pairs": qa_pairs,
                "qa_details": [
                    {
                        "index": i,
                        "question": qa_pair.get("question", ""),
                        "answer": qa_pair.get("answer", "")[:200] + "..." if len(qa_pair.get("answer", "")) > 200 else qa_pair.get("answer", ""),
                        "question_type": qa_pair.get("question_type", ""),
                        "difficulty": qa_pair.get("difficulty", "")
                    }
                    for i, qa_pair in enumerate(qa_pairs)
                ],
                "qa_stats": {
                    "total_qa_pairs": len(qa_pairs),
                    "question_types": list(set(qa.get("question_type", "") for qa in qa_pairs)),
                    "avg_question_length": sum(len(qa.get("question", "")) for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0,
                    "avg_answer_length": sum(len(qa.get("answer", "")) for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0
                },
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            
            self._save_step_result("09_qa_generation", result)
            
            logger.info(f"   ✅ 问答生成完成: {len(qa_pairs)} 个问答对")
            return result
            
        except Exception as e:
            logger.error(f"   ❌ 问答生成失败: {str(e)}")
            result = {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - step_start).total_seconds()
            }
            self._save_step_result("09_qa_generation", result)
            return result
    
    def _save_step_result(self, step_name: str, result: Dict[str, Any], exclude_keys: List[str] = None):
        """保存步骤结果到文件"""
        if exclude_keys is None:
            exclude_keys = []
        
        # 创建可序列化的结果副本
        serializable_result = {}
        for key, value in result.items():
            if key not in exclude_keys:
                try:
                    json.dumps(value)  # 测试是否可序列化
                    serializable_result[key] = value
                except (TypeError, ValueError):
                    serializable_result[key] = str(value)  # 转换为字符串
        
        output_path = os.path.join(self.debug_output_dir, f"{step_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"   💾 步骤结果已保存: {output_path}")
    
    def _save_debug_results(self, debug_results: Dict[str, Any], base_name: str):
        """保存完整的调试结果"""
        output_path = os.path.join(self.debug_output_dir, f"{base_name}_debug_results.json")
        
        # 创建可序列化的结果
        serializable_results = {}
        for key, value in debug_results.items():
            try:
                json.dumps(value)
                serializable_results[key] = value
            except (TypeError, ValueError):
                serializable_results[key] = str(value)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 完整调试结果已保存: {output_path}")
    
    def _generate_debug_report(self, debug_results: Dict[str, Any], base_name: str):
        """生成可读的调试报告"""
        report_path = os.path.join(self.debug_output_dir, f"{base_name}_debug_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Pipeline 调试报告\n\n")
            f.write(f"**文件**: {debug_results['file_info']['file_name']}\n")
            f.write(f"**处理时间**: {debug_results['file_info']['processing_time']:.2f} 秒\n")
            f.write(f"**生成时间**: {debug_results['timestamp']}\n\n")
            
            f.write("## 步骤概览\n\n")
            
            step_names = {
                "document_loading": "📄 文档加载",
                "chunking": "✂️ 文档分块", 
                "embedding": "🔢 Embedding 生成",
                "vector_store": "🗄️ Vector Store 构建",
                "concept_extraction": "🧠 概念提取",
                "concept_merging": "🔗 概念合并",
                "evidence_extraction": "🔍 证据提取",
                "retrieval": "🎯 检索测试",
                "qa_generation": "❓ 问答生成"
            }
            
            for step_key, step_name in step_names.items():
                step_result = debug_results["step_results"].get(step_key, {})
                status = "✅ 成功" if step_result.get("success", False) else "❌ 失败"
                time_taken = step_result.get("processing_time", 0)
                
                f.write(f"- {step_name}: {status} ({time_taken:.2f}s)\n")
            
            f.write("\n## 详细结果\n\n")
            
            # 文档加载详情
            doc_result = debug_results["step_results"].get("document_loading", {})
            if doc_result.get("success"):
                f.write(f"### 📄 文档加载\n")
                f.write(f"- 文档数量: {doc_result.get('documents_count', 0)}\n")
                f.write(f"- 总字符数: {doc_result.get('total_characters', 0):,}\n\n")
            
            # 分块详情
            chunk_result = debug_results["step_results"].get("chunking", {})
            if chunk_result.get("success"):
                f.write(f"### ✂️ 文档分块\n")
                f.write(f"- 块数量: {chunk_result.get('chunks_count', 0)}\n")
                stats = chunk_result.get('chunking_stats', {})
                f.write(f"- 平均块长度: {stats.get('avg_chunk_length', 0):.0f} 字符\n")
                f.write(f"- 最小块长度: {stats.get('min_chunk_length', 0)} 字符\n")
                f.write(f"- 最大块长度: {stats.get('max_chunk_length', 0)} 字符\n\n")
            
            # 概念提取详情
            concept_result = debug_results["step_results"].get("concept_extraction", {})
            if concept_result.get("success"):
                f.write(f"### 🧠 概念提取\n")
                f.write(f"- 提取概念数: {concept_result.get('concepts_count', 0)}\n")
                stats = concept_result.get('concept_stats', {})
                f.write(f"- 平均每块概念数: {stats.get('avg_concepts_per_chunk', 0):.1f}\n\n")
            
            # 概念合并详情
            merge_result = debug_results["step_results"].get("concept_merging", {})
            if merge_result.get("success"):
                f.write(f"### 🔗 概念合并\n")
                f.write(f"- 合并前概念数: {merge_result.get('original_concepts_count', 0)}\n")
                f.write(f"- 合并后概念数: {merge_result.get('merged_concepts_count', 0)}\n")
                stats = merge_result.get('merge_stats', {})
                f.write(f"- 减少比例: {stats.get('reduction_ratio', 0):.1%}\n\n")
            
            # 证据提取详情
            evidence_result = debug_results["step_results"].get("evidence_extraction", {})
            if evidence_result.get("success"):
                f.write(f"### 🔍 证据提取\n")
                f.write(f"- 证据数量: {evidence_result.get('evidence_count', 0)}\n")
                stats = evidence_result.get('evidence_stats', {})
                f.write(f"- 平均每概念证据数: {stats.get('avg_evidence_per_concept', 0):.1f}\n\n")
            
            # 问答生成详情
            qa_result = debug_results["step_results"].get("qa_generation", {})
            if qa_result.get("success"):
                f.write(f"### ❓ 问答生成\n")
                f.write(f"- 问答对数量: {qa_result.get('qa_pairs_count', 0)}\n")
                stats = qa_result.get('qa_stats', {})
                f.write(f"- 问题类型: {', '.join(stats.get('question_types', []))}\n")
                f.write(f"- 平均问题长度: {stats.get('avg_question_length', 0):.0f} 字符\n")
                f.write(f"- 平均答案长度: {stats.get('avg_answer_length', 0):.0f} 字符\n\n")
        
        logger.info(f"📊 调试报告已生成: {report_path}")
        
        # 生成HTML报告
        html_path = self.visualizer.generate_html_report(debug_results, base_name)
        return report_path, html_path


def main():
    """主函数 - 演示如何使用调试器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline 调试工具")
    parser.add_argument("--file", required=True, help="要调试的文件路径")
    parser.add_argument("--config", default="config/config.yml", help="配置文件路径")
    parser.add_argument("--output", default="./debug_output", help="调试输出目录")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建调试器
    debugger = PipelineDebugger(
        config_path=args.config,
        debug_output_dir=args.output
    )
    
    # 执行调试
    results = debugger.debug_full_pipeline(args.file)
    
    if "error" not in results:
        print(f"\n✅ 调试完成！结果保存在: {args.output}")
        print(f"📊 查看调试报告: {args.output}/{os.path.splitext(os.path.basename(args.file))[0]}_debug_report.md")
    else:
        print(f"\n❌ 调试失败: {results['error']}")


if __name__ == "__main__":
    main() 