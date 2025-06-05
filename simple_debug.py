"""
简化版 Pipeline 调试工具
========================

这是一个简化版的调试工具，可以独立运行，展示 Pipeline 各步骤的中间结果。
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

class SimplePipelineDebugger:
    """简化版 Pipeline 调试器"""
    
    def __init__(self, debug_output_dir: str = "./debug_output"):
        """
        初始化调试器
        
        Args:
            debug_output_dir: 调试输出目录
        """
        self.debug_output_dir = debug_output_dir
        
        # 创建调试输出目录
        os.makedirs(debug_output_dir, exist_ok=True)
        
        logger.info(f"🔍 简化版 Pipeline 调试器已初始化")
        logger.info(f"📁 调试输出目录: {debug_output_dir}")
    
    def debug_pipeline_steps(self, file_path: str) -> Dict[str, Any]:
        """
        模拟调试 Pipeline 各个步骤
        
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
            # 模拟各个步骤
            steps_results = {}
            
            # 步骤1: 文档加载
            steps_results["document_loading"] = self._simulate_document_loading(file_path)
            
            # 步骤2: 文档分块
            steps_results["chunking"] = self._simulate_chunking()
            
            # 步骤3: Embedding 生成
            steps_results["embedding"] = self._simulate_embedding()
            
            # 步骤4: Vector Store 构建
            steps_results["vector_store"] = self._simulate_vector_store()
            
            # 步骤5: 概念提取
            steps_results["concept_extraction"] = self._simulate_concept_extraction()
            
            # 步骤6: 概念合并
            steps_results["concept_merging"] = self._simulate_concept_merging()
            
            # 步骤7: 证据提取
            steps_results["evidence_extraction"] = self._simulate_evidence_extraction()
            
            # 步骤8: 检索测试
            steps_results["retrieval"] = self._simulate_retrieval()
            
            # 步骤9: 问答生成
            steps_results["qa_generation"] = self._simulate_qa_generation()
            
            # 汇总结果
            debug_results = {
                "file_info": {
                    "file_path": file_path,
                    "file_name": file_name,
                    "base_name": base_name,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                },
                "step_results": steps_results,
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存结果
            self._save_debug_results(debug_results, base_name)
            self._generate_debug_report(debug_results, base_name)
            
            logger.info(f"✅ Pipeline 调试完成: {file_path}")
            return debug_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline 调试失败: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}
    
    def _simulate_document_loading(self, file_path: str) -> Dict[str, Any]:
        """模拟文档加载步骤"""
        logger.info("📄 步骤1: 模拟文档加载...")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"文件不存在: {file_path}",
                "processing_time": 0.1
            }
        
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        
        result = {
            "success": True,
            "documents_count": 1,
            "file_size_bytes": file_size,
            "estimated_characters": file_size * 2,  # 估算字符数
            "document_details": [
                {
                    "index": 0,
                    "file_size": file_size,
                    "file_type": os.path.splitext(file_path)[1],
                    "text_preview": f"文档: {os.path.basename(file_path)}"
                }
            ],
            "processing_time": 0.5
        }
        
        self._save_step_result("01_document_loading", result)
        logger.info(f"   ✅ 文档加载完成: 文件大小 {file_size} 字节")
        return result
    
    def _simulate_chunking(self) -> Dict[str, Any]:
        """模拟文档分块步骤"""
        logger.info("✂️ 步骤2: 模拟文档分块...")
        
        # 模拟分块结果
        chunks_count = 25
        avg_length = 1500
        
        result = {
            "success": True,
            "chunks_count": chunks_count,
            "chunk_details": [
                {
                    "index": i,
                    "chunk_id": f"chunk_{i:03d}",
                    "text_length": avg_length + (i % 500) - 250,
                    "text_preview": f"这是第 {i+1} 个文档块的内容..."
                }
                for i in range(min(5, chunks_count))  # 只显示前5个
            ],
            "chunking_stats": {
                "avg_chunk_length": avg_length,
                "min_chunk_length": avg_length - 250,
                "max_chunk_length": avg_length + 250
            },
            "processing_time": 1.2
        }
        
        self._save_step_result("02_chunking", result)
        logger.info(f"   ✅ 文档分块完成: {chunks_count} 个块")
        return result
    
    def _simulate_embedding(self) -> Dict[str, Any]:
        """模拟 Embedding 生成步骤"""
        logger.info("🔢 步骤3: 模拟 Embedding 生成...")
        
        embeddings_count = 25
        dimension = 1536
        
        result = {
            "success": True,
            "embeddings_count": embeddings_count,
            "embedding_details": [
                {
                    "chunk_index": i,
                    "chunk_id": f"chunk_{i:03d}",
                    "embedding_dimension": dimension,
                    "embedding_preview": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0],
                    "embedding_norm": 1.0 + (i * 0.1)
                }
                for i in range(min(5, embeddings_count))
            ],
            "embedding_stats": {
                "dimension": dimension,
                "avg_norm": 1.2
            },
            "processing_time": 3.5
        }
        
        self._save_step_result("03_embedding", result)
        logger.info(f"   ✅ Embedding 生成完成: {embeddings_count} 个向量")
        return result
    
    def _simulate_vector_store(self) -> Dict[str, Any]:
        """模拟 Vector Store 构建步骤"""
        logger.info("🗄️ 步骤4: 模拟 Vector Store 构建...")
        
        result = {
            "success": True,
            "vector_store_type": "SimpleVectorStore",
            "stored_nodes_count": 25,
            "vector_store_stats": {
                "total_vectors": 25,
                "vector_dimension": 1536
            },
            "processing_time": 0.8
        }
        
        self._save_step_result("04_vector_store", result)
        logger.info(f"   ✅ Vector Store 构建完成: 25 个向量")
        return result
    
    def _simulate_concept_extraction(self) -> Dict[str, Any]:
        """模拟概念提取步骤"""
        logger.info("🧠 步骤5: 模拟概念提取...")
        
        concepts_count = 45
        
        result = {
            "success": True,
            "concepts_count": concepts_count,
            "concept_details": [
                {
                    "chunk_index": i,
                    "chunk_id": f"chunk_{i:03d}",
                    "concepts_count": 2,
                    "concepts": [
                        {
                            "concept_id": f"concept_{i*2+j:03d}",
                            "concept_text": f"概念 {i*2+j+1}: 示例概念内容",
                            "confidence": 0.8 + (j * 0.1)
                        }
                        for j in range(2)
                    ]
                }
                for i in range(min(5, 25))
            ],
            "concept_stats": {
                "avg_concepts_per_chunk": 1.8,
                "total_concepts": concepts_count
            },
            "processing_time": 15.3
        }
        
        self._save_step_result("05_concept_extraction", result)
        logger.info(f"   ✅ 概念提取完成: {concepts_count} 个概念")
        return result
    
    def _simulate_concept_merging(self) -> Dict[str, Any]:
        """模拟概念合并步骤"""
        logger.info("🔗 步骤6: 模拟概念合并...")
        
        original_count = 45
        merged_count = 32
        
        result = {
            "success": True,
            "original_concepts_count": original_count,
            "merged_concepts_count": merged_count,
            "merge_stats": {
                "reduction_ratio": (original_count - merged_count) / original_count,
                "concepts_before": original_count,
                "concepts_after": merged_count
            },
            "merge_details": [
                {
                    "concept_id": f"merged_concept_{i:03d}",
                    "concept_text": f"合并概念 {i+1}: 这是一个合并后的概念",
                    "source_chunks": [f"chunk_{j:03d}" for j in range(i, min(i+3, 25))],
                    "confidence": 0.85 + (i * 0.01)
                }
                for i in range(min(5, merged_count))
            ],
            "processing_time": 8.7
        }
        
        self._save_step_result("06_concept_merging", result)
        logger.info(f"   ✅ 概念合并完成: {original_count} -> {merged_count} 个概念")
        return result
    
    def _simulate_evidence_extraction(self) -> Dict[str, Any]:
        """模拟证据提取步骤"""
        logger.info("🔍 步骤7: 模拟证据提取...")
        
        evidence_count = 64
        
        result = {
            "success": True,
            "evidence_count": evidence_count,
            "evidence_details": [
                {
                    "concept_id": f"merged_concept_{i:03d}",
                    "concept_text": f"合并概念 {i+1}",
                    "evidence_count": 2,
                    "evidence": [
                        {
                            "evidence_id": f"evidence_{i*2+j:03d}",
                            "evidence_text": f"证据 {i*2+j+1}: 支持概念的证据内容",
                            "relevance_score": 0.75 + (j * 0.1)
                        }
                        for j in range(2)
                    ]
                }
                for i in range(min(5, 32))
            ],
            "evidence_stats": {
                "avg_evidence_per_concept": evidence_count / 32,
                "total_evidence": evidence_count
            },
            "processing_time": 12.1
        }
        
        self._save_step_result("07_evidence_extraction", result)
        logger.info(f"   ✅ 证据提取完成: {evidence_count} 个证据")
        return result
    
    def _simulate_retrieval(self) -> Dict[str, Any]:
        """模拟检索步骤"""
        logger.info("🎯 步骤8: 模拟检索测试...")
        
        result = {
            "success": True,
            "retrieval_results": [
                {
                    "concept_id": f"merged_concept_{i:03d}",
                    "concept_text": f"合并概念 {i+1}",
                    "retrieved_count": 5,
                    "retrieved_chunks": [
                        {
                            "chunk_id": f"chunk_{j:03d}",
                            "chunk_text": f"检索到的块 {j+1} 内容",
                            "similarity_score": 0.9 - (j * 0.1)
                        }
                        for j in range(5)
                    ]
                }
                for i in range(3)  # 测试3个概念
            ],
            "retrieval_stats": {
                "concepts_tested": 3,
                "avg_retrieved_per_concept": 5.0
            },
            "processing_time": 2.3
        }
        
        self._save_step_result("08_retrieval", result)
        logger.info(f"   ✅ 检索测试完成: 测试了 3 个概念")
        return result
    
    def _simulate_qa_generation(self) -> Dict[str, Any]:
        """模拟问答生成步骤"""
        logger.info("❓ 步骤9: 模拟问答生成...")
        
        qa_count = 15
        
        result = {
            "success": True,
            "qa_pairs_count": qa_count,
            "qa_details": [
                {
                    "index": i,
                    "question": f"问题 {i+1}: 这是一个示例问题？",
                    "answer": f"答案 {i+1}: 这是对应的示例答案，包含了详细的解释和说明。",
                    "question_type": ["factual", "analytical", "conceptual"][i % 3],
                    "difficulty": ["easy", "medium", "hard"][i % 3]
                }
                for i in range(qa_count)
            ],
            "qa_stats": {
                "total_qa_pairs": qa_count,
                "question_types": ["factual", "analytical", "conceptual"],
                "avg_question_length": 25.3,
                "avg_answer_length": 156.7
            },
            "processing_time": 18.9
        }
        
        self._save_step_result("09_qa_generation", result)
        logger.info(f"   ✅ 问答生成完成: {qa_count} 个问答对")
        return result
    
    def _save_step_result(self, step_name: str, result: Dict[str, Any]):
        """保存步骤结果到文件"""
        output_path = os.path.join(self.debug_output_dir, f"{step_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"   💾 步骤结果已保存: {output_path}")
    
    def _save_debug_results(self, debug_results: Dict[str, Any], base_name: str):
        """保存完整的调试结果"""
        output_path = os.path.join(self.debug_output_dir, f"{base_name}_debug_results.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(debug_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 完整调试结果已保存: {output_path}")
    
    def _generate_debug_report(self, debug_results: Dict[str, Any], base_name: str):
        """生成可读的调试报告"""
        report_path = os.path.join(self.debug_output_dir, f"{base_name}_debug_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Pipeline 调试报告 (模拟)\n\n")
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
            
            f.write("\n## 详细统计\n\n")
            
            # 添加各步骤的详细统计
            step_results = debug_results["step_results"]
            
            if step_results.get("chunking", {}).get("success"):
                chunk_result = step_results["chunking"]
                f.write(f"### ✂️ 文档分块\n")
                f.write(f"- 块数量: {chunk_result.get('chunks_count', 0)}\n")
                stats = chunk_result.get('chunking_stats', {})
                f.write(f"- 平均块长度: {stats.get('avg_chunk_length', 0):.0f} 字符\n\n")
            
            if step_results.get("concept_extraction", {}).get("success"):
                concept_result = step_results["concept_extraction"]
                f.write(f"### 🧠 概念提取\n")
                f.write(f"- 提取概念数: {concept_result.get('concepts_count', 0)}\n")
                stats = concept_result.get('concept_stats', {})
                f.write(f"- 平均每块概念数: {stats.get('avg_concepts_per_chunk', 0):.1f}\n\n")
            
            if step_results.get("concept_merging", {}).get("success"):
                merge_result = step_results["concept_merging"]
                f.write(f"### 🔗 概念合并\n")
                f.write(f"- 合并前概念数: {merge_result.get('original_concepts_count', 0)}\n")
                f.write(f"- 合并后概念数: {merge_result.get('merged_concepts_count', 0)}\n")
                stats = merge_result.get('merge_stats', {})
                f.write(f"- 减少比例: {stats.get('reduction_ratio', 0):.1%}\n\n")
            
            if step_results.get("qa_generation", {}).get("success"):
                qa_result = step_results["qa_generation"]
                f.write(f"### ❓ 问答生成\n")
                f.write(f"- 问答对数量: {qa_result.get('qa_pairs_count', 0)}\n")
                stats = qa_result.get('qa_stats', {})
                f.write(f"- 问题类型: {', '.join(stats.get('question_types', []))}\n")
                f.write(f"- 平均问题长度: {stats.get('avg_question_length', 0):.0f} 字符\n")
                f.write(f"- 平均答案长度: {stats.get('avg_answer_length', 0):.0f} 字符\n\n")
        
        logger.info(f"📊 调试报告已生成: {report_path}")


def main():
    """主函数 - 演示如何使用简化版调试器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="简化版 Pipeline 调试工具")
    parser.add_argument("--file", required=True, help="要调试的文件路径")
    parser.add_argument("--output", default="./debug_output", help="调试输出目录")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建调试器
    debugger = SimplePipelineDebugger(debug_output_dir=args.output)
    
    # 执行调试
    results = debugger.debug_pipeline_steps(args.file)
    
    if "error" not in results:
        print(f"\n✅ 调试完成！结果保存在: {args.output}")
        print(f"📊 查看调试报告: {args.output}/{os.path.splitext(os.path.basename(args.file))[0]}_debug_report.md")
    else:
        print(f"\n❌ 调试失败: {results['error']}")


if __name__ == "__main__":
    main() 