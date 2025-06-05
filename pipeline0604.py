"""
分步骤Pipeline处理脚本
========================================

将整个处理流程拆分为独立的步骤，每步都生成详细的txt报告
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import traceback
import uuid

# 导入pipeline相关模块
from pipeline_new import ImprovedConceptBasedPipeline
from llama_index.core import Document
from data_generate_0526 import SimpleDataGenerator, FileProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step_by_step_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StepByStepPipelineProcessor:
    """分步骤Pipeline处理器 - 每步单独保存txt结果"""
    
    def __init__(self, 
                 input_file: str,
                 output_dir: str = "step_by_step_results",
                 enable_qa_generation: bool = True,
                 qa_model_name: str = "gpt-4o-mini",
                 questions_per_type: Dict[str, int] = None):
        """
        初始化分步骤处理器
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            enable_qa_generation: 是否启用问答生成
            qa_model_name: QA生成模型
            questions_per_type: 每种类型问题数量
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.enable_qa_generation = enable_qa_generation
        
        # 创建输出目录结构
        self.setup_output_directories()
        
        # 初始化pipeline和QA生成器
        self.pipeline = ImprovedConceptBasedPipeline()
        
        if self.enable_qa_generation:
            self.questions_per_type = questions_per_type or {
                "remember": 2, "understand": 2, "apply": 1,
                "analyze": 1, "evaluate": 1, "create": 1
            }
            self.qa_generator = SimpleDataGenerator(
                model_name=qa_model_name,
                questions_per_type=self.questions_per_type
            )
        
        # 存储各步骤结果
        self.step_results = {}
        self.step_timings = {}
        
        logger.info(f"📄 初始化分步骤处理器")
        logger.info(f"   输入文件: {self.input_file}")
        logger.info(f"   输出目录: {self.output_dir}")
    
    def setup_output_directories(self):
        """设置输出目录结构"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建各步骤的子目录
        self.step_dirs = {
            "step1": self.output_dir / "01_文件加载",
            "step2": self.output_dir / "02_文档分块", 
            "step3": self.output_dir / "03_概念提取",
            "step4": self.output_dir / "04_概念合并",
            "step5": self.output_dir / "05_概念检索",
            "step6": self.output_dir / "06_证据提取",
            "step7": self.output_dir / "07_问答生成",
            "step8": self.output_dir / "08_最终汇总"
        }
        
        for step_dir in self.step_dirs.values():
            step_dir.mkdir(exist_ok=True)
    
    def save_step_txt_report(self, step_name: str, step_num: str, content: str):
        """保存步骤的txt报告"""
        step_dir = self.step_dirs[f"step{step_num}"]
        report_file = step_dir / f"{step_name}_详细报告.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"=" * 80 + "\n")
            f.write(f"{step_name} - 详细报告\n")
            f.write(f"=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理文件: {self.input_file.name}\n")
            f.write(f"=" * 80 + "\n\n")
            f.write(content)
        
        logger.info(f"💾 {step_name}报告已保存: {report_file}")
    
    def step1_load_file(self) -> Dict[str, Any]:
        """步骤1: 文件加载和预处理"""
        print("\n" + "="*60)
        print("🔄 步骤1: 文件加载和预处理")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # 加载文件
            logger.info(f"📄 正在加载文件: {self.input_file}")
            text_content = FileProcessor.extract_text(str(self.input_file))
            
            # 创建Document对象
            document = Document(
                text=text_content,
                metadata={
                    "file_name": self.input_file.name,
                    "file_path": str(self.input_file),
                    "file_type": self.input_file.suffix.lower(),
                    "file_size": self.input_file.stat().st_size,
                    "text_length": len(text_content),
                    "processing_timestamp": datetime.now().isoformat()
                }
            )
            
            processing_time = time.time() - start_time
            
            # 生成详细报告
            report_content = f"""
文件基本信息:
- 文件名: {self.input_file.name}
- 文件类型: {self.input_file.suffix.lower()}
- 文件大小: {self.input_file.stat().st_size / 1024:.2f} KB
- 文本长度: {len(text_content):,} 字符
- 处理时间: {processing_time:.2f} 秒

文本内容统计:
- 总字符数: {len(text_content):,}
- 总行数: {text_content.count(chr(10)) + 1:,}
- 段落数估计: {len([p for p in text_content.split('\n\n') if p.strip()]):,}

文本内容预览 (前500字符):
{'-'*50}
{text_content[:500]}
{'-'*50}

文本内容预览 (后500字符):
{'-'*50}
{text_content[-500:]}
{'-'*50}

元数据信息:
{json.dumps(document.metadata, ensure_ascii=False, indent=2)}
"""
            
            # 保存txt报告
            self.save_step_txt_report("步骤1_文件加载", "1", report_content)
            
            # 保存原始文本
            raw_text_file = self.step_dirs["step1"] / "原始文本内容.txt"
            with open(raw_text_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            result = {
                "success": True,
                "document": document,
                "text_length": len(text_content),
                "processing_time": processing_time,
                "metadata": document.metadata
            }
            
            self.step_results["step1"] = result
            self.step_timings["step1"] = processing_time
            
            print(f"✅ 步骤1完成 - 文件加载成功")
            print(f"   文件大小: {self.input_file.stat().st_size / 1024:.1f} KB")
            print(f"   文本长度: {len(text_content):,} 字符")
            print(f"   处理时间: {processing_time:.2f} 秒")
            
            return result
            
        except Exception as e:
            error_msg = f"文件加载失败: {str(e)}"
            logger.error(error_msg)
            
            error_report = f"""
错误信息:
{error_msg}

错误详情:
{traceback.format_exc()}
"""
            self.save_step_txt_report("步骤1_文件加载_错误", "1", error_report)
            
            result = {
                "success": False,
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
            
            self.step_results["step1"] = result
            return result
    
    def step2_document_chunking(self) -> Dict[str, Any]:
        """步骤2: 文档分块"""
        print("\n" + "="*60)
        print("🔄 步骤2: 文档分块")
        print("="*60)
        
        if not self.step_results.get("step1", {}).get("success"):
            print("❌ 步骤1未成功完成，跳过步骤2")
            return {"success": False, "error": "依赖的步骤1未完成"}
        
        start_time = time.time()
        
        try:
            document = self.step_results["step1"]["document"]
            
            # 执行文档分块
            logger.info("✂️ 开始文档分块...")
            chunk_nodes = self.pipeline.chunker.chunk_and_extract_concepts([document])
            
            processing_time = time.time() - start_time
            
            # 分析chunk统计信息
            chunk_lengths = [len(chunk.text) for chunk in chunk_nodes]
            total_chunks = len(chunk_nodes)
            avg_chunk_length = sum(chunk_lengths) / total_chunks if total_chunks > 0 else 0
            
            # 生成详细报告
            report_content = f"""
分块统计信息:
- 总分块数: {total_chunks}
- 平均分块长度: {avg_chunk_length:.1f} 字符
- 最短分块长度: {min(chunk_lengths) if chunk_lengths else 0} 字符
- 最长分块长度: {max(chunk_lengths) if chunk_lengths else 0} 字符
- 处理时间: {processing_time:.2f} 秒

分块详细信息:
{'-'*60}
"""
            
            # 为每个chunk生成详细信息
            for i, chunk in enumerate(chunk_nodes, 1):
                concepts = chunk.metadata.get("concepts", [])
                report_content += f"""
分块 {i}:
- ID: {chunk.node_id}
- 长度: {len(chunk.text)} 字符
- 概念数量: {len(concepts)}
- 概念列表: {concepts}
- 内容预览: {chunk.text[:100]}...
{'-'*40}
"""
            
            # 保存txt报告
            self.save_step_txt_report("步骤2_文档分块", "2", report_content)
            
            # 保存每个chunk的完整内容
            chunks_detail_file = self.step_dirs["step2"] / "分块详细内容.txt"
            with open(chunks_detail_file, 'w', encoding='utf-8') as f:
                f.write("文档分块详细内容\n")
                f.write("=" * 80 + "\n\n")
                
                for i, chunk in enumerate(chunk_nodes, 1):
                    f.write(f"分块 {i} (ID: {chunk.node_id})\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"长度: {len(chunk.text)} 字符\n")
                    f.write(f"概念: {chunk.metadata.get('concepts', [])}\n")
                    f.write("内容:\n")
                    f.write(chunk.text)
                    f.write("\n\n" + "=" * 80 + "\n\n")
            
            result = {
                "success": True,
                "chunk_nodes": chunk_nodes,
                "chunk_count": total_chunks,
                "avg_chunk_length": avg_chunk_length,
                "processing_time": processing_time
            }
            
            self.step_results["step2"] = result
            self.step_timings["step2"] = processing_time
            
            print(f"✅ 步骤2完成 - 文档分块成功")
            print(f"   分块数量: {total_chunks}")
            print(f"   平均长度: {avg_chunk_length:.1f} 字符")
            print(f"   处理时间: {processing_time:.2f} 秒")
            
            return result
            
        except Exception as e:
            error_msg = f"文档分块失败: {str(e)}"
            logger.error(error_msg)
            
            error_report = f"""
错误信息:
{error_msg}

错误详情:
{traceback.format_exc()}
"""
            self.save_step_txt_report("步骤2_文档分块_错误", "2", error_report)
            
            result = {
                "success": False,
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
            
            self.step_results["step2"] = result
            return result
    
    def step3_concept_extraction(self) -> Dict[str, Any]:
        """步骤3: 概念提取"""
        print("\n" + "="*60)
        print("🔄 步骤3: 概念提取")
        print("="*60)
        
        if not self.step_results.get("step2", {}).get("success"):
            print("❌ 步骤2未成功完成，跳过步骤3")
            return {"success": False, "error": "依赖的步骤2未完成"}
        
        start_time = time.time()
        
        try:
            chunk_nodes = self.step_results["step2"]["chunk_nodes"]
            
            # 从chunk中提取概念
            logger.info("🧠 开始概念提取...")
            all_concepts = []
            
            for chunk in chunk_nodes:
                concepts = chunk.metadata.get("concepts", [])
                all_concepts.extend(concepts)
            
            # 统计概念信息
            unique_concepts = list(set(all_concepts))
            concept_frequency = {concept: all_concepts.count(concept) for concept in unique_concepts}
            
            processing_time = time.time() - start_time
            
            # 生成详细报告
            report_content = f"""
概念提取统计:
- 总概念数 (含重复): {len(all_concepts)}
- 唯一概念数: {len(unique_concepts)}
- 处理时间: {processing_time:.2f} 秒

概念频率分析 (按频率降序):
{'-'*60}
"""
            
            # 按频率排序概念
            sorted_concepts = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)
            
            for concept, freq in sorted_concepts:
                report_content += f"- {concept}: {freq} 次\n"
            
            report_content += f"\n{'-'*60}\n\n概念详细分布:\n"
            
            # 为每个chunk显示其概念
            for i, chunk in enumerate(chunk_nodes, 1):
                concepts = chunk.metadata.get("concepts", [])
                report_content += f"""
分块 {i} 的概念:
- 概念数量: {len(concepts)}
- 概念列表: {concepts}
- 文本预览: {chunk.text[:80]}...
{'-'*40}
"""
            
            # 保存txt报告
            self.save_step_txt_report("步骤3_概念提取", "3", report_content)
            
            # 保存概念列表
            concepts_file = self.step_dirs["step3"] / "概念列表.txt"
            with open(concepts_file, 'w', encoding='utf-8') as f:
                f.write("提取的概念列表\n")
                f.write("=" * 50 + "\n\n")
                f.write("按频率排序的概念:\n")
                f.write("-" * 30 + "\n")
                
                for concept, freq in sorted_concepts:
                    f.write(f"{freq:3d} | {concept}\n")
            
            result = {
                "success": True,
                "all_concepts": all_concepts,
                "unique_concepts": unique_concepts,
                "concept_frequency": concept_frequency,
                "processing_time": processing_time
            }
            
            self.step_results["step3"] = result
            self.step_timings["step3"] = processing_time
            
            print(f"✅ 步骤3完成 - 概念提取成功")
            print(f"   总概念数: {len(all_concepts)}")
            print(f"   唯一概念数: {len(unique_concepts)}")
            print(f"   处理时间: {processing_time:.2f} 秒")
            
            return result
            
        except Exception as e:
            error_msg = f"概念提取失败: {str(e)}"
            logger.error(error_msg)
            
            error_report = f"""
错误信息:
{error_msg}

错误详情:
{traceback.format_exc()}
"""
            self.save_step_txt_report("步骤3_概念提取_错误", "3", error_report)
            
            result = {
                "success": False,
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
            
            self.step_results["step3"] = result
            return result
    
    def step4_concept_merging(self) -> Dict[str, Any]:
        """步骤4: 概念合并"""
        print("\n" + "="*60)
        print("🔄 步骤4: 概念合并")
        print("="*60)
        
        if not self.step_results.get("step2", {}).get("success"):
            print("❌ 步骤2未成功完成，跳过步骤4")
            return {"success": False, "error": "依赖的步骤2未完成"}
        
        start_time = time.time()
        
        try:
            chunk_nodes = self.step_results["step2"]["chunk_nodes"]
            
            # 执行概念合并
            logger.info("🔗 开始概念合并...")
            concept_nodes = self.pipeline.concept_merger.merge_document_concepts(chunk_nodes)
            
            processing_time = time.time() - start_time
            
            # 生成详细报告
            report_content = f"""
概念合并统计:
- 合并后概念数: {len(concept_nodes)}
- 处理时间: {processing_time:.2f} 秒

合并后的概念列表:
{'-'*60}
"""
            
            # 为每个合并后的概念生成详细信息
            for i, concept_node in enumerate(concept_nodes, 1):
                source_chunks = concept_node.metadata.get("source_chunks", [])
                report_content += f"""
概念 {i}:
- ID: {concept_node.node_id}
- 概念文本: {concept_node.text}
- 来源分块数: {len(source_chunks)}
- 来源分块ID: {source_chunks}
{'-'*40}
"""
            
            # 保存txt报告
            self.save_step_txt_report("步骤4_概念合并", "4", report_content)
            
            # 保存合并后概念的详细信息
            merged_concepts_file = self.step_dirs["step4"] / "合并后概念详细信息.txt"
            with open(merged_concepts_file, 'w', encoding='utf-8') as f:
                f.write("合并后的概念详细信息\n")
                f.write("=" * 80 + "\n\n")
                
                for i, concept_node in enumerate(concept_nodes, 1):
                    f.write(f"概念 {i}\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"ID: {concept_node.node_id}\n")
                    f.write(f"概念文本: {concept_node.text}\n")
                    f.write(f"元数据: {json.dumps(concept_node.metadata, ensure_ascii=False, indent=2)}\n")
                    f.write("\n" + "=" * 80 + "\n\n")
            
            result = {
                "success": True,
                "concept_nodes": concept_nodes,
                "concept_count": len(concept_nodes),
                "processing_time": processing_time
            }
            
            self.step_results["step4"] = result
            self.step_timings["step4"] = processing_time
            
            print(f"✅ 步骤4完成 - 概念合并成功")
            print(f"   合并后概念数: {len(concept_nodes)}")
            print(f"   处理时间: {processing_time:.2f} 秒")
            
            return result
            
        except Exception as e:
            error_msg = f"概念合并失败: {str(e)}"
            logger.error(error_msg)
            
            error_report = f"""
错误信息:
{error_msg}

错误详情:
{traceback.format_exc()}
"""
            self.save_step_txt_report("步骤4_概念合并_错误", "4", error_report)
            
            result = {
                "success": False,
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
            
            self.step_results["step4"] = result
            return result
    
    def step5_concept_retrieval(self) -> Dict[str, Any]:
        """步骤5: 概念检索"""
        print("\n" + "="*60)
        print("🔄 步骤5: 概念检索")
        print("="*60)
        
        if not (self.step_results.get("step2", {}).get("success") and 
                self.step_results.get("step4", {}).get("success")):
            print("❌ 依赖的步骤未成功完成，跳过步骤5")
            return {"success": False, "error": "依赖的步骤未完成"}
        
        start_time = time.time()
        
        try:
            chunk_nodes = self.step_results["step2"]["chunk_nodes"]
            concept_nodes = self.step_results["step4"]["concept_nodes"]
            
            # 创建chunk索引
            logger.info("🔍 开始概念检索...")
            from llama_index.core import VectorStoreIndex
            chunk_index = VectorStoreIndex(chunk_nodes)
            
            # 执行概念检索
            concept_to_chunks = self.pipeline.retriever.retrieve_chunks_for_concepts(
                concept_nodes, chunk_index
            )
            
            processing_time = time.time() - start_time
            
            # 生成详细报告
            report_content = f"""
概念检索统计:
- 参与检索的概念数: {len(concept_nodes)}
- 检索结果映射数: {len(concept_to_chunks)}
- 处理时间: {processing_time:.2f} 秒

检索结果详情:
{'-'*60}
"""
            
            # 为每个概念的检索结果生成详细信息
            for concept_node in concept_nodes:
                concept_id = concept_node.node_id
                chunks = concept_to_chunks.get(concept_id, [])
                
                report_content += f"""
概念: {concept_node.text}
- 概念ID: {concept_id}
- 检索到的相关分块数: {len(chunks)}
"""
                
                for j, chunk_with_score in enumerate(chunks, 1):
                    chunk = chunk_with_score.node
                    score = chunk_with_score.score
                    report_content += f"  相关分块 {j}: ID={chunk.node_id}, 相似度={score:.3f}, 内容={chunk.text[:50]}...\n"
                
                report_content += "-" * 40 + "\n"
            
            # 保存txt报告
            self.save_step_txt_report("步骤5_概念检索", "5", report_content)
            
            # 保存检索结果的详细映射
            retrieval_details_file = self.step_dirs["step5"] / "检索结果详细映射.txt"
            with open(retrieval_details_file, 'w', encoding='utf-8') as f:
                f.write("概念检索详细结果\n")
                f.write("=" * 80 + "\n\n")
                
                for concept_node in concept_nodes:
                    concept_id = concept_node.node_id
                    chunks = concept_to_chunks.get(concept_id, [])
                    
                    f.write(f"概念: {concept_node.text}\n")
                    f.write(f"概念ID: {concept_id}\n")
                    f.write(f"检索到 {len(chunks)} 个相关分块:\n")
                    f.write("-" * 60 + "\n")
                    
                    for j, chunk_with_score in enumerate(chunks, 1):
                        chunk = chunk_with_score.node
                        score = chunk_with_score.score
                        f.write(f"\n相关分块 {j}:\n")
                        f.write(f"  - 分块ID: {chunk.node_id}\n")
                        f.write(f"  - 相似度分数: {score:.4f}\n")
                        f.write(f"  - 分块内容:\n{chunk.text}\n")
                        f.write("-" * 30 + "\n")
                    
                    f.write("\n" + "=" * 80 + "\n\n")
            
            result = {
                "success": True,
                "concept_to_chunks": concept_to_chunks,
                "retrieval_count": len(concept_to_chunks),
                "processing_time": processing_time
            }
            
            self.step_results["step5"] = result
            self.step_timings["step5"] = processing_time
            
            print(f"✅ 步骤5完成 - 概念检索成功")
            print(f"   检索映射数: {len(concept_to_chunks)}")
            print(f"   处理时间: {processing_time:.2f} 秒")
            
            return result
            
        except Exception as e:
            error_msg = f"概念检索失败: {str(e)}"
            logger.error(error_msg)
            
            error_report = f"""
错误信息:
{error_msg}

错误详情:
{traceback.format_exc()}
"""
            self.save_step_txt_report("步骤5_概念检索_错误", "5", error_report)
            
            result = {
                "success": False,
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
            
            self.step_results["step5"] = result
            return result
    
    def step6_evidence_extraction(self) -> Dict[str, Any]:
        """步骤6: 证据提取"""
        print("\n" + "="*60)
        print("🔄 步骤6: 证据提取")
        print("="*60)
        
        if not (self.step_results.get("step4", {}).get("success") and 
                self.step_results.get("step5", {}).get("success")):
            print("❌ 依赖的步骤未成功完成，跳过步骤6")
            return {"success": False, "error": "依赖的步骤未完成"}
        
        start_time = time.time()
        
        try:
            concept_nodes = self.step_results["step4"]["concept_nodes"]
            concept_to_chunks = self.step_results["step5"]["concept_to_chunks"]
            
            # 执行证据提取
            logger.info("🔍 开始证据提取...")
            evidence_nodes = self.pipeline.evidence_extractor.extract_evidence_for_concepts(
                concept_nodes, concept_to_chunks
            )
            
            processing_time = time.time() - start_time
            
            # 生成详细报告
            report_content = f"""
证据提取统计:
- 输入概念数: {len(concept_nodes)}
- 提取的证据数: {len(evidence_nodes)}
- 处理时间: {processing_time:.2f} 秒

证据提取详情:
{'-'*60}
"""
            
            # 为每个证据生成详细信息
            for i, evidence_node in enumerate(evidence_nodes, 1):
                concept_text = evidence_node.metadata.get("concept_text", "未知")
                relevance_score = evidence_node.metadata.get("relevance_score", 0.0)
                
                report_content += f"""
证据 {i}:
- 证据ID: {evidence_node.node_id}
- 关联概念: {concept_text}
- 相关性分数: {relevance_score:.3f}
- 证据长度: {len(evidence_node.text)} 字符
- 证据内容: {evidence_node.text[:100]}...
{'-'*40}
"""
            
            # 保存txt报告
            self.save_step_txt_report("步骤6_证据提取", "6", report_content)
            
            # 保存证据的完整内容
            evidence_details_file = self.step_dirs["step6"] / "证据详细内容.txt"
            with open(evidence_details_file, 'w', encoding='utf-8') as f:
                f.write("证据提取详细结果\n")
                f.write("=" * 80 + "\n\n")
                
                for i, evidence_node in enumerate(evidence_nodes, 1):
                    concept_text = evidence_node.metadata.get("concept_text", "未知")
                    relevance_score = evidence_node.metadata.get("relevance_score", 0.0)
                    source_chunks = evidence_node.metadata.get("source_chunks", [])
                    
                    f.write(f"证据 {i}\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"证据ID: {evidence_node.node_id}\n")
                    f.write(f"关联概念: {concept_text}\n")
                    f.write(f"相关性分数: {relevance_score:.4f}\n")
                    f.write(f"来源分块: {source_chunks}\n")
                    f.write(f"证据长度: {len(evidence_node.text)} 字符\n")
                    f.write("证据完整内容:\n")
                    f.write(evidence_node.text)
                    f.write("\n\n" + "=" * 80 + "\n\n")
            
            result = {
                "success": True,
                "evidence_nodes": evidence_nodes,
                "evidence_count": len(evidence_nodes),
                "processing_time": processing_time
            }
            
            self.step_results["step6"] = result
            self.step_timings["step6"] = processing_time
            
            print(f"✅ 步骤6完成 - 证据提取成功")
            print(f"   提取证据数: {len(evidence_nodes)}")
            print(f"   处理时间: {processing_time:.2f} 秒")
            
            return result
            
        except Exception as e:
            error_msg = f"证据提取失败: {str(e)}"
            logger.error(error_msg)
            
            error_report = f"""
错误信息:
{error_msg}

错误详情:
{traceback.format_exc()}
"""
            self.save_step_txt_report("步骤6_证据提取_错误", "6", error_report)
            
            result = {
                "success": False,
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
            
            self.step_results["step6"] = result
            return result
    
    def step7_qa_generation(self) -> Dict[str, Any]:
        """步骤7: 问答生成"""
        print("\n" + "="*60)
        print("🔄 步骤7: 问答生成")
        print("="*60)
        
        if not self.enable_qa_generation:
            print("ℹ️  问答生成已禁用，跳过步骤7")
            return {"success": True, "skipped": True, "reason": "QA generation disabled"}
        
        if not self.step_results.get("step6", {}).get("success"):
            print("❌ 步骤6未成功完成，跳过步骤7")
            return {"success": False, "error": "依赖的步骤6未完成"}
        
        start_time = time.time()
        
        try:
            evidence_nodes = self.step_results["step6"]["evidence_nodes"]
            
            if not evidence_nodes:
                print("⚠️  没有证据可用于问答生成，跳过步骤7")
                return {"success": True, "skipped": True, "reason": "No evidence available"}
            
            # 执行问答生成
            logger.info("❓ 开始问答生成...")
            
            all_qa_pairs = []
            
            for i, evidence_node in enumerate(evidence_nodes):
                try:
                    # 基于每个证据生成问答对
                    qa_pairs = self.qa_generator.generate_qa_pairs_from_text(evidence_node.text)
                    
                    # 添加来源信息
                    for qa_pair in qa_pairs:
                        qa_pair["evidence_source"] = evidence_node.node_id
                        qa_pair["evidence_concept"] = evidence_node.metadata.get("concept_text", "未知")
                        qa_pair["generation_timestamp"] = datetime.now().isoformat()
                    
                    all_qa_pairs.extend(qa_pairs)
                    
                except Exception as e:
                    logger.warning(f"从证据 {i+1} 生成问答对失败: {e}")
                    continue
            
            processing_time = time.time() - start_time
            
            # 统计问答对类型
            qa_types = {}
            for qa_pair in all_qa_pairs:
                qa_type = qa_pair.get("type", "unknown")
                qa_types[qa_type] = qa_types.get(qa_type, 0) + 1
            
            # 生成详细报告
            report_content = f"""
问答生成统计:
- 输入证据数: {len(evidence_nodes)}
- 生成问答对数: {len(all_qa_pairs)}
- 处理时间: {processing_time:.2f} 秒

问答类型分布:
{'-'*60}
"""
            
            for qa_type, count in qa_types.items():
                report_content += f"- {qa_type}: {count} 个\n"
            
            report_content += f"\n{'-'*60}\n问答对详情:\n"
            
            # 为每个问答对生成详细信息
            for i, qa_pair in enumerate(all_qa_pairs, 1):
                report_content += f"""
问答对 {i}:
- 类型: {qa_pair.get('type', '未知')}
- 难度: {qa_pair.get('difficulty', '未知')}
- 来源概念: {qa_pair.get('evidence_concept', '未知')}
- 问题: {qa_pair.get('question', '未知')}
- 答案: {qa_pair.get('answer', '未知')[:100]}...
{'-'*40}
"""
            
            # 保存txt报告
            self.save_step_txt_report("步骤7_问答生成", "7", report_content)
            
            # 保存问答对的完整内容
            qa_details_file = self.step_dirs["step7"] / "问答对详细内容.txt"
            with open(qa_details_file, 'w', encoding='utf-8') as f:
                f.write("问答生成详细结果\n")
                f.write("=" * 80 + "\n\n")
                
                for i, qa_pair in enumerate(all_qa_pairs, 1):
                    f.write(f"问答对 {i}\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"类型: {qa_pair.get('type', '未知')}\n")
                    f.write(f"难度: {qa_pair.get('difficulty', '未知')}\n")
                    f.write(f"来源概念: {qa_pair.get('evidence_concept', '未知')}\n")
                    f.write(f"来源证据ID: {qa_pair.get('evidence_source', '未知')}\n")
                    f.write(f"生成时间: {qa_pair.get('generation_timestamp', '未知')}\n")
                    f.write("问题:\n")
                    f.write(qa_pair.get('question', '未知'))
                    f.write("\n\n答案:\n")
                    f.write(qa_pair.get('answer', '未知'))
                    f.write("\n\n" + "=" * 80 + "\n\n")
            
            # 生成标准格式的训练数据
            training_data = []
            for i, qa_pair in enumerate(all_qa_pairs):
                training_item = {
                    "_id": uuid.uuid4().hex,
                    "Question": qa_pair.get("question", ""),
                    "Answer": qa_pair.get("answer", ""),
                    "Type": qa_pair.get("type", "unknown"),
                    "Difficulty": qa_pair.get("difficulty", "medium"),
                    "Domain": getattr(self.qa_generator, 'domain', 'general'),
                    "Rationale": qa_pair.get("rationale", ""),
                    "Context": []  # 这里可以根据需要构建Context
                }
                training_data.append(training_item)
            
            # 保存训练数据
            training_data_file = self.step_dirs["step7"] / "训练数据.jsonl"
            with open(training_data_file, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            result = {
                "success": True,
                "qa_pairs": all_qa_pairs,
                "training_data": training_data,
                "qa_count": len(all_qa_pairs),
                "qa_types": qa_types,
                "processing_time": processing_time
            }
            
            self.step_results["step7"] = result
            self.step_timings["step7"] = processing_time
            
            print(f"✅ 步骤7完成 - 问答生成成功")
            print(f"   生成问答对: {len(all_qa_pairs)} 个")
            print(f"   问题类型: {list(qa_types.keys())}")
            print(f"   处理时间: {processing_time:.2f} 秒")
            
            return result
            
        except Exception as e:
            error_msg = f"问答生成失败: {str(e)}"
            logger.error(error_msg)
            
            error_report = f"""
错误信息:
{error_msg}

错误详情:
{traceback.format_exc()}
"""
            self.save_step_txt_report("步骤7_问答生成_错误", "7", error_report)
            
            result = {
                "success": False,
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
            
            self.step_results["step7"] = result
            return result
    
    def step8_final_summary(self) -> Dict[str, Any]:
        """步骤8: 最终汇总"""
        print("\n" + "="*60)
        print("🔄 步骤8: 最终汇总")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # 汇总所有步骤的结果
            total_processing_time = sum(self.step_timings.values())
            
            # 🔧 添加这一行：计算当前步骤的处理时间
            processing_time = time.time() - start_time
            
            # 生成最终汇总报告
            report_content = f"""
Pipeline处理完整汇总报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
处理文件: {self.input_file}

总体统计:
- 总处理时间: {total_processing_time:.2f} 秒
- 成功步骤数: {sum(1 for step in self.step_results.values() if step.get('success', False))}
- 失败步骤数: {sum(1 for step in self.step_results.values() if not step.get('success', False))}

各步骤耗时:
{'-'*60}
"""
            
            step_names = [
                "步骤1: 文件加载",
                "步骤2: 文档分块", 
                "步骤3: 概念提取",
                "步骤4: 概念合并",
                "步骤5: 概念检索", 
                "步骤6: 证据提取",
                "步骤7: 问答生成"
            ]
            
            for i, step_name in enumerate(step_names, 1):
                step_key = f"step{i}"
                if step_key in self.step_timings:
                    timing = self.step_timings[step_key]
                    status = "✅ 成功" if self.step_results[step_key]["success"] else "❌ 失败"
                    report_content += f"- {step_name}: {status}, 耗时: {timing:.2f} 秒\n"
            
            # 保存txt报告
            self.save_step_txt_report("步骤8_最终汇总", "8", report_content)
            
            result = {
                "success": True,
                "total_processing_time": total_processing_time,
                "step_results": self.step_results,
                "step_timings": self.step_timings
            }
            
            self.step_results["step8"] = result
            self.step_timings["step8"] = processing_time
            
            print(f"✅ 步骤8完成 - 最终汇总成功")
            print(f"   总处理时间: {total_processing_time:.2f} 秒")
            print(f"   成功步骤数: {sum(1 for step in self.step_results.values() if step.get('success', False))} 个")
            print(f"   失败步骤数: {sum(1 for step in self.step_results.values() if not step.get('success', False))} 个")
            
            return result
            
        except Exception as e:
            error_msg = f"最终汇总失败: {str(e)}"
            logger.error(error_msg)
            
            error_report = f"""
错误信息:
{error_msg}

错误详情:
{traceback.format_exc()}
"""
            self.save_step_txt_report("步骤8_最终汇总_错误", "8", error_report)
            
            result = {
                "success": False,
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
            
            self.step_results["step8"] = result
            return result

def main():
    """主函数 - 执行分步骤处理"""
    print("🔄 分步骤Pipeline处理工具")
    print("=" * 50)
    
    # 🔧 修改：设置默认输入文件
    default_file = "attention is all you need.pdf"
    input_file = input(f"请输入文件路径 (默认: {default_file}): ").strip()
    if not input_file:
        input_file = default_file
        print(f"✅ 使用默认文件: {default_file}")
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return
    
    # 配置输出目录
    output_dir = input("输出目录 (默认: step_by_step_results_en): ").strip()
    if not output_dir:
        output_dir = "step_by_step_results_en"
    
    # 询问是否启用QA生成
    enable_qa = input("是否启用问答生成? (y/N): ").strip().lower() == 'y'
    
    qa_config = {}
    qa_model = "gpt-4o-mini"
    
    if enable_qa:
        print("\n配置问答生成参数:")
        print("1. 使用默认配置")
        print("2. 自定义配置")
        
        config_choice = input("请选择 (1-2): ").strip()
        
        if config_choice == "2":
            print("\n请输入每种类型的问题数量:")
            qa_config = {
                "remember": int(input("Remember类型 (默认2): ") or "2"),
                "understand": int(input("Understand类型 (默认2): ") or "2"),
                "apply": int(input("Apply类型 (默认1): ") or "1"),
                "analyze": int(input("Analyze类型 (默认1): ") or "1"),
                "evaluate": int(input("Evaluate类型 (默认1): ") or "1"),
                "create": int(input("Create类型 (默认1): ") or "1")
            }
        
        qa_model = input("QA生成模型 (默认gpt-4o-mini): ").strip() or "gpt-4o-mini"
    
    print(f"\n📄 开始处理文件: {input_path.name}")
    print(f"📁 输出目录: {output_dir}")
    print(f"❓ 问答生成: {'启用' if enable_qa else '禁用'}")
    
    try:
        # 初始化处理器
        processor = StepByStepPipelineProcessor(
            input_file=str(input_path),
            output_dir=output_dir,
            enable_qa_generation=enable_qa,
            qa_model_name=qa_model,
            questions_per_type=qa_config if qa_config else None
        )
        
        # 执行所有步骤
        print("\n🚀 开始执行分步骤处理...")
        
        # 步骤1: 文件加载
        step1_result = processor.step1_load_file()
        if not step1_result["success"]:
            print("❌ 步骤1失败，终止处理")
            return
        
        # 步骤2: 文档分块
        step2_result = processor.step2_document_chunking()
        if not step2_result["success"]:
            print("❌ 步骤2失败，终止处理")
            return
        
        # 步骤3: 概念提取
        step3_result = processor.step3_concept_extraction()
        if not step3_result["success"]:
            print("❌ 步骤3失败，终止处理")
            return
        
        # 步骤4: 概念合并
        step4_result = processor.step4_concept_merging()
        if not step4_result["success"]:
            print("❌ 步骤4失败，终止处理")
            return
        
        # 步骤5: 概念检索
        step5_result = processor.step5_concept_retrieval()
        if not step5_result["success"]:
            print("❌ 步骤5失败，终止处理")
            return
        
        # 步骤6: 证据提取
        step6_result = processor.step6_evidence_extraction()
        if not step6_result["success"]:
            print("❌ 步骤6失败，终止处理")
            return
        
        # 步骤7: 问答生成
        step7_result = processor.step7_qa_generation()
        if not step7_result["success"] and not step7_result.get("skipped"):
            print("❌ 步骤7失败，继续执行汇总")
        
        # 步骤8: 最终汇总
        step8_result = processor.step8_final_summary()
        
        # 显示最终结果
        print("\n" + "="*60)
        print("🎉 分步骤处理完成!")
        print("="*60)
        
        total_time = sum(processor.step_timings.values())
        success_count = sum(1 for step in processor.step_results.values() if step.get('success', False))
        
        print(f"📊 处理结果:")
        print(f"   - 总处理时间: {total_time:.2f} 秒")
        print(f"   - 成功步骤数: {success_count}/8")
        print(f"   - 输出目录: {processor.output_dir}")
        
        # 显示各步骤结果
        step_names = [
            "文件加载", "文档分块", "概念提取", "概念合并",
            "概念检索", "证据提取", "问答生成", "最终汇总"
        ]
        
        print(f"\n📋 各步骤状态:")
        for i, step_name in enumerate(step_names, 1):
            step_key = f"step{i}"
            if step_key in processor.step_results:
                result = processor.step_results[step_key]
                status = "✅" if result.get("success") else "⚠️" if result.get("skipped") else "❌"
                timing = processor.step_timings.get(step_key, 0)
                print(f"   {status} {step_name}: {timing:.2f}秒")
        
        print(f"\n📁 查看详细结果:")
        print(f"   文件夹: {processor.output_dir}/")
        print(f"   每个步骤都有独立的txt报告")
        
        if enable_qa and step7_result.get("success"):
            qa_count = step7_result.get("qa_count", 0)
            print(f"\n❓ 问答生成结果:")
            print(f"   - 生成问答对: {qa_count} 个")
            print(f"   - 训练数据: {processor.output_dir}/07_问答生成/训练数据.jsonl")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 处理被用户中断")
    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {str(e)}")
        logger.error(f"主程序错误: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
