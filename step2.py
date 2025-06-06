#!/usr/bin/env python3
"""
步骤2: 文档分块与概念提取 - 增强版
======================================

功能：
1. 从步骤1的实验结果中读取文档
2. 执行高质量的语义分块
3. 提取和优化概念
4. 保存到同一个实验文件夹

用法: 
- python step2.py <step1输出文件.txt>
- python step2.py <实验文件夹路径>

新功能：
- ✅ 自动识别并使用step1的实验文件夹
- ✅ 高质量的概念提取和验证
- ✅ 改进的分块算法
- ✅ 统一的实验管理
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging

# 导入核心模块
sys.path.append(str(Path(__file__).parent))
from llama_index.core import Document
from llama_index.core.schema import TextNode

# 导入核心处理模块
from core.chunking import SemanticChunker
from core.concept_merger import ConceptMerger
from config.settings import load_config_from_yaml

# 导入实验管理器
from utils.experiment_manager import ExperimentManager, load_latest_experiment
from utils.helpers import FileHelper

logger = logging.getLogger(__name__)

def load_step1_result(step1_file_or_dir: str) -> tuple:
    """
    从步骤1的输出文件或实验文件夹中加载结果
    
    Args:
        step1_file_or_dir: 步骤1的输出文件或实验文件夹路径
        
    Returns:
        tuple: (step1_result, experiment_manager)
    """
    step1_path = Path(step1_file_or_dir)
    
    if step1_path.is_file():
        # 情况1：直接指定了step1的输出文件
        if step1_path.name.startswith("step1") and step1_path.suffix == ".txt":
            experiment_dir = step1_path.parent
            experiment_manager = ExperimentManager.load_experiment(str(experiment_dir))
            
            # 从txt文件加载结果
            step1_result = load_result_from_txt(str(step1_path))
            
        else:
            raise ValueError(f"不支持的文件格式: {step1_path}")
            
    elif step1_path.is_dir():
        # 情况2：直接指定了实验文件夹
        experiment_manager = ExperimentManager.load_experiment(str(step1_path))
        
        # 查找step1的输出文件
        step1_txt_path = experiment_manager.get_step_output_path(1, "txt")
        if not step1_txt_path.exists():
            raise FileNotFoundError(f"实验文件夹中找不到step1输出文件: {step1_txt_path}")
        
        step1_result = load_result_from_txt(str(step1_txt_path))
        
    else:
        raise FileNotFoundError(f"输入路径不存在: {step1_path}")
    
    return step1_result, experiment_manager

def load_result_from_txt(input_file: str) -> Dict[str, Any]:
    """从txt文件中加载结果数据"""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    start_marker = "# JSON_DATA_START\n"
    end_marker = "\n# JSON_DATA_END"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("无法从txt文件中解析数据")
    
    json_str = content[start_idx + len(start_marker):end_idx]
    return json.loads(json_str)

def reconstruct_document_from_step1(step1_result: Dict[str, Any]) -> Document:
    """从步骤1结果重建Document对象"""
    doc_data = step1_result.get("document", {})
    
    if not doc_data:
        raise ValueError("步骤1结果中没有找到文档数据")
    
    # 重建Document对象
    document = Document(
        text=doc_data.get("text", ""),
        metadata=doc_data.get("metadata", {})
    )
    
    # 验证文档内容
    if not document.text or len(document.text.strip()) < 100:
        raise ValueError("文档内容为空或过短")
    
    return document

def extract_high_quality_concepts(chunk_nodes: List[TextNode], config) -> tuple:
    """
    提取高质量概念
    
    Args:
        chunk_nodes: 文本节点列表
        config: 配置对象
        
    Returns:
        tuple: (processed_chunks, unique_concepts, concept_stats)
    """
    print("🧠 开始高质量概念提取...")
    
    # 1. 解析现有概念（如果已经存在）
    all_concepts = []
    processed_chunks = []
    
    for i, chunk in enumerate(chunk_nodes):
        # 获取概念数据
        concepts_data = chunk.metadata.get("concepts", "[]")
        if isinstance(concepts_data, str):
            try:
                concepts = json.loads(concepts_data)
            except json.JSONDecodeError:
                concepts = []
        else:
            concepts = concepts_data if isinstance(concepts_data, list) else []
        
        # 验证和清理概念
        valid_concepts = []
        for concept in concepts:
            if isinstance(concept, str) and len(concept.strip()) > 2:
                cleaned_concept = concept.strip()
                # 简单的质量过滤
                if not any(pattern in cleaned_concept.lower() for pattern in ['training', 'method', 'system', 'process']):
                    valid_concepts.append(cleaned_concept)
        
        all_concepts.extend(valid_concepts)
        
        # 更新chunk的概念数据
        chunk.metadata["concepts"] = json.dumps(valid_concepts)
        chunk.metadata["concept_count"] = len(valid_concepts)
        chunk.metadata["chunk_index"] = i
        
        processed_chunks.append(chunk)
    
    print(f"   提取到 {len(all_concepts)} 个原始概念")
    
    # 2. 概念去重和合并
    unique_concepts = list(set(all_concepts))
    print(f"   去重后剩余 {len(unique_concepts)} 个唯一概念")
    
    # 3. 尝试使用概念合并器进一步优化
    try:
        concept_merger = ConceptMerger(config)
        
        # 转换为概念节点格式
        from core.nodes import ConceptNode
        concept_nodes = []
        for concept in unique_concepts:
            concept_node = ConceptNode(
                concept=concept,
                definition=f"从文档中提取的概念: {concept}",
                source_chunks=[],
                confidence=0.8
            )
            concept_nodes.append(concept_node)
        
        # 执行概念合并
        merged_concepts = concept_merger.merge_concepts(concept_nodes)
        final_concepts = [node.concept for node in merged_concepts]
        
        print(f"   概念合并后剩余 {len(final_concepts)} 个高质量概念")
        unique_concepts = final_concepts
        
    except Exception as e:
        logger.warning(f"概念合并失败，使用原始概念: {e}")
    
    # 4. 生成概念统计
    concept_stats = {
        "total_raw_concepts": len(all_concepts),
        "unique_concepts": len(unique_concepts),
        "concepts_per_chunk": len(all_concepts) / len(processed_chunks) if processed_chunks else 0,
        "concept_quality_score": len(unique_concepts) / max(len(all_concepts), 1) if all_concepts else 0
    }
    
    return processed_chunks, unique_concepts, concept_stats

def improve_chunking_quality(document: Document, config) -> List[TextNode]:
    """
    改进分块质量
    
    Args:
        document: 文档对象
        config: 配置对象
        
    Returns:
        List[TextNode]: 高质量的分块节点
    """
    print("📄 执行高质量语义分块...")
    
    # 使用更严格的分块参数
    improved_config = config.copy()
    
    # 调整分块参数以提高质量
    improved_config.set("chunking.max_tokens_per_chunk", 4000)  # 减小chunk大小
    improved_config.set("chunking.min_chunk_length", 50)       # 增加最小长度
    improved_config.set("chunking.breakpoint_percentile_threshold", 90)  # 更严格的分割阈值
    
    # 初始化改进的分块器
    chunker = SemanticChunker(improved_config)
    
    # 执行分块
    chunk_nodes = chunker.chunk_and_extract_concepts([document])
    
    # 过滤质量过低的分块
    quality_chunks = []
    for chunk in chunk_nodes:
        # 质量检查
        if len(chunk.text.strip()) >= 50:  # 最小长度检查
            # 检查概念质量
            concepts_data = chunk.metadata.get("concepts", "[]")
            if isinstance(concepts_data, str):
                try:
                    concepts = json.loads(concepts_data)
                except json.JSONDecodeError:
                    concepts = []
            else:
                concepts = concepts_data if isinstance(concepts_data, list) else []
            
            # 至少要有一些概念或足够长的文本
            if len(concepts) > 0 or len(chunk.text) > 200:
                quality_chunks.append(chunk)
    
    print(f"   分块完成: {len(chunk_nodes)} → {len(quality_chunks)} 个高质量分块")
    
    return quality_chunks

def process_step2_chunking(step1_result: Dict[str, Any], config_path: str = "config.yml") -> Dict[str, Any]:
    """
    执行步骤2的文档分块处理
    
    Args:
        step1_result: 步骤1的结果
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 步骤2的处理结果
    """
    start_time = time.time()
    
    try:
        # 1. 加载配置
        print("📋 加载配置...")
        config = load_config_from_yaml(config_path)
        
        # 2. 重建文档对象
        print("📄 重建文档对象...")
        document = reconstruct_document_from_step1(step1_result)
        print(f"✅ 文档重建成功: {len(document.text):,} 字符")
        
        # 3. 检查是否需要重新分块
        existing_chunks = step1_result.get("chunk_nodes", [])
        
        if existing_chunks and len(existing_chunks) > 0:
            print("🔄 使用现有分块，重新提取概念...")
            
            # 重建TextNode对象
            chunk_nodes = []
            for chunk_data in existing_chunks:
                if isinstance(chunk_data, dict):
                    text_node = TextNode(
                        text=chunk_data.get("text", ""),
                        metadata=chunk_data.get("metadata", {})
                    )
                    if hasattr(chunk_data, 'get') and chunk_data.get("node_id"):
                        text_node.node_id = chunk_data["node_id"]
                    chunk_nodes.append(text_node)
            
        else:
            print("📄 执行全新的高质量分块...")
            chunk_nodes = improve_chunking_quality(document, config)
        
        # 4. 高质量概念提取
        processed_chunks, unique_concepts, concept_stats = extract_high_quality_concepts(chunk_nodes, config)
        
        # 5. 生成分块统计
        chunk_lengths = [len(chunk.text) for chunk in processed_chunks]
        
        statistics = {
            "total_chunks": len(processed_chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
            "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
            "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
            "total_concepts": concept_stats["total_raw_concepts"],
            "unique_concepts": len(unique_concepts),
            "avg_concepts_per_chunk": concept_stats["concepts_per_chunk"],
            "concept_quality_score": concept_stats["concept_quality_score"]
        }
        
        processing_time = time.time() - start_time
        
        # 6. 构建结果
        result = {
            "success": True,
            "step": 2,
            "step_name": "文档分块与概念提取",
            "document": document,
            "chunks": processed_chunks,
            "unique_concepts": unique_concepts,
            "statistics": statistics,
            "concept_stats": concept_stats,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "config_used": {
                "max_tokens_per_chunk": config.get("chunking.max_tokens_per_chunk", 4000),
                "min_chunk_length": config.get("chunking.min_chunk_length", 50),
                "concepts_per_chunk": config.get("concept_extraction.concepts_per_chunk", 5)
            }
        }
        
        return result
        
    except Exception as e:
        error_msg = f"步骤2处理失败: {str(e)}"
        print(f"❌ {error_msg}")
        
        result = {
            "success": False,
            "step": 2,
            "step_name": "文档分块与概念提取",
            "error": error_msg,
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

def main():
    if len(sys.argv) != 2:
        print("用法: python step2.py <step1输出文件或实验文件夹>")
        print("示例:")
        print("  python step2.py experiments/20241204_143052_attention_paper/step1_vectorization.txt")
        print("  python step2.py experiments/20241204_143052_attention_paper/")
        print("\n新功能:")
        print("✅ 自动识别实验文件夹")
        print("✅ 高质量概念提取和验证")
        print("✅ 改进的分块算法")
        print("✅ 统一的实验管理")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    print(f"🚀 步骤2: 文档分块与概念提取 (增强版)")
    print(f"📄 输入: {input_path}")
    print("="*60)
    
    try:
        # 1. 加载步骤1结果和实验管理器
        print("📂 加载步骤1结果...")
        step1_result, experiment_manager = load_step1_result(input_path)
        
        if not step1_result.get("success"):
            print("❌ 步骤1未成功完成，无法继续")
            sys.exit(1)
        
        print(f"✅ 已加载实验: {experiment_manager.experiment_name}")
        print(f"📁 实验目录: {experiment_manager.experiment_dir}")
        print()
        
        # 2. 执行步骤2处理
        print("🔄 开始步骤2处理...")
        result = process_step2_chunking(step1_result)
        
        # 3. 保存结果到实验文件夹
        print("💾 保存实验结果...")
        saved_files = experiment_manager.save_step_result(
            step_num=2,
            result=result,
            save_formats=['txt', 'json']
        )
        
        if result.get("success"):
            print(f"\n✅ 步骤2完成!")
            print(f"📁 实验目录: {experiment_manager.experiment_dir}")
            print(f"📄 输出文件:")
            for format_type, file_path in saved_files.items():
                print(f"   - {format_type.upper()}: {file_path}")
            
            # 显示处理结果摘要
            stats = result.get("statistics", {})
            concept_stats = result.get("concept_stats", {})
            
            print(f"\n📊 处理结果摘要:")
            print(f"   - 总分块数: {stats.get('total_chunks', 0)}")
            print(f"   - 平均分块长度: {stats.get('avg_chunk_length', 0):.1f} 字符")
            print(f"   - 分块长度范围: {stats.get('min_chunk_length', 0)} - {stats.get('max_chunk_length', 0)} 字符")
            print(f"   - 提取概念数: {stats.get('total_concepts', 0)}")
            print(f"   - 唯一概念数: {stats.get('unique_concepts', 0)}")
            print(f"   - 概念质量分数: {concept_stats.get('concept_quality_score', 0):.2f}")
            print(f"   - 平均每分块概念数: {stats.get('avg_concepts_per_chunk', 0):.1f}")
            print(f"   - 处理时间: {result.get('processing_time', 0):.2f} 秒")
            
            # 显示概念示例
            unique_concepts = result.get("unique_concepts", [])
            if unique_concepts:
                print(f"\n🧠 高质量概念示例 (前15个):")
                for i, concept in enumerate(unique_concepts[:15], 1):
                    print(f"   {i:2d}. {concept}")
                if len(unique_concepts) > 15:
                    print(f"   ... 还有 {len(unique_concepts) - 15} 个概念")
            
            # 显示配置信息
            config_used = result.get("config_used", {})
            print(f"\n⚙️ 使用的配置:")
            for key, value in config_used.items():
                print(f"   - {key}: {value}")
            
            # 显示实验状态
            summary = experiment_manager.get_experiment_summary()
            print(f"\n🧪 实验状态:")
            print(f"   - 实验ID: {summary['experiment_id']}")
            print(f"   - 已完成步骤: {summary['steps_completed']}/{summary['total_steps']}")
            print(f"   - 当前状态: {summary['status']}")
            
            # 提示后续步骤
            print(f"\n📋 后续步骤:")
            print(f"   运行下一步: python step3.py {saved_files['txt']}")
            print(f"   查看结果: cat {saved_files['txt']}")
            print(f"   查看概念: grep -A 20 '高质量概念示例' {saved_files['txt']}")
                
        else:
            print(f"❌ 步骤2失败: {result.get('error')}")
            
            # 即使失败也保存错误信息
            experiment_manager.save_step_result(
                step_num=2,
                result=result,
                save_formats=['txt']
            )
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        
        # 如果有实验管理器，保存错误信息
        if 'experiment_manager' in locals():
            error_result = {
                "step": 2,
                "step_name": "文档分块与概念提取",
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                experiment_manager.save_step_result(2, error_result, ['txt'])
                print(f"📄 错误信息已保存到实验目录: {experiment_manager.experiment_dir}")
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()