#!/usr/bin/env python3
"""
步骤5: 概念检索与映射 - 增强版
===============================

功能：
1. 从步骤4的结果中读取合并后的概念
2. 从步骤2的结果中重建分块数据
3. 执行概念到分块的智能检索
4. 计算相似度和相关性分数
5. 保存到同一个实验文件夹

用法: 
- python step5.py <step4输出文件.txt>
- python step5.py <实验文件夹路径>

新功能：
- ✅ 自动识别并使用step4的实验文件夹
- ✅ 智能概念到分块的检索算法
- ✅ 多维度相似度计算和评分
- ✅ 统一的实验管理
- ✅ 检索结果质量评估
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
import logging
import numpy as np

# 导入核心模块
sys.path.append(str(Path(__file__).parent))
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import TextNode

# 导入核心处理模块
from config.settings import load_config_from_yaml
from core.nodes import ConceptNode

# 导入实验管理器
from utils.experiment_manager import ExperimentManager
from utils.helpers import FileHelper

logger = logging.getLogger(__name__)

def load_step4_result(step4_file_or_dir: str) -> tuple:
    """
    从步骤4的输出文件或实验文件夹中加载结果
    
    Args:
        step4_file_or_dir: 步骤4的输出文件或实验文件夹路径
        
    Returns:
        tuple: (step4_result, experiment_manager)
    """
    step4_path = Path(step4_file_or_dir)
    
    if step4_path.is_file():
        # 情况1：直接指定了step4的输出文件
        if step4_path.name.startswith("step4") and step4_path.suffix == ".txt":
            experiment_dir = step4_path.parent
            experiment_manager = ExperimentManager.load_experiment(str(experiment_dir))
            
            # 从txt文件加载结果
            step4_result = load_result_from_txt(str(step4_path))
            
        else:
            raise ValueError(f"不支持的文件格式: {step4_path}")
            
    elif step4_path.is_dir():
        # 情况2：直接指定了实验文件夹
        experiment_manager = ExperimentManager.load_experiment(str(step4_path))
        
        # 查找step4的输出文件
        step4_txt_path = experiment_manager.get_step_output_path(4, "txt")
        if not step4_txt_path.exists():
            raise FileNotFoundError(f"实验文件夹中找不到step4输出文件: {step4_txt_path}")
        
        step4_result = load_result_from_txt(str(step4_txt_path))
        
    else:
        raise FileNotFoundError(f"输入路径不存在: {step4_path}")
    
    return step4_result, experiment_manager

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

def load_previous_steps_data(experiment_manager: ExperimentManager) -> Dict[str, Any]:
    """
    加载之前步骤的数据
    
    Args:
        experiment_manager: 实验管理器
        
    Returns:
        Dict[str, Any]: 包含之前步骤数据的字典
    """
    previous_data = {}
    
    # 加载步骤2的数据（需要分块信息）
    step2_path = experiment_manager.get_step_output_path(2, "txt")
    if step2_path.exists():
        try:
            step2_result = load_result_from_txt(str(step2_path))
            previous_data["step2"] = step2_result
            print(f"✅ 加载步骤2数据: {step2_path}")
        except Exception as e:
            print(f"⚠️ 加载步骤2数据失败: {e}")
    
    return previous_data

def extract_concept_nodes_from_step4(step4_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从步骤4结果中提取概念节点
    
    Args:
        step4_result: 步骤4的结果
        
    Returns:
        List[Dict[str, Any]]: 概念节点列表
    """
    print("📊 从步骤4结果中提取概念节点...")
    
    concept_nodes = step4_result.get("concept_nodes", [])
    
    if not concept_nodes:
        raise ValueError("步骤4结果中没有找到概念节点数据")
    
    print(f"   - 概念节点数: {len(concept_nodes)}")
    
    # 验证概念节点格式
    valid_concepts = []
    for concept in concept_nodes:
        if isinstance(concept, dict) and "concept_text" in concept:
            valid_concepts.append(concept)
        else:
            print(f"⚠️ 跳过无效的概念节点: {type(concept)}")
    
    print(f"   - 有效概念节点数: {len(valid_concepts)}")
    
    return valid_concepts

def reconstruct_chunks_from_step2(step2_result: Dict[str, Any]) -> List[TextNode]:
    """
    从步骤2结果重建TextNode对象
    
    Args:
        step2_result: 步骤2的结果
        
    Returns:
        List[TextNode]: 分块节点列表
    """
    print("🔄 从步骤2结果中重建分块数据...")
    
    # 尝试不同的字段名（兼容性）
    chunks_data = None
    for field_name in ["chunks", "chunk_nodes", "processed_chunks"]:
        if field_name in step2_result:
            chunks_data = step2_result[field_name]
            print(f"✅ 找到分块数据字段: {field_name}")
            break
    
    if not chunks_data:
        available_fields = list(step2_result.keys())
        print(f"❌ 未找到分块数据，可用字段: {available_fields}")
        raise ValueError("步骤2结果中没有找到分块数据")
    
    # 重建TextNode对象
    chunk_nodes = []
    for i, chunk_data in enumerate(chunks_data):
        try:
            if isinstance(chunk_data, dict):
                # 从字典数据重建TextNode
                text_node = TextNode(
                    text=chunk_data.get("text", ""),
                    metadata=chunk_data.get("metadata", {})
                )
                
                # 设置node_id
                if "node_id" in chunk_data:
                    text_node.node_id = chunk_data["node_id"]
                
                chunk_nodes.append(text_node)
                
            elif hasattr(chunk_data, 'text'):
                # 已经是TextNode对象
                chunk_nodes.append(chunk_data)
                
            else:
                print(f"⚠️ 跳过无效的chunk数据: {type(chunk_data)}")
                continue
                
        except Exception as e:
            print(f"⚠️ 重建chunk {i} 失败: {e}")
            continue
    
    print(f"✅ 成功重建 {len(chunk_nodes)} 个TextNode对象")
    return chunk_nodes

def calculate_concept_chunk_similarity(concept_text: str, chunk_text: str, chunk_concepts: List[str] = None) -> float:
    """
    计算概念与分块的相似度
    
    Args:
        concept_text: 概念文本
        chunk_text: 分块文本
        chunk_concepts: 分块中的概念列表（可选）
        
    Returns:
        float: 相似度分数 (0-1)
    """
    from difflib import SequenceMatcher
    
    # 1. 直接文本相似度
    text_similarity = SequenceMatcher(None, concept_text.lower(), chunk_text.lower()).ratio()
    
    # 2. 关键词匹配度
    concept_words = set(concept_text.lower().split())
    chunk_words = set(chunk_text.lower().split())
    
    if concept_words and chunk_words:
        keyword_overlap = len(concept_words.intersection(chunk_words)) / len(concept_words.union(chunk_words))
    else:
        keyword_overlap = 0.0
    
    # 3. 概念包含度（如果有chunk的概念信息）
    concept_containment = 0.0
    if chunk_concepts:
        chunk_concepts_lower = [c.lower() for c in chunk_concepts]
        if any(concept_text.lower() in cc or cc in concept_text.lower() for cc in chunk_concepts_lower):
            concept_containment = 0.5
    
    # 4. 长度相关性调整
    length_factor = min(len(concept_text), len(chunk_text)) / max(len(concept_text), len(chunk_text), 1)
    
    # 综合相似度计算
    final_similarity = (
        text_similarity * 0.3 +
        keyword_overlap * 0.4 +
        concept_containment * 0.2 +
        length_factor * 0.1
    )
    
    return min(1.0, max(0.0, final_similarity))

def perform_concept_retrieval(concept_nodes: List[Dict[str, Any]], 
                            chunk_nodes: List[TextNode],
                            config) -> Dict[str, Any]:
    """
    执行概念检索
    
    Args:
        concept_nodes: 概念节点列表
        chunk_nodes: 分块节点列表
        config: 配置对象
        
    Returns:
        Dict[str, Any]: 检索结果
    """
    print("🔍 开始执行概念检索...")
    
    top_k = config.get("retrieval.top_k", 5)
    similarity_threshold = config.get("retrieval.similarity_cutoff", 0.1)
    
    retrieval_results = {}
    total_retrievals = 0
    concepts_with_retrievals = 0
    all_similarities = []
    
    for concept in concept_nodes:
        concept_id = concept.get("concept_id", "unknown")
        concept_text = concept.get("concept_text", "")
        
        print(f"   检索概念: {concept_text}")
        
        # 为当前概念检索相关分块
        chunk_similarities = []
        
        for chunk in chunk_nodes:
            # 获取分块的概念信息
            chunk_concepts = []
            concepts_data = chunk.metadata.get("concepts", "[]")
            if isinstance(concepts_data, str):
                try:
                    chunk_concepts = json.loads(concepts_data)
                except:
                    pass
            elif isinstance(concepts_data, list):
                chunk_concepts = concepts_data
            
            # 计算相似度
            similarity = calculate_concept_chunk_similarity(
                concept_text, 
                chunk.text, 
                chunk_concepts
            )
            
            if similarity >= similarity_threshold:
                chunk_similarities.append({
                    "chunk_id": chunk.metadata.get("chunk_id", chunk.node_id),
                    "chunk_text": chunk.text,
                    "chunk_concepts": chunk_concepts,
                    "similarity_score": similarity,
                    "node_id": chunk.node_id
                })
        
        # 按相似度排序并取前k个
        chunk_similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_chunks = chunk_similarities[:top_k]
        
        # 统计信息
        retrieval_count = len(top_chunks)
        avg_similarity = sum(c["similarity_score"] for c in top_chunks) / len(top_chunks) if top_chunks else 0
        
        all_similarities.extend([c["similarity_score"] for c in top_chunks])
        total_retrievals += retrieval_count
        
        if retrieval_count > 0:
            concepts_with_retrievals += 1
        
        # 保存检索结果
        retrieval_results[concept_id] = {
            "concept_text": concept_text,
            "concept_id": concept_id,
            "retrieval_count": retrieval_count,
            "avg_similarity": avg_similarity,
            "max_similarity": max([c["similarity_score"] for c in top_chunks]) if top_chunks else 0,
            "min_similarity": min([c["similarity_score"] for c in top_chunks]) if top_chunks else 0,
            "retrieved_chunks": top_chunks,
            "source_chunks": concept.get("source_chunks", []),
            "confidence_score": concept.get("confidence_score", 0)
        }
    
    # 计算全局统计
    avg_similarity_all = sum(all_similarities) / len(all_similarities) if all_similarities else 0
    avg_retrievals_per_concept = total_retrievals / len(concept_nodes) if concept_nodes else 0
    
    print(f"   ✅ 检索完成:")
    print(f"      - 总检索结果: {total_retrievals}")
    print(f"      - 有结果的概念: {concepts_with_retrievals}/{len(concept_nodes)}")
    print(f"      - 平均相似度: {avg_similarity_all:.4f}")
    
    return {
        "retrieval_results": retrieval_results,
        "statistics": {
            "concept_count": len(concept_nodes),
            "chunk_count": len(chunk_nodes),
            "total_retrievals": total_retrievals,
            "concepts_with_retrievals": concepts_with_retrievals,
            "avg_retrievals_per_concept": avg_retrievals_per_concept,
            "avg_similarity_all": avg_similarity_all,
            "retrieval_coverage": concepts_with_retrievals / len(concept_nodes) if concept_nodes else 0,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold
        }
    }

def analyze_retrieval_quality(retrieval_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析检索质量
    
    Args:
        retrieval_data: 检索数据
        
    Returns:
        Dict[str, Any]: 质量分析结果
    """
    print("📈 分析检索质量...")
    
    retrieval_results = retrieval_data["retrieval_results"]
    
    # 质量指标计算
    high_quality_retrievals = 0  # 相似度 > 0.5
    medium_quality_retrievals = 0  # 相似度 0.2-0.5
    low_quality_retrievals = 0  # 相似度 < 0.2
    
    concept_coverage_scores = []
    similarity_distributions = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    
    for concept_id, result in retrieval_results.items():
        avg_sim = result["avg_similarity"]
        retrieval_count = result["retrieval_count"]
        
        # 概念覆盖度评分
        if retrieval_count >= 3 and avg_sim >= 0.3:
            coverage_score = 1.0
        elif retrieval_count >= 2 and avg_sim >= 0.2:
            coverage_score = 0.7
        elif retrieval_count >= 1 and avg_sim >= 0.1:
            coverage_score = 0.4
        else:
            coverage_score = 0.0
        
        concept_coverage_scores.append(coverage_score)
        
        # 质量分类
        if avg_sim >= 0.5:
            high_quality_retrievals += 1
        elif avg_sim >= 0.2:
            medium_quality_retrievals += 1
        else:
            low_quality_retrievals += 1
        
        # 相似度分布
        for chunk in result["retrieved_chunks"]:
            sim = chunk["similarity_score"]
            if sim >= 0.8:
                similarity_distributions["0.8-1.0"] += 1
            elif sim >= 0.6:
                similarity_distributions["0.6-0.8"] += 1
            elif sim >= 0.4:
                similarity_distributions["0.4-0.6"] += 1
            elif sim >= 0.2:
                similarity_distributions["0.2-0.4"] += 1
            else:
                similarity_distributions["0.0-0.2"] += 1
    
    # 整体质量评分
    total_concepts = len(retrieval_results)
    avg_coverage_score = sum(concept_coverage_scores) / len(concept_coverage_scores) if concept_coverage_scores else 0
    
    quality_score = (
        (high_quality_retrievals / total_concepts * 1.0) +
        (medium_quality_retrievals / total_concepts * 0.6) +
        (low_quality_retrievals / total_concepts * 0.2)
    ) if total_concepts > 0 else 0
    
    return {
        "overall_quality_score": quality_score,
        "avg_coverage_score": avg_coverage_score,
        "quality_distribution": {
            "high_quality": high_quality_retrievals,
            "medium_quality": medium_quality_retrievals,
            "low_quality": low_quality_retrievals
        },
        "similarity_distribution": similarity_distributions,
        "top_performing_concepts": sorted(
            [(cid, res["avg_similarity"], res["retrieval_count"]) 
             for cid, res in retrieval_results.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
    }

def process_step5_concept_retrieval(step4_result: Dict[str, Any], 
                                   previous_data: Dict[str, Any],
                                   config_path: str = "config.yml") -> Dict[str, Any]:
    """
    执行步骤5的概念检索处理
    
    Args:
        step4_result: 步骤4的结果
        previous_data: 之前步骤的数据
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 步骤5的处理结果
    """
    start_time = time.time()
    
    try:
        # 1. 加载配置
        print("📋 加载配置...")
        config = load_config_from_yaml(config_path)
        
        # 2. 提取概念节点
        print("📊 提取概念节点...")
        concept_nodes = extract_concept_nodes_from_step4(step4_result)
        
        # 3. 重建分块数据
        print("🔄 重建分块数据...")
        if "step2" not in previous_data:
            raise ValueError("缺少步骤2的分块数据")
        
        chunk_nodes = reconstruct_chunks_from_step2(previous_data["step2"])
        
        # 4. 执行概念检索
        print("🔍 执行概念检索...")
        retrieval_data = perform_concept_retrieval(concept_nodes, chunk_nodes, config)
        
        # 5. 分析检索质量
        print("📈 分析检索质量...")
        quality_analysis = analyze_retrieval_quality(retrieval_data)
        
        processing_time = time.time() - start_time
        
        # 6. 构建结果
        result = {
            "success": True,
            "step": 5,
            "step_name": "概念检索与映射",
            "retrieval_results": retrieval_data["retrieval_results"],
            "statistics": retrieval_data["statistics"],
            "quality_analysis": quality_analysis,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "config_used": {
                "top_k": config.get("retrieval.top_k", 5),
                "similarity_threshold": config.get("retrieval.similarity_cutoff", 0.1)
            }
        }
        
        return result
        
    except Exception as e:
        error_msg = f"步骤5处理失败: {str(e)}"
        print(f"❌ {error_msg}")
        
        result = {
            "success": False,
            "step": 5,
            "step_name": "概念检索与映射",
            "error": error_msg,
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

def main():
    if len(sys.argv) != 2:
        print("用法: python step5.py <step4输出文件或实验文件夹>")
        print("示例:")
        print("  python step5.py experiments/20241204_143052_attention_paper/step4_reranking.txt")
        print("  python step5.py experiments/20241204_143052_attention_paper/")
        print("\n新功能:")
        print("✅ 自动识别实验文件夹")
        print("✅ 智能概念到分块的检索算法")
        print("✅ 多维度相似度计算和评分")
        print("✅ 统一的实验管理")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    print(f"🚀 步骤5: 概念检索与映射 (增强版)")
    print(f"📄 输入: {input_path}")
    print("="*60)
    
    try:
        # 1. 加载步骤4结果和实验管理器
        print("📂 加载步骤4结果...")
        step4_result, experiment_manager = load_step4_result(input_path)
        
        if not step4_result.get("success"):
            print("❌ 步骤4未成功完成，无法继续")
            sys.exit(1)
        
        print(f"✅ 已加载实验: {experiment_manager.experiment_name}")
        print(f"📁 实验目录: {experiment_manager.experiment_dir}")
        print()
        
        # 2. 加载之前步骤的数据
        print("📂 加载之前步骤的数据...")
        previous_data = load_previous_steps_data(experiment_manager)
        
        # 3. 执行步骤5处理
        print("🔄 开始步骤5处理...")
        result = process_step5_concept_retrieval(step4_result, previous_data)
        
        # 4. 保存结果到实验文件夹
        print("💾 保存实验结果...")
        saved_files = experiment_manager.save_step_result(
            step_num=5,
            result=result,
            save_formats=['txt', 'json']
        )
        
        if result.get("success"):
            print(f"\n✅ 步骤5完成!")
            print(f"📁 实验目录: {experiment_manager.experiment_dir}")
            print(f"📄 输出文件:")
            for format_type, file_path in saved_files.items():
                print(f"   - {format_type.upper()}: {file_path}")
            
            # 显示处理结果摘要
            stats = result.get("statistics", {})
            quality = result.get("quality_analysis", {})
            
            print(f"\n📊 概念检索结果摘要:")
            print(f"   - 概念数量: {stats.get('concept_count', 0)}")
            print(f"   - 分块数量: {stats.get('chunk_count', 0)}")
            print(f"   - 总检索结果: {stats.get('total_retrievals', 0)}")
            print(f"   - 有结果的概念: {stats.get('concepts_with_retrievals', 0)}")
            print(f"   - 检索覆盖率: {stats.get('retrieval_coverage', 0):.2%}")
            print(f"   - 平均相似度: {stats.get('avg_similarity_all', 0):.4f}")
            print(f"   - 平均每概念检索数: {stats.get('avg_retrievals_per_concept', 0):.2f}")
            print(f"   - 处理时间: {result.get('processing_time', 0):.2f} 秒")
            
            # 显示质量分析
            print(f"\n📈 检索质量分析:")
            print(f"   - 整体质量评分: {quality.get('overall_quality_score', 0):.3f}")
            print(f"   - 平均覆盖度: {quality.get('avg_coverage_score', 0):.3f}")
            
            quality_dist = quality.get("quality_distribution", {})
            print(f"   - 高质量检索: {quality_dist.get('high_quality', 0)} 个")
            print(f"   - 中等质量检索: {quality_dist.get('medium_quality', 0)} 个")
            print(f"   - 低质量检索: {quality_dist.get('low_quality', 0)} 个")
            
            # 显示顶级检索结果
            top_concepts = quality.get("top_performing_concepts", [])
            if top_concepts:
                print(f"\n🌟 顶级检索结果 (前5个):")
                for i, (concept_id, avg_sim, count) in enumerate(top_concepts[:5], 1):
                    retrieval_results = result.get("retrieval_results", {})
                    concept_text = retrieval_results.get(concept_id, {}).get("concept_text", "未知")
                    print(f"   {i}. {concept_text}")
                    print(f"      相似度: {avg_sim:.4f}, 检索数: {count}")
            
            # 显示实验状态
            summary = experiment_manager.get_experiment_summary()
            print(f"\n🧪 实验状态:")
            print(f"   - 实验ID: {summary['experiment_id']}")
            print(f"   - 已完成步骤: {summary['steps_completed']}/{summary['total_steps']}")
            print(f"   - 当前状态: {summary['status']}")
            
            # 提示后续步骤
            print(f"\n📋 后续步骤:")
            print(f"   运行下一步: python step6.py {saved_files['txt']}")
            print(f"   查看结果: cat {saved_files['txt']}")
            print(f"   查看检索: grep -A 10 '顶级检索结果' {saved_files['txt']}")
                
        else:
            print(f"❌ 步骤5失败: {result.get('error')}")
            
            # 即使失败也保存错误信息
            experiment_manager.save_step_result(
                step_num=5,
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
                "step": 5,
                "step_name": "概念检索与映射",
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                experiment_manager.save_step_result(5, error_result, ['txt'])
                print(f"📄 错误信息已保存到实验目录: {experiment_manager.experiment_dir}")
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()