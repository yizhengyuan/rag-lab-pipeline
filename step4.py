#!/usr/bin/env python3
"""
步骤4: 概念合并与优化 - 增强版
===============================

功能：
1. 从步骤3的结果中读取概念数据
2. 执行智能概念合并和去重
3. 优化概念质量和层次结构
4. 保存到同一个实验文件夹

用法: 
- python step4.py <step3输出文件.txt>
- python step4.py <实验文件夹路径>

新功能：
- ✅ 自动识别并使用step3的实验文件夹
- ✅ 智能概念合并和相似度计算
- ✅ 概念层次结构优化
- ✅ 统一的实验管理
- ✅ 概念质量评估和过滤
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Set, Tuple
from collections import Counter, defaultdict
import logging

# 导入核心模块
sys.path.append(str(Path(__file__).parent))
from llama_index.core import Document
from llama_index.core.schema import TextNode

# 导入核心处理模块
from config.settings import load_config_from_yaml
from core.nodes import ConceptNode
from core.concept_merger import ConceptMerger

# 导入实验管理器
from utils.experiment_manager import ExperimentManager
from utils.helpers import FileHelper

logger = logging.getLogger(__name__)

def load_step3_result(step3_file_or_dir: str) -> tuple:
    """
    从步骤3的输出文件或实验文件夹中加载结果
    
    Args:
        step3_file_or_dir: 步骤3的输出文件或实验文件夹路径
        
    Returns:
        tuple: (step3_result, experiment_manager)
    """
    step3_path = Path(step3_file_or_dir)
    
    if step3_path.is_file():
        # 情况1：直接指定了step3的输出文件
        if step3_path.name.startswith("step3") and step3_path.suffix == ".txt":
            experiment_dir = step3_path.parent
            experiment_manager = ExperimentManager.load_experiment(str(experiment_dir))
            
            # 从txt文件加载结果
            step3_result = load_result_from_txt(str(step3_path))
            
        else:
            raise ValueError(f"不支持的文件格式: {step3_path}")
            
    elif step3_path.is_dir():
        # 情况2：直接指定了实验文件夹
        experiment_manager = ExperimentManager.load_experiment(str(step3_path))
        
        # 查找step3的输出文件
        step3_txt_path = experiment_manager.get_step_output_path(3, "txt")
        if not step3_txt_path.exists():
            raise FileNotFoundError(f"实验文件夹中找不到step3输出文件: {step3_txt_path}")
        
        step3_result = load_result_from_txt(str(step3_txt_path))
        
    else:
        raise FileNotFoundError(f"输入路径不存在: {step3_path}")
    
    return step3_result, experiment_manager

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

def extract_concepts_from_step3(step3_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    从步骤3结果中提取概念数据
    
    Args:
        step3_result: 步骤3的结果
        
    Returns:
        Dict[str, Any]: 提取的概念数据
    """
    print("📊 从步骤3结果中提取概念数据...")
    
    concept_analysis = step3_result.get("concept_analysis", {})
    
    if not concept_analysis:
        raise ValueError("步骤3结果中没有找到概念分析数据")
    
    # 提取概念信息
    all_concepts = concept_analysis.get("all_concepts", [])
    unique_concepts = concept_analysis.get("unique_concepts", [])
    concept_frequency = concept_analysis.get("concept_frequency", {})
    quality_scores = concept_analysis.get("quality_scores", {})
    high_quality_concepts = concept_analysis.get("high_quality_concepts", [])
    chunk_concept_map = concept_analysis.get("chunk_concept_map", {})
    
    print(f"   - 总概念数: {len(all_concepts)}")
    print(f"   - 唯一概念数: {len(unique_concepts)}")
    print(f"   - 高质量概念数: {len(high_quality_concepts)}")
    
    return {
        "all_concepts": all_concepts,
        "unique_concepts": unique_concepts,
        "concept_frequency": concept_frequency,
        "quality_scores": quality_scores,
        "high_quality_concepts": high_quality_concepts,
        "chunk_concept_map": chunk_concept_map
    }

def merge_similar_concepts(concepts_data: Dict[str, Any], config) -> Dict[str, Any]:
    """
    合并相似的概念
    
    Args:
        concepts_data: 概念数据
        config: 配置对象
        
    Returns:
        Dict[str, Any]: 合并后的概念数据
    """
    print("🔗 开始智能概念合并...")
    
    unique_concepts = concepts_data["unique_concepts"]
    concept_frequency = concepts_data["concept_frequency"]
    quality_scores = concepts_data["quality_scores"]
    
    # 1. 基于文本相似度进行初步分组
    concept_groups = group_similar_concepts(unique_concepts)
    print(f"   分组完成: {len(unique_concepts)} → {len(concept_groups)} 组")
    
    # 2. 为每组选择最佳代表概念
    merged_concepts = []
    merge_mapping = {}  # 原概念 -> 合并后概念
    
    for group in concept_groups:
        if len(group) == 1:
            # 单独的概念直接保留
            best_concept = group[0]
        else:
            # 从组中选择最佳概念作为代表
            best_concept = select_best_concept_from_group(
                group, concept_frequency, quality_scores
            )
        
        merged_concepts.append(best_concept)
        
        # 记录映射关系
        for concept in group:
            merge_mapping[concept] = best_concept
    
    print(f"   合并完成: {len(merged_concepts)} 个合并后概念")
    
    # 3. 重新计算合并后的统计信息
    merged_frequency = defaultdict(int)
    for original_concept, frequency in concept_frequency.items():
        merged_concept = merge_mapping[original_concept]
        merged_frequency[merged_concept] += frequency
    
    # 4. 计算合并后的质量分数
    merged_quality_scores = {}
    for merged_concept in merged_concepts:
        # 取所有映射到此概念的原概念的最高质量分数
        mapped_scores = [
            quality_scores.get(orig_concept, 0) 
            for orig_concept, merged in merge_mapping.items() 
            if merged == merged_concept
        ]
        merged_quality_scores[merged_concept] = max(mapped_scores) if mapped_scores else 0
    
    # 5. 按质量和频率重新排序
    sorted_by_quality = sorted(
        merged_concepts, 
        key=lambda x: (merged_quality_scores.get(x, 0), merged_frequency.get(x, 0)), 
        reverse=True
    )
    
    sorted_by_frequency = sorted(
        merged_frequency.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return {
        "merged_concepts": merged_concepts,
        "merge_mapping": dict(merge_mapping),
        "merged_frequency": dict(merged_frequency),
        "merged_quality_scores": merged_quality_scores,
        "sorted_by_quality": sorted_by_quality,
        "sorted_by_frequency": sorted_by_frequency,
        "concept_groups": concept_groups,
        "compression_ratio": len(merged_concepts) / len(unique_concepts) if unique_concepts else 1.0
    }

def group_similar_concepts(concepts: List[str], similarity_threshold: float = 0.7) -> List[List[str]]:
    """
    基于文本相似度对概念进行分组
    
    Args:
        concepts: 概念列表
        similarity_threshold: 相似度阈值
        
    Returns:
        List[List[str]]: 概念分组
    """
    from difflib import SequenceMatcher
    
    groups = []
    used = set()
    
    for i, concept1 in enumerate(concepts):
        if concept1 in used:
            continue
        
        group = [concept1]
        used.add(concept1)
        
        for j, concept2 in enumerate(concepts[i+1:], i+1):
            if concept2 in used:
                continue
            
            # 计算文本相似度
            similarity = calculate_text_similarity(concept1, concept2)
            
            if similarity >= similarity_threshold:
                group.append(concept2)
                used.add(concept2)
        
        groups.append(group)
    
    return groups

def calculate_text_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度"""
    from difflib import SequenceMatcher
    
    # 基础字符串相似度
    char_similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    # 词汇重叠相似度
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        word_similarity = 1.0
    elif not words1 or not words2:
        word_similarity = 0.0
    else:
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        word_similarity = len(intersection) / len(union)
    
    # 包含关系检查
    containment_similarity = 0.0
    if text1.lower() in text2.lower() or text2.lower() in text1.lower():
        containment_similarity = 0.5
    
    # 综合相似度
    final_similarity = max(
        char_similarity * 0.4 + word_similarity * 0.6,
        containment_similarity
    )
    
    return final_similarity

def select_best_concept_from_group(group: List[str], 
                                 concept_frequency: Dict[str, int], 
                                 quality_scores: Dict[str, float]) -> str:
    """
    从概念组中选择最佳代表概念
    
    Args:
        group: 概念组
        concept_frequency: 概念频率
        quality_scores: 质量分数
        
    Returns:
        str: 最佳概念
    """
    # 计算综合分数：质量分数 + 频率权重
    best_concept = group[0]
    best_score = 0
    
    for concept in group:
        quality = quality_scores.get(concept, 0)
        frequency = concept_frequency.get(concept, 0)
        
        # 综合分数：质量为主，频率为辅
        combined_score = quality * 0.7 + (frequency / 10) * 0.3
        
        if combined_score > best_score:
            best_score = combined_score
            best_concept = concept
    
    return best_concept

def create_concept_nodes(merged_data: Dict[str, Any], 
                        concepts_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    创建概念节点数据结构
    
    Args:
        merged_data: 合并后的概念数据
        concepts_data: 原始概念数据
        
    Returns:
        List[Dict[str, Any]]: 概念节点列表
    """
    print("🏗️ 创建概念节点...")
    
    merged_concepts = merged_data["merged_concepts"]
    merge_mapping = merged_data["merge_mapping"]
    merged_frequency = merged_data["merged_frequency"]
    merged_quality_scores = merged_data["merged_quality_scores"]
    chunk_concept_map = concepts_data["chunk_concept_map"]
    
    concept_nodes = []
    
    for i, concept in enumerate(merged_concepts):
        # 找到所有映射到此概念的原概念
        source_concepts = [
            orig for orig, merged in merge_mapping.items() 
            if merged == concept
        ]
        
        # 找到包含此概念的所有chunks
        source_chunks = []
        for chunk_id, chunk_concepts in chunk_concept_map.items():
            if any(orig_concept in chunk_concepts for orig_concept in source_concepts):
                source_chunks.append(chunk_id)
        
        # 计算置信度分数
        frequency = merged_frequency.get(concept, 0)
        quality = merged_quality_scores.get(concept, 0)
        coverage = len(source_chunks)
        
        # 置信度 = (质量分数 + 频率权重 + 覆盖度权重) / 3
        confidence_score = (
            quality / 5.0 +  # 质量分数标准化到0-1
            min(frequency / 10.0, 1.0) +  # 频率权重，最大为1
            min(coverage / 5.0, 1.0)  # 覆盖度权重，最大为1
        ) / 3.0
        
        # 创建概念节点
        concept_node = {
            "concept_id": f"merged_concept_{i}",
            "concept_text": concept,
            "concept_name": concept[:50],  # 限制名称长度
            "concept_length": len(concept),
            "source_concepts": source_concepts,
            "source_chunks": source_chunks,
            "frequency": frequency,
            "quality_score": quality,
            "confidence_score": confidence_score,
            "coverage": coverage,
            "merge_group_size": len(source_concepts),
            "created_at": datetime.now().isoformat()
        }
        
        concept_nodes.append(concept_node)
    
    # 按置信度排序
    concept_nodes.sort(key=lambda x: x["confidence_score"], reverse=True)
    
    print(f"   创建了 {len(concept_nodes)} 个概念节点")
    
    return concept_nodes

def process_step4_concept_merging(step3_result: Dict[str, Any], 
                                 previous_data: Dict[str, Any],
                                 config_path: str = "config.yml") -> Dict[str, Any]:
    """
    执行步骤4的概念合并处理
    
    Args:
        step3_result: 步骤3的结果
        previous_data: 之前步骤的数据
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 步骤4的处理结果
    """
    start_time = time.time()
    
    try:
        # 1. 加载配置
        print("📋 加载配置...")
        config = load_config_from_yaml(config_path)
        
        # 2. 提取概念数据
        print("📊 提取概念数据...")
        concepts_data = extract_concepts_from_step3(step3_result)
        
        # 3. 执行概念合并
        print("🔗 执行概念合并...")
        merged_data = merge_similar_concepts(concepts_data, config)
        
        # 4. 创建概念节点
        print("🏗️ 创建概念节点...")
        concept_nodes = create_concept_nodes(merged_data, concepts_data)
        
        # 5. 生成统计信息
        original_count = len(concepts_data["unique_concepts"])
        merged_count = len(merged_data["merged_concepts"])
        compression_ratio = merged_count / original_count if original_count > 0 else 1.0
        
        avg_confidence = sum(node["confidence_score"] for node in concept_nodes) / len(concept_nodes) if concept_nodes else 0
        avg_concept_length = sum(node["concept_length"] for node in concept_nodes) / len(concept_nodes) if concept_nodes else 0
        total_source_chunks = len(set().union(*(node["source_chunks"] for node in concept_nodes)))
        
        statistics = {
            "original_concept_count": original_count,
            "merged_concept_count": merged_count,
            "compression_ratio": compression_ratio,
            "avg_confidence": avg_confidence,
            "avg_concept_length": avg_concept_length,
            "total_source_chunks": total_source_chunks,
            "concept_groups_count": len(merged_data["concept_groups"]),
            "high_confidence_count": len([n for n in concept_nodes if n["confidence_score"] > 0.7]),
            "medium_confidence_count": len([n for n in concept_nodes if 0.4 <= n["confidence_score"] <= 0.7]),
            "low_confidence_count": len([n for n in concept_nodes if n["confidence_score"] < 0.4])
        }
        
        processing_time = time.time() - start_time
        
        # 6. 构建结果
        result = {
            "success": True,
            "step": 4,
            "step_name": "概念合并与优化",
            "concept_nodes": concept_nodes,
            "merged_data": merged_data,
            "input_statistics": {
                "original_concept_count": original_count,
                "compression_ratio": compression_ratio
            },
            "statistics": statistics,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "config_used": {
                "similarity_threshold": config.get("concept_merging.similarity_threshold", 0.7),
                "max_document_concepts": config.get("concept_merging.max_document_concepts", 10)
            }
        }
        
        return result
        
    except Exception as e:
        error_msg = f"步骤4处理失败: {str(e)}"
        print(f"❌ {error_msg}")
        
        result = {
            "success": False,
            "step": 4,
            "step_name": "概念合并与优化",
            "error": error_msg,
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

def main():
    if len(sys.argv) != 2:
        print("用法: python step4.py <step3输出文件或实验文件夹>")
        print("示例:")
        print("  python step4.py experiments/20241204_143052_attention_paper/step3_retrieval.txt")
        print("  python step4.py experiments/20241204_143052_attention_paper/")
        print("\n新功能:")
        print("✅ 自动识别实验文件夹")
        print("✅ 智能概念合并和相似度计算")
        print("✅ 概念层次结构优化")
        print("✅ 统一的实验管理")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    print(f"🚀 步骤4: 概念合并与优化 (增强版)")
    print(f"📄 输入: {input_path}")
    print("="*60)
    
    try:
        # 1. 加载步骤3结果和实验管理器
        print("📂 加载步骤3结果...")
        step3_result, experiment_manager = load_step3_result(input_path)
        
        if not step3_result.get("success"):
            print("❌ 步骤3未成功完成，无法继续")
            sys.exit(1)
        
        print(f"✅ 已加载实验: {experiment_manager.experiment_name}")
        print(f"📁 实验目录: {experiment_manager.experiment_dir}")
        print()
        
        # 2. 加载之前步骤的数据
        print("📂 加载之前步骤的数据...")
        previous_data = load_previous_steps_data(experiment_manager)
        
        # 3. 执行步骤4处理
        print("🔄 开始步骤4处理...")
        result = process_step4_concept_merging(step3_result, previous_data)
        
        # 4. 保存结果到实验文件夹
        print("💾 保存实验结果...")
        saved_files = experiment_manager.save_step_result(
            step_num=4,
            result=result,
            save_formats=['txt', 'json']
        )
        
        if result.get("success"):
            print(f"\n✅ 步骤4完成!")
            print(f"📁 实验目录: {experiment_manager.experiment_dir}")
            print(f"📄 输出文件:")
            for format_type, file_path in saved_files.items():
                print(f"   - {format_type.upper()}: {file_path}")
            
            # 显示处理结果摘要
            stats = result.get("statistics", {})
            input_stats = result.get("input_statistics", {})
            
            print(f"\n📊 概念合并结果摘要:")
            print(f"   - 原始概念数: {input_stats.get('original_concept_count', 0)}")
            print(f"   - 合并后概念数: {stats.get('merged_concept_count', 0)}")
            print(f"   - 压缩比: {input_stats.get('compression_ratio', 0):.2f}")
            print(f"   - 概念组数: {stats.get('concept_groups_count', 0)}")
            print(f"   - 平均置信度: {stats.get('avg_confidence', 0):.3f}")
            print(f"   - 平均概念长度: {stats.get('avg_concept_length', 0):.1f} 字符")
            print(f"   - 涉及分块数: {stats.get('total_source_chunks', 0)}")
            print(f"   - 处理时间: {result.get('processing_time', 0):.2f} 秒")
            
            # 显示概念质量分布
            print(f"\n📈 概念质量分布:")
            print(f"   - 高置信度 (>0.7): {stats.get('high_confidence_count', 0)} 个")
            print(f"   - 中等置信度 (0.4-0.7): {stats.get('medium_confidence_count', 0)} 个")
            print(f"   - 低置信度 (<0.4): {stats.get('low_confidence_count', 0)} 个")
            
            # 显示顶级合并概念
            concept_nodes = result.get("concept_nodes", [])
            if concept_nodes:
                print(f"\n🌟 顶级合并概念 (前10个):")
                for i, concept in enumerate(concept_nodes[:10], 1):
                    print(f"   {i:2d}. {concept['concept_text']}")
                    print(f"       置信度: {concept['confidence_score']:.3f}, "
                          f"频率: {concept['frequency']}, "
                          f"合并数: {concept['merge_group_size']}")
                
                if len(concept_nodes) > 10:
                    print(f"   ... 还有 {len(concept_nodes) - 10} 个合并概念")
            
            # 显示实验状态
            summary = experiment_manager.get_experiment_summary()
            print(f"\n🧪 实验状态:")
            print(f"   - 实验ID: {summary['experiment_id']}")
            print(f"   - 已完成步骤: {summary['steps_completed']}/{summary['total_steps']}")
            print(f"   - 当前状态: {summary['status']}")
            
            # 提示后续步骤
            print(f"\n📋 后续步骤:")
            print(f"   运行下一步: python step5.py {saved_files['txt']}")
            print(f"   查看结果: cat {saved_files['txt']}")
            print(f"   查看概念: grep -A 20 '顶级合并概念' {saved_files['txt']}")
                
        else:
            print(f"❌ 步骤4失败: {result.get('error')}")
            
            # 即使失败也保存错误信息
            experiment_manager.save_step_result(
                step_num=4,
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
                "step": 4,
                "step_name": "概念合并与优化",
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                experiment_manager.save_step_result(4, error_result, ['txt'])
                print(f"📄 错误信息已保存到实验目录: {experiment_manager.experiment_dir}")
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()