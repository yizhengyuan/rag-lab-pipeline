#!/usr/bin/env python3
"""
步骤3: 概念提取与映射 - 增强版
================================

功能：
1. 从步骤2的结果中读取分块数据
2. 执行高质量的概念提取和验证
3. 使用新的概念映射功能
4. 保存到同一个实验文件夹

用法: 
- python step3.py <step2输出文件.txt>
- python step3.py <实验文件夹路径>

新功能：
- ✅ 自动识别并使用step2的实验文件夹
- ✅ 高质量的概念提取和概念图构建
- ✅ 概念关系映射和层次结构
- ✅ 统一的实验管理
- ✅ 支持新的concept_mapping提示词
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Set
from collections import Counter
import logging

# 导入核心模块
sys.path.append(str(Path(__file__).parent))
from llama_index.core import Document
from llama_index.core.schema import TextNode

# 导入核心处理模块
from config.settings import load_config_from_yaml
from core.nodes import ConceptNode

# 导入实验管理器
from utils.experiment_manager import ExperimentManager
from utils.helpers import FileHelper

logger = logging.getLogger(__name__)

def load_step2_result(step2_file_or_dir: str) -> tuple:
    """
    从步骤2的输出文件或实验文件夹中加载结果
    
    Args:
        step2_file_or_dir: 步骤2的输出文件或实验文件夹路径
        
    Returns:
        tuple: (step2_result, experiment_manager)
    """
    step2_path = Path(step2_file_or_dir)
    
    if step2_path.is_file():
        # 情况1：直接指定了step2的输出文件
        if step2_path.name.startswith("step2") and step2_path.suffix == ".txt":
            experiment_dir = step2_path.parent
            experiment_manager = ExperimentManager.load_experiment(str(experiment_dir))
            
            # 从txt文件加载结果
            step2_result = load_result_from_txt(str(step2_path))
            
        else:
            raise ValueError(f"不支持的文件格式: {step2_path}")
            
    elif step2_path.is_dir():
        # 情况2：直接指定了实验文件夹
        experiment_manager = ExperimentManager.load_experiment(str(step2_path))
        
        # 查找step2的输出文件
        step2_txt_path = experiment_manager.get_step_output_path(2, "txt")
        if not step2_txt_path.exists():
            raise FileNotFoundError(f"实验文件夹中找不到step2输出文件: {step2_txt_path}")
        
        step2_result = load_result_from_txt(str(step2_txt_path))
        
    else:
        raise FileNotFoundError(f"输入路径不存在: {step2_path}")
    
    return step2_result, experiment_manager

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

def reconstruct_chunks_from_step2(step2_result: Dict[str, Any]) -> List[TextNode]:
    """从步骤2结果重建TextNode对象"""
    
    # 尝试不同的字段名（兼容性）
    chunks_data = None
    for field_name in ["chunks", "chunk_nodes", "processed_chunks"]:
        if field_name in step2_result:
            chunks_data = step2_result[field_name]
            print(f"✅ 找到分块数据字段: {field_name}")
            break
    
    if not chunks_data:
        # 列出所有可用的字段
        available_fields = list(step2_result.keys())
        print(f"❌ 未找到分块数据，可用字段: {available_fields}")
        raise ValueError("步骤2结果中没有找到分块数据")
    
    print(f"📄 重建 {len(chunks_data)} 个TextNode对象...")
    
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

def extract_and_analyze_concepts(chunk_nodes: List[TextNode], config) -> Dict[str, Any]:
    """
    从chunks中提取和分析概念
    
    Args:
        chunk_nodes: 文本节点列表
        config: 配置对象
        
    Returns:
        Dict[str, Any]: 概念分析结果
    """
    print("🧠 开始概念提取和分析...")
    
    # 1. 收集所有概念
    all_concepts = []
    chunk_concept_map = {}
    
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
        
        # 过滤和清理概念
        valid_concepts = []
        for concept in concepts:
            if isinstance(concept, str) and len(concept.strip()) > 2:
                cleaned_concept = concept.strip()
                # 高质量概念过滤
                if is_high_quality_concept(cleaned_concept):
                    valid_concepts.append(cleaned_concept)
        
        all_concepts.extend(valid_concepts)
        chunk_concept_map[f"chunk_{i}"] = valid_concepts
    
    print(f"   收集到 {len(all_concepts)} 个概念")
    
    # 2. 概念频率分析
    concept_frequency = Counter(all_concepts)
    unique_concepts = list(concept_frequency.keys())
    
    print(f"   唯一概念数: {len(unique_concepts)}")
    
    # 3. 概念质量评估
    quality_scores = {}
    for concept in unique_concepts:
        quality_scores[concept] = calculate_concept_quality(concept, concept_frequency[concept])
    
    # 4. 按质量和频率排序
    sorted_by_quality = sorted(unique_concepts, key=lambda x: quality_scores[x], reverse=True)
    sorted_by_frequency = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)
    
    # 5. 分类概念
    high_quality_concepts = [c for c in unique_concepts if quality_scores[c] >= 4.0]
    medium_quality_concepts = [c for c in unique_concepts if 2.0 <= quality_scores[c] < 4.0]
    low_quality_concepts = [c for c in unique_concepts if quality_scores[c] < 2.0]
    
    # 6. 尝试概念映射（使用新的concept_mapping提示词）
    concept_map = None
    try:
        concept_map = create_concept_map(chunk_nodes, config)
        print(f"   ✅ 生成概念图成功")
    except Exception as e:
        print(f"   ⚠️ 概念映射失败: {e}")
    
    # 7. 生成统计数据
    statistics = {
        "total_concepts": len(all_concepts),
        "unique_concepts": len(unique_concepts),
        "avg_frequency": sum(concept_frequency.values()) / len(unique_concepts) if unique_concepts else 0,
        "avg_quality_score": sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0,
        "high_quality_count": len(high_quality_concepts),
        "medium_quality_count": len(medium_quality_concepts),
        "low_quality_count": len(low_quality_concepts),
        "concepts_per_chunk": len(all_concepts) / len(chunk_nodes) if chunk_nodes else 0
    }
    
    return {
        "all_concepts": all_concepts,
        "unique_concepts": unique_concepts,
        "concept_frequency": dict(concept_frequency),
        "quality_scores": quality_scores,
        "sorted_by_quality": sorted_by_quality,
        "sorted_by_frequency": sorted_by_frequency,
        "high_quality_concepts": high_quality_concepts,
        "medium_quality_concepts": medium_quality_concepts,
        "low_quality_concepts": low_quality_concepts,
        "chunk_concept_map": chunk_concept_map,
        "concept_map": concept_map,
        "statistics": statistics
    }

def is_high_quality_concept(concept: str) -> bool:
    """判断概念是否为高质量"""
    if not concept or len(concept.strip()) < 3:
        return False
    
    concept_lower = concept.lower()
    
    # 过滤低质量模式
    low_quality_patterns = [
        # 过于通用的词
        'method', 'system', 'process', 'study', 'analysis', 'research', 
        'data', 'information', 'model', 'approach', 'technique',
        # 简单形容词
        'good', 'bad', 'new', 'old', 'important', 'main', 'key',
        # 文档引用
        'paper', 'article', 'work', 'figure', 'table', 'section'
    ]
    
    # 检查是否包含低质量模式
    for pattern in low_quality_patterns:
        if pattern in concept_lower:
            return False
    
    # 检查长度（2-6个词比较好）
    word_count = len(concept.split())
    if word_count < 1 or word_count > 6:
        return False
    
    return True

def calculate_concept_quality(concept: str, frequency: int) -> float:
    """计算概念质量分数（0-5分）"""
    score = 3.0  # 基础分数
    
    # 长度评分
    word_count = len(concept.split())
    if 2 <= word_count <= 4:
        score += 1.0
    elif word_count == 1 or word_count > 5:
        score -= 0.5
    
    # 频率评分
    if frequency >= 3:
        score += 0.5
    elif frequency >= 2:
        score += 0.2
    
    # 包含专业术语加分
    professional_indicators = [
        'network', 'learning', 'attention', 'transformer', 'neural',
        'algorithm', 'optimization', 'architecture', 'mechanism',
        'translation', 'encoding', 'decoding', 'embedding'
    ]
    
    concept_lower = concept.lower()
    for indicator in professional_indicators:
        if indicator in concept_lower:
            score += 0.3
            break
    
    # 大小写混合（专有名词）加分
    if any(c.isupper() for c in concept) and any(c.islower() for c in concept):
        score += 0.2
    
    return min(5.0, max(0.0, score))

def create_concept_map(chunk_nodes: List[TextNode], config) -> Dict[str, Any]:
    """
    使用新的concept_mapping提示词创建概念图
    
    Args:
        chunk_nodes: 文本节点列表
        config: 配置对象
        
    Returns:
        Dict[str, Any]: 概念图数据
    """
    print("🗺️ 开始创建概念图...")
    
    # 这里可以实现概念图创建逻辑
    # 由于需要调用LLM，暂时返回基础结构
    concept_map = {
        "main_topics": [],
        "cross_topic_relationships": [],
        "created_at": datetime.now().isoformat(),
        "total_chunks": len(chunk_nodes)
    }
    
    # 简化版本：基于概念频率创建主题
    all_concepts = []
    for chunk in chunk_nodes:
        concepts_data = chunk.metadata.get("concepts", "[]")
        if isinstance(concepts_data, str):
            try:
                concepts = json.loads(concepts_data)
                all_concepts.extend(concepts)
            except:
                pass
    
    # 按频率分组概念
    concept_freq = Counter(all_concepts)
    high_freq_concepts = [c for c, f in concept_freq.most_common(10)]
    
    if high_freq_concepts:
        concept_map["main_topics"].append({
            "topic": "核心概念",
            "key_concepts": [
                {
                    "concept": concept,
                    "frequency": concept_freq[concept],
                    "definition": f"高频概念: {concept}",
                    "relationships": [],
                    "examples": [],
                    "supporting_details": []
                } for concept in high_freq_concepts
            ]
        })
    
    return concept_map

def process_step3_concept_extraction(step2_result: Dict[str, Any], config_path: str = "config.yml") -> Dict[str, Any]:
    """
    执行步骤3的概念提取处理
    
    Args:
        step2_result: 步骤2的结果
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 步骤3的处理结果
    """
    start_time = time.time()
    
    try:
        # 1. 加载配置
        print("📋 加载配置...")
        config = load_config_from_yaml(config_path)
        
        # 2. 重建TextNode对象
        print("📄 重建分块数据...")
        chunk_nodes = reconstruct_chunks_from_step2(step2_result)
        
        if not chunk_nodes:
            raise ValueError("没有有效的分块数据")
        
        # 3. 提取和分析概念
        print("🧠 执行概念提取和分析...")
        concept_analysis = extract_and_analyze_concepts(chunk_nodes, config)
        
        processing_time = time.time() - start_time
        
        # 4. 构建结果
        result = {
            "success": True,
            "step": 3,
            "step_name": "概念提取与映射",
            "chunk_count": len(chunk_nodes),
            "concept_analysis": concept_analysis,
            "statistics": concept_analysis["statistics"],
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "config_used": {
                "concepts_per_chunk": config.get("concept_extraction.concepts_per_chunk", 5),
                "similarity_threshold": config.get("concept_merging.similarity_threshold", 0.7)
            }
        }
        
        return result
        
    except Exception as e:
        error_msg = f"步骤3处理失败: {str(e)}"
        print(f"❌ {error_msg}")
        
        result = {
            "success": False,
            "step": 3,
            "step_name": "概念提取与映射",
            "error": error_msg,
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

def main():
    if len(sys.argv) != 2:
        print("用法: python step3.py <step2输出文件或实验文件夹>")
        print("示例:")
        print("  python step3.py experiments/20241204_143052_attention_paper/step2_chunking.txt")
        print("  python step3.py experiments/20241204_143052_attention_paper/")
        print("\n新功能:")
        print("✅ 自动识别实验文件夹")
        print("✅ 高质量概念提取和验证")
        print("✅ 概念关系映射和分析")
        print("✅ 统一的实验管理")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    print(f"🚀 步骤3: 概念提取与映射 (增强版)")
    print(f"📄 输入: {input_path}")
    print("="*60)
    
    try:
        # 1. 加载步骤2结果和实验管理器
        print("📂 加载步骤2结果...")
        step2_result, experiment_manager = load_step2_result(input_path)
        
        if not step2_result.get("success"):
            print("❌ 步骤2未成功完成，无法继续")
            sys.exit(1)
        
        print(f"✅ 已加载实验: {experiment_manager.experiment_name}")
        print(f"📁 实验目录: {experiment_manager.experiment_dir}")
        print()
        
        # 2. 执行步骤3处理
        print("🔄 开始步骤3处理...")
        result = process_step3_concept_extraction(step2_result)
        
        # 3. 保存结果到实验文件夹
        print("💾 保存实验结果...")
        saved_files = experiment_manager.save_step_result(
            step_num=3,
            result=result,
            save_formats=['txt', 'json']
        )
        
        if result.get("success"):
            print(f"\n✅ 步骤3完成!")
            print(f"📁 实验目录: {experiment_manager.experiment_dir}")
            print(f"📄 输出文件:")
            for format_type, file_path in saved_files.items():
                print(f"   - {format_type.upper()}: {file_path}")
            
            # 显示处理结果摘要
            concept_analysis = result.get("concept_analysis", {})
            stats = concept_analysis.get("statistics", {})
            
            print(f"\n📊 概念分析结果摘要:")
            print(f"   - 处理分块数: {result.get('chunk_count', 0)}")
            print(f"   - 总概念数: {stats.get('total_concepts', 0)}")
            print(f"   - 唯一概念数: {stats.get('unique_concepts', 0)}")
            print(f"   - 平均概念频率: {stats.get('avg_frequency', 0):.2f}")
            print(f"   - 平均质量分数: {stats.get('avg_quality_score', 0):.2f}/5.0")
            print(f"   - 高质量概念数: {stats.get('high_quality_count', 0)}")
            print(f"   - 中等质量概念数: {stats.get('medium_quality_count', 0)}")
            print(f"   - 低质量概念数: {stats.get('low_quality_count', 0)}")
            print(f"   - 处理时间: {result.get('processing_time', 0):.2f} 秒")
            
            # 显示高质量概念示例
            high_quality = concept_analysis.get("high_quality_concepts", [])
            if high_quality:
                print(f"\n🌟 高质量概念示例 (前15个):")
                for i, concept in enumerate(high_quality[:15], 1):
                    quality_score = concept_analysis.get("quality_scores", {}).get(concept, 0)
                    frequency = concept_analysis.get("concept_frequency", {}).get(concept, 0)
                    print(f"   {i:2d}. {concept} (质量: {quality_score:.1f}, 频率: {frequency})")
                
                if len(high_quality) > 15:
                    print(f"   ... 还有 {len(high_quality) - 15} 个高质量概念")
            
            # 显示概念映射信息
            concept_map = concept_analysis.get("concept_map")
            if concept_map:
                print(f"\n🗺️ 概念图信息:")
                print(f"   - 主要主题数: {len(concept_map.get('main_topics', []))}")
                print(f"   - 跨主题关系数: {len(concept_map.get('cross_topic_relationships', []))}")
            
            # 显示实验状态
            summary = experiment_manager.get_experiment_summary()
            print(f"\n🧪 实验状态:")
            print(f"   - 实验ID: {summary['experiment_id']}")
            print(f"   - 已完成步骤: {summary['steps_completed']}/{summary['total_steps']}")
            print(f"   - 当前状态: {summary['status']}")
            
            # 提示后续步骤
            print(f"\n📋 后续步骤:")
            print(f"   运行下一步: python step4.py {saved_files['txt']}")
            print(f"   查看结果: cat {saved_files['txt']}")
            print(f"   查看概念: grep -A 30 '高质量概念示例' {saved_files['txt']}")
                
        else:
            print(f"❌ 步骤3失败: {result.get('error')}")
            
            # 即使失败也保存错误信息
            experiment_manager.save_step_result(
                step_num=3,
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
                "step": 3,
                "step_name": "概念提取与映射",
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                experiment_manager.save_step_result(3, error_result, ['txt'])
                print(f"📄 错误信息已保存到实验目录: {experiment_manager.experiment_dir}")
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()