#!/usr/bin/env python3
"""
步骤6: 证据提取与质量评估 - 增强版
=====================================

功能：
1. 从步骤5的检索结果中获取概念-分块映射
2. 智能提取每个概念的支持证据
3. 多维度证据质量评估和分类
4. 保存到同一个实验文件夹

用法: 
- python step6.py <step5输出文件.txt>
- python step6.py <实验文件夹路径>

新功能：
- ✅ 自动识别并使用step5的实验文件夹
- ✅ 智能证据提取和相关性计算
- ✅ 多类型证据识别（定义、例子、解释等）
- ✅ 证据质量评分和过滤
- ✅ 统一的实验管理
"""

import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
import logging

# 导入核心模块
sys.path.append(str(Path(__file__).parent))
from llama_index.core import Document
from llama_index.core.schema import TextNode

# 导入核心处理模块
from config.settings import load_config_from_yaml
from core.nodes import EvidenceNode

# 导入实验管理器
from utils.experiment_manager import ExperimentManager
from utils.helpers import FileHelper

logger = logging.getLogger(__name__)

def load_step5_result(step5_file_or_dir: str) -> tuple:
    """
    从步骤5的输出文件或实验文件夹中加载结果
    
    Args:
        step5_file_or_dir: 步骤5的输出文件或实验文件夹路径
        
    Returns:
        tuple: (step5_result, experiment_manager)
    """
    step5_path = Path(step5_file_or_dir)
    
    if step5_path.is_file():
        # 情况1：直接指定了step5的输出文件
        if step5_path.name.startswith("step5") and step5_path.suffix == ".txt":
            experiment_dir = step5_path.parent
            experiment_manager = ExperimentManager.load_experiment(str(experiment_dir))
            
            # 从txt文件加载结果
            step5_result = load_result_from_txt(str(step5_path))
            
        else:
            raise ValueError(f"不支持的文件格式: {step5_path}")
            
    elif step5_path.is_dir():
        # 情况2：直接指定了实验文件夹
        experiment_manager = ExperimentManager.load_experiment(str(step5_path))
        
        # 查找step5的输出文件
        step5_txt_path = experiment_manager.get_step_output_path(5, "txt")
        if not step5_txt_path.exists():
            raise FileNotFoundError(f"实验文件夹中找不到step5输出文件: {step5_txt_path}")
        
        step5_result = load_result_from_txt(str(step5_txt_path))
        
    else:
        raise FileNotFoundError(f"输入路径不存在: {step5_path}")
    
    return step5_result, experiment_manager

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
    
    # 加载步骤2的数据（需要分块文本）
    step2_path = experiment_manager.get_step_output_path(2, "txt")
    if step2_path.exists():
        try:
            step2_result = load_result_from_txt(str(step2_path))
            previous_data["step2"] = step2_result
            print(f"✅ 加载步骤2数据: {step2_path}")
        except Exception as e:
            print(f"⚠️ 加载步骤2数据失败: {e}")
    
    return previous_data

def extract_retrieval_results_from_step5(step5_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    从步骤5结果中提取检索数据
    
    Args:
        step5_result: 步骤5的结果
        
    Returns:
        Dict[str, Any]: 检索结果数据
    """
    print("📊 从步骤5结果中提取检索数据...")
    
    retrieval_results = step5_result.get("retrieval_results", {})
    
    if not retrieval_results:
        raise ValueError("步骤5结果中没有找到检索数据")
    
    print(f"   - 检索结果数: {len(retrieval_results)}")
    
    # 统计检索覆盖情况
    concepts_with_chunks = 0
    total_chunk_mappings = 0
    
    for concept_id, result in retrieval_results.items():
        retrieved_chunks = result.get("retrieved_chunks", [])
        if retrieved_chunks:
            concepts_with_chunks += 1
            total_chunk_mappings += len(retrieved_chunks)
    
    print(f"   - 有检索结果的概念: {concepts_with_chunks}")
    print(f"   - 总概念-分块映射数: {total_chunk_mappings}")
    
    return retrieval_results

def get_chunk_text_mapping(step2_result: Dict[str, Any]) -> Dict[str, str]:
    """
    创建分块ID到文本的映射
    
    Args:
        step2_result: 步骤2的结果
        
    Returns:
        Dict[str, str]: 分块ID到文本的映射
    """
    print("🗂️ 创建分块文本映射...")
    
    # 尝试不同的字段名
    chunks_data = None
    for field_name in ["chunks", "chunk_nodes", "processed_chunks"]:
        if field_name in step2_result:
            chunks_data = step2_result[field_name]
            break
    
    if not chunks_data:
        raise ValueError("步骤2结果中没有找到分块数据")
    
    chunk_text_mapping = {}
    
    for chunk_data in chunks_data:
        if isinstance(chunk_data, dict):
            chunk_id = chunk_data.get("metadata", {}).get("chunk_id") or chunk_data.get("node_id", "unknown")
            chunk_text = chunk_data.get("text", "")
            chunk_text_mapping[chunk_id] = chunk_text
        else:
            print(f"⚠️ 跳过无效的chunk数据: {type(chunk_data)}")
    
    print(f"   ✅ 创建了 {len(chunk_text_mapping)} 个分块文本映射")
    return chunk_text_mapping

def classify_evidence_type(evidence_text: str, concept_text: str) -> str:
    """
    分类证据类型
    
    Args:
        evidence_text: 证据文本
        concept_text: 概念文本
        
    Returns:
        str: 证据类型
    """
    evidence_lower = evidence_text.lower()
    concept_lower = concept_text.lower()
    
    # 定义类型模式
    definition_patterns = [
        r'\bis\s+(?:a|an|the)?\s*\w+',
        r'\bdefine[ds]?\s+as',
        r'\brefer[s]?\s+to',
        r'\bmean[s]?\s+that',
        r'\bconsist[s]?\s+of',
        r'\btype\s+of',
        r'\bkind\s+of'
    ]
    
    example_patterns = [
        r'\bfor\s+example',
        r'\bsuch\s+as',
        r'\bincluding',
        r'\be\.g\.',
        r'\bnamely',
        r'\bspecifically',
        r'\bin\s+particular'
    ]
    
    explanation_patterns = [
        r'\bbecause',
        r'\bsince',
        r'\btherefore',
        r'\bthus',
        r'\bhence',
        r'\bas\s+a\s+result',
        r'\bdue\s+to',
        r'\bowing\s+to'
    ]
    
    procedure_patterns = [
        r'\bstep[s]?',
        r'\bprocess',
        r'\bmethod',
        r'\bprocedure',
        r'\balgorithm',
        r'\btechnique',
        r'\bapproach'
    ]
    
    # 检查模式匹配
    for pattern in definition_patterns:
        if re.search(pattern, evidence_lower):
            return "definition"
    
    for pattern in example_patterns:
        if re.search(pattern, evidence_lower):
            return "example"
    
    for pattern in explanation_patterns:
        if re.search(pattern, evidence_lower):
            return "explanation"
    
    for pattern in procedure_patterns:
        if re.search(pattern, evidence_lower):
            return "procedure"
    
    # 检查是否包含概念名称
    if concept_lower in evidence_lower:
        return "reference"
    
    return "general"

def calculate_evidence_relevance(evidence_text: str, 
                               concept_text: str, 
                               similarity_score: float) -> float:
    """
    计算证据相关性分数
    
    Args:
        evidence_text: 证据文本
        concept_text: 概念文本
        similarity_score: 检索相似度分数
        
    Returns:
        float: 相关性分数 (0-1)
    """
    from difflib import SequenceMatcher
    
    # 基础分数来自检索相似度
    base_score = similarity_score
    
    # 1. 概念词汇匹配度
    concept_words = set(concept_text.lower().split())
    evidence_words = set(evidence_text.lower().split())
    
    if concept_words and evidence_words:
        word_overlap = len(concept_words.intersection(evidence_words)) / len(concept_words)
    else:
        word_overlap = 0.0
    
    # 2. 文本长度适中性（太短或太长都减分）
    evidence_length = len(evidence_text)
    if 50 <= evidence_length <= 300:
        length_factor = 1.0
    elif 20 <= evidence_length < 50 or 300 < evidence_length <= 500:
        length_factor = 0.8
    elif evidence_length > 500:
        length_factor = 0.6
    else:
        length_factor = 0.4
    
    # 3. 直接包含概念名称加分
    concept_containment = 0.0
    if concept_text.lower() in evidence_text.lower():
        concept_containment = 0.3
    
    # 4. 完整句子加分
    sentence_completeness = 0.0
    if evidence_text.strip().endswith(('.', '!', '?', ':')):
        sentence_completeness = 0.1
    
    # 综合相关性分数
    relevance_score = (
        base_score * 0.4 +
        word_overlap * 0.3 +
        concept_containment * 0.2 +
        length_factor * 0.1 +
        sentence_completeness
    )
    
    return min(1.0, max(0.0, relevance_score))

def extract_evidence_from_text(chunk_text: str, 
                             concept_text: str, 
                             similarity_score: float,
                             min_length: int = 20,
                             max_length: int = 400) -> List[Dict[str, Any]]:
    """
    从分块文本中提取证据
    
    Args:
        chunk_text: 分块文本
        concept_text: 概念文本
        similarity_score: 相似度分数
        min_length: 最小证据长度
        max_length: 最大证据长度
        
    Returns:
        List[Dict[str, Any]]: 提取的证据列表
    """
    # 将文本分解为句子
    sentences = re.split(r'[.!?]+', chunk_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    evidences = []
    concept_lower = concept_text.lower()
    
    # 1. 寻找包含概念的句子
    for i, sentence in enumerate(sentences):
        if len(sentence) < min_length or len(sentence) > max_length:
            continue
        
        sentence_lower = sentence.lower()
        
        # 检查是否与概念相关
        if any(word in sentence_lower for word in concept_lower.split()):
            # 尝试扩展上下文
            extended_evidence = sentence
            
            # 向前扩展一句（如果合适）
            if i > 0 and len(extended_evidence + " " + sentences[i-1]) <= max_length:
                extended_evidence = sentences[i-1] + " " + extended_evidence
            
            # 向后扩展一句（如果合适）
            if i < len(sentences) - 1 and len(extended_evidence + " " + sentences[i+1]) <= max_length:
                extended_evidence = extended_evidence + " " + sentences[i+1]
            
            # 计算相关性
            relevance_score = calculate_evidence_relevance(
                extended_evidence, concept_text, similarity_score
            )
            
            # 分类证据类型
            evidence_type = classify_evidence_type(extended_evidence, concept_text)
            
            evidences.append({
                "evidence_text": extended_evidence,
                "evidence_length": len(extended_evidence),
                "relevance_score": relevance_score,
                "evidence_type": evidence_type,
                "sentence_index": i,
                "is_extended": len(extended_evidence) > len(sentence)
            })
    
    # 2. 寻找定义性段落（连续2-3句话）
    for i in range(len(sentences) - 1):
        if i + 2 < len(sentences):
            paragraph = " ".join(sentences[i:i+3])
        else:
            paragraph = " ".join(sentences[i:i+2])
        
        if min_length <= len(paragraph) <= max_length:
            paragraph_lower = paragraph.lower()
            
            # 检查是否像定义或解释
            if any(pattern in paragraph_lower for pattern in [
                concept_lower, "define", "refer to", "means", "is a", "consist of"
            ]):
                relevance_score = calculate_evidence_relevance(
                    paragraph, concept_text, similarity_score
                )
                
                if relevance_score >= 0.3:  # 只保留较高相关性的段落
                    evidence_type = classify_evidence_type(paragraph, concept_text)
                    
                    evidences.append({
                        "evidence_text": paragraph,
                        "evidence_length": len(paragraph),
                        "relevance_score": relevance_score,
                        "evidence_type": evidence_type,
                        "sentence_index": i,
                        "is_extended": True
                    })
    
    # 去重并按相关性排序
    unique_evidences = []
    seen_texts = set()
    
    for evidence in evidences:
        text_key = evidence["evidence_text"][:100]  # 使用前100字符作为去重键
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            unique_evidences.append(evidence)
    
    # 按相关性排序
    unique_evidences.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return unique_evidences[:5]  # 最多返回5个证据

def perform_evidence_extraction(retrieval_results: Dict[str, Any],
                              chunk_text_mapping: Dict[str, str],
                              config) -> Dict[str, Any]:
    """
    执行证据提取
    
    Args:
        retrieval_results: 检索结果
        chunk_text_mapping: 分块文本映射
        config: 配置对象
        
    Returns:
        Dict[str, Any]: 证据提取结果
    """
    print("🔍 开始执行证据提取...")
    
    min_length = config.get("evidence_extraction.min_length", 20)
    max_length = config.get("evidence_extraction.max_length", 400)
    min_relevance = 0.2  # 最小相关性阈值
    
    evidence_nodes = []
    evidence_by_concept = {}
    evidence_type_counts = defaultdict(int)
    
    total_evidence = 0
    concepts_with_evidence = 0
    
    for concept_id, retrieval_data in retrieval_results.items():
        concept_text = retrieval_data.get("concept_text", "")
        retrieved_chunks = retrieval_data.get("retrieved_chunks", [])
        
        print(f"   提取概念证据: {concept_text}")
        
        concept_evidences = []
        
        for chunk_data in retrieved_chunks:
            chunk_id = chunk_data.get("chunk_id", "")
            similarity_score = chunk_data.get("similarity_score", 0.0)
            
            # 获取分块文本
            chunk_text = chunk_text_mapping.get(chunk_id, "")
            if not chunk_text:
                continue
            
            # 提取证据
            evidences = extract_evidence_from_text(
                chunk_text, concept_text, similarity_score, min_length, max_length
            )
            
            for evidence in evidences:
                if evidence["relevance_score"] >= min_relevance:
                    evidence_id = f"evidence_{len(evidence_nodes)}"
                    
                    evidence_node = {
                        "evidence_id": evidence_id,
                        "concept_id": concept_id,
                        "concept_text": concept_text,
                        "evidence_text": evidence["evidence_text"],
                        "evidence_length": evidence["evidence_length"],
                        "relevance_score": evidence["relevance_score"],
                        "evidence_type": evidence["evidence_type"],
                        "source_chunk_id": chunk_id,
                        "chunk_similarity": similarity_score,
                        "sentence_index": evidence["sentence_index"],
                        "is_extended": evidence["is_extended"],
                        "created_at": datetime.now().isoformat()
                    }
                    
                    evidence_nodes.append(evidence_node)
                    concept_evidences.append(evidence_node)
                    evidence_type_counts[evidence["evidence_type"]] += 1
                    total_evidence += 1
        
        if concept_evidences:
            concepts_with_evidence += 1
            evidence_by_concept[concept_id] = concept_evidences
        else:
            evidence_by_concept[concept_id] = []
    
    # 计算统计信息
    if evidence_nodes:
        avg_relevance = sum(e["relevance_score"] for e in evidence_nodes) / len(evidence_nodes)
        avg_length = sum(e["evidence_length"] for e in evidence_nodes) / len(evidence_nodes)
    else:
        avg_relevance = 0.0
        avg_length = 0.0
    
    avg_evidence_per_concept = total_evidence / len(retrieval_results) if retrieval_results else 0
    
    print(f"   ✅ 证据提取完成:")
    print(f"      - 总证据数: {total_evidence}")
    print(f"      - 有证据的概念: {concepts_with_evidence}/{len(retrieval_results)}")
    print(f"      - 平均相关性: {avg_relevance:.4f}")
    
    return {
        "evidence_nodes": evidence_nodes,
        "evidence_by_concept": evidence_by_concept,
        "statistics": {
            "total_evidence": total_evidence,
            "concepts_with_evidence": concepts_with_evidence,
            "avg_evidence_per_concept": avg_evidence_per_concept,
            "avg_relevance_score": avg_relevance,
            "avg_evidence_length": avg_length,
            "evidence_type_distribution": dict(evidence_type_counts),
            "min_relevance_threshold": min_relevance,
            "processed_concepts": len(retrieval_results)
        }
    }

def analyze_evidence_quality(evidence_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析证据质量
    
    Args:
        evidence_data: 证据数据
        
    Returns:
        Dict[str, Any]: 质量分析结果
    """
    print("📈 分析证据质量...")
    
    evidence_nodes = evidence_data["evidence_nodes"]
    
    if not evidence_nodes:
        return {
            "overall_quality_score": 0.0,
            "quality_distribution": {"high": 0, "medium": 0, "low": 0},
            "type_quality_scores": {},
            "top_evidences": []
        }
    
    # 质量分类
    high_quality = [e for e in evidence_nodes if e["relevance_score"] >= 0.7]
    medium_quality = [e for e in evidence_nodes if 0.4 <= e["relevance_score"] < 0.7]
    low_quality = [e for e in evidence_nodes if e["relevance_score"] < 0.4]
    
    # 按类型分析质量
    type_scores = defaultdict(list)
    for evidence in evidence_nodes:
        type_scores[evidence["evidence_type"]].append(evidence["relevance_score"])
    
    type_quality_scores = {}
    for evidence_type, scores in type_scores.items():
        type_quality_scores[evidence_type] = {
            "avg_score": sum(scores) / len(scores),
            "count": len(scores),
            "max_score": max(scores),
            "min_score": min(scores)
        }
    
    # 整体质量评分
    total_count = len(evidence_nodes)
    overall_quality_score = (
        (len(high_quality) / total_count * 1.0) +
        (len(medium_quality) / total_count * 0.6) +
        (len(low_quality) / total_count * 0.2)
    )
    
    # 顶级证据
    top_evidences = sorted(
        evidence_nodes,
        key=lambda x: x["relevance_score"],
        reverse=True
    )[:10]
    
    return {
        "overall_quality_score": overall_quality_score,
        "quality_distribution": {
            "high": len(high_quality),
            "medium": len(medium_quality),
            "low": len(low_quality)
        },
        "type_quality_scores": type_quality_scores,
        "top_evidences": top_evidences,
        "coverage_analysis": {
            "concepts_with_high_quality": len([
                cid for cid, evidences in evidence_data["evidence_by_concept"].items()
                if any(e["relevance_score"] >= 0.7 for e in evidences)
            ]),
            "concepts_without_evidence": len([
                cid for cid, evidences in evidence_data["evidence_by_concept"].items()
                if not evidences
            ])
        }
    }

def process_step6_evidence_extraction(step5_result: Dict[str, Any], 
                                     previous_data: Dict[str, Any],
                                     config_path: str = "config.yml") -> Dict[str, Any]:
    """
    执行步骤6的证据提取处理
    
    Args:
        step5_result: 步骤5的结果
        previous_data: 之前步骤的数据
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 步骤6的处理结果
    """
    start_time = time.time()
    
    try:
        # 1. 加载配置
        print("📋 加载配置...")
        config = load_config_from_yaml(config_path)
        
        # 2. 提取检索结果
        print("📊 提取检索结果...")
        retrieval_results = extract_retrieval_results_from_step5(step5_result)
        
        # 3. 创建分块文本映射
        print("🗂️ 创建分块文本映射...")
        if "step2" not in previous_data:
            raise ValueError("缺少步骤2的分块数据")
        
        chunk_text_mapping = get_chunk_text_mapping(previous_data["step2"])
        
        # 4. 执行证据提取
        print("🔍 执行证据提取...")
        evidence_data = perform_evidence_extraction(retrieval_results, chunk_text_mapping, config)
        
        # 5. 分析证据质量
        print("📈 分析证据质量...")
        quality_analysis = analyze_evidence_quality(evidence_data)
        
        processing_time = time.time() - start_time
        
        # 6. 构建结果
        result = {
            "success": True,
            "step": 6,
            "step_name": "证据提取与质量评估",
            "evidence_nodes": evidence_data["evidence_nodes"],
            "evidence_by_concept": evidence_data["evidence_by_concept"],
            "statistics": evidence_data["statistics"],
            "quality_analysis": quality_analysis,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "config_used": {
                "min_length": config.get("evidence_extraction.min_length", 20),
                "max_length": config.get("evidence_extraction.max_length", 400)
            }
        }
        
        return result
        
    except Exception as e:
        error_msg = f"步骤6处理失败: {str(e)}"
        print(f"❌ {error_msg}")
        
        result = {
            "success": False,
            "step": 6,
            "step_name": "证据提取与质量评估",
            "error": error_msg,
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

def main():
    if len(sys.argv) != 2:
        print("用法: python step6.py <step5输出文件或实验文件夹>")
        print("示例:")
        print("  python step6.py experiments/20241204_143052_attention_paper/step5_answer_generation.txt")
        print("  python step6.py experiments/20241204_143052_attention_paper/")
        print("\n新功能:")
        print("✅ 自动识别实验文件夹")
        print("✅ 智能证据提取和相关性计算")
        print("✅ 多类型证据识别和分类")
        print("✅ 证据质量评分和过滤")
        print("✅ 统一的实验管理")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    print(f"🚀 步骤6: 证据提取与质量评估 (增强版)")
    print(f"📄 输入: {input_path}")
    print("="*60)
    
    try:
        # 1. 加载步骤5结果和实验管理器
        print("📂 加载步骤5结果...")
        step5_result, experiment_manager = load_step5_result(input_path)
        
        if not step5_result.get("success"):
            print("❌ 步骤5未成功完成，无法继续")
            sys.exit(1)
        
        print(f"✅ 已加载实验: {experiment_manager.experiment_name}")
        print(f"📁 实验目录: {experiment_manager.experiment_dir}")
        print()
        
        # 2. 加载之前步骤的数据
        print("📂 加载之前步骤的数据...")
        previous_data = load_previous_steps_data(experiment_manager)
        
        # 3. 执行步骤6处理
        print("🔄 开始步骤6处理...")
        result = process_step6_evidence_extraction(step5_result, previous_data)
        
        # 4. 保存结果到实验文件夹
        print("💾 保存实验结果...")
        saved_files = experiment_manager.save_step_result(
            step_num=6,
            result=result,
            save_formats=['txt', 'json']
        )
        
        if result.get("success"):
            print(f"\n✅ 步骤6完成!")
            print(f"📁 实验目录: {experiment_manager.experiment_dir}")
            print(f"📄 输出文件:")
            for format_type, file_path in saved_files.items():
                print(f"   - {format_type.upper()}: {file_path}")
            
            # 显示处理结果摘要
            stats = result.get("statistics", {})
            quality = result.get("quality_analysis", {})
            
            print(f"\n📊 证据提取结果摘要:")
            print(f"   - 总证据数: {stats.get('total_evidence', 0)}")
            print(f"   - 有证据的概念数: {stats.get('concepts_with_evidence', 0)}")
            print(f"   - 平均每概念证据数: {stats.get('avg_evidence_per_concept', 0):.2f}")
            print(f"   - 平均相关性分数: {stats.get('avg_relevance_score', 0):.4f}")
            print(f"   - 平均证据长度: {stats.get('avg_evidence_length', 0):.1f} 字符")
            print(f"   - 处理时间: {result.get('processing_time', 0):.2f} 秒")
            
            # 显示证据类型分布
            type_dist = stats.get("evidence_type_distribution", {})
            if type_dist:
                print(f"\n📈 证据类型分布:")
                for evidence_type, count in type_dist.items():
                    print(f"   - {evidence_type}: {count} 个")
            
            # 显示质量分析
            print(f"\n📈 证据质量分析:")
            print(f"   - 整体质量评分: {quality.get('overall_quality_score', 0):.3f}")
            
            quality_dist = quality.get("quality_distribution", {})
            print(f"   - 高质量证据: {quality_dist.get('high', 0)} 个")
            print(f"   - 中等质量证据: {quality_dist.get('medium', 0)} 个")
            print(f"   - 低质量证据: {quality_dist.get('low', 0)} 个")
            
            # 显示覆盖分析
            coverage = quality.get("coverage_analysis", {})
            print(f"   - 有高质量证据的概念: {coverage.get('concepts_with_high_quality', 0)} 个")
            print(f"   - 无证据的概念: {coverage.get('concepts_without_evidence', 0)} 个")
            
            # 显示顶级证据
            top_evidences = quality.get("top_evidences", [])
            if top_evidences:
                print(f"\n🌟 顶级证据 (前5个):")
                for i, evidence in enumerate(top_evidences[:5], 1):
                    print(f"   {i}. 概念: {evidence['concept_text'][:30]}...")
                    print(f"      相关性: {evidence['relevance_score']:.4f}")
                    print(f"      类型: {evidence['evidence_type']}")
                    print(f"      内容: {evidence['evidence_text'][:100]}...")
                    print()
            
            # 显示实验状态
            summary = experiment_manager.get_experiment_summary()
            print(f"🧪 实验状态:")
            print(f"   - 实验ID: {summary['experiment_id']}")
            print(f"   - 已完成步骤: {summary['steps_completed']}/{summary['total_steps']}")
            print(f"   - 当前状态: {summary['status']}")
            
            # 提示后续步骤
            print(f"\n📋 后续步骤:")
            print(f"   运行下一步: python step7.py {saved_files['txt']}")
            print(f"   查看结果: cat {saved_files['txt']}")
            print(f"   查看证据: grep -A 5 '顶级证据' {saved_files['txt']}")
                
        else:
            print(f"❌ 步骤6失败: {result.get('error')}")
            
            # 即使失败也保存错误信息
            experiment_manager.save_step_result(
                step_num=6,
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
                "step": 6,
                "step_name": "证据提取与质量评估",
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                experiment_manager.save_step_result(6, error_result, ['txt'])
                print(f"📄 错误信息已保存到实验目录: {experiment_manager.experiment_dir}")
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()