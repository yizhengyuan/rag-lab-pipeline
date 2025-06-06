#!/usr/bin/env python3
"""
步骤7: 问答生成 - 增强版
==========================

功能：
1. 从步骤6的结果中读取证据数据
2. 基于每个证据生成高质量问答对
3. 支持多种问答类型和难度级别
4. 保存到同一个实验文件夹

用法: 
- python step7.py <step6输出文件.txt>
- python step7.py <实验文件夹路径>

新功能：
- ✅ 自动识别并使用step6的实验文件夹
- ✅ 智能问答生成，支持多种认知层次
- ✅ 问答质量评估和分类
- ✅ 统一的实验管理
- ✅ 支持多种输出格式
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Set
from collections import defaultdict
import logging

# 导入核心模块
sys.path.append(str(Path(__file__).parent))
from llama_index.core import Document
from llama_index.core.schema import TextNode

# 导入核心处理模块
from config.settings import load_config_from_yaml

# 🔧 修正：导入QA生成器
from data_generate_0526 import SimpleDataGenerator as QAGenerator

# 导入实验管理器
from utils.experiment_manager import ExperimentManager
from utils.helpers import FileHelper

logger = logging.getLogger(__name__)

def load_step6_result(step6_file_or_dir: str) -> tuple:
    """
    从步骤6的输出文件或实验文件夹中加载结果
    
    Args:
        step6_file_or_dir: 步骤6的输出文件或实验文件夹路径
        
    Returns:
        tuple: (step6_result, experiment_manager)
    """
    step6_path = Path(step6_file_or_dir)
    
    if step6_path.is_file():
        # 情况1：直接指定了step6的输出文件
        if step6_path.name.startswith("step6") and step6_path.suffix == ".txt":
            experiment_dir = step6_path.parent
            experiment_manager = ExperimentManager.load_experiment(str(experiment_dir))
            
            # 从txt文件加载结果
            step6_result = load_result_from_txt(str(step6_path))
            
        else:
            raise ValueError(f"不支持的文件格式: {step6_path}")
            
    elif step6_path.is_dir():
        # 情况2：直接指定了实验文件夹
        experiment_manager = ExperimentManager.load_experiment(str(step6_path))
        
        # 查找step6的输出文件
        step6_txt_path = experiment_manager.get_step_output_path(6, "txt")
        if not step6_txt_path.exists():
            raise FileNotFoundError(f"实验文件夹中找不到step6输出文件: {step6_txt_path}")
        
        step6_result = load_result_from_txt(str(step6_txt_path))
        
    else:
        raise FileNotFoundError(f"输入路径不存在: {step6_path}")
    
    return step6_result, experiment_manager

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

def extract_evidence_nodes_from_step6(step6_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从步骤6结果中提取证据节点
    
    Args:
        step6_result: 步骤6的结果
        
    Returns:
        List[Dict[str, Any]]: 证据节点列表
    """
    print("📊 从步骤6结果中提取证据节点...")
    
    evidence_nodes = step6_result.get("evidence_nodes", [])
    
    if not evidence_nodes:
        raise ValueError("步骤6结果中没有找到证据数据")
    
    print(f"   - 证据节点数: {len(evidence_nodes)}")
    
    # 验证证据节点格式
    valid_evidences = []
    for evidence in evidence_nodes:
        if isinstance(evidence, dict) and "evidence_text" in evidence:
            valid_evidences.append(evidence)
        else:
            print(f"⚠️ 跳过无效的证据节点: {type(evidence)}")
    
    print(f"   - 有效证据节点数: {len(valid_evidences)}")
    
    return valid_evidences

def generate_qa_pairs_from_evidence(evidence_nodes: List[Dict[str, Any]], 
                                  config) -> Dict[str, Any]:
    """
    从证据中生成问答对
    
    Args:
        evidence_nodes: 证据节点列表
        config: 配置对象
        
    Returns:
        Dict[str, Any]: 问答生成结果
    """
    print("❓ 开始问答生成...")
    
    # 🔧 初始化QA生成器，使用配置中的参数
    questions_per_type = config.get("qa_generation.questions_per_type", {
        "remember": 2,
        "understand": 2,
        "apply": 1,
        "analyze": 1,
        "evaluate": 1,
        "create": 1
    })
    
    qa_generator = QAGenerator(
        model_name=config.get("openai.model", "gpt-4o-mini"),
        questions_per_type=questions_per_type
    )
    
    all_qa_pairs = []
    qa_stats = defaultdict(int)
    failed_evidences = 0
    
    for i, evidence_node in enumerate(evidence_nodes):
        try:
            evidence_text = evidence_node.get("evidence_text", "")
            concept_text = evidence_node.get("concept_text", "未知概念")
            evidence_id = evidence_node.get("evidence_id", f"evidence_{i}")
            
            print(f"   生成证据 {i+1}/{len(evidence_nodes)} 的问答: {concept_text}")
            
            if not evidence_text or len(evidence_text.strip()) < 20:
                print(f"   ⚠️ 跳过证据 {i+1}: 文本太短或为空")
                failed_evidences += 1
                continue
            
            # 🔧 修复：使用字典数据生成问答对
            qa_pairs = qa_generator.generate_qa_pairs_from_text(evidence_text)
            
            # 为每个问答对添加元数据
            for qa_pair in qa_pairs:
                qa_pair["evidence_source"] = evidence_id
                qa_pair["evidence_concept"] = concept_text
                qa_pair["evidence_relevance"] = evidence_node.get("relevance_score", 0.0)
                qa_pair["evidence_type"] = evidence_node.get("evidence_type", "general")
                qa_pair["generation_timestamp"] = datetime.now().isoformat()
                
                # 统计QA类型
                qa_type = qa_pair.get("type", "unknown")
                qa_stats[qa_type] += 1
            
            all_qa_pairs.extend(qa_pairs)
            print(f"   ✅ 生成了 {len(qa_pairs)} 个问答对")
            
        except Exception as e:
            print(f"   ❌ 证据 {i+1} 生成失败: {e}")
            failed_evidences += 1
            continue
    
    print(f"   ✅ 问答生成完成:")
    print(f"      - 总问答对数: {len(all_qa_pairs)}")
    print(f"      - 成功处理证据: {len(evidence_nodes) - failed_evidences}/{len(evidence_nodes)}")
    print(f"      - 失败证据数: {failed_evidences}")
    
    return {
        "qa_pairs": all_qa_pairs,
        "qa_stats": dict(qa_stats),
        "total_qa_pairs": len(all_qa_pairs),
        "processed_evidences": len(evidence_nodes) - failed_evidences,
        "failed_evidences": failed_evidences
    }

def analyze_qa_quality(qa_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析问答质量
    
    Args:
        qa_data: 问答数据
        
    Returns:
        Dict[str, Any]: 质量分析结果
    """
    print("📈 分析问答质量...")
    
    qa_pairs = qa_data["qa_pairs"]
    
    if not qa_pairs:
        return {
            "overall_quality_score": 0.0,
            "difficulty_distribution": {},
            "type_distribution": {},
            "avg_question_length": 0.0,
            "avg_answer_length": 0.0
        }
    
    # 难度分布
    difficulty_counts = defaultdict(int)
    type_counts = defaultdict(int)
    question_lengths = []
    answer_lengths = []
    
    for qa_pair in qa_pairs:
        difficulty = qa_pair.get("difficulty", "unknown")
        qa_type = qa_pair.get("type", "unknown")
        question = qa_pair.get("question", "")
        answer = qa_pair.get("answer", "")
        
        difficulty_counts[difficulty] += 1
        type_counts[qa_type] += 1
        question_lengths.append(len(question))
        answer_lengths.append(len(answer))
    
    # 计算平均长度
    avg_question_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0
    avg_answer_length = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
    
    # 计算质量分数（基于多样性和完整性）
    type_diversity = len(type_counts) / 6.0  # 最多6种类型
    difficulty_diversity = len(difficulty_counts) / 3.0  # 3种难度
    completeness_score = len(qa_pairs) / max(len(qa_pairs), 10)  # 相对于预期数量
    
    overall_quality_score = (type_diversity + difficulty_diversity + min(completeness_score, 1.0)) / 3.0
    
    return {
        "overall_quality_score": overall_quality_score,
        "difficulty_distribution": dict(difficulty_counts),
        "type_distribution": dict(type_counts),
        "avg_question_length": avg_question_length,
        "avg_answer_length": avg_answer_length,
        "type_diversity": type_diversity,
        "difficulty_diversity": difficulty_diversity
    }

def process_step7_qa_generation(step6_result: Dict[str, Any], 
                               config_path: str = "config.yml") -> Dict[str, Any]:
    """
    执行步骤7的问答生成处理
    
    Args:
        step6_result: 步骤6的结果
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 步骤7的处理结果
    """
    start_time = time.time()
    
    try:
        # 1. 加载配置
        print("📋 加载配置...")
        config = load_config_from_yaml(config_path)
        
        # 2. 提取证据节点
        print("📊 提取证据节点...")
        evidence_nodes = extract_evidence_nodes_from_step6(step6_result)
        
        if not evidence_nodes:
            return {
                "success": True,
                "skipped": True,
                "reason": "没有可用的证据数据",
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
        
        # 3. 生成问答对
        print("❓ 生成问答对...")
        qa_data = generate_qa_pairs_from_evidence(evidence_nodes, config)
        
        if qa_data["total_qa_pairs"] == 0:
            return {
                "success": True,
                "skipped": True,
                "reason": "未能生成任何问答对",
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
        
        # 4. 分析问答质量
        print("📈 分析问答质量...")
        quality_analysis = analyze_qa_quality(qa_data)
        
        processing_time = time.time() - start_time
        
        # 5. 构建结果
        result = {
            "success": True,
            "step": 7,
            "step_name": "问答生成",
            "qa_pairs": qa_data["qa_pairs"],
            "statistics": {
                "total_qa_pairs": qa_data["total_qa_pairs"],
                "processed_evidences": qa_data["processed_evidences"],
                "failed_evidences": qa_data["failed_evidences"],
                "qa_type_distribution": qa_data["qa_stats"]
            },
            "quality_analysis": quality_analysis,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "config_used": {
                "questions_per_type": config.get("qa_generation.questions_per_type", {}),
                "evidence_count": len(evidence_nodes)
            }
        }
        
        return result
        
    except Exception as e:
        error_msg = f"步骤7处理失败: {str(e)}"
        print(f"❌ {error_msg}")
        
        result = {
            "success": False,
            "step": 7,
            "step_name": "问答生成",
            "error": error_msg,
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

def main():
    if len(sys.argv) != 2:
        print("用法: python step7.py <step6输出文件或实验文件夹>")
        print("示例:")
        print("  python step7.py experiments/20241204_143052_attention_paper/step6_evaluation.txt")
        print("  python step7.py experiments/20241204_143052_attention_paper/")
        print("\n新功能:")
        print("✅ 自动识别实验文件夹")
        print("✅ 智能问答生成，支持多种认知层次")
        print("✅ 问答质量评估和分类")
        print("✅ 统一的实验管理")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    print(f"🚀 步骤7: 问答生成 (增强版)")
    print(f"📄 输入: {input_path}")
    print("="*60)
    
    try:
        # 1. 加载步骤6结果和实验管理器
        print("📂 加载步骤6结果...")
        step6_result, experiment_manager = load_step6_result(input_path)
        
        if not step6_result.get("success"):
            print("❌ 步骤6未成功完成，无法继续")
            sys.exit(1)
        
        print(f"✅ 已加载实验: {experiment_manager.experiment_name}")
        print(f"📁 实验目录: {experiment_manager.experiment_dir}")
        print()
        
        # 2. 执行步骤7处理
        print("🔄 开始步骤7处理...")
        result = process_step7_qa_generation(step6_result)
        
        # 3. 保存结果到实验文件夹
        print("💾 保存实验结果...")
        saved_files = experiment_manager.save_step_result(
            step_num=7,
            result=result,
            save_formats=['txt', 'json']
        )
        
        if result.get("success"):
            if result.get("skipped"):
                print(f"\n⏭️ 步骤7已跳过!")
                print(f"📁 实验目录: {experiment_manager.experiment_dir}")
                print(f"⚠️ 跳过原因: {result.get('reason')}")
            else:
                print(f"\n✅ 步骤7完成!")
                print(f"📁 实验目录: {experiment_manager.experiment_dir}")
                print(f"📄 输出文件:")
                for format_type, file_path in saved_files.items():
                    print(f"   - {format_type.upper()}: {file_path}")
                
                # 显示处理结果摘要
                stats = result.get("statistics", {})
                quality = result.get("quality_analysis", {})
                
                print(f"\n📊 问答生成结果摘要:")
                print(f"   - 总问答对数: {stats.get('total_qa_pairs', 0)}")
                print(f"   - 处理证据数: {stats.get('processed_evidences', 0)}")
                print(f"   - 失败证据数: {stats.get('failed_evidences', 0)}")
                print(f"   - 整体质量分数: {quality.get('overall_quality_score', 0):.3f}")
                print(f"   - 平均问题长度: {quality.get('avg_question_length', 0):.1f} 字符")
                print(f"   - 平均答案长度: {quality.get('avg_answer_length', 0):.1f} 字符")
                print(f"   - 处理时间: {result.get('processing_time', 0):.2f} 秒")
                
                # 显示问答类型分布
                qa_type_dist = stats.get("qa_type_distribution", {})
                if qa_type_dist:
                    print(f"\n📈 问答类型分布:")
                    for qa_type, count in qa_type_dist.items():
                        print(f"   - {qa_type}: {count} 个")
                
                # 显示难度分布
                difficulty_dist = quality.get("difficulty_distribution", {})
                if difficulty_dist:
                    print(f"\n📊 难度分布:")
                    for difficulty, count in difficulty_dist.items():
                        print(f"   - {difficulty}: {count} 个")
                
                # 显示问答对示例
                qa_pairs = result.get("qa_pairs", [])
                if qa_pairs:
                    print(f"\n🌟 问答对示例 (前3个):")
                    for i, qa_pair in enumerate(qa_pairs[:3], 1):
                        print(f"   {i}. 类型: {qa_pair.get('type', '未知')}, 难度: {qa_pair.get('difficulty', '未知')}")
                        print(f"      概念: {qa_pair.get('evidence_concept', '未知')}")
                        print(f"      问题: {qa_pair.get('question', '未知')[:80]}...")
                        print(f"      答案: {qa_pair.get('answer', '未知')[:100]}...")
                        print()
                
                if len(qa_pairs) > 3:
                    print(f"   ... 还有 {len(qa_pairs) - 3} 个问答对")
            
            # 显示实验状态
            summary = experiment_manager.get_experiment_summary()
            print(f"\n🧪 实验状态:")
            print(f"   - 实验ID: {summary['experiment_id']}")
            print(f"   - 已完成步骤: {summary['steps_completed']}/{summary['total_steps']}")
            print(f"   - 当前状态: {summary['status']}")
            
            # 提示后续步骤
            if not result.get("skipped"):
                print(f"\n🎉 流水线处理完成!")
                print(f"   查看结果: cat {saved_files['txt']}")
                print(f"   查看问答: grep -A 10 '问答对示例' {saved_files['txt']}")
                print(f"   实验目录: ls {experiment_manager.experiment_dir}")
                
        else:
            print(f"❌ 步骤7失败: {result.get('error')}")
            
            # 即使失败也保存错误信息
            experiment_manager.save_step_result(
                step_num=7,
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
                "step": 7,
                "step_name": "问答生成",
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                experiment_manager.save_step_result(7, error_result, ['txt'])
                print(f"📄 错误信息已保存到实验目录: {experiment_manager.experiment_dir}")
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()