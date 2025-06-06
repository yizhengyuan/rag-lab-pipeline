#!/usr/bin/env python3
"""
步骤8: 最终汇总与报告 - 增强版
================================

功能：
1. 从步骤7的结果或实验文件夹中读取所有步骤结果
2. 生成完整的流水线执行汇总报告
3. 分析各步骤性能和成功率
4. 保存到同一个实验文件夹

用法: 
- python step8.py <step7输出文件.txt>
- python step8.py <实验文件夹路径>

新功能：
- ✅ 自动识别并使用step7的实验文件夹
- ✅ 完整的流水线执行报告
- ✅ 详细的性能分析和统计
- ✅ 统一的实验管理
- ✅ 支持多种输出格式
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict
import logging

# 导入核心模块
sys.path.append(str(Path(__file__).parent))

# 导入核心处理模块
from config.settings import load_config_from_yaml

# 导入实验管理器
from utils.experiment_manager import ExperimentManager
from utils.helpers import FileHelper

logger = logging.getLogger(__name__)

def load_step7_result(step7_file_or_dir: str) -> tuple:
    """
    从步骤7的输出文件或实验文件夹中加载结果
    
    Args:
        step7_file_or_dir: 步骤7的输出文件或实验文件夹路径
        
    Returns:
        tuple: (step7_result, experiment_manager)
    """
    step7_path = Path(step7_file_or_dir)
    
    if step7_path.is_file():
        # 情况1：直接指定了step7的输出文件
        if step7_path.name.startswith("step7") and step7_path.suffix == ".txt":
            experiment_dir = step7_path.parent
            experiment_manager = ExperimentManager.load_experiment(str(experiment_dir))
            
            # 从txt文件加载结果
            step7_result = load_result_from_txt(str(step7_path))
            
        else:
            raise ValueError(f"不支持的文件格式: {step7_path}")
            
    elif step7_path.is_dir():
        # 情况2：直接指定了实验文件夹
        experiment_manager = ExperimentManager.load_experiment(str(step7_path))
        
        # 查找step7的输出文件
        step7_txt_path = experiment_manager.get_step_output_path(7, "txt")
        if not step7_txt_path.exists():
            raise FileNotFoundError(f"实验文件夹中找不到step7输出文件: {step7_txt_path}")
        
        step7_result = load_result_from_txt(str(step7_txt_path))
        
    else:
        raise FileNotFoundError(f"输入路径不存在: {step7_path}")
    
    return step7_result, experiment_manager

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

def load_all_steps_data(experiment_manager: ExperimentManager) -> Dict[str, Any]:
    """
    加载所有步骤的数据
    
    Args:
        experiment_manager: 实验管理器
        
    Returns:
        Dict[str, Any]: 包含所有步骤数据的字典
    """
    print("📂 加载所有步骤的数据...")
    
    all_steps_data = {}
    step_timings = {}
    
    # Step信息映射
    step_info = {
        1: {"name": "文档加载与向量化存储", "file_prefix": "step1"},
        2: {"name": "文档分块与概念提取", "file_prefix": "step2"},
        3: {"name": "概念提取与映射", "file_prefix": "step3"},
        4: {"name": "概念合并与优化", "file_prefix": "step4"},
        5: {"name": "概念检索与映射", "file_prefix": "step5"},
        6: {"name": "证据提取与质量评估", "file_prefix": "step6"},
        7: {"name": "问答生成", "file_prefix": "step7"}
    }
    
    for step_num, step_info_item in step_info.items():
        try:
            step_txt_path = experiment_manager.get_step_output_path(step_num, "txt")
            
            if step_txt_path.exists():
                step_result = load_result_from_txt(str(step_txt_path))
                all_steps_data[f"step{step_num}"] = step_result
                step_timings[f"step{step_num}"] = step_result.get("processing_time", 0.0)
                print(f"   ✅ 加载步骤{step_num}: {step_info_item['name']}")
            else:
                print(f"   ⚠️ 步骤{step_num}文件不存在: {step_txt_path}")
                all_steps_data[f"step{step_num}"] = {"success": False, "error": "文件不存在"}
                step_timings[f"step{step_num}"] = 0.0
                
        except Exception as e:
            print(f"   ❌ 加载步骤{step_num}失败: {e}")
            all_steps_data[f"step{step_num}"] = {"success": False, "error": str(e)}
            step_timings[f"step{step_num}"] = 0.0
    
    print(f"   ✅ 共加载 {len([k for k, v in all_steps_data.items() if v.get('success')])} 个成功步骤")
    
    return {
        "step_results": all_steps_data,
        "step_timings": step_timings
    }

def analyze_pipeline_performance(step_results: Dict[str, Any], 
                               step_timings: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析流水线性能
    
    Args:
        step_results: 步骤结果
        step_timings: 步骤时间
        
    Returns:
        Dict[str, Any]: 性能分析结果
    """
    print("📈 分析流水线性能...")
    
    # 基本统计
    total_steps = len(step_results)
    successful_steps = len([r for r in step_results.values() if r.get("success")])
    failed_steps = len([r for r in step_results.values() if r.get("success") == False])
    skipped_steps = len([r for r in step_results.values() if r.get("skipped")])
    
    total_time = sum(step_timings.values())
    avg_time_per_step = total_time / total_steps if total_steps > 0 else 0
    
    # 步骤状态分析
    step_status = {}
    for step_key, result in step_results.items():
        if result.get("success"):
            status = "success"
        elif result.get("skipped"):
            status = "skipped"
        else:
            status = "failed"
        step_status[step_key] = status
    
    # 时间分析
    slowest_step = max(step_timings.items(), key=lambda x: x[1]) if step_timings else ("", 0)
    fastest_step = min(step_timings.items(), key=lambda x: x[1]) if step_timings else ("", 0)
    
    # 数据流分析
    data_flow = extract_data_flow_metrics(step_results)
    
    return {
        "basic_stats": {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "skipped_steps": skipped_steps,
            "success_rate": successful_steps / total_steps if total_steps > 0 else 0
        },
        "timing_stats": {
            "total_time": total_time,
            "avg_time_per_step": avg_time_per_step,
            "slowest_step": slowest_step,
            "fastest_step": fastest_step
        },
        "step_status": step_status,
        "data_flow": data_flow
    }

def extract_data_flow_metrics(step_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    提取数据流指标
    
    Args:
        step_results: 步骤结果
        
    Returns:
        Dict[str, Any]: 数据流指标
    """
    data_flow = {}
    
    # Step1: 文档信息
    step1_result = step_results.get("step1", {})
    if step1_result.get("success"):
        stats = step1_result.get("statistics", {})
        data_flow["document_length"] = step1_result.get("document", {}).get("metadata", {}).get("text_length", 0)
        data_flow["total_chunks"] = stats.get("total_chunks", 0)
        data_flow["vector_nodes"] = step1_result.get("vector_info", {}).get("vectorized_nodes", 0)
    
    # Step2: 分块和概念
    step2_result = step_results.get("step2", {})
    if step2_result.get("success"):
        stats = step2_result.get("statistics", {})
        data_flow["processed_chunks"] = stats.get("total_chunks", 0)
        data_flow["extracted_concepts"] = stats.get("unique_concepts", 0)
    
    # Step3: 概念分析
    step3_result = step_results.get("step3", {})
    if step3_result.get("success"):
        stats = step3_result.get("statistics", {})
        data_flow["analyzed_concepts"] = stats.get("unique_concepts", 0)
        data_flow["high_quality_concepts"] = stats.get("high_quality_count", 0)
    
    # Step4: 概念合并
    step4_result = step_results.get("step4", {})
    if step4_result.get("success"):
        stats = step4_result.get("statistics", {})
        data_flow["merged_concepts"] = stats.get("merged_concept_count", 0)
        data_flow["compression_ratio"] = step4_result.get("input_statistics", {}).get("compression_ratio", 0)
    
    # Step5: 概念检索
    step5_result = step_results.get("step5", {})
    if step5_result.get("success"):
        stats = step5_result.get("statistics", {})
        data_flow["total_retrievals"] = stats.get("total_retrievals", 0)
        data_flow["retrieval_coverage"] = stats.get("retrieval_coverage", 0)
    
    # Step6: 证据提取
    step6_result = step_results.get("step6", {})
    if step6_result.get("success"):
        stats = step6_result.get("statistics", {})
        data_flow["total_evidence"] = stats.get("total_evidence", 0)
        data_flow["concepts_with_evidence"] = stats.get("concepts_with_evidence", 0)
    
    # Step7: 问答生成
    step7_result = step_results.get("step7", {})
    if step7_result.get("success") and not step7_result.get("skipped"):
        stats = step7_result.get("statistics", {})
        data_flow["total_qa_pairs"] = stats.get("total_qa_pairs", 0)
        data_flow["processed_evidences"] = stats.get("processed_evidences", 0)
    
    return data_flow

def generate_final_summary(step_results: Dict[str, Any], 
                         step_timings: Dict[str, Any],
                         performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成最终汇总
    
    Args:
        step_results: 步骤结果
        step_timings: 步骤时间
        performance_analysis: 性能分析
        
    Returns:
        Dict[str, Any]: 最终汇总结果
    """
    print("📊 生成最终汇总...")
    
    # 步骤名称映射
    step_names = {
        "step1": "文档加载与向量化存储",
        "step2": "文档分块与概念提取", 
        "step3": "概念提取与映射",
        "step4": "概念合并与优化",
        "step5": "概念检索与映射",
        "step6": "证据提取与质量评估",
        "step7": "问答生成"
    }
    
    # 生成步骤详情
    step_details = []
    for step_key in ["step1", "step2", "step3", "step4", "step5", "step6", "step7"]:
        step_result = step_results.get(step_key, {})
        step_time = step_timings.get(step_key, 0.0)
        step_name = step_names.get(step_key, f"步骤{step_key[-1]}")
        
        if step_result.get("success"):
            status = "✅ 成功"
            status_code = "success"
        elif step_result.get("skipped"):
            status = "⏭️ 跳过"
            status_code = "skipped"
        else:
            status = "❌ 失败"
            status_code = "failed"
        
        step_details.append({
            "step_key": step_key,
            "step_name": step_name,
            "status": status,
            "status_code": status_code,
            "processing_time": step_time,
            "error": step_result.get("error", None) if not step_result.get("success") else None
        })
    
    # 关键指标提取
    data_flow = performance_analysis["data_flow"]
    key_metrics = {
        "document_processing": {
            "input_document_length": data_flow.get("document_length", 0),
            "generated_chunks": data_flow.get("total_chunks", 0),
            "vectorized_nodes": data_flow.get("vector_nodes", 0)
        },
        "concept_processing": {
            "extracted_concepts": data_flow.get("extracted_concepts", 0),
            "high_quality_concepts": data_flow.get("high_quality_concepts", 0),
            "merged_concepts": data_flow.get("merged_concepts", 0),
            "compression_ratio": data_flow.get("compression_ratio", 0)
        },
        "evidence_and_qa": {
            "total_evidence": data_flow.get("total_evidence", 0),
            "concepts_with_evidence": data_flow.get("concepts_with_evidence", 0),
            "total_qa_pairs": data_flow.get("total_qa_pairs", 0),
            "processed_evidences": data_flow.get("processed_evidences", 0)
        }
    }
    
    # 生成总结信息
    basic_stats = performance_analysis["basic_stats"]
    timing_stats = performance_analysis["timing_stats"]
    
    summary_info = {
        "pipeline_status": "complete" if basic_stats["failed_steps"] == 0 else "partial",
        "overall_success": basic_stats["failed_steps"] == 0,
        "execution_summary": {
            "total_steps": basic_stats["total_steps"],
            "successful_steps": basic_stats["successful_steps"],
            "failed_steps": basic_stats["failed_steps"],
            "skipped_steps": basic_stats["skipped_steps"],
            "success_rate": basic_stats["success_rate"]
        },
        "performance_summary": {
            "total_processing_time": timing_stats["total_time"],
            "average_time_per_step": timing_stats["avg_time_per_step"],
            "slowest_step": timing_stats["slowest_step"],
            "fastest_step": timing_stats["fastest_step"]
        }
    }
    
    return {
        "step_details": step_details,
        "key_metrics": key_metrics,
        "summary_info": summary_info,
        "data_flow_metrics": data_flow
    }

def process_step8_final_summary(step7_result: Dict[str, Any],
                               all_steps_data: Dict[str, Any],
                               config_path: str = "config.yml") -> Dict[str, Any]:
    """
    执行步骤8的最终汇总处理
    
    Args:
        step7_result: 步骤7的结果
        all_steps_data: 所有步骤的数据
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 步骤8的处理结果
    """
    start_time = time.time()
    
    try:
        # 1. 加载配置
        print("📋 加载配置...")
        config = load_config_from_yaml(config_path)
        
        # 2. 添加step7结果到所有步骤数据中
        all_steps_data["step_results"]["step7"] = step7_result
        all_steps_data["step_timings"]["step7"] = step7_result.get("processing_time", 0.0)
        
        # 3. 分析流水线性能
        print("📈 分析流水线性能...")
        performance_analysis = analyze_pipeline_performance(
            all_steps_data["step_results"],
            all_steps_data["step_timings"]
        )
        
        # 4. 生成最终汇总
        print("📊 生成最终汇总...")
        final_summary = generate_final_summary(
            all_steps_data["step_results"],
            all_steps_data["step_timings"],
            performance_analysis
        )
        
        processing_time = time.time() - start_time
        
        # 5. 构建结果
        result = {
            "success": True,
            "step": 8,
            "step_name": "最终汇总与报告",
            "pipeline_summary": final_summary,
            "performance_analysis": performance_analysis,
            "all_step_results": all_steps_data["step_results"],
            "all_step_timings": all_steps_data["step_timings"],
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "config_used": {
                "total_steps_analyzed": len(all_steps_data["step_results"]),
                "experiment_management": True
            }
        }
        
        return result
        
    except Exception as e:
        error_msg = f"步骤8处理失败: {str(e)}"
        print(f"❌ {error_msg}")
        
        result = {
            "success": False,
            "step": 8,
            "step_name": "最终汇总与报告",
            "error": error_msg,
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

def main():
    if len(sys.argv) != 2:
        print("用法: python step8.py <step7输出文件或实验文件夹>")
        print("示例:")
        print("  python step8.py experiments/20241204_143052_attention_paper/step7_qa_generation.txt")
        print("  python step8.py experiments/20241204_143052_attention_paper/")
        print("\n新功能:")
        print("✅ 自动识别实验文件夹")
        print("✅ 完整的流水线执行报告")
        print("✅ 详细的性能分析和统计")
        print("✅ 统一的实验管理")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    print(f"🚀 步骤8: 最终汇总与报告 (增强版)")
    print(f"📄 输入: {input_path}")
    print("="*60)
    
    try:
        # 1. 加载步骤7结果和实验管理器
        print("📂 加载步骤7结果...")
        step7_result, experiment_manager = load_step7_result(input_path)
        
        print(f"✅ 已加载实验: {experiment_manager.experiment_name}")
        print(f"📁 实验目录: {experiment_manager.experiment_dir}")
        print()
        
        # 2. 加载所有步骤的数据
        print("📂 加载所有步骤数据...")
        all_steps_data = load_all_steps_data(experiment_manager)
        
        # 3. 执行步骤8处理
        print("🔄 开始步骤8处理...")
        result = process_step8_final_summary(step7_result, all_steps_data)
        
        # 4. 保存结果到实验文件夹
        print("💾 保存实验结果...")
        saved_files = experiment_manager.save_step_result(
            step_num=8,
            result=result,
            save_formats=['txt', 'json']
        )
        
        if result.get("success"):
            print(f"\n✅ 步骤8完成!")
            print(f"📁 实验目录: {experiment_manager.experiment_dir}")
            print(f"📄 输出文件:")
            for format_type, file_path in saved_files.items():
                print(f"   - {format_type.upper()}: {file_path}")
            
            # 显示处理结果摘要
            pipeline_summary = result.get("pipeline_summary", {})
            performance_analysis = result.get("performance_analysis", {})
            
            summary_info = pipeline_summary.get("summary_info", {})
            execution_summary = summary_info.get("execution_summary", {})
            performance_summary = summary_info.get("performance_summary", {})
            
            print(f"\n🎉 流水线执行汇总:")
            print(f"   - 流水线状态: {'✅ 完整' if summary_info.get('overall_success') else '⚠️ 部分成功'}")
            print(f"   - 总步骤数: {execution_summary.get('total_steps', 0)}")
            print(f"   - 成功步骤: {execution_summary.get('successful_steps', 0)}")
            print(f"   - 失败步骤: {execution_summary.get('failed_steps', 0)}")
            print(f"   - 跳过步骤: {execution_summary.get('skipped_steps', 0)}")
            print(f"   - 成功率: {execution_summary.get('success_rate', 0):.1%}")
            print(f"   - 总处理时间: {performance_summary.get('total_processing_time', 0):.2f} 秒")
            print(f"   - 平均每步时间: {performance_summary.get('average_time_per_step', 0):.2f} 秒")
            
            # 显示数据流指标
            key_metrics = pipeline_summary.get("key_metrics", {})
            
            doc_metrics = key_metrics.get("document_processing", {})
            concept_metrics = key_metrics.get("concept_processing", {})
            qa_metrics = key_metrics.get("evidence_and_qa", {})
            
            print(f"\n📊 数据流指标:")
            print(f"   📄 文档处理:")
            print(f"      - 输入文档长度: {doc_metrics.get('input_document_length', 0):,} 字符")
            print(f"      - 生成分块数: {doc_metrics.get('generated_chunks', 0)}")
            print(f"      - 向量化节点数: {doc_metrics.get('vectorized_nodes', 0)}")
            
            print(f"   🧠 概念处理:")
            print(f"      - 提取概念数: {concept_metrics.get('extracted_concepts', 0)}")
            print(f"      - 高质量概念数: {concept_metrics.get('high_quality_concepts', 0)}")
            print(f"      - 合并后概念数: {concept_metrics.get('merged_concepts', 0)}")
            print(f"      - 压缩比: {concept_metrics.get('compression_ratio', 0):.2f}")
            
            print(f"   📋 证据与问答:")
            print(f"      - 总证据数: {qa_metrics.get('total_evidence', 0)}")
            print(f"      - 有证据概念数: {qa_metrics.get('concepts_with_evidence', 0)}")
            print(f"      - 生成问答对数: {qa_metrics.get('total_qa_pairs', 0)}")
            print(f"      - 处理证据数: {qa_metrics.get('processed_evidences', 0)}")
            
            # 显示步骤详情
            step_details = pipeline_summary.get("step_details", [])
            print(f"\n📋 各步骤执行状态:")
            for step_detail in step_details:
                print(f"   {step_detail['step_name']}: {step_detail['status']}, 耗时: {step_detail['processing_time']:.2f} 秒")
                if step_detail.get('error'):
                    print(f"      错误: {step_detail['error']}")
            
            # 显示实验状态
            summary = experiment_manager.get_experiment_summary()
            print(f"\n🧪 实验状态:")
            print(f"   - 实验ID: {summary['experiment_id']}")
            print(f"   - 已完成步骤: {summary['steps_completed']}/{summary['total_steps']}")
            print(f"   - 当前状态: {summary['status']}")
            
            # 最终完成提示
            print(f"\n🎉 恭喜！完整的文档处理流水线已执行完成!")
            print(f"   查看详细报告: cat {saved_files['txt']}")
            print(f"   实验完整目录: ls {experiment_manager.experiment_dir}")
            print(f"   JSON数据: {saved_files['json']}")
                
        else:
            print(f"❌ 步骤8失败: {result.get('error')}")
            
            # 即使失败也保存错误信息
            experiment_manager.save_step_result(
                step_num=8,
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
                "step": 8,
                "step_name": "最终汇总与报告",
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                experiment_manager.save_step_result(8, error_result, ['txt'])
                print(f"📄 错误信息已保存到实验目录: {experiment_manager.experiment_dir}")
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()
