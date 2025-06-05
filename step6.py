#!/usr/bin/env python3
"""
步骤6: 证据提取 - 基于pipeline0604.py
================

用法: python step6.py step5Out.txt
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 导入现有的处理器类
sys.path.append(str(Path(__file__).parent))
from pipeline0604 import StepByStepPipelineProcessor

def save_result_as_txt(result, output_file):
    """保存结果为txt格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("步骤6: 证据提取 - 处理结果\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理状态: {'✅ 成功' if result.get('success') else '❌ 失败'}\n")
        
        if result.get("success"):
            stats = result.get("statistics", {})
            
            f.write(f"\n📊 证据提取统计:\n")
            f.write(f"- 总证据数: {stats.get('total_evidence', 0)}\n")
            f.write(f"- 有证据的概念数: {stats.get('concepts_with_evidence', 0)}\n")
            f.write(f"- 平均每概念证据数: {stats.get('avg_evidence_per_concept', 0):.2f}\n")
            f.write(f"- 平均证据长度: {stats.get('avg_evidence_length', 0):.1f} 字符\n")
            f.write(f"- 平均相关性分数: {stats.get('avg_relevance_score', 0):.4f}\n")
            f.write(f"- 处理时间: {result.get('processing_time', 0):.2f} 秒\n")
            
            # 显示证据类型分布
            evidence_type_dist = stats.get("evidence_type_distribution", {})
            if evidence_type_dist:
                f.write(f"\n📈 证据类型分布:\n")
                for evidence_type, count in evidence_type_dist.items():
                    f.write(f"  - {evidence_type}: {count} 个\n")
            
            # 显示证据详情
            evidence_nodes = result.get("evidence_nodes", [])
            f.write(f"\n🔍 证据详细信息:\n")
            f.write("-" * 80 + "\n")
            
            for i, evidence in enumerate(evidence_nodes, 1):
                f.write(f"\n证据 {i}:\n")
                f.write(f"  ID: {evidence.get('evidence_id', '未知')}\n")
                f.write(f"  关联概念: {evidence.get('concept_text', '未知')}\n")
                f.write(f"  证据类型: {evidence.get('evidence_type', '未知')}\n")
                f.write(f"  相关性分数: {evidence.get('relevance_score', 0):.4f}\n")
                f.write(f"  证据长度: {evidence.get('evidence_length', 0)} 字符\n")
                f.write(f"  来源分块: {evidence.get('source_chunks', [])}\n")
                f.write(f"  证据内容: {evidence.get('evidence_text', '')[:200]}...\n")
                f.write("-" * 60 + "\n")
            
            # 按概念分组显示证据
            evidence_by_concept = result.get("evidence_by_concept", {})
            f.write(f"\n📋 按概念分组的证据:\n")
            f.write("-" * 80 + "\n")
            
            for concept_id, evidences in evidence_by_concept.items():
                if evidences:
                    concept_text = evidences[0].get('concept_text', '未知概念')
                    f.write(f"\n概念: {concept_text}\n")
                    f.write(f"  概念ID: {concept_id}\n")
                    f.write(f"  证据数量: {len(evidences)}\n")
                    
                    for j, evidence in enumerate(evidences, 1):
                        f.write(f"  证据 {j}: 相关性 {evidence.get('relevance_score', 0):.3f} - {evidence.get('evidence_text', '')[:100]}...\n")
                    
                    f.write("-" * 40 + "\n")
            
            # 显示证据质量分析
            f.write(f"\n📈 证据质量分析:\n")
            f.write("-" * 60 + "\n")
            
            if evidence_nodes:
                # 相关性分数分析
                relevance_scores = [e.get('relevance_score', 0) for e in evidence_nodes]
                high_quality = [e for e in evidence_nodes if e.get('relevance_score', 0) >= 0.8]
                medium_quality = [e for e in evidence_nodes if 0.5 <= e.get('relevance_score', 0) < 0.8]
                low_quality = [e for e in evidence_nodes if e.get('relevance_score', 0) < 0.5]
                
                f.write(f"相关性分数分布:\n")
                f.write(f"  - 高质量 (≥0.8): {len(high_quality)} 个\n")
                f.write(f"  - 中等质量 (0.5-0.8): {len(medium_quality)} 个\n")
                f.write(f"  - 低质量 (<0.5): {len(low_quality)} 个\n")
                
                if relevance_scores:
                    max_score = max(relevance_scores)
                    min_score = min(relevance_scores)
                    f.write(f"  - 最高分数: {max_score:.4f}\n")
                    f.write(f"  - 最低分数: {min_score:.4f}\n")
                
                # 证据长度分析
                evidence_lengths = [e.get('evidence_length', 0) for e in evidence_nodes]
                if evidence_lengths:
                    avg_length = sum(evidence_lengths) / len(evidence_lengths)
                    max_length = max(evidence_lengths)
                    min_length = min(evidence_lengths)
                    f.write(f"\n证据长度分析:\n")
                    f.write(f"  - 平均长度: {avg_length:.1f} 字符\n")
                    f.write(f"  - 最长证据: {max_length} 字符\n")
                    f.write(f"  - 最短证据: {min_length} 字符\n")
                
                # 显示高质量证据示例
                if high_quality:
                    f.write(f"\n🌟 高质量证据示例 (相关性 ≥ 0.8):\n")
                    f.write("-" * 40 + "\n")
                    for i, evidence in enumerate(high_quality[:3], 1):
                        f.write(f"{i}. 概念: {evidence.get('concept_text', '未知')[:30]}...\n")
                        f.write(f"   相关性: {evidence.get('relevance_score', 0):.4f}\n")
                        f.write(f"   内容: {evidence.get('evidence_text', '')[:150]}...\n\n")
            
        else:
            f.write(f"\n❌ 错误信息: {result.get('error', '未知错误')}\n")
        
        # 在文件末尾添加机器可读的数据
        f.write(f"\n" + "="*80 + "\n")
        f.write("# 机器可读数据 (请勿手动修改)\n")
        f.write("# JSON_DATA_START\n")
        f.write(json.dumps(result, ensure_ascii=False, indent=2))
        f.write("\n# JSON_DATA_END\n")

def load_result_from_txt(input_file):
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

def main():
    if len(sys.argv) != 2:
        print("用法: python step6.py <step5输出文件>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = "step6Out.txt"
    
    print(f"🔄 步骤6: 证据提取")
    print(f"📄 输入: {input_file}")
    print(f"📄 输出: {output_file}")
    
    try:
        # 读取step5的结果
        step5_result = load_result_from_txt(input_file)
        
        if not step5_result.get("success"):
            print("❌ 输入文件显示step5失败")
            sys.exit(1)
        
        # 读取之前步骤的结果
        step_files = [
            (1, input_file.replace("step5Out.txt", "step1Out.txt")),
            (2, input_file.replace("step5Out.txt", "step2Out.txt")),
            (3, input_file.replace("step5Out.txt", "step3Out.txt")),
            (4, input_file.replace("step5Out.txt", "step4Out.txt"))
        ]
        
        previous_results = {}
        for step_num, file_path in step_files:
            if Path(file_path).exists():
                try:
                    previous_results[f"step{step_num}"] = load_result_from_txt(file_path)
                    print(f"✅ 成功读取 step{step_num} 结果")
                except Exception as e:
                    print(f"⚠️ 读取step{step_num}文件失败: {e}")
                    previous_results[f"step{step_num}"] = {"success": True}
            else:
                print(f"⚠️ 找不到step{step_num}输出文件: {file_path}")
                previous_results[f"step{step_num}"] = {"success": True}
        
        # 使用现有的处理器
        processor = StepByStepPipelineProcessor(
            input_file="dummy",
            output_dir="temp_step6"
        )
        
        # 手动设置之前步骤的结果
        for step_key, result in previous_results.items():
            processor.step_results[step_key] = result
        processor.step_results["step5"] = step5_result
        
        print("🔧 开始执行证据提取...")
        
        # 调用现有的方法
        result = processor.step6_evidence_extraction()
        
        # 保存结果为txt
        save_result_as_txt(result, output_file)
        
        if result.get("success"):
            print(f"✅ 步骤6完成: {output_file}")
            
            # 显示简要统计
            stats = result.get("statistics", {})
            print(f"📊 证据提取结果:")
            print(f"   - 总证据数: {stats.get('total_evidence', 0)}")
            print(f"   - 有证据的概念数: {stats.get('concepts_with_evidence', 0)}")
            print(f"   - 平均相关性: {stats.get('avg_relevance_score', 0):.4f}")
            print(f"   - 处理时间: {result.get('processing_time', 0):.2f} 秒")
            
        else:
            print(f"❌ 步骤6失败: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        
        # 保存错误信息到输出文件
        error_result = {
            "step": 6,
            "step_name": "证据提取",
            "success": False,
            "error": str(e),
            "processing_time": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            save_result_as_txt(error_result, output_file)
            print(f"📄 错误信息已保存到: {output_file}")
        except:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()
