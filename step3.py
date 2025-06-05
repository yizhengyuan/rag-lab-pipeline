#!/usr/bin/env python3
"""
步骤3: 概念提取 - 基于pipeline0604.py
================

用法: python step3.py step2Out.txt
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

# 导入现有的处理器类
sys.path.append(str(Path(__file__).parent))
from pipeline0604 import StepByStepPipelineProcessor

def save_result_as_txt(result, output_file):
    """保存结果为txt格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("步骤3: 概念提取 - 处理结果\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理状态: {'✅ 成功' if result.get('success') else '❌ 失败'}\n")
        
        if result.get("success"):
            stats = result.get("statistics", {})
            
            f.write(f"\n📊 概念提取统计:\n")
            f.write(f"- 总概念数 (含重复): {stats.get('total_concepts', 0)}\n")
            f.write(f"- 唯一概念数: {stats.get('unique_concepts', 0)}\n")
            f.write(f"- 平均频率: {stats.get('avg_frequency', 0):.2f}\n")
            f.write(f"- 质量评分: {stats.get('quality_score', 0):.2f}/5.0\n")
            f.write(f"- 处理时间: {result.get('processing_time', 0):.2f} 秒\n")
            
            # 显示概念频率分析
            concepts_data = result.get("concepts", {})
            sorted_concepts = concepts_data.get("sorted_by_frequency", [])
            
            f.write(f"\n🔍 概念频率分析 (前20个高频概念):\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'排名':<4} {'频率':<6} {'概念'}\n")
            f.write("-" * 60 + "\n")
            
            for i, (concept, freq) in enumerate(sorted_concepts[:20], 1):
                f.write(f"{i:<4} {freq:<6} {concept}\n")
            
            # 显示质量分析
            quality_analysis = result.get("quality_analysis", {})
            if quality_analysis:
                f.write(f"\n📈 概念质量分析:\n")
                f.write("-" * 40 + "\n")
                f.write(f"平均质量分数: {quality_analysis.get('avg_quality_score', 0):.2f}/5.0\n")
                
                score_dist = quality_analysis.get("score_distribution", {})
                f.write(f"质量分布:\n")
                f.write(f"  - 优秀 (4.0+): {score_dist.get('excellent', 0)} 个\n")
                f.write(f"  - 良好 (3.0-4.0): {score_dist.get('good', 0)} 个\n")
                f.write(f"  - 一般 (2.0-3.0): {score_dist.get('average', 0)} 个\n")
                f.write(f"  - 较差 (<2.0): {score_dist.get('poor', 0)} 个\n")
                
                high_quality = quality_analysis.get("high_quality", [])
                if high_quality:
                    f.write(f"\n🌟 高质量概念示例:\n")
                    for concept in high_quality[:10]:
                        f.write(f"  ✓ {concept}\n")
                
                low_quality = quality_analysis.get("low_quality", [])
                if low_quality:
                    f.write(f"\n⚠️ 低质量概念示例:\n")
                    for concept in low_quality[:10]:
                        f.write(f"  ✗ {concept}\n")
            
            # 显示所有唯一概念
            unique_concepts = concepts_data.get("unique_concepts", [])
            f.write(f"\n📝 所有唯一概念列表 ({len(unique_concepts)} 个):\n")
            f.write("-" * 60 + "\n")
            for i, concept in enumerate(unique_concepts, 1):
                f.write(f"{i:3d}. {concept}\n")
                
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
        print("用法: python step3.py <step2输出文件>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = "step3Out.txt"
    
    print(f"🔄 步骤3: 概念提取")
    print(f"📄 输入: {input_file}")
    print(f"📄 输出: {output_file}")
    
    try:
        # 从txt文件读取step2的结果
        step2_result = load_result_from_txt(input_file)
        
        if not step2_result.get("success"):
            print("❌ 输入文件显示step2失败")
            sys.exit(1)
        
        # 还需要读取step1的结果
        step1_file = input_file.replace("step2Out.txt", "step1Out.txt")
        if Path(step1_file).exists():
            step1_result = load_result_from_txt(step1_file)
        else:
            print("⚠️ 找不到step1输出文件，使用默认配置")
            step1_result = {"success": True}
        
        # 使用现有的处理器
        processor = StepByStepPipelineProcessor(
            input_file="dummy",
            output_dir="temp_step3"
        )
        
        # 手动设置之前步骤的结果
        processor.step_results["step1"] = step1_result
        processor.step_results["step2"] = step2_result
        
        # 调用现有的方法
        result = processor.step3_concept_extraction()
        
        # 保存结果为txt
        save_result_as_txt(result, output_file)
        
        if result.get("success"):
            print(f"✅ 步骤3完成: {output_file}")
        else:
            print(f"❌ 步骤3失败: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
