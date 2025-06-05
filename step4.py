#!/usr/bin/env python3
"""
步骤4: 概念合并 - 基于pipeline0604.py
================

用法: python step4.py step3Out.txt
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
        f.write("步骤4: 概念合并 - 处理结果\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理状态: {'✅ 成功' if result.get('success') else '❌ 失败'}\n")
        
        if result.get("success"):
            stats = result.get("statistics", {})
            input_stats = result.get("input_statistics", {})
            
            f.write(f"\n📊 概念合并统计:\n")
            f.write(f"- 合并后概念数: {stats.get('merged_concept_count', 0)}\n")
            f.write(f"- 原始概念数: {input_stats.get('original_concept_count', 0)}\n")
            f.write(f"- 压缩比: {input_stats.get('compression_ratio', 0):.2f}\n")
            f.write(f"- 平均概念长度: {stats.get('avg_concept_length', 0):.1f} 字符\n")
            f.write(f"- 总来源分块数: {stats.get('total_source_chunks', 0)}\n")
            f.write(f"- 平均置信度: {stats.get('avg_confidence', 0):.3f}\n")
            f.write(f"- 处理时间: {result.get('processing_time', 0):.2f} 秒\n")
            
            # 显示合并后的概念
            concept_nodes = result.get("concept_nodes", [])
            f.write(f"\n🔗 合并后的概念列表 ({len(concept_nodes)} 个):\n")
            f.write("-" * 80 + "\n")
            
            for i, concept in enumerate(concept_nodes, 1):
                f.write(f"\n概念 {i}:\n")
                f.write(f"  ID: {concept.get('concept_id', '未知')}\n")
                f.write(f"  文本: {concept.get('concept_text', '未知')}\n")
                f.write(f"  长度: {concept.get('concept_length', 0)} 字符\n")
                f.write(f"  来源分块: {concept.get('source_chunks', [])}\n")
                f.write(f"  置信度: {concept.get('confidence_score', 0):.3f}\n")
                f.write("-" * 60 + "\n")
                
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
        print("用法: python step4.py <step3输出文件>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = "step4Out.txt"
    
    print(f"🔄 步骤4: 概念合并")
    print(f"📄 输入: {input_file}")
    print(f"📄 输出: {output_file}")
    
    try:
        # 读取step3的结果
        step3_result = load_result_from_txt(input_file)
        
        if not step3_result.get("success"):
            print("❌ 输入文件显示step3失败")
            sys.exit(1)
        
        # 读取之前步骤的结果
        step_files = [
            (1, input_file.replace("step3Out.txt", "step1Out.txt")),
            (2, input_file.replace("step3Out.txt", "step2Out.txt"))
        ]
        
        previous_results = {}
        for step_num, file_path in step_files:
            if Path(file_path).exists():
                previous_results[f"step{step_num}"] = load_result_from_txt(file_path)
            else:
                print(f"⚠️ 找不到step{step_num}输出文件")
                previous_results[f"step{step_num}"] = {"success": True}
        
        # 使用现有的处理器
        processor = StepByStepPipelineProcessor(
            input_file="dummy",
            output_dir="temp_step4"
        )
        
        # 手动设置之前步骤的结果
        for step_key, result in previous_results.items():
            processor.step_results[step_key] = result
        processor.step_results["step3"] = step3_result
        
        # 调用现有的方法
        result = processor.step4_concept_merging()
        
        # 保存结果为txt
        save_result_as_txt(result, output_file)
        
        if result.get("success"):
            print(f"✅ 步骤4完成: {output_file}")
        else:
            print(f"❌ 步骤4失败: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
