#!/usr/bin/env python3
"""
步骤8: 最终汇总 - 基于pipeline0604.py
================

用法: python step8.py step7Out.txt
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
        f.write("步骤8: 最终汇总 - 处理结果\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理状态: {'✅ 成功' if result.get('success') else '❌ 失败'}\n")
        
        if result.get("success"):
            f.write(f"\n🎉 Pipeline处理完整汇总报告\n")
            f.write("="*80 + "\n")
            
            total_time = result.get("total_processing_time", 0)
            step_results = result.get("step_results", {})
            step_timings = result.get("step_timings", {})
            
            f.write(f"总处理时间: {total_time:.2f} 秒\n")
            
            # 统计成功和失败的步骤
            success_count = 0
            failed_count = 0
            skipped_count = 0
            
            for step_result in step_results.values():
                if step_result.get("success"):
                    success_count += 1
                elif step_result.get("skipped"):
                    skipped_count += 1
                else:
                    failed_count += 1
            
            f.write(f"成功步骤数: {success_count}\n")
            f.write(f"跳过步骤数: {skipped_count}\n")
            f.write(f"失败步骤数: {failed_count}\n")
            
            f.write(f"\n📊 各步骤详细信息:\n")
            f.write("-" * 80 + "\n")
            
            step_names = [
                "步骤1: 文件加载",
                "步骤2: 文档分块", 
                "步骤3: 概念提取",
                "步骤4: 概念合并",
                "步骤5: 概念检索", 
                "步骤6: 证据提取",
                "步骤7: 问答生成",
                "步骤8: 最终汇总"
            ]
            
            for i, step_name in enumerate(step_names, 1):
                step_key = f"step{i}"
                if step_key in step_timings:
                    timing = step_timings[step_key]
                    step_result = step_results.get(step_key, {})
                    
                    if step_result.get("success"):
                        status = "✅ 成功"
                    elif step_result.get("skipped"):
                        status = "⏭️ 跳过"
                    else:
                        status = "❌ 失败"
                    
                    f.write(f"{step_name}: {status}, 耗时: {timing:.2f} 秒\n")
                else:
                    f.write(f"{step_name}: ⚠️ 未执行\n")
            
            # 显示关键统计信息
            f.write(f"\n📈 处理结果统计:\n")
            f.write("-" * 60 + "\n")
            
            # 从各步骤提取关键信息
            step1_result = step_results.get("step1", {})
            if step1_result.get("success"):
                text_length = step1_result.get("text_length", 0)
                f.write(f"原始文档长度: {text_length:,} 字符\n")
            
            step2_result = step_results.get("step2", {})
            if step2_result.get("success"):
                chunk_count = step2_result.get("chunk_count", 0)
                f.write(f"生成分块数: {chunk_count}\n")
            
            step3_result = step_results.get("step3", {})
            if step3_result.get("success"):
                stats = step3_result.get("statistics", {})
                unique_concepts = stats.get("unique_concepts", 0)
                f.write(f"提取概念数: {unique_concepts}\n")
            
            step4_result = step_results.get("step4", {})
            if step4_result.get("success"):
                stats = step4_result.get("statistics", {})
                merged_concepts = stats.get("merged_concept_count", 0)
                f.write(f"合并后概念数: {merged_concepts}\n")
            
            step6_result = step_results.get("step6", {})
            if step6_result.get("success"):
                stats = step6_result.get("statistics", {})
                total_evidence = stats.get("total_evidence", 0)
                f.write(f"提取证据数: {total_evidence}\n")
            
            step7_result = step_results.get("step7", {})
            if step7_result.get("success"):
                stats = step7_result.get("statistics", {})
                total_qa = stats.get("total_qa_pairs", 0)
                f.write(f"生成问答对数: {total_qa}\n")
            elif step7_result.get("skipped"):
                f.write(f"问答生成: 已跳过\n")
            
            f.write(f"\n🎯 Pipeline执行总结:\n")
            f.write("-" * 60 + "\n")
            
            if failed_count == 0:
                f.write("🎉 所有步骤执行成功！\n")
            else:
                f.write(f"⚠️ 有 {failed_count} 个步骤执行失败\n")
            
            f.write(f"总耗时: {total_time:.2f} 秒\n")
            f.write(f"平均每步耗时: {total_time / len(step_timings):.2f} 秒\n" if step_timings else "平均每步耗时: 0.00 秒\n")
                
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
        print("用法: python step8.py <step7输出文件>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = "step8Out.txt"
    
    print(f"🔄 步骤8: 最终汇总")
    print(f"📄 输入: {input_file}")
    print(f"📄 输出: {output_file}")
    
    try:
        # 读取step7的结果
        step7_result = load_result_from_txt(input_file)
        
        if not (step7_result.get("success") or step7_result.get("skipped")):
            print("❌ 输入文件显示step7失败")
            sys.exit(1)
        
        # 读取所有之前步骤的结果
        step_files = [
            (1, input_file.replace("step7Out.txt", "step1Out.txt")),
            (2, input_file.replace("step7Out.txt", "step2Out.txt")),
            (3, input_file.replace("step7Out.txt", "step3Out.txt")),
            (4, input_file.replace("step7Out.txt", "step4Out.txt")),
            (5, input_file.replace("step7Out.txt", "step5Out.txt")),
            (6, input_file.replace("step7Out.txt", "step6Out.txt"))
        ]
        
        previous_results = {}
        step_timings = {}
        
        for step_num, file_path in step_files:
            if Path(file_path).exists():
                result = load_result_from_txt(file_path)
                previous_results[f"step{step_num}"] = result
                step_timings[f"step{step_num}"] = result.get("processing_time", 0.0)
            else:
                print(f"⚠️ 找不到step{step_num}输出文件")
                previous_results[f"step{step_num}"] = {"success": True}
                step_timings[f"step{step_num}"] = 0.0
        
        # 使用现有的处理器
        processor = StepByStepPipelineProcessor(
            input_file="dummy",
            output_dir="temp_step8"
        )
        
        # 手动设置之前步骤的结果和时间
        for step_key, result in previous_results.items():
            processor.step_results[step_key] = result
        for step_key, timing in step_timings.items():
            processor.step_timings[step_key] = timing
        
        processor.step_results["step7"] = step7_result
        processor.step_timings["step7"] = step7_result.get("processing_time", 0.0)
        
        # 调用现有的方法
        result = processor.step8_final_summary()
        
        # 保存结果为txt
        save_result_as_txt(result, output_file)
        
        if result.get("success"):
            print(f"✅ 步骤8完成: {output_file}")
            
            # 显示汇总信息
            total_time = result.get("total_processing_time", 0)
            success_count = sum(1 for r in previous_results.values() if r.get("success", False))
            success_count += 1 if step7_result.get("success") or step7_result.get("skipped") else 0
            success_count += 1  # step8
            
            print(f"\n🎉 Pipeline执行完成!")
            print(f"📊 总处理时间: {total_time:.2f} 秒")
            print(f"📊 成功步骤数: {success_count}/8")
            print(f"📁 详细结果: {processor.output_dir}")
            
        else:
            print(f"❌ 步骤8失败: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
