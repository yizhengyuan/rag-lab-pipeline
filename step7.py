#!/usr/bin/env python3
"""
步骤7: 问答生成 - 基于pipeline0604.py
================

用法: python step7.py step6Out.txt [--skip-qa]
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
        f.write("步骤7: 问答生成 - 处理结果\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if result.get("skipped"):
            f.write(f"处理状态: ⏭️ 已跳过\n")
            f.write(f"跳过原因: {result.get('reason', '用户指定跳过')}\n")
        else:
            f.write(f"处理状态: {'✅ 成功' if result.get('success') else '❌ 失败'}\n")
        
        if result.get("success") and not result.get("skipped"):
            stats = result.get("statistics", {})
            
            f.write(f"\n📊 问答生成统计:\n")
            f.write(f"- 总问答对数: {stats.get('total_qa_pairs', 0)}\n")
            f.write(f"- 处理时间: {result.get('processing_time', 0):.2f} 秒\n")
            
            # 显示问答类型分布
            qa_types = stats.get("qa_types", {})
            if qa_types:
                f.write(f"\n📈 问答类型分布:\n")
                for qa_type, count in qa_types.items():
                    f.write(f"  - {qa_type}: {count} 个\n")
            
            # 显示问答对详情
            qa_pairs = result.get("qa_pairs", [])
            f.write(f"\n❓ 问答对详细信息:\n")
            f.write("-" * 80 + "\n")
            
            for i, qa_pair in enumerate(qa_pairs, 1):
                f.write(f"\n问答对 {i}:\n")
                f.write(f"  类型: {qa_pair.get('type', '未知')}\n")
                f.write(f"  难度: {qa_pair.get('difficulty', '未知')}\n")
                f.write(f"  来源概念: {qa_pair.get('evidence_concept', '未知')}\n")
                f.write(f"  来源证据ID: {qa_pair.get('evidence_source', '未知')}\n")
                f.write(f"  问题: {qa_pair.get('question', '未知')}\n")
                f.write(f"  答案: {qa_pair.get('answer', '未知')[:200]}...\n")
                f.write("-" * 60 + "\n")
            
            # 显示训练数据格式
            training_data = result.get("training_data", [])
            if training_data:
                f.write(f"\n📝 训练数据样例 (标准格式):\n")
                f.write("-" * 80 + "\n")
                
                for i, item in enumerate(training_data[:3], 1):  # 只显示前3个
                    f.write(f"\n训练数据 {i}:\n")
                    f.write(f"  ID: {item.get('_id', '未知')}\n")
                    f.write(f"  Question: {item.get('Question', '未知')}\n")
                    f.write(f"  Answer: {item.get('Answer', '未知')[:150]}...\n")
                    f.write(f"  Type: {item.get('Type', '未知')}\n")
                    f.write(f"  Difficulty: {item.get('Difficulty', '未知')}\n")
                    f.write(f"  Domain: {item.get('Domain', '未知')}\n")
                    f.write("-" * 40 + "\n")
                
                if len(training_data) > 3:
                    f.write(f"\n... 还有 {len(training_data) - 3} 个训练数据项\n")
                    
        elif result.get("skipped"):
            f.write(f"\n⏭️ 问答生成已跳过\n")
            f.write(f"处理时间: {result.get('processing_time', 0):.2f} 秒\n")
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
    if len(sys.argv) < 2:
        print("用法: python step7.py <step6输出文件> [--skip-qa]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = "step7Out.txt"
    skip_qa = "--skip-qa" in sys.argv
    
    print(f"🔄 步骤7: 问答生成")
    print(f"📄 输入: {input_file}")
    print(f"📄 输出: {output_file}")
    print(f"🎛️  跳过QA: {skip_qa}")
    
    try:
        # 读取step6的结果
        step6_result = load_result_from_txt(input_file)
        
        if not step6_result.get("success"):
            print("❌ 输入文件显示step6失败")
            sys.exit(1)
        
        # 读取之前步骤的结果
        step_files = [
            (1, input_file.replace("step6Out.txt", "step1Out.txt")),
            (2, input_file.replace("step6Out.txt", "step2Out.txt")),
            (3, input_file.replace("step6Out.txt", "step3Out.txt")),
            (4, input_file.replace("step6Out.txt", "step4Out.txt")),
            (5, input_file.replace("step6Out.txt", "step5Out.txt"))
        ]
        
        previous_results = {}
        for step_num, file_path in step_files:
            if Path(file_path).exists():
                previous_results[f"step{step_num}"] = load_result_from_txt(file_path)
            else:
                print(f"⚠️ 找不到step{step_num}输出文件")
                previous_results[f"step{step_num}"] = {"success": True}
        
        # 使用现有的处理器
        enable_qa = not skip_qa
        processor = StepByStepPipelineProcessor(
            input_file="dummy",
            output_dir="temp_step7",
            enable_qa_generation=enable_qa
        )
        
        # 手动设置之前步骤的结果
        for step_key, result in previous_results.items():
            processor.step_results[step_key] = result
        processor.step_results["step6"] = step6_result
        
        # 调用现有的方法
        result = processor.step7_qa_generation()
        
        # 保存结果为txt
        save_result_as_txt(result, output_file)
        
        if result.get("success") or result.get("skipped"):
            status = "⏭️ 跳过" if result.get("skipped") else "✅ 完成"
            print(f"{status} 步骤7: {output_file}")
        else:
            print(f"❌ 步骤7失败: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
