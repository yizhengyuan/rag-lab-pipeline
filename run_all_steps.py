#!/usr/bin/env python3
"""
便捷执行脚本 - 一键运行所有步骤 (txt输出版本)
================

用法: python run_all_steps.py [输入文件] [--skip-qa]
"""

import sys
import subprocess
from pathlib import Path

def main():
    # 解析参数
    input_file = "attention is all you need.pdf"  # 默认文件
    skip_qa = False
    
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        input_file = sys.argv[1]
    
    if "--skip-qa" in sys.argv:
        skip_qa = True
    
    print("🚀 开始执行完整Pipeline流程 (txt输出版本)")
    print(f"📄 输入文件: {input_file}")
    print(f"🎛️  跳过QA: {skip_qa}")
    print("=" * 60)
    
    if not Path(input_file).exists():
        print(f"❌ 输入文件不存在: {input_file}")
        sys.exit(1)
    
    # 步骤定义
    steps = [
        ("python", "step1.py", input_file),
        ("python", "step2.py", "step1Out.txt"),
        ("python", "step3.py", "step2Out.txt"),
        ("python", "step4.py", "step3Out.txt"),
        ("python", "step5.py", "step4Out.txt"),
        ("python", "step6.py", "step5Out.txt"),
        ("python", "step7.py", "step6Out.txt") + (("--skip-qa",) if skip_qa else ()),
        ("python", "step8.py", "step7Out.txt")
    ]
    
    # 输出文件列表
    output_files = [
        "step1Out.txt", "step2Out.txt", "step3Out.txt", "step4Out.txt",
        "step5Out.txt", "step6Out.txt", "step7Out.txt", "step8Out.txt"
    ]
    
    # 执行所有步骤
    completed_steps = []
    
    for i, step_cmd in enumerate(steps, 1):
        print(f"\n🔄 执行步骤{i}...")
        
        try:
            result = subprocess.run(step_cmd, capture_output=True, text=True, encoding='utf-8')
            
            # 显示输出
            if result.stdout:
                print(result.stdout)
            
            if result.returncode != 0:
                print(f"❌ 步骤{i}失败!")
                if result.stderr:
                    print(f"错误信息: {result.stderr}")
                
                # 显示已完成的步骤
                if completed_steps:
                    print(f"\n✅ 已完成的步骤输出文件:")
                    for step_num, output_file in completed_steps:
                        print(f"   步骤{step_num}: {output_file}")
                
                sys.exit(1)
            else:
                completed_steps.append((i, output_files[i-1]))
                
        except Exception as e:
            print(f"❌ 步骤{i}执行异常: {e}")
            
            # 显示已完成的步骤
            if completed_steps:
                print(f"\n✅ 已完成的步骤输出文件:")
                for step_num, output_file in completed_steps:
                    print(f"   步骤{step_num}: {output_file}")
            
            sys.exit(1)
    
    print("\n🎉 所有步骤执行完成!")
    print("\n📄 生成的输出文件:")
    for i, output_file in enumerate(output_files, 1):
        if Path(output_file).exists():
            size = Path(output_file).stat().st_size / 1024
            print(f"   步骤{i}: {output_file} ({size:.1f} KB)")
        else:
            print(f"   步骤{i}: {output_file} (文件不存在)")
    
    print(f"\n💡 您可以查看任何步骤的txt文件来了解详细结果")
    print(f"💡 最终汇总报告: step8Out.txt")

if __name__ == "__main__":
    main()
