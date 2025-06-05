#!/usr/bin/env python3
"""
Pipeline命令行控制脚本 - 基于pipeline0604.py
========================================

用法:
1. 全流程执行: python pipeline_cli.py --input "attention is all you need.pdf" --full
2. 分步执行: python pipeline_cli.py --step 1 --input "source.pdf"
3. 继续执行: python pipeline_cli.py --step 3 --resume
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# 导入现有的处理器类
from pipeline0604 import StepByStepPipelineProcessor

class PipelineCLI:
    """Pipeline命令行接口 - 基于现有的StepByStepPipelineProcessor"""
    
    def __init__(self):
        self.steps = [
            {"name": "文件加载", "method": "step1_load_file"},
            {"name": "文档分块", "method": "step2_document_chunking"},
            {"name": "概念提取", "method": "step3_concept_extraction"},
            {"name": "概念合并", "method": "step4_concept_merging"},
            {"name": "概念检索", "method": "step5_concept_retrieval"},
            {"name": "证据提取", "method": "step6_evidence_extraction"},
            {"name": "问答生成", "method": "step7_qa_generation"},
            {"name": "最终汇总", "method": "step8_final_summary"}
        ]
        
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def run_full_pipeline(self, input_file: str, enable_qa: bool = True, output_dir: str = "step_by_step_results_cli") -> bool:
        """执行完整的pipeline流程"""
        print("🚀 开始执行完整Pipeline流程")
        print("=" * 60)
        
        try:
            # 初始化现有的处理器
            processor = StepByStepPipelineProcessor(
                input_file=input_file,
                output_dir=output_dir,
                enable_qa_generation=enable_qa
            )
            
            # 按顺序执行所有步骤
            steps_methods = [
                processor.step1_load_file,
                processor.step2_document_chunking,
                processor.step3_concept_extraction,
                processor.step4_concept_merging,
                processor.step5_concept_retrieval,
                processor.step6_evidence_extraction,
                processor.step7_qa_generation,
                processor.step8_final_summary
            ]
            
            for i, (step_info, step_method) in enumerate(zip(self.steps, steps_methods), 1):
                print(f"\n🔄 执行步骤{i}: {step_info['name']}")
                
                try:
                    result = step_method()
                    
                    if not result.get("success", False) and not result.get("skipped", False):
                        print(f"❌ 步骤{i}失败: {result.get('error', '未知错误')}")
                        return False
                    
                    # 保存步骤结果到标准输出文件
                    step_output_file = self.output_dir / f"step{i}_output.json"
                    with open(step_output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    status = "⏭️ 跳过" if result.get("skipped") else "✅ 完成"
                    print(f"{status} 步骤{i}")
                    
                except Exception as e:
                    print(f"❌ 步骤{i}执行异常: {e}")
                    return False
            
            print("\n🎉 完整Pipeline执行成功!")
            print(f"📁 详细结果目录: {processor.output_dir}")
            print(f"📁 JSON输出目录: {self.output_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Pipeline初始化失败: {e}")
            return False
    
    def run_single_step(self, step_num: int, input_file: str = None, resume: bool = False, output_dir: str = "step_by_step_results_cli") -> bool:
        """执行单个步骤"""
        if step_num < 1 or step_num > len(self.steps):
            print(f"❌ 无效的步骤号: {step_num}，有效范围: 1-{len(self.steps)}")
            return False
        
        step_info = self.steps[step_num - 1]
        
        # 确定输入文件
        if resume and not input_file:
            if step_num == 1:
                print("❌ 步骤1需要指定输入文件")
                return False
            # 对于resume模式，检查之前步骤是否完成
            prev_output = self.output_dir / f"step{step_num-1}_output.json"
            if not prev_output.exists():
                print(f"❌ 找不到上一步的输出文件: {prev_output}")
                return False
        
        if step_num == 1 and not input_file:
            print("❌ 步骤1需要指定输入文件")
            return False
        
        if input_file and not Path(input_file).exists():
            print(f"❌ 输入文件不存在: {input_file}")
            return False
        
        print(f"🔄 执行步骤{step_num}: {step_info['name']}")
        if input_file:
            print(f"📄 输入文件: {input_file}")
        
        try:
            # 初始化处理器
            actual_input = input_file if step_num == 1 else "dummy"  # 非步骤1的输入文件在step_results中
            processor = StepByStepPipelineProcessor(
                input_file=actual_input,
                output_dir=output_dir,
                enable_qa_generation=True  # 默认启用，步骤7会根据情况处理
            )
            
            # 如果是resume模式，需要加载之前的步骤结果
            if resume and step_num > 1:
                self._load_previous_results(processor, step_num)
            
            # 获取对应的方法并执行
            step_method = getattr(processor, step_info["method"])
            result = step_method()
            
            # 保存结果到标准输出文件
            step_output_file = self.output_dir / f"step{step_num}_output.json"
            step_output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(step_output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            if result.get("success", False) or result.get("skipped", False):
                status = "⏭️ 跳过" if result.get("skipped") else "✅ 完成"
                print(f"{status} 步骤{step_num}")
                print(f"📄 输出文件: {step_output_file}")
                print(f"📁 详细结果: {processor.output_dir}")
                return True
            else:
                print(f"❌ 步骤{step_num}失败: {result.get('error', '未知错误')}")
                return False
                
        except Exception as e:
            print(f"❌ 步骤{step_num}执行异常: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return False
    
    def _load_previous_results(self, processor, step_num):
        """加载之前步骤的结果到processor中"""
        for i in range(1, step_num):
            result_file = self.output_dir / f"step{i}_output.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    processor.step_results[f"step{i}"] = result
                    processor.step_timings[f"step{i}"] = result.get("processing_time", 0.0)
    
    def list_steps(self):
        """列出所有步骤"""
        print("📋 Pipeline步骤列表:")
        print("=" * 50)
        
        for i, step in enumerate(self.steps, 1):
            output_file = self.output_dir / f"step{i}_output.json"
            status = "✅" if output_file.exists() else "⏳"
            print(f"{status} 步骤{i}: {step['name']} ({step['method']})")
        
        print(f"\n📁 JSON输出目录: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline命令行执行工具 - 基于pipeline0604.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  1. 执行完整流程:
     python pipeline_cli.py --input "attention is all you need.pdf" --full
  
  2. 执行单个步骤:
     python pipeline_cli.py --step 1 --input "attention is all you need.pdf"
     python pipeline_cli.py --step 2 --resume
  
  3. 从指定步骤继续:
     python pipeline_cli.py --step 3 --resume
  
  4. 列出所有步骤:
     python pipeline_cli.py --list
        """
    )
    
    parser.add_argument("--input", "-i", help="输入文件路径")
    parser.add_argument("--step", "-s", type=int, help="执行指定步骤 (1-8)")
    parser.add_argument("--full", "-f", action="store_true", help="执行完整流程")
    parser.add_argument("--resume", "-r", action="store_true", help="从上一步的输出继续执行")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有步骤")
    parser.add_argument("--no-qa", action="store_true", help="禁用问答生成")
    parser.add_argument("--output-dir", "-o", default="step_by_step_results_cli", help="输出目录")
    
    args = parser.parse_args()
    
    cli = PipelineCLI()
    
    # 列出步骤
    if args.list:
        cli.list_steps()
        return
    
    # 执行完整流程
    if args.full:
        if not args.input:
            # 使用默认文件
            default_input = "attention is all you need.pdf"
            if Path(default_input).exists():
                args.input = default_input
                print(f"✅ 使用默认输入文件: {default_input}")
            else:
                print("❌ 执行完整流程需要指定输入文件，或确保默认文件存在")
                sys.exit(1)
        
        success = cli.run_full_pipeline(args.input, enable_qa=not args.no_qa, output_dir=args.output_dir)
        sys.exit(0 if success else 1)
    
    # 执行单个步骤
    if args.step:
        success = cli.run_single_step(args.step, args.input, args.resume, args.output_dir)
        sys.exit(0 if success else 1)
    
    # 如果没有指定动作，显示帮助
    parser.print_help()

if __name__ == "__main__":
    main()
