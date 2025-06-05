#!/usr/bin/env python3
"""
步骤5: 概念检索 - 基于pipeline0604.py
================

用法: python step5.py step4Out.txt
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
        f.write("步骤5: 概念检索 - 处理结果\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理状态: {'✅ 成功' if result.get('success') else '❌ 失败'}\n")
        
        if result.get("success"):
            stats = result.get("statistics", {})
            
            f.write(f"\n📊 概念检索统计:\n")
            f.write(f"- 概念数量: {stats.get('concept_count', 0)}\n")
            f.write(f"- 分块数量: {stats.get('chunk_count', 0)}\n")
            f.write(f"- 总检索结果: {stats.get('total_retrievals', 0)}\n")
            f.write(f"- 平均每概念检索数: {stats.get('avg_retrievals_per_concept', 0):.2f}\n")
            f.write(f"- 有检索结果的概念: {stats.get('concepts_with_retrievals', 0)}/{stats.get('concept_count', 0)}\n")
            f.write(f"- 平均相似度: {stats.get('avg_similarity_all', 0):.4f}\n")
            f.write(f"- 处理时间: {result.get('processing_time', 0):.2f} 秒\n")
            
            # 显示检索结果详情
            retrieval_results = result.get("retrieval_results", {})
            f.write(f"\n🔍 概念检索详细结果:\n")
            f.write("-" * 80 + "\n")
            
            for i, (concept_id, retrieval_data) in enumerate(retrieval_results.items(), 1):
                concept_text = retrieval_data.get("concept_text", "未知")
                retrieval_count = retrieval_data.get("retrieval_count", 0)
                avg_similarity = retrieval_data.get("avg_similarity", 0)
                
                f.write(f"\n概念 {i}: {concept_text}\n")
                f.write(f"  概念ID: {concept_id}\n")
                f.write(f"  检索到相关分块数: {retrieval_count}\n")
                f.write(f"  平均相似度: {avg_similarity:.4f}\n")
                
                retrieved_chunks = retrieval_data.get("retrieved_chunks", [])
                for j, chunk in enumerate(retrieved_chunks[:3], 1):  # 只显示前3个
                    f.write(f"    相关分块 {j}:\n")
                    f.write(f"      ID: {chunk.get('chunk_id', '未知')}\n")
                    f.write(f"      相似度: {chunk.get('similarity_score', 0):.4f}\n")
                    f.write(f"      内容: {chunk.get('chunk_text', '')[:80]}...\n")
                
                if len(retrieved_chunks) > 3:
                    f.write(f"    ... 还有 {len(retrieved_chunks) - 3} 个相关分块\n")
                
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
        print("用法: python step5.py <step4输出文件>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = "step5Out.txt"
    
    print(f"🔄 步骤5: 概念检索")
    print(f"📄 输入: {input_file}")
    print(f"📄 输出: {output_file}")
    
    try:
        # 读取step4的结果
        step4_result = load_result_from_txt(input_file)
        
        if not step4_result.get("success"):
            print("❌ 输入文件显示step4失败")
            sys.exit(1)
        
        # 读取之前步骤的结果
        step_files = [
            (1, input_file.replace("step4Out.txt", "step1Out.txt")),
            (2, input_file.replace("step4Out.txt", "step2Out.txt")),
            (3, input_file.replace("step4Out.txt", "step3Out.txt"))
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
            output_dir="temp_step5"
        )
        
        # 手动设置之前步骤的结果
        for step_key, result in previous_results.items():
            processor.step_results[step_key] = result
        processor.step_results["step4"] = step4_result
        
        # 调用现有的方法
        result = processor.step5_concept_retrieval()
        
        # 保存结果为txt
        save_result_as_txt(result, output_file)
        
        if result.get("success"):
            print(f"✅ 步骤5完成: {output_file}")
        else:
            print(f"❌ 步骤5失败: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
