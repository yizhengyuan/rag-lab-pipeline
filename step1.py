#!/usr/bin/env python3
"""
步骤1: 文件加载 - 基于pipeline0604.py (修复版)
================

用法: python step1.py sourceDoc.txt
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
        f.write("步骤1: 文件加载 - 处理结果\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理状态: {'✅ 成功' if result.get('success') else '❌ 失败'}\n")
        
        if result.get("success"):
            # 🔧 修复：正确处理Document对象
            doc_data = result.get("document", {})
            
            # 安全获取文档信息
            if hasattr(doc_data, 'metadata'):
                # 如果是Document对象
                metadata = doc_data.metadata
                text_content = doc_data.text
            elif isinstance(doc_data, dict):
                # 如果已经是字典
                metadata = doc_data.get("metadata", {})
                text_content = doc_data.get("text", "")
            else:
                # 兜底情况
                metadata = {}
                text_content = str(doc_data) if doc_data else ""
            
            f.write(f"\n📊 文件信息:\n")
            f.write(f"- 文件名: {metadata.get('file_name', '未知')}\n")
            f.write(f"- 文件类型: {metadata.get('file_type', '未知')}\n")
            f.write(f"- 文件大小: {metadata.get('file_size', 0) / 1024:.2f} KB\n")
            f.write(f"- 文本长度: {len(text_content):,} 字符\n")
            f.write(f"- 处理时间: {result.get('processing_time', 0):.2f} 秒\n")
            
            f.write(f"\n📄 文本内容预览 (前500字符):\n")
            f.write("-" * 50 + "\n")
            f.write(text_content[:500])
            f.write("\n" + "-" * 50 + "\n")
            
            if len(text_content) > 500:
                f.write(f"\n📄 文本内容预览 (后500字符):\n")
                f.write("-" * 50 + "\n")
                f.write(text_content[-500:])
                f.write("\n" + "-" * 50 + "\n")
        else:
            f.write(f"\n❌ 错误信息: {result.get('error', '未知错误')}\n")
        
        # 🔧 修复：确保JSON数据可序列化
        json_result = convert_result_to_serializable(result)
        
        # 在文件末尾添加机器可读的数据
        f.write(f"\n" + "="*80 + "\n")
        f.write("# 机器可读数据 (请勿手动修改)\n")
        f.write("# JSON_DATA_START\n")
        f.write(json.dumps(json_result, ensure_ascii=False, indent=2))
        f.write("\n# JSON_DATA_END\n")

def convert_result_to_serializable(result):
    """将结果转换为可序列化的格式"""
    serializable_result = {}
    
    for key, value in result.items():
        if key == "document":
            # 特殊处理Document对象
            if hasattr(value, 'text') and hasattr(value, 'metadata'):
                # 是Document对象
                serializable_result[key] = {
                    "text": value.text,
                    "metadata": dict(value.metadata)
                }
            elif isinstance(value, dict):
                # 已经是字典
                serializable_result[key] = value
            else:
                # 其他情况，转换为字符串
                serializable_result[key] = str(value)
        else:
            # 其他键值直接复制
            serializable_result[key] = value
    
    return serializable_result

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
        print("用法: python step1.py <输入文件>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = "step1Out.txt"
    
    print(f"🔄 步骤1: 文件加载")
    print(f"📄 输入: {input_file}")
    print(f"📄 输出: {output_file}")
    
    try:
        # 使用现有的处理器
        processor = StepByStepPipelineProcessor(
            input_file=input_file,
            output_dir="temp_step1"
        )
        
        # 调用现有的方法
        result = processor.step1_load_file()
        
        # 保存结果为txt
        save_result_as_txt(result, output_file)
        
        if result.get("success"):
            print(f"✅ 步骤1完成: {output_file}")
            
            # 显示简要统计
            doc_data = result.get("document", {})
            if hasattr(doc_data, 'text'):
                text_length = len(doc_data.text)
            elif isinstance(doc_data, dict):
                text_length = len(doc_data.get("text", ""))
            else:
                text_length = 0
            
            print(f"📊 处理结果:")
            print(f"   - 文本长度: {text_length:,} 字符")
            print(f"   - 处理时间: {result.get('processing_time', 0):.2f} 秒")
        else:
            print(f"❌ 步骤1失败: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
