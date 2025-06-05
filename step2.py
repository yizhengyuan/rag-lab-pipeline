#!/usr/bin/env python3
"""
步骤2: 文档分块 - 基于pipeline0604.py (完全修复版)
================

用法: python step2.py step1Out.txt
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 导入现有的处理器类
sys.path.append(str(Path(__file__).parent))
from pipeline0604 import StepByStepPipelineProcessor
from llama_index.core import Document
from llama_index.core.schema import TextNode

def convert_result_to_serializable(result):
    """将结果转换为可序列化的格式 - 递归处理所有对象"""
    
    def convert_object(obj):
        """递归转换对象为可序列化格式"""
        if isinstance(obj, TextNode):
            # 转换TextNode对象
            return {
                "chunk_id": obj.node_id,
                "text": obj.text,
                "text_length": len(obj.text),
                "concepts": obj.metadata.get("concepts", []),
                "metadata": dict(obj.metadata)
            }
        elif isinstance(obj, Document):
            # 转换Document对象
            return {
                "text": obj.text,
                "metadata": dict(obj.metadata)
            }
        elif isinstance(obj, dict):
            # 递归处理字典
            return {k: convert_object(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # 递归处理列表
            return [convert_object(item) for item in obj]
        elif isinstance(obj, tuple):
            # 转换元组为列表
            return [convert_object(item) for item in obj]
        elif isinstance(obj, set):
            # 转换集合为列表
            return [convert_object(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # 处理其他有属性的对象
            return convert_object(obj.__dict__)
        else:
            # 基本类型直接返回
            return obj
    
    return convert_object(result)

def save_result_as_txt(result, output_file):
    """保存结果为txt格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("步骤2: 文档分块 - 处理结果\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理状态: {'✅ 成功' if result.get('success') else '❌ 失败'}\n")
        
        if result.get("success"):
            stats = result.get("statistics", {})
            
            f.write(f"\n📊 分块统计信息:\n")
            f.write(f"- 总分块数: {stats.get('total_chunks', 0)}\n")
            f.write(f"- 平均分块长度: {stats.get('avg_chunk_length', 0):.1f} 字符\n")
            f.write(f"- 最短分块: {stats.get('min_chunk_length', 0)} 字符\n")
            f.write(f"- 最长分块: {stats.get('max_chunk_length', 0)} 字符\n")
            f.write(f"- 总概念数: {stats.get('total_concepts', 0)}\n")
            f.write(f"- 唯一概念数: {stats.get('unique_concepts', 0)}\n")
            f.write(f"- 平均每分块概念数: {stats.get('avg_concepts_per_chunk', 0):.1f}\n")
            f.write(f"- 处理时间: {result.get('processing_time', 0):.2f} 秒\n")
            
            # 显示分块详情 - 安全处理chunks
            chunks = result.get("chunks", [])
            f.write(f"\n📄 分块详细信息:\n")
            f.write("-" * 60 + "\n")
            
            for i, chunk in enumerate(chunks[:10], 1):  # 只显示前10个
                # 安全获取chunk信息
                if isinstance(chunk, TextNode):
                    chunk_id = chunk.node_id
                    text_length = len(chunk.text)
                    concepts = chunk.metadata.get("concepts", [])
                    text_preview = chunk.text[:100]
                elif isinstance(chunk, dict):
                    chunk_id = chunk.get('chunk_id', '未知')
                    text_length = chunk.get('text_length', 0)
                    concepts = chunk.get('concepts', [])
                    text_preview = chunk.get('text', '')[:100]
                else:
                    chunk_id = f"chunk_{i}"
                    text_length = 0
                    concepts = []
                    text_preview = str(chunk)[:100]
                
                f.write(f"\n分块 {i}:\n")
                f.write(f"  ID: {chunk_id}\n")
                f.write(f"  长度: {text_length} 字符\n")
                f.write(f"  概念数: {len(concepts)}\n")
                f.write(f"  概念: {concepts}\n")
                f.write(f"  内容预览: {text_preview}...\n")
                f.write("-" * 40 + "\n")
            
            if len(chunks) > 10:
                f.write(f"\n... 还有 {len(chunks) - 10} 个分块 (详细信息见JSON数据)\n")
            
            # 显示所有概念
            unique_concepts = result.get("unique_concepts", [])
            f.write(f"\n🧠 提取的概念列表 ({len(unique_concepts)} 个):\n")
            f.write("-" * 60 + "\n")
            for i, concept in enumerate(unique_concepts, 1):
                f.write(f"{i:3d}. {concept}\n")
                
        else:
            f.write(f"\n❌ 错误信息: {result.get('error', '未知错误')}\n")
        
        # 🔧 修复：完全转换为可序列化格式
        try:
            json_result = convert_result_to_serializable(result)
            
            # 在文件末尾添加机器可读的数据
            f.write(f"\n" + "="*80 + "\n")
            f.write("# 机器可读数据 (请勿手动修改)\n")
            f.write("# JSON_DATA_START\n")
            f.write(json.dumps(json_result, ensure_ascii=False, indent=2))
            f.write("\n# JSON_DATA_END\n")
            
        except Exception as json_error:
            f.write(f"\n⚠️ JSON序列化失败: {json_error}\n")
            f.write("# 简化的机器可读数据\n")
            f.write("# JSON_DATA_START\n")
            simplified_result = {
                "step": result.get("step", 2),
                "success": result.get("success", False),
                "error": result.get("error", ""),
                "processing_time": result.get("processing_time", 0),
                "statistics": result.get("statistics", {}),
                "timestamp": datetime.now().isoformat()
            }
            f.write(json.dumps(simplified_result, ensure_ascii=False, indent=2))
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
        print("用法: python step2.py <step1输出文件>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = "step2Out.txt"
    
    print(f"🔄 步骤2: 文档分块")
    print(f"📄 输入: {input_file}")
    print(f"📄 输出: {output_file}")
    
    try:
        # 从txt文件读取step1的结果
        step1_result = load_result_from_txt(input_file)
        
        if not step1_result.get("success"):
            print("❌ 输入文件显示step1失败")
            sys.exit(1)
        
        # 重建Document对象
        doc_data = step1_result.get("document", {})
        if isinstance(doc_data, dict):
            document = Document(
                text=doc_data.get("text", ""),
                metadata=doc_data.get("metadata", {})
            )
        else:
            print("❌ 无法从step1结果中重建Document对象")
            sys.exit(1)
        
        print(f"✅ 成功重建Document对象，文本长度: {len(document.text):,} 字符")
        
        # 更新step1_result中的document为真正的Document对象
        step1_result["document"] = document
        
        # 使用现有的处理器
        processor = StepByStepPipelineProcessor(
            input_file="dummy",
            output_dir="temp_step2"
        )
        
        # 手动设置step1结果
        processor.step_results["step1"] = step1_result
        
        print("🔧 开始执行文档分块...")
        
        # 调用现有的方法
        result = processor.step2_document_chunking()
        
        # 保存结果为txt
        save_result_as_txt(result, output_file)
        
        if result.get("success"):
            print(f"✅ 步骤2完成: {output_file}")
            
            # 显示简要统计
            stats = result.get("statistics", {})
            print(f"📊 分块结果:")
            print(f"   - 总分块数: {stats.get('total_chunks', 0)}")
            print(f"   - 平均分块长度: {stats.get('avg_chunk_length', 0):.1f} 字符")
            print(f"   - 提取概念数: {stats.get('total_concepts', 0)} (唯一: {stats.get('unique_concepts', 0)})")
            print(f"   - 处理时间: {result.get('processing_time', 0):.2f} 秒")
            
            # 显示概念示例
            unique_concepts = result.get("unique_concepts", [])
            if unique_concepts:
                print(f"📝 概念示例 (前10个):")
                for i, concept in enumerate(unique_concepts[:10], 1):
                    print(f"   {i}. {concept}")
                    
        else:
            print(f"❌ 步骤2失败: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        
        # 保存错误信息
        error_result = {
            "step": 2,
            "step_name": "文档分块",
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
