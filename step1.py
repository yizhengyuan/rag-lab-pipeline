#!/usr/bin/env python3
"""
步骤1: 文件加载 + 向量化存储 - 增强版
==========================================

功能：
1. 加载各种格式的文档文件
2. 进行语义分块和概念提取
3. 向量化并存储到本地Chroma数据库
4. 生成详细的向量化报告

用法: python step1.py sourceDoc.txt

新功能：
- ✅ 自动创建基于时间戳的实验文件夹
- ✅ 统一的输出文件管理
- ✅ 支持多种输出格式（txt, json, md）
- ✅ 实验元数据自动记录
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 导入现有的处理器类
sys.path.append(str(Path(__file__).parent))
from pipeline0604 import FileProcessor
from llama_index.core import Document

# 导入核心模块
from core.chunking import SemanticChunker
from core.vector_store import VectorStoreManager
from config.settings import load_config_from_yaml

# 🆕 导入实验管理器
from utils.experiment_manager import create_experiment_manager

def convert_result_to_serializable(result):
    """将结果转换为可序列化的格式"""
    serializable_result = {}
    
    for key, value in result.items():
        if key == "document":
            # 特殊处理Document对象
            if hasattr(value, 'text') and hasattr(value, 'metadata'):
                serializable_result[key] = {
                    "text": value.text,
                    "metadata": dict(value.metadata)
                }
            elif isinstance(value, dict):
                serializable_result[key] = value
            else:
                serializable_result[key] = str(value)
        elif key == "chunk_nodes":
            # 处理文本节点列表
            serializable_result[key] = []
            for node in value:
                # 🔧 修复：正确解析概念JSON字符串
                concepts_data = node.metadata.get("concepts", "[]")
                if isinstance(concepts_data, str):
                    try:
                        concepts = json.loads(concepts_data)
                    except json.JSONDecodeError:
                        concepts = []
                else:
                    concepts = concepts_data if isinstance(concepts_data, list) else []
                
                node_data = {
                    "node_id": node.node_id,
                    "text": node.text,
                    "text_length": len(node.text),
                    "concepts": concepts,  # 使用解析后的概念列表
                    "metadata": dict(node.metadata)
                }
                serializable_result[key].append(node_data)
        elif key == "vector_info":
            # 向量信息已经是可序列化的
            serializable_result[key] = value
        else:
            # 其他键值直接复制
            serializable_result[key] = value
    
    return serializable_result

def save_result_as_txt(result, output_file):
    """保存结果为txt格式，包含向量化信息 - 兼容性函数"""
    
    # 🔧 自动序列化原始结果
    if 'chunk_nodes' in result and result['chunk_nodes']:
        first_chunk = result['chunk_nodes'][0]
        if hasattr(first_chunk, 'text'):  # 是TextNode对象
            result = convert_result_to_serializable(result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("步骤1: 文件加载 + 向量化存储 - 处理结果\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理状态: {'✅ 成功' if result.get('success') else '❌ 失败'}\n")
        
        if result.get("success"):
            # 文档基本信息
            doc_data = result.get("document", {})
            if hasattr(doc_data, 'text') and hasattr(doc_data, 'metadata'):
                # Document对象
                metadata = doc_data.metadata
                text_content = doc_data.text
            elif isinstance(doc_data, dict):
                metadata = doc_data.get("metadata", {})
                text_content = doc_data.get("text", "")
            else:
                metadata = {}
                text_content = str(doc_data) if doc_data else ""
            
            f.write(f"\n📊 文档基本信息:\n")
            f.write(f"- 文件名: {metadata.get('file_name', '未知')}\n")
            f.write(f"- 文件类型: {metadata.get('file_type', '未知')}\n")
            f.write(f"- 文件大小: {metadata.get('file_size', 0) / 1024:.2f} KB\n")
            f.write(f"- 文本长度: {len(text_content):,} 字符\n")
            f.write(f"- 处理时间: {result.get('processing_time', 0):.2f} 秒\n")
            
            # 分块信息
            chunks = result.get("chunk_nodes", [])
            f.write(f"\n📄 文档分块信息:\n")
            f.write(f"- 总分块数: {len(chunks)}\n")
            
            if chunks:
                # 🔧 修复：正确处理概念数据
                chunk_lengths = []
                concept_counts = []
                all_concepts = set()
                
                for chunk in chunks:
                    if hasattr(chunk, 'text'):
                        # TextNode对象 - 需要解析JSON字符串
                        chunk_lengths.append(len(chunk.text))
                        concepts_data = chunk.metadata.get("concepts", "[]")
                        if isinstance(concepts_data, str):
                            try:
                                concepts = json.loads(concepts_data)
                            except json.JSONDecodeError:
                                concepts = []
                        else:
                            concepts = concepts_data if isinstance(concepts_data, list) else []
                        concept_counts.append(len(concepts))
                        all_concepts.update(concepts)
                    else:
                        # 字典对象（向后兼容）
                        chunk_lengths.append(chunk.get('text_length', 0))
                        concepts = chunk.get('concepts', [])
                        if isinstance(concepts, list):
                            concept_counts.append(len(concepts))
                            all_concepts.update(concepts)
                        else:
                            concept_counts.append(0)
                
                if chunk_lengths:
                    f.write(f"- 平均分块长度: {sum(chunk_lengths) / len(chunk_lengths):.1f} 字符\n")
                    f.write(f"- 最短分块: {min(chunk_lengths)} 字符\n")
                    f.write(f"- 最长分块: {max(chunk_lengths)} 字符\n")
                
                f.write(f"- 总概念数: {sum(concept_counts)}\n")
                f.write(f"- 唯一概念数: {len(all_concepts)}\n")
                if concept_counts:
                    f.write(f"- 平均每分块概念数: {sum(concept_counts) / len(concept_counts):.1f}\n")
            
            # 向量化信息
            vector_info = result.get("vector_info", {})
            f.write(f"\n🗃️  向量化存储信息:\n")
            f.write(f"- 向量数据库类型: {vector_info.get('store_type', '未知')}\n")
            f.write(f"- 存储目录: {vector_info.get('persist_directory', '未知')}\n")
            f.write(f"- 集合名称: {vector_info.get('collection_name', '未知')}\n")
            f.write(f"- 向量维度: {vector_info.get('dimension', '未知')}\n")
            f.write(f"- 向量化节点数: {vector_info.get('vectorized_nodes', 0)}\n")
            f.write(f"- 存储大小: {vector_info.get('storage_size_mb', 0):.2f} MB\n")
            f.write(f"- 向量化时间: {vector_info.get('vectorization_time', 0):.2f} 秒\n")
            
            # 缓存信息
            cache_info = vector_info.get('cache_info', {})
            if cache_info:
                f.write(f"\n💾 缓存信息:\n")
                f.write(f"- 缓存条目数: {cache_info.get('total_entries', 0)}\n")
                f.write(f"- 缓存大小: {cache_info.get('estimated_size_mb', 0):.2f} MB\n")
                f.write(f"- 缓存命中率: {cache_info.get('hit_rate', 0):.1%}\n")
            
            # 显示分块详情（前10个）
            f.write(f"\n📝 分块详细信息 (前10个):\n")
            f.write("-" * 60 + "\n")
            
            for i, chunk in enumerate(chunks[:10], 1):
                if hasattr(chunk, 'text'):
                    # TextNode对象 - 需要解析JSON字符串
                    concepts_data = chunk.metadata.get('concepts', "[]")
                    if isinstance(concepts_data, str):
                        try:
                            concepts = json.loads(concepts_data)
                        except json.JSONDecodeError:
                            concepts = []
                    else:
                        concepts = concepts_data if isinstance(concepts_data, list) else []
                    text_preview = chunk.text[:100]
                    node_id = chunk.node_id
                    text_length = len(chunk.text)
                else:
                    # 字典对象（向后兼容）
                    concepts = chunk.get('concepts', [])
                    if not isinstance(concepts, list):
                        concepts = []
                    text_preview = chunk.get('text', '')[:100]
                    node_id = chunk.get('node_id', '未知')
                    text_length = chunk.get('text_length', 0)
                
                f.write(f"\n分块 {i}:\n")
                f.write(f"  ID: {node_id}\n")
                f.write(f"  长度: {text_length} 字符\n")
                f.write(f"  概念数: {len(concepts)}\n")
                f.write(f"  概念: {concepts}\n")
                f.write(f"  内容预览: {text_preview}...\n")
                f.write("-" * 40 + "\n")
            
            if len(chunks) > 10:
                f.write(f"\n... 还有 {len(chunks) - 10} 个分块\n")
            
            # 显示所有概念
            all_concepts = set()
            for chunk in chunks:
                if hasattr(chunk, 'metadata'):
                    # TextNode对象 - 需要解析JSON字符串
                    concepts_data = chunk.metadata.get('concepts', "[]")
                    if isinstance(concepts_data, str):
                        try:
                            concepts = json.loads(concepts_data)
                        except json.JSONDecodeError:
                            concepts = []
                    else:
                        concepts = concepts_data if isinstance(concepts_data, list) else []
                    all_concepts.update(concepts)
                else:
                    # 字典对象
                    concepts = chunk.get('concepts', [])
                    if isinstance(concepts, list):
                        all_concepts.update(concepts)
            
            f.write(f"\n🧠 提取的概念列表 ({len(all_concepts)} 个):\n")
            f.write("-" * 60 + "\n")
            for i, concept in enumerate(sorted(all_concepts), 1):
                f.write(f"{i:3d}. {concept}\n")
            
            # 文本内容预览
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
        
        # 在文件末尾添加机器可读的数据
        try:
            json_result = convert_result_to_serializable(result)
            
            f.write(f"\n" + "="*80 + "\n")
            f.write("# 机器可读数据 (请勿手动修改)\n")
            f.write("# JSON_DATA_START\n")
            f.write(json.dumps(json_result, ensure_ascii=False, indent=2))
            f.write("\n# JSON_DATA_END\n")
            
        except Exception as json_error:
            f.write(f"\n⚠️ JSON序列化失败: {json_error}\n")

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

def process_file_with_vectorization(input_file: str, config_path: str = "config.yml") -> Dict[str, Any]:
    """
    处理文件并进行向量化存储
    
    Args:
        input_file: 输入文件路径
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    start_time = time.time()
    
    try:
        # 1. 加载配置
        print("📋 加载配置...")
        config = load_config_from_yaml(config_path)
        
        # 2. 文件加载
        print(f"📄 加载文件: {input_file}")
        text_content = FileProcessor.extract_text(input_file)
        
        # 创建Document对象
        input_path = Path(input_file)
        document = Document(
            text=text_content,
            metadata={
                "file_name": input_path.name,
                "file_path": str(input_path),
                "file_type": input_path.suffix.lower(),
                "file_size": input_path.stat().st_size,
                "text_length": len(text_content),
                "processing_timestamp": datetime.now().isoformat(),
                "vectorized": True  # 标记为已向量化
            }
        )
        
        print(f"✅ 文件加载成功: {len(text_content):,} 字符")
        
        # 3. 初始化向量化组件
        print("🔧 初始化向量化组件...")
        chunker = SemanticChunker(config)
        vector_manager = VectorStoreManager(config)
        
        # 4. 语义分块和概念提取
        print("✂️ 执行语义分块和概念提取...")
        chunk_start_time = time.time()
        
        chunk_nodes = chunker.chunk_and_extract_concepts([document])
        
        chunk_time = time.time() - chunk_start_time
        print(f"✅ 分块完成: {len(chunk_nodes)} 个chunks，耗时 {chunk_time:.2f} 秒")
        
        # 5. 向量化存储
        print("🗃️  开始向量化存储...")
        vector_start_time = time.time()
        
        # 创建向量索引并存储到Chroma
        chunk_index = vector_manager.create_chunk_index(chunk_nodes, persist=True)
        
        vector_time = time.time() - vector_start_time
        print(f"✅ 向量化存储完成，耗时 {vector_time:.2f} 秒")
        
        # 6. 收集向量化信息
        index_info = vector_manager.get_index_info()
        storage_info = vector_manager.get_storage_size()
        
        # 获取缓存信息
        cache_info = {}
        if chunker.embedding_cache:
            cache_stats = chunker.embedding_cache.get_cache_stats()
            cache_info = cache_stats
        
        vector_info = {
            "store_type": index_info['store_type'],
            "persist_directory": index_info['persist_directory'],
            "collection_name": index_info['collection_name'],
            "dimension": index_info['dimension'],
            "vectorized_nodes": len(chunk_nodes),
            "storage_size_mb": storage_info.get('total', {}).get('size_mb', 0),
            "vectorization_time": vector_time,
            "cache_info": cache_info,
            "index_metadata": index_info['indexes'].get('chunks', {}).get('metadata', {}),
        }
        
        processing_time = time.time() - start_time
        
        # 7. 构建结果
        result = {
            "success": True,
            "step": 1,
            "step_name": "文件加载与向量化存储",
            "document": document,
            "chunk_nodes": chunk_nodes,
            "vector_info": vector_info,
            "statistics": {
                "total_chunks": len(chunk_nodes),
                "total_concepts": sum(len(node.metadata.get("concepts", [])) for node in chunk_nodes),
                "unique_concepts": len(set().union(*[node.metadata.get("concepts", []) for node in chunk_nodes])),
                "avg_chunk_length": sum(len(node.text) for node in chunk_nodes) / len(chunk_nodes) if chunk_nodes else 0,
                "chunk_time": chunk_time,
                "vector_time": vector_time,
            },
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        error_msg = f"文件处理失败: {str(e)}"
        print(f"❌ {error_msg}")
        
        result = {
            "success": False,
            "step": 1,
            "step_name": "文件加载与向量化存储",
            "error": error_msg,
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

def main():
    if len(sys.argv) != 2:
        print("用法: python step1.py <输入文件>")
        print("示例: python step1.py 'attention is all you need.pdf'")
        print("\n新功能:")
        print("✅ 自动创建实验文件夹（基于时间戳）")
        print("✅ 统一的文件输出管理")
        print("✅ 支持多种格式（txt, json, md）")
        print("✅ 实验元数据自动记录")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    print(f"🚀 步骤1: 文件加载 + 向量化存储 (新版)")
    print(f"📄 输入文件: {input_file}")
    print("="*60)
    
    try:
        # 检查输入文件是否存在
        if not Path(input_file).exists():
            print(f"❌ 输入文件不存在: {input_file}")
            sys.exit(1)
        
        # 🆕 创建实验管理器
        print("🧪 创建实验环境...")
        experiment_manager = create_experiment_manager(input_file)
        
        print(f"✅ 实验环境创建完成:")
        print(f"   实验ID: {experiment_manager.experiment_name}")
        print(f"   实验目录: {experiment_manager.experiment_dir}")
        print()
        
        # 处理文件并向量化
        print("🔄 开始处理文件...")
        result = process_file_with_vectorization(input_file)
        
        # 🆕 使用实验管理器保存结果
        print("💾 保存实验结果...")
        saved_files = experiment_manager.save_step_result(
            step_num=1,
            result=result,
            save_formats=['txt', 'json']  # 可以选择保存的格式
        )
        
        if result.get("success"):
            print(f"\n✅ 步骤1完成!")
            print(f"📁 实验目录: {experiment_manager.experiment_dir}")
            print(f"📄 输出文件:")
            for format_type, file_path in saved_files.items():
                print(f"   - {format_type.upper()}: {file_path}")
            
            # 显示简要统计
            stats = result.get("statistics", {})
            vector_info = result.get("vector_info", {})
            
            print(f"\n📊 处理结果摘要:")
            print(f"   - 文档分块数: {stats.get('total_chunks', 0)}")
            print(f"   - 提取概念数: {stats.get('total_concepts', 0)} (唯一: {stats.get('unique_concepts', 0)})")
            print(f"   - 向量化节点数: {vector_info.get('vectorized_nodes', 0)}")
            print(f"   - 存储大小: {vector_info.get('storage_size_mb', 0):.2f} MB")
            print(f"   - 总处理时间: {result.get('processing_time', 0):.2f} 秒")
            print(f"   - 向量数据库: {vector_info.get('store_type', '未知')} @ {vector_info.get('persist_directory', '未知')}")
            
            # 显示缓存信息
            cache_info = vector_info.get('cache_info', {})
            if cache_info:
                print(f"   - 缓存状态: {cache_info.get('total_entries', 0)} 条记录, {cache_info.get('estimated_size_mb', 0):.1f} MB")
            
            # 🆕 显示实验信息
            summary = experiment_manager.get_experiment_summary()
            print(f"\n🧪 实验信息:")
            print(f"   - 实验ID: {summary['experiment_id']}")
            print(f"   - 已完成步骤: {summary['steps_completed']}/{summary['total_steps']}")
            print(f"   - 实验状态: {summary['status']}")
            
            # 🆕 提示后续步骤
            print(f"\n📋 后续步骤:")
            print(f"   运行下一步: python step2.py {saved_files['txt']}")
            print(f"   查看结果: cat {saved_files['txt']}")
            print(f"   实验目录: ls {experiment_manager.experiment_dir}")
                
        else:
            print(f"❌ 步骤1失败: {result.get('error')}")
            
            # 即使失败也保存错误信息
            experiment_manager.save_step_result(
                step_num=1,
                result=result,
                save_formats=['txt']
            )
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        
        # 🆕 保存错误信息到实验目录
        if 'experiment_manager' in locals():
            error_result = {
                "step": 1,
                "step_name": "文件加载与向量化存储",
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                experiment_manager.save_step_result(1, error_result, ['txt'])
                print(f"📄 错误信息已保存到实验目录: {experiment_manager.experiment_dir}")
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()