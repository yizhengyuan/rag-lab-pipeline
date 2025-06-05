"""
Chroma 向量数据库详细使用示例
展示如何在概念提取管道中使用 Chroma 进行本地向量存储

本文件演示了：
1. 基础配置和初始化
2. 文档处理和分块
3. 向量存储和检索
4. 缓存机制的使用
5. 数据管理和维护
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.chunking import SemanticChunker, EmbeddingCache
from core.vector_store import VectorStoreManager
from utils.config_loader import ConfigLoader
from llama_index.core import Document

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_basic_usage():
    """演示基本的 Chroma 使用流程"""
    
    print("=" * 60)
    print("🎯 Chroma 基本使用流程演示")
    print("=" * 60)
    
    # 1. 加载配置
    print("\n📋 步骤 1: 加载配置")
    config = ConfigLoader.load_config()
    
    # 检查配置
    vector_type = config.get('vector_store.type')
    persist_dir = config.get('vector_store.persist_directory')
    collection_name = config.get('vector_store.collection_name')
    
    print(f"   ✅ 向量数据库类型: {vector_type}")
    print(f"   📂 存储目录: {persist_dir}")
    print(f"   🏷️  集合名称: {collection_name}")
    
    # 2. 初始化组件
    print("\n🔧 步骤 2: 初始化组件")
    
    # 初始化语义分块器（自动启用embedding缓存）
    chunker = SemanticChunker(config)
    print("   ✅ SemanticChunker 初始化完成")
    
    # 初始化向量存储管理器（自动连接Chroma）
    vector_manager = VectorStoreManager(config)
    print("   ✅ VectorStoreManager 初始化完成")
    
    # 3. 准备测试文档
    print("\n📝 步骤 3: 准备测试文档")
    documents = [
        Document(
            text="Transformer架构是现代自然语言处理的基础，通过自注意力机制实现了并行计算。",
            metadata={"source": "transformer_paper", "type": "technical"}
        ),
        Document(
            text="BERT模型基于Transformer的编码器结构，通过掩码语言模型进行预训练。",
            metadata={"source": "bert_paper", "type": "technical"}
        ),
        Document(
            text="GPT系列模型采用Transformer的解码器结构，专注于文本生成任务。",
            metadata={"source": "gpt_paper", "type": "technical"}
        )
    ]
    print(f"   📄 准备了 {len(documents)} 个测试文档")
    
    # 4. 第一次处理 - 会调用API并缓存
    print("\n🔄 步骤 4: 第一次处理（会调用OpenAI API）")
    
    chunk_nodes = chunker.chunk_and_extract_concepts(documents)
    print(f"   ✅ 语义分块完成: {len(chunk_nodes)} 个chunks")
    
    # 显示概念提取结果
    for i, node in enumerate(chunk_nodes):
        concepts = chunker.get_concepts_from_node(node)
        text_preview = node.text[:50] + "..." if len(node.text) > 50 else node.text
        print(f"   📍 Chunk {i+1}: {len(concepts)} 个概念")
        print(f"      📝 文本: {text_preview}")
        print(f"      🏷️  概念: {concepts}")
    
    # 5. 创建向量索引并存储到Chroma
    print("\n🗃️  步骤 5: 创建向量索引并存储")
    
    chunk_index = vector_manager.create_chunk_index(chunk_nodes, persist=True)
    print("   ✅ 向量索引已创建并存储到Chroma")
    
    # 6. 验证存储
    print("\n🔍 步骤 6: 验证数据存储")
    
    # 检查Chroma数据库目录
    if os.path.exists(persist_dir):
        db_files = list(Path(persist_dir).glob("*"))
        print(f"   📁 Chroma数据库文件: {len(db_files)} 个")
        for file in db_files[:3]:  # 只显示前3个
            print(f"      📄 {file.name}")
    
    # 检查索引信息
    index_info = vector_manager.get_index_info()
    for index_type, info in index_info['indexes'].items():
        if info['exists']:
            node_count = info['metadata'].get('node_count', 0)
            print(f"   📊 {index_type} 索引: {node_count} 个节点")
    
    # 7. 测试向量检索
    print("\n🔍 步骤 7: 测试向量检索")
    
    query_engine = chunk_index.as_query_engine(
        similarity_top_k=3,
        verbose=True
    )
    
    test_query = "Transformer模型的核心机制是什么？"
    print(f"   ❓ 查询: {test_query}")
    
    try:
        response = query_engine.query(test_query)
        print(f"   ✅ 检索成功")
        print(f"   📝 回答: {response.response[:100]}...")
        
        # 显示检索到的源文档
        if hasattr(response, 'source_nodes'):
            print(f"   📚 检索到 {len(response.source_nodes)} 个相关文档片段")
    except Exception as e:
        print(f"   ⚠️  检索测试失败: {e}")
    
    return chunker, vector_manager, chunk_nodes

def demo_caching_mechanism():
    """演示缓存机制的工作原理"""
    
    print("\n" + "=" * 60)
    print("💾 缓存机制演示")
    print("=" * 60)
    
    config = ConfigLoader.load_config()
    chunker = SemanticChunker(config)
    
    # 准备相同的文档
    documents = [
        Document(text="这是一个测试文档，用于演示缓存机制的工作原理。"),
        Document(text="第二个文档包含不同的内容，但同样会被缓存。")
    ]
    
    print("\n🔄 第一次处理（会调用API）...")
    import time
    start_time = time.time()
    
    chunk_nodes_1 = chunker.chunk_and_extract_concepts(documents)
    first_time = time.time() - start_time
    
    print(f"   ⏱️  第一次处理耗时: {first_time:.2f} 秒")
    
    # 显示缓存统计
    if chunker.embedding_cache:
        stats = chunker.embedding_cache.get_cache_stats()
        print(f"   💾 缓存条目: {stats['total_entries']}")
        print(f"   💽 缓存大小: {stats['estimated_size_mb']:.2f} MB")
    
    print("\n🔄 第二次处理（使用缓存）...")
    
    # 重置分块器状态但保留缓存
    chunker.chunk_nodes = []
    chunker.chunk_index = None
    
    start_time = time.time()
    chunk_nodes_2 = chunker.chunk_and_extract_concepts(documents)
    second_time = time.time() - start_time
    
    print(f"   ⏱️  第二次处理耗时: {second_time:.2f} 秒")
    print(f"   🚀 速度提升: {first_time/second_time:.1f}x")
    
    # 验证结果一致性
    if len(chunk_nodes_1) == len(chunk_nodes_2):
        print("   ✅ 两次处理结果一致")
    else:
        print("   ⚠️  结果不一致，可能存在问题")

def demo_data_management():
    """演示数据管理功能"""
    
    print("\n" + "=" * 60)
    print("🛠️  数据管理功能演示")
    print("=" * 60)
    
    config = ConfigLoader.load_config()
    vector_manager = VectorStoreManager(config)
    
    # 1. 查看存储信息
    print("\n📊 当前存储状态:")
    
    try:
        storage_info = vector_manager.get_storage_size()
        total_size = storage_info.get('total', {}).get('size_mb', 0)
        print(f"   💽 总存储大小: {total_size:.2f} MB")
        
        for index_type, info in storage_info.items():
            if index_type != 'total' and isinstance(info, dict):
                size_mb = info.get('size_mb', 0)
                if size_mb > 0:
                    print(f"   📁 {index_type}: {size_mb:.2f} MB")
    except Exception as e:
        print(f"   ⚠️  获取存储信息失败: {e}")
    
    # 2. 索引信息
    print("\n📋 索引信息:")
    index_info = vector_manager.get_index_info()
    
    print(f"   🗃️  数据库类型: {index_info['store_type']}")
    print(f"   📂 存储目录: {index_info['persist_directory']}")
    
    for index_type, info in index_info['indexes'].items():
        exists = info['exists']
        persisted = info['persisted']
        node_count = info['metadata'].get('node_count', 0)
        
        status = "✅" if exists else "❌"
        persist_status = "💾" if persisted else "❌"
        
        print(f"   {status} {index_type}: 内存中={exists}, 持久化={persisted}, 节点数={node_count}")
    
    # 3. 备份演示
    print("\n💾 备份功能演示:")
    
    backup_dir = "./demo_backup"
    try:
        success = vector_manager.backup_indexes(backup_dir)
        if success:
            print(f"   ✅ 备份创建成功: {backup_dir}")
            
            # 显示备份内容
            backup_path = Path(backup_dir)
            if backup_path.exists():
                backup_folders = [d for d in backup_path.iterdir() if d.is_dir()]
                if backup_folders:
                    latest_backup = max(backup_folders, key=lambda x: x.stat().st_mtime)
                    print(f"   📁 最新备份: {latest_backup.name}")
        else:
            print("   ❌ 备份创建失败")
    except Exception as e:
        print(f"   ⚠️  备份操作失败: {e}")

def demo_advanced_usage():
    """演示高级使用场景"""
    
    print("\n" + "=" * 60)
    print("🚀 高级使用场景演示")
    print("=" * 60)
    
    config = ConfigLoader.load_config()
    vector_manager = VectorStoreManager(config)
    
    # 1. 多类型数据存储
    print("\n🗂️  多类型数据存储:")
    
    from core.nodes import ConceptNode
    from llama_index.core.schema import TextNode
    
    # 创建不同类型的节点
    concept_nodes = [
        ConceptNode(text="深度学习", concept_type="技术概念"),
        ConceptNode(text="神经网络", concept_type="技术概念"),
        ConceptNode(text="Transformer", concept_type="模型架构")
    ]
    
    evidence_nodes = [
        TextNode(
            text="Transformer模型在机器翻译任务上取得了显著的性能提升",
            metadata={"type": "evidence", "concept": "Transformer", "source": "research"}
        )
    ]
    
    try:
        # 分别存储不同类型的数据
        concept_index = vector_manager.create_concept_index(concept_nodes)
        evidence_index = vector_manager.create_evidence_index(evidence_nodes)
        
        print("   ✅ 概念索引创建成功")
        print("   ✅ 证据索引创建成功")
        
        # 显示各索引的节点数
        info = vector_manager.get_index_info()
        for index_type, index_info in info['indexes'].items():
            if index_info['exists']:
                count = index_info['metadata'].get('node_count', 0)
                print(f"   📊 {index_type} 索引: {count} 个节点")
    
    except Exception as e:
        print(f"   ⚠️  多类型存储失败: {e}")
    
    # 2. 缓存清理演示
    print("\n🧹 缓存清理:")
    
    chunker = SemanticChunker(config)
    if chunker.embedding_cache:
        initial_stats = chunker.embedding_cache.get_cache_stats()
        print(f"   📊 清理前: {initial_stats['total_entries']} 个缓存条目")
        
        # 清理过期缓存
        chunker.embedding_cache.clear_expired()
        
        final_stats = chunker.embedding_cache.get_cache_stats()
        print(f"   📊 清理后: {final_stats['total_entries']} 个缓存条目")
        
        cleaned = initial_stats['total_entries'] - final_stats['total_entries']
        if cleaned > 0:
            print(f"   🗑️  清理了 {cleaned} 个过期缓存")
        else:
            print("   ✅ 没有过期缓存需要清理")

def main():
    """主演示函数"""
    
    print("🎯 Chroma 向量数据库完整使用演示")
    print("=" * 70)
    
    try:
        # 基本使用流程
        chunker, vector_manager, chunk_nodes = demo_basic_usage()
        
        # 缓存机制演示
        demo_caching_mechanism()
        
        # 数据管理演示  
        demo_data_management()
        
        # 高级使用演示
        demo_advanced_usage()
        
        print("\n" + "=" * 70)
        print("✅ 所有演示完成！")
        print("\n💡 关键要点:")
        print("   🔥 首次处理时会调用API并缓存结果")
        print("   ⚡ 后续相同内容直接使用缓存，速度提升数十倍")
        print("   💾 所有向量数据自动持久化到本地Chroma数据库")
        print("   🗂️  支持多种数据类型的分离存储")
        print("   🛠️  提供完整的数据管理和备份功能")
        
        print(f"\n📂 生成的文件和目录:")
        print(f"   🗃️  向量数据库: ./vector_db/")
        print(f"   💾 Embedding缓存: ./embedding_cache/")
        print(f"   📦 备份文件: ./demo_backup/")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 