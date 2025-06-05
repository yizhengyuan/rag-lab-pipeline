"""
Chroma 实际使用指南
在你的项目中如何直接使用 Chroma 向量数据库

这个文件展示了最常用的使用场景和代码模式
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.chunking import SemanticChunker
from core.vector_store import VectorStoreManager
from utils.config_loader import ConfigLoader
from llama_index.core import Document

# ============================================================================
# 🚀 场景1：基本文档处理流程
# ============================================================================

def basic_document_processing():
    """最基本的文档处理流程"""
    
    print("🚀 基本文档处理流程")
    
    # 1. 加载配置（确保 vector_store.type = "chroma"）
    config = ConfigLoader.load_config()
    
    # 2. 初始化组件
    chunker = SemanticChunker(config)
    vector_manager = VectorStoreManager(config)
    
    # 3. 准备你的文档
    documents = [
        Document(text="你的文档内容1..."),
        Document(text="你的文档内容2..."),
        # 更多文档...
    ]
    
    # 4. 处理文档（第一次会调用API，后续使用缓存）
    chunk_nodes = chunker.chunk_and_extract_concepts(documents)
    
    # 5. 存储到Chroma向量数据库
    chunk_index = vector_manager.create_chunk_index(chunk_nodes, persist=True)
    
    print(f"✅ 处理完成: {len(chunk_nodes)} 个chunks已存储到Chroma")
    
    return chunk_index

# ============================================================================
# 🔍 场景2：向量检索和查询
# ============================================================================

def vector_search_example(chunk_index):
    """向量检索示例"""
    
    print("\n🔍 向量检索示例")
    
    # 创建查询引擎
    query_engine = chunk_index.as_query_engine(
        similarity_top_k=5,  # 返回最相似的5个结果
        verbose=True
    )
    
    # 执行查询
    queries = [
        "什么是深度学习？",
        "Transformer模型的优势",
        "人工智能的应用领域"
    ]
    
    for query in queries:
        print(f"\n❓ 查询: {query}")
        try:
            response = query_engine.query(query)
            print(f"📝 回答: {response.response[:100]}...")
            
            # 显示检索到的源文档
            if hasattr(response, 'source_nodes'):
                print(f"📚 找到 {len(response.source_nodes)} 个相关文档")
                
        except Exception as e:
            print(f"⚠️ 查询失败: {e}")

# ============================================================================
# 💾 场景3：缓存状态检查
# ============================================================================

def check_cache_status():
    """检查缓存状态"""
    
    print("\n💾 缓存状态检查")
    
    config = ConfigLoader.load_config()
    chunker = SemanticChunker(config)
    vector_manager = VectorStoreManager(config)
    
    # 检查embedding缓存
    if chunker.embedding_cache:
        cache_stats = chunker.embedding_cache.get_cache_stats()
        print(f"📊 Embedding缓存:")
        print(f"   条目数: {cache_stats['total_entries']}")
        print(f"   大小: {cache_stats['estimated_size_mb']:.2f} MB")
        print(f"   目录: {cache_stats['cache_directory']}")
    
    # 检查向量数据库状态
    index_info = vector_manager.get_index_info()
    print(f"\n🗃️ 向量数据库状态:")
    print(f"   类型: {index_info['store_type']}")
    print(f"   目录: {index_info['persist_directory']}")
    
    for index_type, info in index_info['indexes'].items():
        if info['exists'] or info['persisted']:
            node_count = info['metadata'].get('node_count', 0)
            print(f"   {index_type}: {node_count} 个节点")

# ============================================================================
# 🗂️ 场景4：多类型数据存储
# ============================================================================

def multi_type_storage():
    """多类型数据存储示例"""
    
    print("\n🗂️ 多类型数据存储")
    
    config = ConfigLoader.load_config()
    vector_manager = VectorStoreManager(config)
    
    from core.nodes import ConceptNode, EvidenceNode
    from llama_index.core.schema import TextNode
    
    # 1. 存储概念
    concept_nodes = [
        ConceptNode(text="机器学习", concept_type="技术"),
        ConceptNode(text="深度学习", concept_type="技术"),
    ]
    concept_index = vector_manager.create_concept_index(concept_nodes)
    print("✅ 概念数据已存储")
    
    # 2. 存储证据
    evidence_nodes = [
        TextNode(
            text="研究表明，深度学习在图像识别任务上表现卓越",
            metadata={"type": "evidence", "concept": "深度学习"}
        )
    ]
    evidence_index = vector_manager.create_evidence_index(evidence_nodes)
    print("✅ 证据数据已存储")
    
    return concept_index, evidence_index

# ============================================================================
# 📦 场景5：数据备份和恢复
# ============================================================================

def backup_and_restore():
    """数据备份和恢复"""
    
    print("\n📦 数据备份和恢复")
    
    config = ConfigLoader.load_config()
    vector_manager = VectorStoreManager(config)
    
    # 创建备份
    backup_dir = "./my_backup"
    success = vector_manager.backup_indexes(backup_dir)
    
    if success:
        print(f"✅ 数据已备份到: {backup_dir}")
        
        # 显示备份大小
        storage_info = vector_manager.get_storage_size()
        total_size = storage_info.get('total', {}).get('size_mb', 0)
        print(f"💽 备份大小: {total_size:.2f} MB")
    else:
        print("❌ 备份失败")

# ============================================================================
# 🧹 场景6：数据清理和维护
# ============================================================================

def data_maintenance():
    """数据清理和维护"""
    
    print("\n🧹 数据清理和维护")
    
    config = ConfigLoader.load_config()
    chunker = SemanticChunker(config)
    vector_manager = VectorStoreManager(config)
    
    # 1. 清理过期缓存
    if chunker.embedding_cache:
        initial_count = chunker.embedding_cache.get_cache_stats()['total_entries']
        chunker.embedding_cache.clear_expired()
        final_count = chunker.embedding_cache.get_cache_stats()['total_entries']
        cleaned = initial_count - final_count
        print(f"🗑️ 清理了 {cleaned} 个过期缓存")
    
    # 2. 优化索引（可选，用于大量数据时提升性能）
    try:
        success = vector_manager.optimize_indexes()
        if success:
            print("⚡ 索引优化完成")
        else:
            print("⚠️ 索引优化失败")
    except Exception as e:
        print(f"⚠️ 索引优化失败: {e}")

# ============================================================================
# 📋 常用的实用函数
# ============================================================================

def quick_setup():
    """快速设置 - 一键初始化所有组件"""
    
    config = ConfigLoader.load_config()
    chunker = SemanticChunker(config)
    vector_manager = VectorStoreManager(config)
    
    print("✅ Chroma组件初始化完成")
    return config, chunker, vector_manager

def process_documents_with_cache_check(documents):
    """处理文档并检查缓存使用情况"""
    
    config, chunker, vector_manager = quick_setup()
    
    # 检查缓存状态
    initial_cache = 0
    if chunker.embedding_cache:
        initial_cache = chunker.embedding_cache.get_cache_stats()['total_entries']
    
    # 处理文档
    chunk_nodes = chunker.chunk_and_extract_concepts(documents)
    
    # 检查缓存变化
    final_cache = 0
    if chunker.embedding_cache:
        final_cache = chunker.embedding_cache.get_cache_stats()['total_entries']
    
    new_cache_entries = final_cache - initial_cache
    
    print(f"📊 处理结果:")
    print(f"   文档数: {len(documents)}")
    print(f"   Chunks: {len(chunk_nodes)}")
    print(f"   新增缓存: {new_cache_entries}")
    
    # 存储到向量数据库
    chunk_index = vector_manager.create_chunk_index(chunk_nodes)
    
    return chunk_index, chunk_nodes

def get_storage_summary():
    """获取存储摘要信息"""
    
    config = ConfigLoader.load_config()
    vector_manager = VectorStoreManager(config)
    chunker = SemanticChunker(config)
    
    # 向量数据库信息
    index_info = vector_manager.get_index_info()
    storage_info = vector_manager.get_storage_size()
    
    # 缓存信息
    cache_stats = {}
    if chunker.embedding_cache:
        cache_stats = chunker.embedding_cache.get_cache_stats()
    
    summary = {
        "vector_db": {
            "type": index_info['store_type'],
            "directory": index_info['persist_directory'],
            "total_size_mb": storage_info.get('total', {}).get('size_mb', 0),
            "indexes": {}
        },
        "embedding_cache": cache_stats
    }
    
    # 索引详情
    for index_type, info in index_info['indexes'].items():
        if info['exists'] or info['persisted']:
            summary["vector_db"]["indexes"][index_type] = {
                "node_count": info['metadata'].get('node_count', 0),
                "last_updated": info['metadata'].get('last_updated', 'N/A')
            }
    
    return summary

# ============================================================================
# 💡 使用示例
# ============================================================================

if __name__ == "__main__":
    print("🎯 Chroma 实际使用演示")
    print("=" * 50)
    
    # 示例文档
    sample_documents = [
        Document(
            text="GPT是一种基于Transformer架构的大型语言模型，具有强大的文本生成能力。",
            metadata={"source": "ai_textbook", "chapter": "language_models"}
        ),
        Document(
            text="BERT通过双向编码器表示来理解语言上下文，在多种NLP任务中表现优异。",
            metadata={"source": "ai_textbook", "chapter": "language_models"}
        )
    ]
    
    try:
        # 1. 基本处理流程
        chunk_index = basic_document_processing()
        
        # 2. 向量检索测试
        vector_search_example(chunk_index)
        
        # 3. 检查状态
        check_cache_status()
        
        # 4. 多类型存储
        concept_index, evidence_index = multi_type_storage()
        
        # 5. 备份数据
        backup_and_restore()
        
        # 6. 数据维护
        data_maintenance()
        
        # 7. 存储摘要
        print("\n📋 存储摘要:")
        summary = get_storage_summary()
        print(f"   向量数据库: {summary['vector_db']['type']}")
        print(f"   总大小: {summary['vector_db']['total_size_mb']:.2f} MB")
        if summary['embedding_cache']:
            print(f"   缓存条目: {summary['embedding_cache']['total_entries']}")
        
        print("\n✅ 所有功能演示完成！")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n💡 下一步:")
    print("   1. 将你的实际文档替换 sample_documents")
    print("   2. 根据需要调整 similarity_top_k 参数")
    print("   3. 定期运行数据维护功能")
    print("   4. 查看生成的 ./vector_db/ 和 ./embedding_cache/ 目录") 