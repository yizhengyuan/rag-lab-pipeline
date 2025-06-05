"""
本地向量数据库测试脚本
演示如何使用 Chroma 本地存储 embedding，避免重复调用 OpenAI API

使用方法：
1. 安装新依赖：pip install chromadb llama-index-vector-stores-chroma
2. 确保 config.yml 中 vector_store.type 设为 "chroma"
3. 运行此脚本
"""

import os
import sys
import logging
from pathlib import Path

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

def test_local_vector_store():
    """测试本地向量存储功能"""
    
    print("🚀 开始测试本地向量数据库功能...")
    
    # 1. 加载配置
    try:
        config = ConfigLoader.load_config()
        print(f"✅ 配置加载成功")
        print(f"📂 向量数据库类型: {config.get('vector_store.type')}")
        print(f"📂 存储路径: {config.get('vector_store.persist_directory')}")
        print(f"🔧 缓存启用: {config.get('vector_store.enable_embedding_cache')}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False
    
    # 2. 检查向量数据库配置
    vector_store_type = config.get('vector_store.type', 'simple')
    if vector_store_type != 'chroma':
        print(f"⚠️  当前向量存储类型为: {vector_store_type}")
        print("💡 建议在 config.yml 中设置 vector_store.type: 'chroma' 以启用本地存储")
    
    # 3. 初始化组件
    try:
        print("\n🔧 初始化组件...")
        
        # 初始化语义分块器（包含embedding缓存）
        chunker = SemanticChunker(config)
        
        # 初始化向量存储管理器
        vector_manager = VectorStoreManager(config)
        
        print("✅ 组件初始化成功")
        
    except Exception as e:
        print(f"❌ 组件初始化失败: {e}")
        return False
    
    # 4. 创建测试文档
    test_documents = [
        Document(
            text="人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
        ),
        Document(
            text="深度学习是机器学习的一个子领域，它模拟人脑的神经网络结构，通过多层神经网络来学习数据的表示。"
        ),
        Document(
            text="自然语言处理是人工智能的一个重要分支，旨在让计算机能够理解、解释和生成人类语言。"
        )
    ]
    
    print(f"\n📝 创建了 {len(test_documents)} 个测试文档")
    
    # 5. 第一次处理（会调用API并缓存）
    print("\n🔄 第一次处理文档（会调用 OpenAI API）...")
    try:
        chunk_nodes = chunker.chunk_and_extract_concepts(test_documents)
        print(f"✅ 分块完成: {len(chunk_nodes)} 个 chunks")
        
        # 显示概念提取结果
        for i, node in enumerate(chunk_nodes[:2]):  # 只显示前2个
            concepts = chunker.get_concepts_from_node(node)  # 🔑 使用新方法获取concepts
            print(f"📍 Chunk {i+1}: {len(concepts)} 个概念 - {concepts}")
        
    except Exception as e:
        print(f"❌ 第一次处理失败: {e}")
        return False
    
    # 6. 显示缓存统计
    if chunker.embedding_cache:
        cache_stats = chunker.embedding_cache.get_cache_stats()
        print(f"\n💾 Embedding缓存统计:")
        print(f"   📊 缓存条目: {cache_stats['total_entries']}")
        print(f"   💽 估算大小: {cache_stats['estimated_size_mb']:.2f} MB")
        print(f"   📂 缓存目录: {cache_stats['cache_directory']}")
    
    # 7. 创建向量索引并持久化
    try:
        print(f"\n🗃️  创建向量索引...")
        chunk_index = vector_manager.create_chunk_index(chunk_nodes, persist=True)
        print(f"✅ 向量索引创建成功，已持久化到本地")
        
    except Exception as e:
        print(f"❌ 向量索引创建失败: {e}")
        return False
    
    # 8. 重置并测试第二次处理（应该使用缓存）
    print(f"\n🔄 重置后第二次处理（应该使用缓存）...")
    try:
        chunker.reset()
        
        # 第二次处理相同文档
        start_time = time.time() if 'time' in globals() else None
        chunk_nodes_2 = chunker.chunk_and_extract_concepts(test_documents)
        
        print(f"✅ 第二次处理完成: {len(chunk_nodes_2)} 个 chunks")
        
        if chunker.embedding_cache:
            cache_stats_2 = chunker.embedding_cache.get_cache_stats()
            print(f"💾 缓存命中情况: {cache_stats_2['total_entries']} 条缓存记录")
        
    except Exception as e:
        print(f"❌ 第二次处理失败: {e}")
    
    # 9. 显示向量数据库信息
    try:
        print(f"\n📋 向量数据库信息:")
        index_info = vector_manager.get_index_info()
        
        for index_type, info in index_info['indexes'].items():
            if info['exists'] or info['persisted']:
                print(f"   📁 {index_type}: 存在={info['exists']}, 持久化={info['persisted']}")
                if info['metadata'].get('node_count', 0) > 0:
                    print(f"      📊 节点数: {info['metadata']['node_count']}")
                    print(f"      🕐 更新时间: {info['metadata'].get('last_updated', 'N/A')}")
        
        # 显示存储大小
        storage_info = vector_manager.get_storage_size()
        print(f"   💽 总存储大小: {storage_info['total_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"⚠️  获取向量数据库信息失败: {e}")
    
    print(f"\n✅ 本地向量数据库测试完成！")
    print(f"\n💡 主要优势:")
    print(f"   🚀 首次处理后，embedding 已保存到本地")
    print(f"   💰 后续相同文档处理将使用缓存，节省 API 费用")
    print(f"   ⚡ 大幅提升处理速度")
    print(f"   🔒 数据完全存储在本地，保护隐私")
    
    return True

def check_dependencies():
    """检查必要的依赖是否已安装"""
    
    print("🔍 检查依赖...")
    
    missing_deps = []
    
    try:
        import chromadb
        print("✅ chromadb 已安装")
    except ImportError:
        missing_deps.append("chromadb")
    
    try:
        from llama_index.vector_stores.chroma import ChromaVectorStore
        print("✅ llama-index-vector-stores-chroma 已安装")
    except ImportError:
        missing_deps.append("llama-index-vector-stores-chroma")
    
    if missing_deps:
        print(f"\n❌ 缺少依赖: {', '.join(missing_deps)}")
        print(f"请运行: pip install {' '.join(missing_deps)}")
        return False
    
    print("✅ 所有依赖已安装")
    return True

if __name__ == "__main__":
    print("🎯 本地向量数据库测试工具")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 运行测试
    try:
        import time  # 用于计时
        success = test_local_vector_store()
        if success:
            print("\n🎉 测试成功完成！")
        else:
            print("\n❌ 测试失败")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 测试被用户中断")
    except Exception as e:
        print(f"\n💥 测试出现异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 