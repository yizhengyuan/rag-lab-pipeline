# 本地向量数据库使用指南

## 🎯 概述

本功能允许你将 embedding 向量存储在本地数据库中，避免重复调用 OpenAI API，从而：

- 💰 **节省成本**：已处理的文档不会重复计算 embedding
- ⚡ **提升速度**：从本地加载比 API 调用快数十倍
- 🔒 **保护隐私**：所有数据完全存储在本地
- 📦 **离线使用**：处理过的数据无需网络连接

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install chromadb llama-index-vector-stores-chroma
```

### 2. 更新配置

在 `config.yml` 中启用本地向量存储：

```yaml
# 向量数据库设置
vector_store:
  # 🔧 更改为 chroma 启用本地存储
  type: "chroma"
  # 本地存储路径
  persist_directory: "./vector_db"
  # 集合名称
  collection_name: "concept_pipeline"
  # 向量维度 (OpenAI text-embedding-ada-002 是 1536)
  dimension: 1536
  # 🔧 启用embedding缓存
  enable_embedding_cache: true
  # embedding缓存目录
  embedding_cache_dir: "./embedding_cache"
  # 缓存过期时间（天）
  cache_expiry_days: 30
```

### 3. 测试配置

运行测试脚本验证配置：

```bash
python test_local_vector_store.py
```

## 📁 目录结构

启用本地存储后，会创建以下目录：

```
./vector_db/                 # Chroma 向量数据库
├── chroma.sqlite3          # SQLite 数据库文件
└── ...                     # 其他 Chroma 文件

./embedding_cache/          # Embedding 缓存
├── embedding_cache.pkl     # 缓存数据
└── cache_metadata.json     # 缓存元数据
```

## 🔧 使用方法

### 基本使用

```python
from core.chunking import SemanticChunker
from core.vector_store import VectorStoreManager
from utils.config_loader import ConfigLoader
from llama_index.core import Document

# 加载配置
config = ConfigLoader.load_config()

# 初始化组件
chunker = SemanticChunker(config)
vector_manager = VectorStoreManager(config)

# 处理文档（第一次会调用 API）
documents = [Document(text="你的文档内容...")]
chunk_nodes = chunker.chunk_and_extract_concepts(documents)

# 创建并持久化向量索引
chunk_index = vector_manager.create_chunk_index(chunk_nodes, persist=True)

# 第二次处理相同文档将使用缓存！
```

### 检查缓存状态

```python
# 获取缓存统计
if chunker.embedding_cache:
    stats = chunker.embedding_cache.get_cache_stats()
    print(f"缓存条目: {stats['total_entries']}")
    print(f"缓存大小: {stats['estimated_size_mb']:.1f} MB")
```

### 管理向量数据库

```python
# 获取数据库信息
info = vector_manager.get_index_info()
print(f"向量数据库类型: {info['store_type']}")

# 获取存储大小
storage = vector_manager.get_storage_size()
print(f"存储大小: {storage['total_size_mb']:.1f} MB")

# 备份数据库
vector_manager.backup_indexes("./backup")

# 清理过期缓存
if chunker.embedding_cache:
    chunker.embedding_cache.clear_expired()
```

## 🔍 缓存机制详解

### 双重缓存策略

1. **Embedding 缓存**：在文件系统中缓存文本的 embedding 向量
2. **向量数据库**：使用 Chroma 存储向量索引，支持高效检索

### 缓存命中判断

- 对文本内容计算 SHA256 哈希值
- 检查哈希值是否在缓存中
- 验证缓存是否过期（可配置过期天数）

### 自动过期清理

- 默认 30 天过期
- 自动清理过期缓存
- 支持手动清理

## 📊 性能对比

| 处理方式 | 第一次处理 | 第二次处理 | 节省成本 |
|---------|-----------|-----------|----------|
| 无缓存   | 100% API调用 | 100% API调用 | 0% |
| 启用缓存 | 100% API调用 | ~5% API调用 | ~95% |

## 🛠️ 故障排除

### 常见问题

1. **依赖缺失**
   ```bash
   pip install chromadb llama-index-vector-stores-chroma
   ```

2. **权限问题**
   ```bash
   # 确保有写入权限
   chmod 755 ./vector_db
   chmod 755 ./embedding_cache
   ```

3. **数据库损坏**
   ```python
   # 删除并重建数据库
   vector_manager.delete_persisted_indexes()
   ```

### 日志查看

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

关键日志信息：
- ✅ `Chroma客户端初始化成功` - Chroma 工作正常
- 💾 `embedding缓存已保存` - 缓存功能正常
- 🎯 `检测到相同文档已处理` - 成功避免重复处理

## 🔧 高级配置

### 自定义向量数据库

```yaml
vector_store:
  type: "chroma"
  persist_directory: "./custom_vector_db"
  collection_name: "my_project"
  # 为不同项目使用不同的集合名称
```

### 调整缓存策略

```yaml
vector_store:
  enable_embedding_cache: true
  cache_expiry_days: 60  # 延长到60天
  embedding_cache_dir: "./cache"
```

### 多项目隔离

为不同项目使用不同的配置：

```yaml
# project_a.yml
vector_store:
  collection_name: "project_a"
  persist_directory: "./vector_db_a"

# project_b.yml  
vector_store:
  collection_name: "project_b"
  persist_directory: "./vector_db_b"
```

## 📈 最佳实践

1. **定期备份**：重要数据定期备份向量数据库
2. **监控大小**：定期检查缓存和数据库大小
3. **清理过期**：定期清理过期缓存释放空间
4. **项目隔离**：不同项目使用不同的集合名称
5. **版本控制**：向量数据库目录加入 `.gitignore`

```gitignore
# 向量数据库和缓存
/vector_db/
/embedding_cache/
```

## 🎉 开始使用

1. 运行测试脚本：`python test_local_vector_store.py`
2. 验证配置正确
3. 开始享受本地向量存储的便利！

有问题？查看测试脚本的输出或启用详细日志进行调试。 