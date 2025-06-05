# ConceptBasedPipeline - 基于概念的问题生成管道

## 项目简介

这是一个基于 LlamaIndex 框架实现的多阶段问题生成 Pipeline，参考了 Savaal 的 3-stage 思路，专注于实现前4个核心步骤：

1. **Semantic Chunking + Chunk-Level-Concepts 提取**
2. **Document-Level-Concept 合并和后处理**  
3. **基于 Doc-Concept 的相关 chunks 检索**
4. **Evidence 提取**

## 核心特性

🔍 **智能语义分块**: 使用 LlamaIndex 的 SemanticSplitterNodeParser 进行语义感知的文档分块

🧠 **概念提取**: 从每个 chunk 中提取核心概念，支持关键词和短句形式

🔗 **概念合并**: 使用相似性聚类将相关概念合并为文档级别概念

🎯 **精准检索**: 基于文档概念进行跨chunk的相关内容检索

💡 **证据提取**: 从检索结果中提取高质量的证据片段，去除噪声

## 安装依赖

```bash
pip install -r requirements.txt
```

## 📁 支持的文档格式

LlamaIndex 原生支持多种文档格式，无需额外配置：

- ✅ **PDF 文档** (.pdf) - 自动文本提取
- ✅ **文本文件** (.txt, .md) 
- ✅ **Word 文档** (.docx)
- ✅ **网页内容** (.html)

### PDF 文档使用示例

```python
from llama_index.core import SimpleDirectoryReader

# 从目录批量加载 PDF
reader = SimpleDirectoryReader(input_dir="./pdfs/", required_exts=[".pdf"])
documents = reader.load_data()

# 或指定特定 PDF 文件
reader = SimpleDirectoryReader(input_files=["paper.pdf", "report.pdf"])
documents = reader.load_data()

# 运行 pipeline
pipeline = ConceptBasedPipeline(openai_api_key="your-api-key")
results = pipeline.run_pipeline(documents)
```

运行 PDF 专用示例：
```bash
python pdf_example.py
```

## 快速开始

### 1. 配置 API 密钥

```python
# 设置环境变量
export OPENAI_API_KEY="your-api-key-here"
```

### 2. 运行示例

```bash
python example_usage.py
```

### 3. 基本使用

```python
from llama_index.core import Document
from pipeline import ConceptBasedPipeline

# 初始化 pipeline
pipeline = ConceptBasedPipeline(openai_api_key="your-api-key")

# 准备文档
documents = [Document(text="您的文档内容")]

# 运行完整 pipeline
results = pipeline.run_pipeline(documents)

# 保存结果
pipeline.save_results(results, "output.json")
```

## 技术架构

```
输入文档 → Semantic Chunking → Concept 提取 → 概念合并 → 概念检索 → Evidence 提取
```

## 📝 配置管理

### YAML 配置系统

项目使用 YAML 格式进行配置管理，提供更好的可读性和灵活性：

**主配置文件**: `config.yml`

```yaml
# OpenAI 设置
openai:
  api_key: "your-openai-api-key-here"
  model: "gpt-3.5-turbo"
  embedding_model: "text-embedding-ada-002"
  temperature: 0.1

# 概念合并设置
concept_merging:
  similarity_threshold: 0.7  # 概念相似性阈值
  max_document_concepts: 10  # 最大文档概念数量

# 检索设置
retrieval:
  top_k: 5  # 检索的 chunk 数量

# Evidence 提取设置
evidence_extraction:
  min_length: 20   # 最小证据长度
  max_length: 200  # 最大证据长度
```

### 配置加载和使用

```python
from config_loader import load_config

# 加载配置
config = load_config("config.yml")

# 使用配置
pipeline = ConceptBasedPipeline(
    openai_api_key=config.openai.api_key,
    model_name=config.openai.model
)
```

### 环境变量覆盖

支持使用环境变量覆盖配置：

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4"
export CONCEPT_SIMILARITY_THRESHOLD="0.8"
```

### 多环境配置

```bash
# 运行配置演示，生成不同环境的配置文件
python config_example.py

# 使用特定环境配置
python your_script.py --config config_dev.yml  # 开发环境
python your_script.py --config config_prod.yml # 生产环境
```

## 📂 项目结构

```
llamaindex-pipeline/
├── 🔧 配置文件
│   ├── config.yml              # 主配置文件 (YAML)
│   ├── config_loader.py        # YAML 配置加载器
│   ├── config_dev.yml          # 开发环境配置
│   ├── config_prod.yml         # 生产环境配置
│   └── config.py               # 原配置文件 (已废弃)
│
├── 🚀 核心 Pipeline
│   ├── pipeline.py             # 主 Pipeline 实现
│   └── pipeline_improved.py    # 改进版 Pipeline
│
├── 📖 使用示例
│   ├── example_usage.py        # 基础使用示例
│   ├── pdf_example.py          # PDF 处理示例
│   ├── demo_with_pdf.py        # PDF 演示脚本
│   └── config_example.py       # 配置系统演示
│
├── 🧪 测试文件
│   ├── test_pipeline.py        # 模拟测试
│   └── test_no_api.py          # 无 API 测试
│
├── 📄 文档
│   ├── README.md               # 项目说明
│   ├── environment_setup.md    # 环境配置
│   └── requirements.txt        # 依赖列表
│
└── 📊 示例数据
    └── attention is all you need.pdf  # 示例 PDF 文档
```

## 🆕 新功能特性

### ✅ YAML 配置系统
- 📝 可视化配置编辑
- 🔄 环境变量覆盖
- 🌍 多环境支持
- 🔒 配置验证

### ✅ PDF 文档支持
- 📄 原生 PDF 处理
- 📚 批量文档加载
- 🔍 自动文本提取

### ✅ 改进版 Pipeline
- 🧠 更好的 LlamaIndex 集成
- 🚀 优化的性能
- 📊 详细的结果输出

## 依赖版本

- Python >= 3.8
- llama-index >= 0.10.0
- openai >= 1.12.0
- sentence-transformers >= 2.2.0
- PyYAML >= 6.0

## 许可证

MIT License 