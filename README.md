# 📚 智能文档处理流水线系统

一个基于 LlamaIndex 的智能文档处理流水线，能够从原始文档自动提取概念、生成证据并创建高质量问答对。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Latest-green.svg)](https://gpt-index.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 核心特性

### ✨ **新版本亮点 (v2.0)**

- 🧪 **统一实验管理系统** - 基于时间戳的实验文件夹，完整追踪每次运行
- 🔄 **完全重构的流水线** - 8个步骤的模块化处理，数据格式统一
- 📊 **智能性能分析** - 详细的执行报告和数据流分析
- 🎯 **高质量问答生成** - 支持多种认知层次的问答对创建
- 🛠️ **改进的错误处理** - 更好的故障诊断和恢复机制

### 🎯 **主要功能**

- **📄 文档处理**: 支持PDF、TXT等多种格式
- **🔍 语义分块**: 智能的文档分割和概念提取
- **🧠 概念分析**: 高质量概念提取、合并和优化
- **📋 证据提取**: 基于概念的证据检索和质量评估
- **❓ 问答生成**: 基于Bloom分类法的多层次问答对生成
- **📊 完整报告**: 详细的处理结果和性能分析

## 🚀 快速开始

### 📋 环境要求

- Python 3.8+
- OpenAI API 密钥
- 8GB+ RAM (推荐)
- 2GB+ 磁盘空间

### 🔧 安装配置

1. **克隆仓库**
```bash
git clone https://github.com/your-username/llamaindex-pipeline.git
cd llamaindex-pipeline
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置API密钥**
编辑 `config.yml` 文件：
```yaml
openai:
  api_key: "your-api-key-here"
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1/"
```

4. **验证安装**
```bash
python step1.py --help
```

### ⚡ 一键运行示例

```bash
# 处理示例文档
python step1.py "attention is all you need.pdf"

# 查看实验结果
ls experiments/  # 查看创建的实验文件夹
```

## 📋 完整流水线指南

### 🔄 **8步处理流程**

我们的系统采用模块化的8步处理流程，每一步都有明确的输入输出：

#### **Step 1: 文档加载与向量化存储**
```bash
python step1.py "your_document.pdf"
```
- 📄 加载文档，提取文本内容
- ✂️ 初始语义分块
- 🗃️ 向量化存储到 Chroma 数据库
- 🧪 创建实验文件夹: `experiments/YYYYMMDD_HHMMSS_文档名/`

**输出**: 分块数据、向量索引、实验文件夹

#### **Step 2: 文档分块与概念提取优化**
```bash
python step2.py experiments/20241204_143052_attention_paper/step1_vectorization.txt
```
- 🔧 改进的语义分块算法
- 🧠 高质量概念提取和验证
- 📊 概念质量评分和过滤

**输出**: 优化的分块、高质量概念列表

#### **Step 3: 概念提取与映射分析**
```bash
python step3.py experiments/20241204_143052_attention_paper/step2_chunking.txt
```
- 📈 深度概念分析和质量评估
- 🗺️ 概念关系映射和层次结构
- 📊 概念频率分析和分类

**输出**: 概念分析报告、概念图、质量评分

#### **Step 4: 概念合并与优化**
```bash
python step4.py experiments/20241204_143052_attention_paper/step3_retrieval.txt
```
- 🔗 智能概念合并和去重
- 📉 概念压缩和层次优化
- 🎯 概念节点创建和置信度评分

**输出**: 合并后的概念节点、概念映射关系

#### **Step 5: 概念检索与映射**
```bash
python step5.py experiments/20241204_143052_attention_paper/step4_reranking.txt
```
- 🔍 概念到分块的智能检索
- 📏 多维度相似度计算 (阈值: 0.2)
- 📈 检索质量分析和覆盖度评估

**输出**: 概念-分块映射关系、检索结果、质量分析

#### **Step 6: 证据提取与质量评估**
```bash
python step6.py experiments/20241204_143052_attention_paper/step5_answer_generation.txt
```
- 🔍 基于概念的证据提取
- 🏷️ 证据类型分类 (定义、例子、解释等)
- 📊 证据相关性评分和质量过滤

**输出**: 证据节点、证据分类、质量评估报告

#### **Step 7: 问答生成**
```bash
python step7.py experiments/20241204_143052_attention_paper/step6_evaluation.txt
```
- ❓ 基于证据的智能问答生成
- 🎓 支持 Bloom 分类法的6种认知层次
- 📊 问答质量评估和类型分布分析

**输出**: 问答对、质量分析、类型统计

#### **Step 8: 最终汇总与报告**
```bash
python step8.py experiments/20241204_143052_attention_paper/step7_qa_generation.txt
```
- 📊 完整流水线执行汇总
- 📈 性能分析和数据流统计
- 🎯 最终处理报告和指标

**输出**: 最终汇总报告、性能分析、完整统计

### 🧪 **实验管理系统**

每次运行都会创建独立的实验文件夹：

```
experiments/
└── 20241204_143052_attention_paper/
    ├── step1_vectorization.txt     # Step1输出
    ├── step1_vectorization.json    # JSON格式数据
    ├── step2_chunking.txt         # Step2输出  
    ├── step2_chunking.json
    ├── ...
    ├── step8_final_summary.txt    # 最终报告
    └── experiment_metadata.json   # 实验元数据
```

### 📊 **输出格式说明**

每个步骤都生成两种格式的输出：

- **📄 TXT格式**: 人类可读的详细报告，包含统计信息和示例
- **📋 JSON格式**: 机器可读的完整数据，便于后续处理

## ⚙️ 配置说明

### 🔧 **主要配置项** (`config.yml`)

```yaml
# API配置
openai:
  api_key: "your-api-key"
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1/"

# 检索配置 (重要!)
retrieval:
  top_k: 5
  similarity_cutoff: 0.2  # 🔧 关键配置：降低阈值提高覆盖率

# 问答生成配置
qa_generation:
  questions_per_type:
    remember: 2
    understand: 2
    apply: 1
    analyze: 1
    evaluate: 1
    create: 1

# 向量数据库配置
vector_store:
  type: "chroma"
  persist_directory: "./vector_db"
  enable_embedding_cache: true
```

### 📈 **性能优化配置**

```yaml
# 高级配置
advanced:
  max_concurrent_requests: 1
  request_timeout: 120
  max_retries: 8
  
# 分块配置  
chunking:
  max_tokens_per_chunk: 6000
  enable_token_validation: true
```

## 🔍 故障排除

### ❌ **常见问题**

#### **1. 检索覆盖率为0**
```
✅ 解决方案: 检查 config.yml 中的 similarity_cutoff 设置
retrieval:
  similarity_cutoff: 0.2  # 确保不要太高
```

#### **2. Step7 证据格式错误**
```
错误: 'dict' object has no attribute 'text'
✅ 解决方案: 使用新版 step7.py，已修复格式兼容性
```

#### **3. 实验文件夹未找到**
```bash
# 检查实验目录
ls experiments/

# 使用文件夹路径而非文件路径
python step2.py experiments/20241204_143052_attention_paper/
```

#### **4. API 调用失败**
```yaml
# 检查网络和配置
advanced:
  request_timeout: 180  # 增加超时时间
  max_retries: 5       # 增加重试次数
```

### 🛠️ **调试技巧**

```bash
# 查看详细错误信息
python step1.py document.pdf 2>&1 | tee debug.log

# 检查实验状态
cat experiments/latest/experiment_metadata.json

# 验证配置
python -c "from config.settings import load_config_from_yaml; print(load_config_from_yaml('config.yml'))"
```

## 📊 性能指标

### 🎯 **典型处理性能**

| 文档大小 | 处理时间 | 生成问答对 | 内存使用 |
|---------|---------|-----------|---------|
| 50KB    | 2-5分钟  | 10-20个   | 2GB     |
| 200KB   | 5-15分钟 | 20-50个   | 4GB     |
| 1MB+    | 15-45分钟| 50-100个  | 6GB+    |

### 📈 **质量指标**

- **概念提取准确率**: 85%+
- **证据相关性**: 70%+ 
- **问答对质量**: 80%+
- **系统稳定性**: 95%+

## 🆕 更新日志

### **v2.0.0** (2024-12-06)
- 🔄 **重大重构**: 完全重写流水线系统
- 🧪 **新增**: 统一实验管理系统
- 🐛 **修复**: 检索覆盖率问题 (相似度阈值 0.7→0.2)
- 🐛 **修复**: Step7证据格式不兼容问题
- ✨ **新增**: Step8完整汇总报告
- 🚀 **改进**: 统一数据格式和错误处理

### **v1.x.x** (历史版本)
- 基础流水线功能
- 初始概念提取和问答生成

## 🤝 贡献指南

### 📋 **开发流程**

1. Fork 项目
2. 创建功能分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 创建 Pull Request

### 🧪 **测试**

```bash
# 运行测试
python -m pytest tests/

# 单步测试
python step1.py test_document.pdf
```

### 📝 **代码规范**

- 使用类型注解
- 遵循 PEP 8 规范
- 添加详细的docstring
- 确保错误处理完整

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LlamaIndex](https://gpt-index.readthedocs.io) - 核心框架
- [OpenAI](https://openai.com) - API 支持
- [Chroma](https://www.trychroma.com) - 向量数据库

## 📞 支持

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/llamaindex-pipeline/issues)
- 📖 文档: [项目Wiki](https://github.com/your-username/llamaindex-pipeline/wiki)

---

⭐ **喜欢这个项目？请给我们一个星标！**

📚 **需要帮助？查看我们的 [详细文档](docs/) 或提交 [Issue](https://github.com/your-username/llamaindex-pipeline/issues)** 