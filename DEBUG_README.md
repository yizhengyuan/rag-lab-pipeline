# Pipeline 调试工具使用指南

## 概述

这套调试工具能够详细展示文档处理 Pipeline 中每个步骤的中间产出，帮助您观察和评估每一步的工作质量。

## 功能特性

### 🔍 全流程调试
- **文档加载**: 展示文档解析结果、字符统计
- **文档分块**: 显示分块策略效果、块大小分布
- **Embedding 生成**: 查看向量维度、相似度分布
- **Vector Store 构建**: 验证向量存储状态
- **概念提取**: 观察提取的概念质量和数量
- **概念合并**: 查看合并效果和去重情况
- **证据提取**: 检查证据与概念的关联度
- **检索测试**: 验证检索准确性
- **问答生成**: 评估生成的问答对质量

### 📊 多种输出格式
- **JSON 文件**: 每个步骤的详细数据
- **Markdown 报告**: 可读性强的文本报告
- **HTML 报告**: 交互式可视化报告

## 快速开始

### 1. 基本使用

```python
from debug_pipeline import PipelineDebugger

# 创建调试器
debugger = PipelineDebugger(
    config_path="config/config.yml",
    debug_output_dir="./debug_output"
)

# 调试单个文件
results = debugger.debug_full_pipeline("your_document.pdf")
```

### 2. 命令行使用

```bash
# 调试单个文件
python debug_pipeline.py --file "attention is all you need.pdf" --output "./debug_output"

# 使用自定义配置
python debug_pipeline.py --file "document.pdf" --config "custom_config.yml"
```

### 3. 使用示例脚本

```bash
# 运行预设的示例
python debug_example.py
```

## 输出文件说明

调试完成后，会在输出目录生成以下文件：

### 步骤详细文件
- `01_document_loading.json` - 文档加载结果
- `02_chunking.json` - 文档分块结果  
- `03_embedding.json` - Embedding 生成结果
- `04_vector_store.json` - Vector Store 构建结果
- `05_concept_extraction.json` - 概念提取结果
- `06_concept_merging.json` - 概念合并结果
- `07_evidence_extraction.json` - 证据提取结果
- `08_retrieval.json` - 检索测试结果
- `09_qa_generation.json` - 问答生成结果

### 汇总报告
- `{filename}_debug_results.json` - 完整调试结果
- `{filename}_debug_report.md` - Markdown 格式报告
- `{filename}_debug_report.html` - 交互式 HTML 报告

## 详细使用说明

### 文档加载调试

查看文档是否正确解析：
```json
{
  "success": true,
  "documents_count": 1,
  "total_characters": 45678,
  "document_details": [
    {
      "index": 0,
      "text_length": 45678,
      "text_preview": "Attention Is All You Need...",
      "metadata": {...}
    }
  ]
}
```

**关注点**:
- 文档是否完整加载
- 字符数是否合理
- 文本预览是否正确

### 文档分块调试

检查分块策略效果：
```json
{
  "success": true,
  "chunks_count": 23,
  "chunking_stats": {
    "avg_chunk_length": 1987.2,
    "min_chunk_length": 856,
    "max_chunk_length": 3421
  }
}
```

**关注点**:
- 块数量是否合适
- 块大小分布是否均匀
- 是否有过小或过大的块

### 概念提取调试

评估概念提取质量：
```json
{
  "success": true,
  "concepts_count": 45,
  "concept_stats": {
    "avg_concepts_per_chunk": 1.96,
    "total_concepts": 45
  },
  "concept_details": [
    {
      "chunk_index": 0,
      "concepts_count": 2,
      "concepts": [
        {
          "concept_text": "attention mechanism",
          "confidence": 0.85
        }
      ]
    }
  ]
}
```

**关注点**:
- 概念数量是否合理
- 概念质量和相关性
- 置信度分布

### 概念合并调试

检查去重和合并效果：
```json
{
  "success": true,
  "original_concepts_count": 45,
  "merged_concepts_count": 32,
  "merge_stats": {
    "reduction_ratio": 0.289,
    "concepts_before": 45,
    "concepts_after": 32
  }
}
```

**关注点**:
- 合并比例是否合理
- 是否过度合并或合并不足
- 重要概念是否被保留

### 问答生成调试

评估问答对质量：
```json
{
  "success": true,
  "qa_pairs_count": 15,
  "qa_stats": {
    "total_qa_pairs": 15,
    "question_types": ["factual", "analytical", "conceptual"],
    "avg_question_length": 67.3,
    "avg_answer_length": 234.7
  }
}
```

**关注点**:
- 问答对数量
- 问题类型分布
- 问答长度是否合适

## 质量评估指标

### 1. 文档处理质量
- **完整性**: 文档是否完整加载
- **分块合理性**: 块大小分布是否均匀
- **信息保留**: 重要信息是否被保留

### 2. 概念提取质量  
- **覆盖度**: 重要概念是否被提取
- **准确性**: 提取的概念是否准确
- **去重效果**: 重复概念是否被合并

### 3. 检索质量
- **相关性**: 检索结果是否相关
- **排序质量**: 相似度排序是否合理
- **召回率**: 相关内容是否被检索到

### 4. 问答质量
- **多样性**: 问题类型是否多样
- **准确性**: 答案是否准确
- **完整性**: 答案是否完整

## 常见问题排查

### 1. 文档加载失败
- 检查文件路径是否正确
- 确认文件格式是否支持
- 查看错误日志了解具体原因

### 2. 概念提取效果差
- 调整概念提取的 prompt
- 检查模型配置是否正确
- 考虑调整分块大小

### 3. 检索结果不准确
- 检查 embedding 模型配置
- 调整相似度阈值
- 验证向量存储是否正确构建

### 4. 问答质量不高
- 优化问答生成的 prompt
- 调整问题类型配置
- 检查输入文本质量

## 配置优化建议

### 1. 分块参数调整
```yaml
chunking:
  chunk_size: 1000      # 根据文档类型调整
  chunk_overlap: 200    # 保证信息连续性
  separator: "\n\n"     # 选择合适的分隔符
```

### 2. 概念提取优化
```yaml
concept_extraction:
  max_concepts_per_chunk: 3  # 控制概念数量
  min_confidence: 0.7        # 设置置信度阈值
```

### 3. 检索参数调整
```yaml
retrieval:
  top_k: 5              # 检索结果数量
  similarity_threshold: 0.7  # 相似度阈值
```

## 高级功能

### 1. 自定义调试步骤

```python
# 只调试特定步骤
debugger = PipelineDebugger()

# 单独调试文档加载
doc_result = debugger._debug_document_loading("file.pdf")

# 单独调试概念提取
concept_result = debugger._debug_concept_extraction(chunks)
```

### 2. 批量调试

```python
import os

debugger = PipelineDebugger()

# 批量处理目录中的文件
for filename in os.listdir("documents/"):
    if filename.endswith(".pdf"):
        results = debugger.debug_full_pipeline(f"documents/{filename}")
```

### 3. 结果对比

```python
# 对比不同配置的效果
config1_results = debugger.debug_full_pipeline("doc.pdf")
debugger.config_path = "config2.yml"
config2_results = debugger.debug_full_pipeline("doc.pdf")

# 对比关键指标
print(f"配置1概念数: {config1_results['step_results']['concept_extraction']['concepts_count']}")
print(f"配置2概念数: {config2_results['step_results']['concept_extraction']['concepts_count']}")
```

## 最佳实践

### 1. 调试流程
1. 先运行完整调试，查看整体状况
2. 重点关注失败的步骤
3. 逐步优化配置参数
4. 重新调试验证改进效果

### 2. 结果分析
1. 查看 HTML 报告获得直观概览
2. 检查 JSON 文件了解详细数据
3. 关注关键质量指标
4. 对比不同配置的效果

### 3. 性能优化
1. 根据文档特点调整分块策略
2. 优化概念提取的准确性
3. 调整检索参数提高相关性
4. 持续监控和改进

## 技术支持

如果遇到问题，请：
1. 查看生成的错误日志
2. 检查配置文件格式
3. 确认依赖包版本
4. 参考示例代码

---

通过这套调试工具，您可以深入了解 Pipeline 每个步骤的工作情况，及时发现问题并进行优化，确保最终输出的质量。 