"""
概念合并结果导出工具
====================================

这个脚本将概念合并(concept merge)的结果导出为可读的文本文件
包含：
- 合并前后的概念对比
- 每个合并后概念的详细信息
- 来源chunk和置信度信息
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from llama_index.core import SimpleDirectoryReader

# 导入您的模块
from core.chunking import SemanticChunker  
from core.concept_merger import ConceptMerger
from config import load_config_from_yaml

logger = logging.getLogger(__name__)

def export_concept_merge_results(pdf_path: str, config_path: str = "config/config.yml", output_dir: str = "./concept_merge_export"):
    """
    导出概念合并的完整处理结果
    
    Args:
        pdf_path: PDF文件路径
        config_path: 配置文件路径  
        output_dir: 输出目录
    """
    
    file_name = os.path.basename(pdf_path)
    base_name = os.path.splitext(file_name)[0]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "concepts"), exist_ok=True)
    
    logger.info(f"🔄 开始概念合并处理: {pdf_path}")
    logger.info(f"⚙️ 配置文件: {config_path}")
    logger.info(f"📁 输出目录: {output_dir}")
    logger.info("=" * 80)
    
    try:
        # 1. 加载配置
        logger.info("⚙️ 步骤1: 加载配置...")
        config = load_config_from_yaml(config_path)
        logger.info("   ✅ 配置加载完成")
        
        # 2. 初始化处理器
        logger.info("🧠 步骤2: 初始化处理器...")
        chunker = SemanticChunker(config)
        concept_merger = ConceptMerger(config)
        logger.info("   ✅ 处理器初始化完成")
        
        # 3. 加载PDF文档
        logger.info("📄 步骤3: 加载PDF文档...")
        reader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = reader.load_data()
        logger.info(f"   ✅ PDF加载完成: {len(documents)} 个文档")
        
        # 4. 执行语义分块和概念提取
        logger.info("✂️ 步骤4: 执行语义分块和概念提取...")
        chunk_nodes = chunker.chunk_and_extract_concepts(documents)
        logger.info(f"   ✅ 分块完成: {len(chunk_nodes)} 个chunk")
        
        # 5. 执行概念合并
        logger.info("🔗 步骤5: 执行概念合并...")
        merged_concept_nodes = concept_merger.merge_document_concepts(chunk_nodes)
        logger.info(f"   ✅ 概念合并完成: {len(merged_concept_nodes)} 个合并后概念")
        
        # 6. 收集统计信息
        logger.info("📊 步骤6: 收集统计信息...")
        merge_stats = collect_merge_statistics(chunk_nodes, merged_concept_nodes, concept_merger)
        logger.info(f"   ✅ 统计完成")
        
        # 7. 导出概念合并总览
        logger.info("💾 步骤7: 导出概念合并总览...")
        overview_path = export_merge_overview(merged_concept_nodes, merge_stats, base_name, output_dir)
        logger.info(f"   ✅ 总览文件已保存: {overview_path}")
        
        # 8. 导出每个合并后概念的详细信息
        logger.info("📄 步骤8: 导出详细概念文件...")
        concept_files = export_individual_concepts(merged_concept_nodes, chunk_nodes, base_name, output_dir)
        logger.info(f"   ✅ 已导出 {len(concept_files)} 个概念文件")
        
        # 9. 导出JSON格式的完整数据
        logger.info("📋 步骤9: 导出JSON格式...")
        json_path = export_merge_json(merged_concept_nodes, merge_stats, base_name, output_dir)
        logger.info(f"   ✅ JSON文件已保存: {json_path}")
        
        # 10. 生成详细报告
        logger.info("📈 步骤10: 生成详细报告...")
        report_path = generate_merge_report(
            pdf_path, config_path, merge_stats, concept_files, 
            overview_path, json_path, base_name, output_dir
        )
        logger.info(f"   ✅ 报告生成完成: {report_path}")
        
        print(f"\n✅ 概念合并结果导出完成！")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 总览文件: {overview_path}")
        print(f"📋 详细报告: {report_path}")
        print(f"📄 JSON格式: {json_path}")
        print(f"📁 单独概念文件: {output_dir}/concepts/")
        print(f"\n📈 合并统计:")
        print(f"   - 原始概念数: {merge_stats['original_concepts_count']}")
        print(f"   - 合并后概念数: {merge_stats['merged_concepts_count']}")
        print(f"   - 合并减少率: {merge_stats['reduction_ratio']:.1%}")
        print(f"   - 平均置信度: {merge_stats['avg_confidence']:.3f}")
        
        return {
            "success": True,
            "overview_file": overview_path,
            "concept_files": concept_files,
            "json_export": json_path,
            "report": report_path,
            "stats": merge_stats
        }
        
    except Exception as e:
        logger.error(f"❌ 处理失败: {str(e)}")
        print(f"\n❌ 处理失败: {str(e)}")
        return {"success": False, "error": str(e)}

def collect_merge_statistics(chunk_nodes: List, merged_concept_nodes: List, concept_merger: ConceptMerger) -> Dict[str, Any]:
    """收集概念合并的统计信息"""
    
    # 收集所有原始概念
    original_concepts = []
    for node in chunk_nodes:
        concepts = node.metadata.get("concepts", [])
        original_concepts.extend(concepts)
    
    # 基本统计
    original_count = len(original_concepts)
    merged_count = len(merged_concept_nodes)
    reduction_ratio = (original_count - merged_count) / original_count if original_count > 0 else 0
    
    # 置信度统计
    confidence_scores = [node.confidence_score for node in merged_concept_nodes if hasattr(node, 'confidence_score')]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    max_confidence = max(confidence_scores) if confidence_scores else 0
    min_confidence = min(confidence_scores) if confidence_scores else 0
    
    # 类别统计
    categories = [getattr(node, 'category', '未分类') for node in merged_concept_nodes]
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # 来源chunk统计
    chunk_coverage = set()
    for node in merged_concept_nodes:
        if hasattr(node, 'source_chunks'):
            chunk_coverage.update(node.source_chunks)
    
    return {
        "original_concepts_count": original_count,
        "merged_concepts_count": merged_count,
        "reduction_ratio": reduction_ratio,
        "avg_confidence": avg_confidence,
        "max_confidence": max_confidence,
        "min_confidence": min_confidence,
        "category_distribution": category_counts,
        "chunk_coverage": len(chunk_coverage),
        "total_chunks": len(chunk_nodes),
        "processing_time": datetime.now().isoformat()
    }

def export_merge_overview(merged_concept_nodes: List, merge_stats: Dict[str, Any], base_name: str, output_dir: str) -> str:
    """导出概念合并总览文件"""
    
    overview_path = os.path.join(output_dir, f"{base_name}_concept_merge_overview.txt")
    
    with open(overview_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("概念合并(Concept Merge)结果总览\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"文档: {base_name}\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理器: core/concept_merger.py ConceptMerger\n")
        f.write("=" * 80 + "\n\n")
        
        # 统计信息
        f.write("📊 合并统计信息:\n\n")
        f.write(f"  原始概念总数: {merge_stats['original_concepts_count']}\n")
        f.write(f"  合并后概念数: {merge_stats['merged_concepts_count']}\n")
        f.write(f"  概念减少数量: {merge_stats['original_concepts_count'] - merge_stats['merged_concepts_count']}\n")
        f.write(f"  合并减少率: {merge_stats['reduction_ratio']:.1%}\n")
        f.write(f"  涉及chunk数: {merge_stats['chunk_coverage']} / {merge_stats['total_chunks']}\n")
        f.write(f"  平均置信度: {merge_stats['avg_confidence']:.3f}\n")
        f.write(f"  最高置信度: {merge_stats['max_confidence']:.3f}\n")
        f.write(f"  最低置信度: {merge_stats['min_confidence']:.3f}\n\n")
        
        # 类别分布
        f.write("📂 概念类别分布:\n\n")
        for category, count in merge_stats['category_distribution'].items():
            percentage = count / merge_stats['merged_concepts_count'] * 100
            f.write(f"  {category}: {count} 个概念 ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n\n")
        
        # 合并后的概念列表
        f.write("🧠 合并后的概念列表:\n\n")
        
        # 按置信度排序
        sorted_concepts = sorted(merged_concept_nodes, key=lambda x: getattr(x, 'confidence_score', 0), reverse=True)
        
        for i, node in enumerate(sorted_concepts, 1):
            concept_name = getattr(node, 'concept_name', node.text[:50])
            confidence = getattr(node, 'confidence_score', 0)
            category = getattr(node, 'category', '未分类')
            source_chunks = getattr(node, 'source_chunks', [])
            keywords = getattr(node, 'keywords', [])
            
            f.write(f"{i:3d}. 【{category}】{concept_name}\n")
            f.write(f"     置信度: {confidence:.3f}\n")
            f.write(f"     来源chunks: {len(source_chunks)} 个\n")
            if keywords:
                f.write(f"     关键词: {', '.join(keywords[:5])}\n")
            f.write(f"     概念文本: {node.text[:100]}{'...' if len(node.text) > 100 else ''}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("📄 查看详细信息: 请参考 concepts/ 目录下的单独概念文件\n")
        f.write("📋 JSON格式数据: 请参考对应的 _merge_data.json 文件\n")
        f.write("=" * 80 + "\n")
    
    return overview_path

def export_individual_concepts(merged_concept_nodes: List, chunk_nodes: List, base_name: str, output_dir: str) -> List[str]:
    """导出每个合并后概念的详细文件"""
    
    concept_files = []
    concepts_dir = os.path.join(output_dir, "concepts")
    
    # 创建chunk文本映射
    chunk_text_map = {}
    for node in chunk_nodes:
        chunk_id = node.metadata.get("chunk_id", node.node_id)
        chunk_text_map[chunk_id] = node.text
    
    for i, concept_node in enumerate(merged_concept_nodes):
        # 创建文件名
        concept_name = getattr(concept_node, 'concept_name', f'concept_{i}')
        safe_name = "".join(c for c in concept_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')[:50]  # 限制长度
        
        concept_filename = f"{base_name}_concept_{i:03d}_{safe_name}.txt"
        concept_path = os.path.join(concepts_dir, concept_filename)
        
        with open(concept_path, 'w', encoding='utf-8') as f:
            # 头部信息
            f.write("=" * 80 + "\n")
            f.write("概念合并(Concept Merge)详细结果\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"概念编号: {i+1}\n")
            f.write(f"概念ID: {getattr(concept_node, 'node_id', 'N/A')}\n")
            f.write(f"概念名称: {getattr(concept_node, 'concept_name', 'N/A')}\n")
            f.write(f"概念类别: {getattr(concept_node, 'category', '未分类')}\n")
            f.write(f"置信度分数: {getattr(concept_node, 'confidence_score', 0):.3f}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理器: core/concept_merger.py ConceptMerger\n")
            f.write("=" * 80 + "\n\n")
            
            # 概念详细信息
            f.write("🧠 概念详细信息:\n\n")
            f.write(f"完整概念文本:\n")
            f.write(f"{concept_node.text}\n\n")
            
            if hasattr(concept_node, 'definition') and concept_node.definition:
                f.write(f"概念定义:\n")
                f.write(f"{concept_node.definition}\n\n")
            
            # 关键词
            keywords = getattr(concept_node, 'keywords', [])
            if keywords:
                f.write(f"🔑 关键词:\n")
                for j, keyword in enumerate(keywords, 1):
                    f.write(f"  {j}. {keyword}\n")
                f.write("\n")
            
            # 来源信息
            source_chunks = getattr(concept_node, 'source_chunks', [])
            if source_chunks:
                f.write(f"📄 来源chunk信息 (共{len(source_chunks)}个):\n\n")
                for j, chunk_id in enumerate(source_chunks, 1):
                    f.write(f"  {j}. Chunk ID: {chunk_id}\n")
                    if chunk_id in chunk_text_map:
                        chunk_text = chunk_text_map[chunk_id]
                        preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                        f.write(f"     文本预览: {preview}\n")
                    f.write("\n")
            
            # 元数据信息
            if hasattr(concept_node, 'metadata') and concept_node.metadata:
                f.write(f"🔧 技术元数据:\n\n")
                for key, value in concept_node.metadata.items():
                    if key not in ['concept_name', 'definition', 'keywords', 'category']:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("说明: 此概念是通过ConceptMerger合并相似概念后生成的\n")
            f.write("=" * 80 + "\n")
        
        concept_files.append(concept_path)
        logger.info(f"   💾 已保存概念文件: {concept_filename}")
    
    return concept_files

def export_merge_json(merged_concept_nodes: List, merge_stats: Dict[str, Any], base_name: str, output_dir: str) -> str:
    """导出JSON格式的概念合并数据"""
    
    json_path = os.path.join(output_dir, f"{base_name}_concept_merge_data.json")
    
    # 构建JSON数据
    json_data = {
        "metadata": {
            "document_name": base_name,
            "processing_time": datetime.now().isoformat(),
            "processor": "core/concept_merger.py ConceptMerger",
            "statistics": merge_stats
        },
        "merged_concepts": []
    }
    
    for i, node in enumerate(merged_concept_nodes):
        concept_data = {
            "index": i,
            "concept_id": getattr(node, 'node_id', f'concept_{i}'),
            "concept_name": getattr(node, 'concept_name', ''),
            "concept_text": node.text,
            "definition": getattr(node, 'definition', ''),
            "category": getattr(node, 'category', '未分类'),
            "confidence_score": getattr(node, 'confidence_score', 0),
            "keywords": getattr(node, 'keywords', []),
            "source_chunks": getattr(node, 'source_chunks', []),
            "metadata": getattr(node, 'metadata', {})
        }
        json_data["merged_concepts"].append(concept_data)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    return json_path

def generate_merge_report(pdf_path: str, config_path: str, merge_stats: Dict[str, Any], 
                         concept_files: List[str], overview_path: str, json_path: str, 
                         base_name: str, output_dir: str) -> str:
    """生成概念合并的详细报告"""
    
    report_path = os.path.join(output_dir, f"{base_name}_concept_merge_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 概念合并(Concept Merge)处理报告\n\n")
        f.write(f"**处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## 📄 输入信息\n\n")
        f.write(f"- **PDF文件**: `{pdf_path}`\n")
        f.write(f"- **配置文件**: `{config_path}`\n")
        f.write(f"- **处理流程**: Semantic Chunking → Concept Extraction → Concept Merging\n")
        f.write(f"- **核心处理器**: `core/concept_merger.py` 中的 `ConceptMerger`\n\n")
        
        f.write(f"## 📊 合并统计信息\n\n")
        f.write(f"### 概念数量变化\n")
        f.write(f"- **原始概念总数**: {merge_stats['original_concepts_count']}\n")
        f.write(f"- **合并后概念数**: {merge_stats['merged_concepts_count']}\n")
        f.write(f"- **减少的概念数**: {merge_stats['original_concepts_count'] - merge_stats['merged_concepts_count']}\n")
        f.write(f"- **合并减少率**: {merge_stats['reduction_ratio']:.1%}\n\n")
        
        f.write(f"### 质量指标\n")
        f.write(f"- **平均置信度**: {merge_stats['avg_confidence']:.3f}\n")
        f.write(f"- **最高置信度**: {merge_stats['max_confidence']:.3f}\n")
        f.write(f"- **最低置信度**: {merge_stats['min_confidence']:.3f}\n")
        f.write(f"- **涉及chunk数**: {merge_stats['chunk_coverage']} / {merge_stats['total_chunks']} ({merge_stats['chunk_coverage']/merge_stats['total_chunks']:.1%})\n\n")
        
        f.write(f"### 概念类别分布\n")
        for category, count in merge_stats['category_distribution'].items():
            percentage = count / merge_stats['merged_concepts_count'] * 100
            f.write(f"- **{category}**: {count} 个概念 ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write(f"## 📁 生成的文件\n\n")
        f.write(f"### 总览文件\n")
        f.write(f"- `{os.path.basename(overview_path)}` - 概念合并总览和概念列表\n\n")
        
        f.write(f"### 详细概念文件 ({len(concept_files)}个)\n")
        for i, concept_file in enumerate(concept_files, 1):
            filename = os.path.basename(concept_file)
            f.write(f"{i:3d}. `{filename}`\n")
        f.write("\n")
        
        f.write(f"### 数据文件\n")
        f.write(f"- `{os.path.basename(json_path)}` - JSON格式的完整概念合并数据\n\n")
        
        f.write(f"## 🔧 技术说明\n\n")
        f.write(f"此概念合并结果是通过以下步骤生成的：\n\n")
        f.write(f"1. **文档分块**: 使用 `SemanticChunker` 将文档分解为语义块\n")
        f.write(f"2. **概念提取**: 从每个chunk中提取概念\n")
        f.write(f"3. **概念收集**: 收集所有chunk级别的概念\n")
        f.write(f"4. **相似度计算**: 使用LlamaIndex的嵌入系统计算概念相似度\n")
        f.write(f"5. **概念聚类**: 基于相似度阈值进行概念聚类\n")
        f.write(f"6. **概念合并**: 使用LLM合并相似概念的文本表述\n")
        f.write(f"7. **质量评估**: 计算置信度分数和概念分类\n\n")
        
        f.write(f"## 📖 文件说明\n\n")
        f.write(f"### 总览文件内容\n")
        f.write(f"- 📊 完整的合并统计信息\n")
        f.write(f"- 📂 概念类别分布\n")
        f.write(f"- 🧠 所有合并后概念的列表（按置信度排序）\n\n")
        
        f.write(f"### 单独概念文件内容\n")
        f.write(f"每个概念文件 (`concepts/` 目录) 包含：\n")
        f.write(f"- 🧠 完整的概念文本和定义\n")
        f.write(f"- 🔑 提取的关键词\n")
        f.write(f"- 📄 来源chunk信息和文本预览\n")
        f.write(f"- 🔧 技术元数据和处理信息\n\n")
        
        f.write(f"### 使用建议\n")
        f.write(f"1. **快速浏览**: 先查看总览文件了解整体情况\n")
        f.write(f"2. **深入研究**: 查看单独概念文件了解具体概念\n")
        f.write(f"3. **数据分析**: 使用JSON文件进行程序化分析\n")
        f.write(f"4. **质量检查**: 关注置信度较低的概念，可能需要手动调整\n\n")
        
        f.write(f"## 🎯 概念合并的价值\n\n")
        f.write(f"通过概念合并，我们实现了：\n")
        f.write(f"- **去重**: 消除了重复和相似的概念\n")
        f.write(f"- **抽象化**: 将具体概念提升到更通用的层次\n")
        f.write(f"- **结构化**: 建立了清晰的概念层次和分类\n")
        f.write(f"- **可检索**: 提供了高质量的概念索引用于后续检索\n\n")
        
        f.write(f"这些合并后的概念可以用于：\n")
        f.write(f"- 📝 文档摘要生成\n")
        f.write(f"- 🔍 智能检索和问答\n")
        f.write(f"- 📊 知识图谱构建\n")
        f.write(f"- 🎓 自动化学习内容生成\n")
    
    return report_path

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="导出概念合并(Concept Merge)的处理结果")
    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--config", default="config/config.yml", help="配置文件路径")
    parser.add_argument("--output", default="./concept_merge_export", help="输出目录")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 执行导出
    result = export_concept_merge_results(args.pdf, args.config, args.output)
    
    if not result.get("success"):
        exit(1)

if __name__ == "__main__":
    main() 