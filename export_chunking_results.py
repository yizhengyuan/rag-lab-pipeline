"""
直接导出 chunking.py 的真实处理结果
====================================

这个脚本直接使用您的：
- core/chunking.py 中的 SemanticChunker
- config/config.yml 中的配置
- 完全相同的初始化和处理流程

然后将结果导出为单独的文本文件
"""

import os
import logging
from datetime import datetime
from llama_index.core import SimpleDirectoryReader

# 直接导入您的模块
from core.chunking import SemanticChunker  
from config import load_config_from_yaml

logger = logging.getLogger(__name__)

def export_real_chunking_results(pdf_path: str, config_path: str = "config/config.yml", output_dir: str = "./chunking_export"):
    """
    直接导出 chunking.py 的真实处理结果
    
    Args:
        pdf_path: PDF文件路径
        config_path: 配置文件路径  
        output_dir: 输出目录
    """
    
    file_name = os.path.basename(pdf_path)
    base_name = os.path.splitext(file_name)[0]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "chunks"), exist_ok=True)
    
    logger.info(f"🔄 开始使用 chunking.py 处理: {pdf_path}")
    logger.info(f"⚙️ 配置文件: {config_path}")
    logger.info(f"📁 输出目录: {output_dir}")
    logger.info("=" * 80)
    
    try:
        # 1. 加载您的配置
        logger.info("⚙️ 步骤1: 加载配置...")
        config = load_config_from_yaml(config_path)
        logger.info("   ✅ 配置加载完成")
        
        # 2. 初始化您的 SemanticChunker
        logger.info("🧠 步骤2: 初始化 SemanticChunker...")
        chunker = SemanticChunker(config)
        logger.info("   ✅ SemanticChunker 初始化完成")
        
        # 3. 加载PDF文档
        logger.info("📄 步骤3: 加载PDF文档...")
        reader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = reader.load_data()
        full_text = "\n\n".join([doc.text for doc in documents])
        logger.info(f"   ✅ PDF加载完成: {len(full_text)} 字符")
        
        # 4. 使用您的 chunking.py 进行处理
        logger.info("✂️ 步骤4: 执行真实的语义分块和概念提取...")
        chunk_nodes = chunker.chunk_and_extract_concepts(documents)
        logger.info(f"   ✅ 处理完成: {len(chunk_nodes)} 个chunk")
        
        # 5. 获取统计信息
        logger.info("📊 步骤5: 获取统计信息...")
        stats = chunker.get_chunking_statistics()
        logger.info(f"   ✅ 统计完成: 总概念数 {stats.get('total_concepts', 0)}")
        
        # 6. 导出为单独的文本文件
        logger.info("💾 步骤6: 导出chunk为单独文件...")
        chunk_files = export_chunks_to_individual_files(chunk_nodes, base_name, output_dir)
        logger.info(f"   ✅ 已导出 {len(chunk_files)} 个文件")
        
        # 7. 同时使用原有的导出方法生成JSON
        logger.info("📄 步骤7: 生成JSON格式的完整导出...")
        json_export_path = os.path.join(output_dir, f"{base_name}_chunks_complete.json")
        chunker.export_chunks_with_concepts(json_export_path)
        logger.info(f"   ✅ JSON导出完成: {json_export_path}")
        
        # 8. 生成详细报告
        logger.info("📋 步骤8: 生成详细报告...")
        report_path = generate_detailed_report(
            pdf_path, config_path, stats, chunk_files, json_export_path, base_name, output_dir
        )
        logger.info(f"   ✅ 报告生成完成: {report_path}")
        
        print(f"\n✅ chunking.py 真实处理结果导出完成！")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 详细报告: {report_path}")
        print(f"📄 JSON完整导出: {json_export_path}")
        print(f"📁 单独chunk文件: {output_dir}/chunks/")
        print(f"\n📈 处理统计:")
        print(f"   - 总chunk数: {len(chunk_nodes)}")
        print(f"   - 总概念数: {stats.get('total_concepts', 0)}")
        print(f"   - 平均chunk长度: {stats.get('avg_chunk_length', 0):.0f} 字符")
        print(f"   - 平均每块概念数: {stats.get('avg_concepts_per_chunk', 0):.1f}")
        
        return {
            "success": True,
            "chunk_files": chunk_files,
            "json_export": json_export_path,
            "report": report_path,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"❌ 处理失败: {str(e)}")
        print(f"\n❌ 处理失败: {str(e)}")
        return {"success": False, "error": str(e)}

def export_chunks_to_individual_files(chunk_nodes, base_name: str, output_dir: str):
    """将chunk_nodes导出为单独的文本文件"""
    chunk_files = []
    chunks_dir = os.path.join(output_dir, "chunks")
    
    for i, node in enumerate(chunk_nodes):
        # 获取chunk信息
        chunk_id = node.metadata.get("chunk_id", f"chunk_{i}")
        concepts = node.metadata.get("concepts", [])
        
        # 创建文件名
        chunk_filename = f"{base_name}_{chunk_id}.txt"
        chunk_path = os.path.join(chunks_dir, chunk_filename)
        
        # 写入文件
        with open(chunk_path, 'w', encoding='utf-8') as f:
            # 头部信息
            f.write(f"============================================================\n")
            f.write(f"chunking.py 真实处理结果\n")
            f.write(f"============================================================\n\n")
            f.write(f"Chunk ID: {chunk_id}\n")
            f.write(f"Chunk Index: {i}\n")
            f.write(f"Text Length: {len(node.text)} characters\n")
            f.write(f"Word Count: {len(node.text.split())} words\n")
            f.write(f"Concept Count: {len(concepts)}\n")
            f.write(f"Node ID: {getattr(node, 'node_id', 'N/A')}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source: core/chunking.py SemanticChunker\n")
            f.write(f"============================================================\n\n")
            
            # 提取的概念
            if concepts:
                f.write(f"🧠 提取的概念 (由chunking.py生成):\n")
                for j, concept in enumerate(concepts, 1):
                    f.write(f"  {j:2d}. {concept}\n")
                f.write(f"\n")
            else:
                f.write(f"🧠 提取的概念: (无概念被提取)\n\n")
            
            f.write(f"============================================================\n\n")
            
            # 完整文本内容
            f.write(f"📄 Chunk 完整文本内容:\n\n")
            f.write(node.text)
            
            # 如果有额外的metadata，也记录下来
            f.write(f"\n\n============================================================\n")
            f.write(f"🔧 技术信息:\n\n")
            for key, value in node.metadata.items():
                if key not in ['concepts', 'chunk_id', 'chunk_length', 'concept_count']:
                    f.write(f"  {key}: {value}\n")
        
        chunk_files.append(chunk_path)
        logger.info(f"   💾 已保存: {chunk_filename} ({len(concepts)} 个概念)")
    
    return chunk_files

def generate_detailed_report(pdf_path, config_path, stats, chunk_files, json_export_path, base_name, output_dir):
    """生成详细的处理报告"""
    report_path = os.path.join(output_dir, f"{base_name}_chunking_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# chunking.py 真实处理结果报告\n\n")
        f.write(f"**处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## 📄 输入信息\n\n")
        f.write(f"- **PDF文件**: `{pdf_path}`\n")
        f.write(f"- **配置文件**: `{config_path}`\n")
        f.write(f"- **处理器**: `core/chunking.py` 中的 `SemanticChunker`\n\n")
        
        f.write(f"## 📊 处理统计\n\n")
        f.write(f"- **总chunk数**: {stats.get('total_chunks', 0)}\n")
        f.write(f"- **总概念数**: {stats.get('total_concepts', 0)}\n")  
        f.write(f"- **平均chunk长度**: {stats.get('avg_chunk_length', 0):.0f} 字符\n")
        f.write(f"- **最小chunk长度**: {stats.get('min_chunk_length', 0)} 字符\n")
        f.write(f"- **最大chunk长度**: {stats.get('max_chunk_length', 0)} 字符\n")
        f.write(f"- **平均每块概念数**: {stats.get('avg_concepts_per_chunk', 0):.1f}\n")
        f.write(f"- **最少概念数**: {stats.get('min_concepts_per_chunk', 0)}\n")
        f.write(f"- **最多概念数**: {stats.get('max_concepts_per_chunk', 0)}\n\n")
        
        f.write(f"## 📁 生成的文件\n\n")
        f.write(f"### 单独的chunk文件 ({len(chunk_files)}个)\n\n")
        for i, chunk_file in enumerate(chunk_files, 1):
            filename = os.path.basename(chunk_file)
            f.write(f"{i:3d}. `{filename}`\n")
        
        f.write(f"\n### 完整JSON导出\n\n")
        f.write(f"- `{os.path.basename(json_export_path)}` - 完整的JSON格式导出\n\n")
        
        f.write(f"## 🔧 技术说明\n\n")
        f.write(f"此结果是通过直接调用您的 `core/chunking.py` 生成的，确保100%与您的pipeline一致：\n\n")
        f.write(f"1. **使用您的配置**: 从 `{config_path}` 加载所有参数\n")
        f.write(f"2. **使用您的chunker**: `SemanticChunker` 类的 `chunk_and_extract_concepts` 方法\n")
        f.write(f"3. **相同的初始化**: 完全相同的配置和LLM设置\n")
        f.write(f"4. **真实的概念提取**: 每个chunk的概念都是由您的LLM实际生成的\n\n")
        
        f.write(f"## 📖 查看说明\n\n")
        f.write(f"每个chunk文件 (`chunks/` 目录) 包含：\n")
        f.write(f"- Chunk基本信息和统计\n")
        f.write(f"- 🧠 由您的LLM实际提取的概念\n")
        f.write(f"- 📄 完整的chunk文本内容\n")
        f.write(f"- 🔧 技术元数据\n\n")
        
        f.write(f"这些就是您的 `chunking.py` 的真实输出结果！\n")
    
    return report_path

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="导出 chunking.py 的真实处理结果")
    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--config", default="config/config.yml", help="配置文件路径")
    parser.add_argument("--output", default="./chunking_export", help="输出目录")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 执行导出
    result = export_real_chunking_results(args.pdf, args.config, args.output)
    
    if not result.get("success"):
        exit(1)

if __name__ == "__main__":
    main()