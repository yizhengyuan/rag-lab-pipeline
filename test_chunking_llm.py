"""
测试修改后的 chunking.py 是否正确使用大模型
"""

import os
import logging
import json
from datetime import datetime
from llama_index.core import SimpleDirectoryReader, Document

# 导入修改后的模块
from core.chunking import SemanticChunker
from config.settings import load_config_from_yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_chunking_with_llm(pdf_path: str = None, config_path: str = "config/config.yml"):
    """
    测试修改后的chunking.py是否正确使用LLM
    
    Args:
        pdf_path: PDF文件路径（可选，如果没有将使用模拟文本）
        config_path: 配置文件路径
    """
    
    print("🚀 开始测试修改后的 chunking.py")
    print("=" * 80)
    
    try:
        # 1. 加载配置
        print("⚙️ 步骤1: 加载配置...")
        config = load_config_from_yaml(config_path)
        print(f"   ✅ 配置加载完成")
        print(f"   📝 使用模型: {config.get('api.model')}")
        print(f"   📝 API地址: {config.get('api.base_url')}")
        
        # 2. 初始化SemanticChunker（这里会自动初始化LLM）
        print("\n🧠 步骤2: 初始化SemanticChunker...")
        chunker = SemanticChunker(config)
        print("   ✅ SemanticChunker初始化完成")
        
        # 3. 准备测试文档
        print("\n📄 步骤3: 准备测试文档...")
        if pdf_path and os.path.exists(pdf_path):
            print(f"   📁 使用PDF文件: {pdf_path}")
            reader = SimpleDirectoryReader(input_files=[pdf_path])
            documents = reader.load_data()
            test_source = f"PDF文件: {os.path.basename(pdf_path)}"
        else:
            # 使用模拟的中文技术文档
            test_text = """
            人工智能技术正在快速发展，深度学习和机器学习算法在各个领域都有重要应用。
            自然语言处理技术使得计算机能够理解和生成人类语言，这对于智能对话系统、
            文本分析和信息检索都有重要意义。同时，计算机视觉技术让机器能够识别和
            理解图像内容，在自动驾驶、医疗诊断、安防监控等领域发挥重要作用。
            
            神经网络架构的创新推动了人工智能的突破性进展。卷积神经网络在图像识别
            任务中表现优异，循环神经网络适合处理序列数据，而Transformer架构则在
            自然语言处理领域带来了革命性变化。这些技术的组合应用使得智能系统
            能够处理更加复杂的多模态任务。
            
            大数据和云计算为人工智能提供了强大的基础设施支持。分布式计算框架
            使得大规模机器学习训练成为可能，而边缘计算技术则将智能推理能力
            扩展到终端设备。这种计算模式的演进正在重塑整个技术生态系统。
            """
            documents = [Document(text=test_text)]
            test_source = "模拟中文技术文档"
            
        print(f"   ✅ 测试文档准备完成: {test_source}")
        full_text = "\n\n".join([doc.text for doc in documents])
        print(f"   📊 文档长度: {len(full_text)} 字符")
        
        # 4. 执行概念提取
        print("\n🎯 步骤4: 执行语义分块和概念提取...")
        chunk_nodes = chunker.chunk_and_extract_concepts(documents)
        print(f"   ✅ 处理完成: {len(chunk_nodes)} 个chunk")
        
        # 5. 分析概念质量
        print("\n🔍 步骤5: 分析概念提取质量...")
        analysis_results = analyze_concept_quality(chunk_nodes)
        
        # 6. 生成详细报告
        print("\n📋 步骤6: 生成测试报告...")
        report_path = generate_test_report(
            test_source, config, chunk_nodes, analysis_results
        )
        
        # 7. 输出测试结果
        print_test_results(analysis_results, report_path)
        
        return {
            "success": True,
            "chunks_count": len(chunk_nodes),
            "analysis": analysis_results,
            "report_path": report_path
        }
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        logger.error(f"测试失败: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

def analyze_concept_quality(chunk_nodes):
    """分析概念提取的质量"""
    
    total_chunks = len(chunk_nodes)
    total_concepts = 0
    single_word_concepts = 0
    phrase_concepts = 0
    empty_chunks = 0
    
    chunk_details = []
    
    for i, node in enumerate(chunk_nodes):
        concepts = node.metadata.get("concepts", [])
        chunk_length = len(node.text)
        
        if not concepts:
            empty_chunks += 1
            
        chunk_single_words = 0
        chunk_phrases = 0
        
        for concept in concepts:
            total_concepts += 1
            if len(concept.split()) == 1:
                single_word_concepts += 1
                chunk_single_words += 1
            else:
                phrase_concepts += 1
                chunk_phrases += 1
        
        chunk_details.append({
            "chunk_id": i,
            "text_length": chunk_length,
            "concepts": concepts,
            "concept_count": len(concepts),
            "single_words": chunk_single_words,
            "phrases": chunk_phrases
        })
    
    # 计算百分比
    phrase_percentage = (phrase_concepts / total_concepts * 100) if total_concepts > 0 else 0
    single_word_percentage = (single_word_concepts / total_concepts * 100) if total_concepts > 0 else 0
    
    # 判断LLM工作状态
    if phrase_percentage > 60:
        llm_status = "✅ LLM正常工作"
        llm_working = True
    elif phrase_percentage > 30:
        llm_status = "⚠️ LLM部分工作"
        llm_working = True
    else:
        llm_status = "❌ LLM可能未工作"
        llm_working = False
    
    return {
        "total_chunks": total_chunks,
        "total_concepts": total_concepts,
        "single_word_concepts": single_word_concepts,
        "phrase_concepts": phrase_concepts,
        "empty_chunks": empty_chunks,
        "phrase_percentage": phrase_percentage,
        "single_word_percentage": single_word_percentage,
        "llm_status": llm_status,
        "llm_working": llm_working,
        "chunk_details": chunk_details
    }

def print_test_results(analysis, report_path):
    """打印测试结果"""
    
    print("\n" + "="*80)
    print("🎯 测试结果摘要")
    print("="*80)
    
    print(f"\n📊 基本统计:")
    print(f"   - 总chunk数: {analysis['total_chunks']}")
    print(f"   - 总概念数: {analysis['total_concepts']}")
    print(f"   - 空概念chunk: {analysis['empty_chunks']}")
    
    print(f"\n🔍 概念质量分析:")
    print(f"   - 单词概念: {analysis['single_word_concepts']} ({analysis['single_word_percentage']:.1f}%)")
    print(f"   - 词组概念: {analysis['phrase_concepts']} ({analysis['phrase_percentage']:.1f}%)")
    
    print(f"\n🤖 LLM工作状态:")
    print(f"   {analysis['llm_status']}")
    
    if analysis['llm_working']:
        print(f"\n✅ 测试通过！大模型正在正常工作")
        if analysis['phrase_percentage'] > 80:
            print(f"   🌟 概念质量优秀！")
        elif analysis['phrase_percentage'] > 60:
            print(f"   👍 概念质量良好")
    else:
        print(f"\n❌ 测试失败！大模型可能未正常工作")
        print(f"   💡 建议检查:")
        print(f"      - API密钥是否正确")
        print(f"      - 网络连接是否正常") 
        print(f"      - Settings.llm是否正确初始化")
    
    print(f"\n📋 详细报告: {report_path}")
    
    # 显示几个概念示例
    print(f"\n📝 概念示例:")
    for i, chunk in enumerate(analysis['chunk_details'][:3]):
        if chunk['concepts']:
            print(f"   Chunk {i+1}: {chunk['concepts'][:3]}...")
    
    print("\n" + "="*80)

def generate_test_report(test_source, config, chunk_nodes, analysis):
    """生成详细的测试报告"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"./chunking_llm_test_report_{timestamp}.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# chunking.py LLM测试报告\n\n")
        f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**测试文档**: {test_source}\n\n")
        
        f.write(f"## ⚙️ 配置信息\n\n")
        f.write(f"- **模型**: {config.get('api.model')}\n")
        f.write(f"- **API地址**: {config.get('api.base_url')}\n")
        f.write(f"- **Temperature**: {config.get('api.temperature')}\n")
        f.write(f"- **每块概念数**: {config.get('concepts.concepts_per_chunk')}\n\n")
        
        f.write(f"## 📊 测试结果\n\n")
        f.write(f"- **总chunk数**: {analysis['total_chunks']}\n")
        f.write(f"- **总概念数**: {analysis['total_concepts']}\n")
        f.write(f"- **单词概念**: {analysis['single_word_concepts']} ({analysis['single_word_percentage']:.1f}%)\n")
        f.write(f"- **词组概念**: {analysis['phrase_concepts']} ({analysis['phrase_percentage']:.1f}%)\n")
        f.write(f"- **空概念chunk**: {analysis['empty_chunks']}\n\n")
        
        f.write(f"## 🤖 LLM状态\n\n")
        f.write(f"{analysis['llm_status']}\n\n")
        
        f.write(f"## 📝 详细chunk分析\n\n")
        for chunk in analysis['chunk_details']:
            f.write(f"### Chunk {chunk['chunk_id'] + 1}\n\n")
            f.write(f"- **文本长度**: {chunk['text_length']} 字符\n")
            f.write(f"- **概念数量**: {chunk['concept_count']}\n")
            f.write(f"- **单词**: {chunk['single_words']}, **词组**: {chunk['phrases']}\n")
            f.write(f"- **概念列表**: {chunk['concepts']}\n\n")
        
        f.write(f"## 🎯 判断标准\n\n")
        f.write(f"- **LLM正常工作**: 词组概念占比 > 60%\n")
        f.write(f"- **LLM部分工作**: 词组概念占比 30-60%\n") 
        f.write(f"- **LLM未工作**: 词组概念占比 < 30%\n\n")
        
        if not analysis['llm_working']:
            f.write(f"## 🔧 问题排查\n\n")
            f.write(f"如果LLM未正常工作，请检查:\n\n")
            f.write(f"1. **配置文件**: API密钥、模型名称、API地址是否正确\n")
            f.write(f"2. **网络连接**: 能否访问API地址\n")
            f.write(f"3. **Settings初始化**: chunking.py中的_setup_llamaindex_settings是否成功\n")
            f.write(f"4. **日志信息**: 查看是否有LLM调用错误信息\n\n")
    
    return report_path

def main():
    """主函数 - 支持命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试修改后的chunking.py LLM功能")
    parser.add_argument("--pdf", help="PDF文件路径（可选，不提供则使用模拟文档）")
    parser.add_argument("--config", default="config/config.yml", help="配置文件路径")
    
    args = parser.parse_args()
    
    # 执行测试
    result = test_chunking_with_llm(args.pdf, args.config)
    
    if not result.get("success"):
        exit(1)

if __name__ == "__main__":
    main() 