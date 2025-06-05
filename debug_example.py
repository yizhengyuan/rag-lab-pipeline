"""
Pipeline 调试工具使用示例
========================

这个脚本展示如何使用 PipelineDebugger 来调试您的文档处理流程
"""

import os
import logging
from debug_pipeline import PipelineDebugger

def main():
    """主函数 - 使用调试器处理示例文档"""
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🔍 Pipeline 调试工具演示")
    print("=" * 50)
    
    # 检查是否有示例文档
    sample_file = "attention is all you need.pdf"
    if not os.path.exists(sample_file):
        print(f"❌ 找不到示例文档: {sample_file}")
        print("请确保工作目录中有要处理的文档文件")
        return
    
    print(f"📄 将处理文档: {sample_file}")
    
    # 创建调试器
    debugger = PipelineDebugger(
        config_path="config/config.yml",
        debug_output_dir="./debug_output"
    )
    
    print("🚀 开始调试 Pipeline...")
    print("这将展示每个步骤的详细中间结果:")
    print("  📄 文档加载")
    print("  ✂️ 文档分块") 
    print("  🔢 Embedding 生成")
    print("  🗄️ Vector Store 构建")
    print("  🧠 概念提取")
    print("  🔗 概念合并")
    print("  🔍 证据提取")
    print("  🎯 检索测试")
    print("  ❓ 问答生成")
    print()
    
    # 执行调试
    try:
        results = debugger.debug_full_pipeline(sample_file)
        
        if "error" not in results:
            print("\n" + "=" * 50)
            print("✅ 调试完成！")
            print(f"📁 结果保存在: ./debug_output")
            print()
            print("📊 生成的文件:")
            print("  - 01_document_loading.json    # 文档加载结果")
            print("  - 02_chunking.json           # 分块结果")
            print("  - 03_embedding.json          # Embedding 结果")
            print("  - 04_vector_store.json       # Vector Store 结果")
            print("  - 05_concept_extraction.json # 概念提取结果")
            print("  - 06_concept_merging.json    # 概念合并结果")
            print("  - 07_evidence_extraction.json# 证据提取结果")
            print("  - 08_retrieval.json          # 检索测试结果")
            print("  - 09_qa_generation.json      # 问答生成结果")
            print("  - *_debug_results.json       # 完整调试结果")
            print("  - *_debug_report.md          # 可读调试报告")
            print()
            print("💡 建议:")
            print("  1. 查看 debug_report.md 获取概览")
            print("  2. 检查各步骤的 JSON 文件了解详细信息")
            print("  3. 关注失败的步骤并查看错误信息")
            print("  4. 根据结果质量调整配置参数")
            
        else:
            print(f"\n❌ 调试失败: {results['error']}")
            print("请检查配置文件和依赖是否正确安装")
            
    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")
        print("请检查:")
        print("  1. 配置文件 config/config.yml 是否存在")
        print("  2. 所有依赖是否已安装")
        print("  3. API 密钥是否正确配置")

if __name__ == "__main__":
    main() 