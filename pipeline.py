"""
重构后的 ConceptBasedPipeline - 模块化设计，保持向后兼容

这是重构后的版本，内部使用模块化实现，但保持原有的 API 接口不变
"""

# 导入重构后的实现
from pipeline_new import ImprovedConceptBasedPipeline

# 为了完全向后兼容，我们保持原有的导入方式
__all__ = ['ImprovedConceptBasedPipeline']

# 如果有人直接运行这个文件，执行示例
if __name__ == "__main__":
    import logging
    from llama_index.core import SimpleDirectoryReader
    
logging.basicConfig(level=logging.INFO)
    
    print("🚀 运行模块化 ConceptBasedPipeline 示例")
    
    # 使用默认配置
    pipeline = ImprovedConceptBasedPipeline()
    
    try:
        # 加载测试文档
        reader = SimpleDirectoryReader(input_files=["attention is all you need.pdf"])
        documents = reader.load_data()
        
        print(f"📄 加载了 {len(documents)} 个文档")
        
        # 运行 Pipeline
        results = pipeline.run_pipeline(documents)
        
        # 保存结果
        pipeline.save_results(results, "modular_pipeline_results.json")
        
        # 显示统计信息
        stats = pipeline.get_pipeline_statistics()
        print("📊 Pipeline 统计信息:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        
        print("✅ 模块化 Pipeline 运行成功！")
                    
            except Exception as e:
        print(f"❌ Pipeline 运行失败: {e}")