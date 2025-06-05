#!/usr/bin/env python3
"""
测试修复后的pipeline功能
"""

import sys
import os
import logging
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_validators():
    """测试ConceptValidator功能"""
    logger.info("🧪 测试ConceptValidator...")
    
    try:
        from utils import ConceptValidator
        from llama_index.core import Document
        
        # 创建测试文档
        test_docs = [
            Document(text="This is a test document with some content."),
            Document(text="Another document for testing purposes.")
        ]
        
        # 测试验证方法
        result = ConceptValidator.validate_documents(test_docs)
        logger.info(f"✅ ConceptValidator.validate_documents 测试成功: {result}")
        
        # 测试空文档
        empty_result = ConceptValidator.validate_documents([])
        logger.info(f"✅ 空文档测试: {empty_result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ConceptValidator测试失败: {e}")
        return False

def test_api_config():
    """测试API配置"""
    logger.info("🧪 测试API配置...")
    
    try:
        from integrated_pipeline_new import ModularIntegratedPipeline
        
        # 创建pipeline实例
        pipeline = ModularIntegratedPipeline(
            config_path="config.yml",
            output_dir="./test_simple_output"
        )
        
        # 检查配置
        logger.info(f"✅ Pipeline创建成功")
        logger.info(f"   模型: {pipeline.model_name}")
        logger.info(f"   API密钥: {pipeline.concept_pipeline.config.openai_api_key[:10]}...")
        
        # 检查环境变量是否正确设置
        if 'OPENAI_API_KEY' in os.environ:
            logger.info(f"   环境变量API密钥: {os.environ['OPENAI_API_KEY'][:10]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API配置测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_simple_processing():
    """测试简单的文档处理"""
    logger.info("🧪 测试简单文档处理...")
    
    try:
        from integrated_pipeline_new import ModularIntegratedPipeline
        
        # 创建pipeline
        pipeline = ModularIntegratedPipeline(
            config_path="config.yml",
            output_dir="./test_simple_output",
            questions_per_type={
                "remember": 1,
                "understand": 1,
                "apply": 0,
                "analyze": 0,
                "evaluate": 0,
                "create": 0
            }
        )
        
        # 处理IEEE文档
        logger.info("📄 开始处理IEEE Frank Rosenblatt Award文档...")
        
        results = pipeline.process_single_file(
            "en.wikipedia.org_wiki_IEEE_Frank_Rosenblatt_Award.md",
            save_intermediate=True
        )
        
        if results:
            logger.info("✅ 文档处理成功!")
            logger.info(f"   - 问答对数量: {len(results.get('qa_pairs', []))}")
            logger.info(f"   - 训练数据数量: {len(results.get('training_data', []))}")
            
            # 显示生成的问答对
            for i, qa in enumerate(results.get('qa_pairs', [])[:3]):
                logger.info(f"   问题{i+1}: {qa.get('question', 'N/A')}")
                logger.info(f"   答案{i+1}: {qa.get('answer', 'N/A')[:100]}...")
                
            return True
        else:
            logger.error("❌ 文档处理返回空结果")
            return False
            
    except Exception as e:
        logger.error(f"❌ 简单文档处理测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始测试修复后的pipeline...")
    
    tests = [
        ("ConceptValidator功能", test_validators),
        ("API配置", test_api_config),
        ("简单文档处理", test_simple_processing)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"🧪 执行测试: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ 测试 {test_name} 发生异常: {e}")
            results[test_name] = False
    
    # 总结测试结果
    logger.info(f"\n{'='*50}")
    logger.info("📊 测试结果总结:")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试都通过了！")
        return True
    else:
        logger.error(f"⚠️  有 {total - passed} 个测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 