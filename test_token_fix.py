#!/usr/bin/env python3
"""
Token限制修复测试脚本
=====================================

用于验证chunking模块的token限制修复是否有效
"""

import sys
import logging
from pathlib import Path
from llama_index.core import Document

# 导入我们的模块
from core.chunking import SemanticChunker
from utils.config_loader import ConfigLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('token_fix_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def test_token_limit_fix():
    """测试token限制修复"""
    
    logger.info("🧪 开始Token限制修复测试")
    logger.info("=" * 60)
    
    try:
        # 1. 加载配置
        logger.info("📋 步骤1: 加载配置...")
        config = ConfigLoader.load_config()
        logger.info(f"✅ 配置加载成功")
        logger.info(f"   - 最大tokens: {config.get('chunking.max_tokens_per_chunk', 6000)}")
        logger.info(f"   - 最大字符: {config.get('chunking.max_char_per_chunk', 24000)}")
        logger.info(f"   - 启用验证: {config.get('chunking.enable_token_validation', True)}")
        
        # 2. 创建测试文档
        logger.info("\n📄 步骤2: 创建测试文档...")
        
        # 创建一个超长的测试文档
        large_text = """
这是一个测试文档，用于验证token限制修复。""" * 2000  # 重复很多次来创建超长文本
        
        small_text = "这是一个正常大小的测试文档。"
        
        test_documents = [
            Document(
                text=large_text,
                metadata={"source": "large_test_doc", "type": "stress_test"}
            ),
            Document(
                text=small_text,
                metadata={"source": "small_test_doc", "type": "normal_test"}
            )
        ]
        
        logger.info(f"✅ 创建了 {len(test_documents)} 个测试文档")
        logger.info(f"   - 大文档字符数: {len(large_text)}")
        logger.info(f"   - 小文档字符数: {len(small_text)}")
        
        # 3. 初始化Chunker
        logger.info("\n🔧 步骤3: 初始化语义分块器...")
        chunker = SemanticChunker(config)
        logger.info("✅ 分块器初始化成功")
        
        # 4. 执行分块测试
        logger.info("\n✂️ 步骤4: 执行分块和概念提取...")
        chunk_nodes = chunker.chunk_and_extract_concepts(test_documents)
        
        # 5. 分析结果
        logger.info(f"\n📊 步骤5: 分析测试结果...")
        logger.info(f"✅ 成功生成 {len(chunk_nodes)} 个chunk")
        
        if chunk_nodes:
            # 分析chunk大小分布
            token_counts = []
            char_counts = []
            oversized_chunks = 0
            max_tokens = config.get('chunking.max_tokens_per_chunk', 6000)
            
            for i, node in enumerate(chunk_nodes):
                token_count = chunker._count_tokens(node.text)
                char_count = len(node.text)
                
                token_counts.append(token_count)
                char_counts.append(char_count)
                
                if token_count > max_tokens:
                    oversized_chunks += 1
                    logger.warning(f"⚠️ Chunk {i} 仍然超过限制: {token_count} tokens")
                else:
                    logger.debug(f"✅ Chunk {i}: {token_count} tokens, {char_count} 字符")
            
            # 统计信息
            logger.info(f"\n📈 Chunk大小统计:")
            logger.info(f"   - 总chunk数: {len(chunk_nodes)}")
            logger.info(f"   - 平均token数: {sum(token_counts) / len(token_counts):.1f}")
            logger.info(f"   - 最大token数: {max(token_counts)}")
            logger.info(f"   - 最小token数: {min(token_counts)}")
            logger.info(f"   - 超过限制的chunk数: {oversized_chunks}")
            logger.info(f"   - 成功率: {((len(chunk_nodes) - oversized_chunks) / len(chunk_nodes) * 100):.1f}%")
            
            # 检查索引创建
            logger.info(f"\n🔍 步骤6: 检查向量索引...")
            if chunker.chunk_index is not None:
                logger.info("✅ 向量索引创建成功")
            else:
                logger.error("❌ 向量索引创建失败")
            
        else:
            logger.error("❌ 没有生成任何chunk")
            return False
        
        # 6. 测试结论
        logger.info(f"\n🎉 测试结论:")
        if oversized_chunks == 0 and chunker.chunk_index is not None:
            logger.info("✅ Token限制修复测试通过！")
            logger.info("   - 所有chunk都在token限制内")
            logger.info("   - 向量索引创建成功")
            logger.info("   - 没有出现'NoneType' object is not iterable错误")
            return True
        else:
            logger.error("❌ Token限制修复测试失败")
            if oversized_chunks > 0:
                logger.error(f"   - 仍有 {oversized_chunks} 个chunk超过限制")
            if chunker.chunk_index is None:
                logger.error("   - 向量索引创建失败")
            return False
        
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
        import traceback
        logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        return False

def main():
    """主函数"""
    print("🧪 Token限制修复测试")
    print("=" * 40)
    
    success = test_token_limit_fix()
    
    if success:
        print("\n🎉 测试成功！修复生效。")
        sys.exit(0)
    else:
        print("\n❌ 测试失败！需要进一步调试。")
        sys.exit(1)

if __name__ == "__main__":
    main() 