"""
整合版 Pipeline - 使用新的模块化结构
==============================================================

功能：
1. 输入 PDF 或其他格式文档
2. 使用模块化的 ImprovedConceptBasedPipeline 进行概念提取
3. 使用 data_generate_0526.py 的 SimpleDataGenerator 生成问答对
4. 输出完整的训练数据

更新：使用新的模块化结构，支持配置文件
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

# 使用新的模块化 Pipeline
from pipeline_new import ImprovedConceptBasedPipeline
from config import load_config_from_yaml

# 导入 LlamaIndex 组件
from llama_index.core import SimpleDirectoryReader, Document

# 导入 data_generate_0526.py 的类和函数
from data_generate_0526 import (
    SimpleDataGenerator, 
    FileProcessor, 
    QUESTIONS_PER_TYPE
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModularIntegratedPipeline:
    """整合版 Pipeline - 使用模块化结构"""
    
    def __init__(self, 
                 config_path: str = "config/config.yml",
                 output_dir: str = "./integrated_output",
                 questions_per_type: Dict[str, int] = None,
                 # 向后兼容参数
                 openai_api_key: str = None,
                 base_url: str = None,
                 model_name: str = None):
        """
        初始化整合 Pipeline
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
            questions_per_type: 每种认知层次的问题数量配置
            openai_api_key: OpenAI API 密钥（向后兼容）
            base_url: API 基础 URL（向后兼容）
            model_name: 模型名称（向后兼容）
        """
        self.output_dir = output_dir
        self.questions_per_type = questions_per_type or QUESTIONS_PER_TYPE
        
        logger.info("🔧 初始化模块化整合 Pipeline")
        
        # 初始化概念提取 Pipeline
        if openai_api_key or base_url or model_name:
            # 向后兼容模式
            self.concept_pipeline = ImprovedConceptBasedPipeline(
                openai_api_key=openai_api_key,
                base_url=base_url,
                model_name=model_name
            )
            # 从 Pipeline 配置中获取模型信息
            self.model_name = self.concept_pipeline.config.model_name
        else:
            # 新的配置文件模式
            self.concept_pipeline = ImprovedConceptBasedPipeline(config_path=config_path)
            self.model_name = self.concept_pipeline.config.model_name
        
        # 初始化问答生成器
        # 使用概念管道的配置来初始化问答生成器
        # 临时设置环境变量以传递配置
        os.environ['OPENAI_API_KEY'] = self.concept_pipeline.config.openai_api_key
        if hasattr(self.concept_pipeline.config, 'base_url'):
            os.environ['BASE_URL'] = self.concept_pipeline.config.base_url
        
        self.qa_generator = SimpleDataGenerator(
            model_name=self.model_name,
            questions_per_type=self.questions_per_type
        )
        
        self.file_processor = FileProcessor()
        
        # 确保输出目录存在
        self.ensure_output_dirs()
        
        logger.info(f"   ✅ 使用模型: {self.model_name}")
        logger.info(f"   ✅ 输出目录: {self.output_dir}")
    
    def ensure_output_dirs(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def process_single_file(self, file_path: str, save_intermediate: bool = True) -> Dict[str, Any]:
        """
        处理单个文件的完整流程
        
        Args:
            file_path: 输入文件路径
            save_intermediate: 是否保存中间结果
            
        Returns:
            包含所有结果的字典
        """
        try:
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            
            logger.info(f"🚀 开始处理文件: {file_path}")
            logger.info(f"📁 输出目录: {self.output_dir}")
            logger.info("=" * 80)
            
            # 步骤1: 加载文档
            logger.info("📄 步骤1: 加载文档...")
            documents = self._load_document(file_path)
            
            # 提取完整文本用于问答生成
            full_text = "\n\n".join([doc.text for doc in documents])
            logger.info(f"   ✅ 文档加载完成，总字符数: {len(full_text)}")
            
            # 步骤2: 概念提取 Pipeline
            logger.info("🔬 步骤2: 执行模块化概念提取 Pipeline...")
            try:
                concept_results = self.concept_pipeline.run_pipeline(documents)
                
                logger.info(f"   ✅ 概念提取完成:")
                logger.info(f"      - Chunks: {len(concept_results['chunk_nodes'])}")
                logger.info(f"      - Concepts: {len(concept_results['concept_nodes'])}")
                logger.info(f"      - Evidence: {len(concept_results['evidence_nodes'])}")
                
                # 获取 Pipeline 统计信息
                stats = self.concept_pipeline.get_pipeline_statistics()
                logger.info(f"   📊 Pipeline 统计: {stats}")
                
            except Exception as e:
                logger.error(f"   ❌ 概念提取失败: {e}")
                logger.info(f"   🔄 跳过概念提取，继续问答生成...")
                concept_results = {
                    "chunk_nodes": [],
                    "concept_nodes": [],
                    "concept_to_chunks": {},
                    "evidence_nodes": [],
                    "indexes": {}
                }
            
            # 步骤3: 问答对生成
            logger.info("❓ 步骤3: 生成问答对...")
            qa_pairs = self.qa_generator.generate_qa_pairs_from_text(full_text)
            logger.info(f"   ✅ 问答对生成完成，共 {len(qa_pairs)} 个")
            
            # 步骤4: 创建训练数据
            logger.info("📊 步骤4: 创建训练数据...")
            training_data = []
            for qa_pair in qa_pairs:
                training_item = self.qa_generator.create_training_data(qa_pair, full_text, file_name)
                if training_item:
                    training_data.append(training_item)
            logger.info(f"   ✅ 训练数据创建完成，共 {len(training_data)} 条")
            
            # 步骤5: 保存结果
            logger.info("💾 步骤5: 保存结果...")
            results = {
                "file_info": {
                    "file_path": file_path,
                    "file_name": file_name,
                    "base_name": base_name,
                    "text_length": len(full_text)
                },
                "config_info": {
                    "model_name": self.model_name,
                    "questions_per_type": self.questions_per_type,
                    "pipeline_stats": stats if 'stats' in locals() else {}
                },
                "concept_results": concept_results,
                "qa_pairs": qa_pairs,
                "training_data": training_data,
                "processing_time": datetime.now().isoformat()
            }
            
            if save_intermediate:
                self._save_results(results, base_name)
            
            logger.info(f"✅ 文件处理完成: {file_path}")
            logger.info(f"📈 生成统计:")
            logger.info(f"   - 概念数量: {len(concept_results['concept_nodes'])}")
            logger.info(f"   - 证据数量: {len(concept_results['evidence_nodes'])}")
            logger.info(f"   - 问答对数量: {len(qa_pairs)}")
            logger.info(f"   - 训练数据条数: {len(training_data)}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 处理文件失败 {file_path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def _load_document(self, file_path: str) -> List[Document]:
        """加载文档"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            # 对于 PDF，使用 LlamaIndex 的 SimpleDirectoryReader
            reader = SimpleDirectoryReader(input_files=[file_path])
            return reader.load_data()
        else:
            # 对于其他格式，使用 FileProcessor 提取文本，然后创建 Document
            text = self.file_processor.extract_text(file_path)
            return [Document(text=text, metadata={"file_path": file_path})]
    
    def _save_results(self, results: Dict[str, Any], base_name: str):
        """保存单个文件的结果"""
        # 保存概念提取结果（如果有的话）
        if results["concept_results"]["concept_nodes"]:
            concept_output_path = os.path.join(self.output_dir, f"{base_name}_concepts.json")
            self.concept_pipeline.save_results(results["concept_results"], concept_output_path)
        
        # 保存问答对（JSON格式）
        qa_output_path = os.path.join(self.output_dir, f"{base_name}_qa_pairs.json")
        with open(qa_output_path, 'w', encoding='utf-8') as f:
            json.dump(results["qa_pairs"], f, ensure_ascii=False, indent=2)
        
        # 保存训练数据（JSONL格式）
        training_output_path = os.path.join(self.output_dir, f"{base_name}_training.jsonl")
        with open(training_output_path, 'w', encoding='utf-8') as f:
            for data in results["training_data"]:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        # 保存完整结果摘要
        summary_output_path = os.path.join(self.output_dir, f"{base_name}_summary.json")
        summary = {
            "file_info": results["file_info"],
            "config_info": results["config_info"],
            "statistics": {
                "concepts_count": len(results["concept_results"]["concept_nodes"]),
                "evidence_count": len(results["concept_results"]["evidence_nodes"]),
                "qa_pairs_count": len(results["qa_pairs"]),
                "training_data_count": len(results["training_data"])
            },
            "processing_time": results["processing_time"]
        }
        with open(summary_output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"   ✅ 结果已保存:")
        if results["concept_results"]["concept_nodes"]:
            logger.info(f"      - 概念提取: {concept_output_path}")
        logger.info(f"      - 问答对: {qa_output_path}")
        logger.info(f"      - 训练数据: {training_output_path}")
        logger.info(f"      - 摘要: {summary_output_path}")

def add_cli():
    """添加命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模块化整合版 Pipeline - 概念提取 + 问答对生成')
    
    # 基本参数
    parser.add_argument('--input', '-i', required=True, help='输入文件或目录路径')
    parser.add_argument('--output', '-o', default='./integrated_output', help='输出目录路径')
    parser.add_argument('--config', '-c', default='config/config.yml', help='配置文件路径')
    
    # 向后兼容参数
    parser.add_argument('--api-key', '-k', help='OpenAI API 密钥（向后兼容）')
    parser.add_argument('--base-url', '-b', help='API 基础 URL（向后兼容）')
    parser.add_argument('--model', '-m', help='模型名称（向后兼容）')
    
    # 问题数量配置
    parser.add_argument('--remember', type=int, default=2, help='Remember类型问题数量')
    parser.add_argument('--understand', type=int, default=2, help='Understand类型问题数量')
    parser.add_argument('--apply', type=int, default=1, help='Apply类型问题数量')
    parser.add_argument('--analyze', type=int, default=1, help='Analyze类型问题数量')
    parser.add_argument('--evaluate', type=int, default=1, help='Evaluate类型问题数量')
    parser.add_argument('--create', type=int, default=1, help='Create类型问题数量')
    
    args = parser.parse_args()
    
    # 构建问题数量配置
    questions_per_type = {
        "remember": args.remember,
        "understand": args.understand,
        "apply": args.apply,
        "analyze": args.analyze,
        "evaluate": args.evaluate,
        "create": args.create
    }
    
    # 创建整合 Pipeline
    pipeline = ModularIntegratedPipeline(
        config_path=args.config,
        output_dir=args.output,
        questions_per_type=questions_per_type,
        # 向后兼容参数
        openai_api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model
    )
    
    logger.info(f"配置的问题数量: {questions_per_type}")
    
    # 处理输入
    if os.path.isdir(args.input):
        # 这里可以添加目录处理逻辑
        logger.info("目录处理功能待实现")
    else:
        pipeline.process_single_file(args.input)

if __name__ == "__main__":
    add_cli() 