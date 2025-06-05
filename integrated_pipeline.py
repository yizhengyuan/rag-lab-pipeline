"""
整合版 Pipeline - 将概念提取和问答对生成合并在一个脚本中
==============================================================

功能：
1. 输入 PDF 或其他格式文档
2. 使用 pipeline.py 的 ImprovedConceptBasedPipeline 进行概念提取
3. 使用 data_generate_0526.py 的 SimpleDataGenerator 生成问答对
4. 输出完整的训练数据
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

# 默认配置
DEFAULT_CONFIG = {
    "llm_model": "gpt-4o-mini",
    "OPENAI_API_KEY": "sk-zk2884399e3bbb43b998bd31be7b517f82f67bb0e95df2a1",
    "BASE_URL": "https://api.zhizengzeng.com/v1/"
}

# 设置环境变量
os.environ["OPENAI_API_KEY"] = DEFAULT_CONFIG["OPENAI_API_KEY"]
if DEFAULT_CONFIG["BASE_URL"]:
    os.environ["BASE_URL"] = DEFAULT_CONFIG["BASE_URL"]

# 导入 LlamaIndex 组件
from llama_index.core import SimpleDirectoryReader, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 导入 pipeline.py 的类和函数
from pipeline import ImprovedConceptBasedPipeline

# 导入 data_generate_0526.py 的类和函数
from data_generate_0526 import (
    SimpleDataGenerator, 
    FileProcessor, 
    QUESTIONS_PER_TYPE
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedPipeline:
    """整合版 Pipeline - 概念提取 + 问答对生成"""
    
    def __init__(self, 
                 openai_api_key: str = None,
                 base_url: str = None,
                 model_name: str = None,
                 embedding_model: str = "text-embedding-ada-002",
                 output_dir: str = "./integrated_output",
                 questions_per_type: Dict[str, int] = None):
        """
        初始化整合 Pipeline
        
        Args:
            openai_api_key: OpenAI API 密钥（如果不提供则使用默认配置）
            base_url: API 基础 URL（如果不提供则使用默认配置）
            model_name: 用于问答生成的模型名称（如果不提供则使用默认配置）
            embedding_model: 用于概念提取的嵌入模型
            output_dir: 输出目录
            questions_per_type: 每种认知层次的问题数量配置
        """
        # 使用提供的参数或默认配置
        self.openai_api_key = openai_api_key or DEFAULT_CONFIG["OPENAI_API_KEY"]
        self.base_url = base_url or DEFAULT_CONFIG["BASE_URL"]
        self.model_name = model_name or DEFAULT_CONFIG["llm_model"]
        self.embedding_model = embedding_model
        self.output_dir = output_dir
        self.questions_per_type = questions_per_type or QUESTIONS_PER_TYPE
        
        # 设置环境变量
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.base_url:
            os.environ["BASE_URL"] = self.base_url
        
        logger.info(f"🔧 配置信息:")
        logger.info(f"   - 模型: {self.model_name}")
        logger.info(f"   - API Key: {self.openai_api_key[:10]}...{self.openai_api_key[-4:]}")
        logger.info(f"   - Base URL: {self.base_url}")
        logger.info(f"   - 嵌入模型: {self.embedding_model}")
        logger.info(f"   - 输出目录: {self.output_dir}")
        
        # 配置 LlamaIndex 全局设置，确保使用自定义 base_url
        self._setup_llamaindex_settings()
        
        # 初始化两个核心组件
        self.concept_pipeline = ImprovedConceptBasedPipeline(
            openai_api_key=self.openai_api_key,
            model_name=self.model_name,
            embedding_model=self.embedding_model
        )
        
        self.qa_generator = SimpleDataGenerator(
            model_name=self.model_name,
            questions_per_type=self.questions_per_type
        )
        
        self.file_processor = FileProcessor()
        
        # 确保输出目录存在
        self.ensure_output_dirs()
    
    def _setup_llamaindex_settings(self):
        """配置 LlamaIndex 全局设置，确保使用自定义 base_url"""
        try:
            # 创建带有自定义 base_url 的 LLM 和嵌入模型
            if self.base_url:
                # 配置 LLM
                Settings.llm = OpenAI(
                    model=self.model_name,
                    api_key=self.openai_api_key,
                    api_base=self.base_url,
                    temperature=0.1
                )
                
                # 配置嵌入模型
                Settings.embed_model = OpenAIEmbedding(
                    model=self.embedding_model,
                    api_key=self.openai_api_key,
                    api_base=self.base_url
                )
                
                logger.info(f"   ✅ LlamaIndex 配置完成，使用自定义 API: {self.base_url}")
            else:
                # 使用默认配置
                Settings.llm = OpenAI(
                    model=self.model_name,
                    api_key=self.openai_api_key,
                    temperature=0.1
                )
                
                Settings.embed_model = OpenAIEmbedding(
                    model=self.embedding_model,
                    api_key=self.openai_api_key
                )
                
                logger.info(f"   ✅ LlamaIndex 配置完成，使用默认 OpenAI API")
                
        except Exception as e:
            logger.warning(f"   ⚠️ LlamaIndex 配置警告: {e}")
            logger.info(f"   🔄 将使用环境变量配置")
    
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
            logger.info("🔬 步骤2: 执行概念提取 Pipeline...")
            try:
                concept_results = self.concept_pipeline.run_pipeline(documents)
                self.concept_pipeline.save_results(concept_results, f"{base_name}_concepts.json")
                
                logger.info(f"   ✅ 概念提取完成:")
                logger.info(f"      - Chunks: {len(concept_results['chunk_nodes'])}")
                logger.info(f"      - Concepts: {len(concept_results['concept_nodes'])}")
                logger.info(f"      - Evidence: {len(concept_results['evidence_nodes'])}")
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
                    "embedding_model": self.embedding_model,
                    "base_url": self.base_url,
                    "questions_per_type": self.questions_per_type
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
    
    def process_directory(self, dir_path: str, file_extensions: List[str] = None) -> Dict[str, Any]:
        """
        处理目录中的所有文件
        
        Args:
            dir_path: 目录路径
            file_extensions: 支持的文件扩展名列表
            
        Returns:
            包含所有结果的字典
        """
        if file_extensions is None:
            file_extensions = ['.pdf', '.txt', '.md', '.json', '.jsonl']
        
        logger.info(f"🚀 开始处理目录: {dir_path}")
        logger.info(f"📁 支持的文件类型: {file_extensions}")
        logger.info("=" * 80)
        
        all_results = []
        all_training_data = []
        
        # 遍历目录中的所有文件
        for root, _, files in os.walk(dir_path):
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        result = self.process_single_file(file_path, save_intermediate=True)
                        if result:
                            all_results.append(result)
                            all_training_data.extend(result.get("training_data", []))
                    except Exception as e:
                        logger.error(f"处理文件失败 {file_path}: {e}")
                        continue
        
        # 保存合并的结果
        combined_results = {
            "directory_info": {
                "directory_path": dir_path,
                "total_files": len(all_results),
                "total_training_data": len(all_training_data)
            },
            "config_info": {
                "model_name": self.model_name,
                "embedding_model": self.embedding_model,
                "base_url": self.base_url,
                "questions_per_type": self.questions_per_type
            },
            "individual_results": all_results,
            "combined_training_data": all_training_data,
            "processing_time": datetime.now().isoformat()
        }
        
        self._save_combined_results(combined_results)
        
        logger.info(f"✅ 目录处理完成: {dir_path}")
        logger.info(f"📈 总体统计:")
        logger.info(f"   - 处理文件数: {len(all_results)}")
        logger.info(f"   - 总训练数据条数: {len(all_training_data)}")
        
        return combined_results
    
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
    
    def _save_combined_results(self, combined_results: Dict[str, Any]):
        """保存合并的结果"""
        # 保存合并的训练数据
        combined_training_path = os.path.join(self.output_dir, "combined_training_data.jsonl")
        with open(combined_training_path, 'w', encoding='utf-8') as f:
            for data in combined_results["combined_training_data"]:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        # 保存完整的合并结果
        combined_summary_path = os.path.join(self.output_dir, "combined_summary.json")
        with open(combined_summary_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"   ✅ 合并结果已保存:")
        logger.info(f"      - 合并训练数据: {combined_training_path}")
        logger.info(f"      - 合并摘要: {combined_summary_path}")

def add_cli():
    """添加命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='整合版 Pipeline - 概念提取 + 问答对生成')
    
    # 基本参数
    parser.add_argument('--input', '-i', required=True, help='输入文件或目录路径')
    parser.add_argument('--output', '-o', default='./integrated_output', help='输出目录路径')
    parser.add_argument('--api-key', '-k', default=DEFAULT_CONFIG["OPENAI_API_KEY"], help=f'OpenAI API 密钥（默认: {DEFAULT_CONFIG["OPENAI_API_KEY"][:10]}...）')
    parser.add_argument('--base-url', '-b', default=DEFAULT_CONFIG["BASE_URL"], help=f'API 基础 URL（默认: {DEFAULT_CONFIG["BASE_URL"]}）')
    parser.add_argument('--model', '-m', default=DEFAULT_CONFIG["llm_model"], help=f'用于问答生成的模型名称（默认: {DEFAULT_CONFIG["llm_model"]}）')
    parser.add_argument('--embedding-model', '-e', default='text-embedding-ada-002', help='用于概念提取的嵌入模型')
    
    # 问题数量配置
    parser.add_argument('--remember', type=int, default=2, help='Remember类型问题数量')
    parser.add_argument('--understand', type=int, default=2, help='Understand类型问题数量')
    parser.add_argument('--apply', type=int, default=1, help='Apply类型问题数量')
    parser.add_argument('--analyze', type=int, default=1, help='Analyze类型问题数量')
    parser.add_argument('--evaluate', type=int, default=1, help='Evaluate类型问题数量')
    parser.add_argument('--create', type=int, default=1, help='Create类型问题数量')
    
    # 新增：仅问答生成模式
    parser.add_argument('--qa-only', action='store_true', help='仅执行问答生成，跳过概念提取')
    
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
    pipeline = IntegratedPipeline(
        openai_api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        embedding_model=args.embedding_model,
        output_dir=args.output,
        questions_per_type=questions_per_type
    )
    
    logger.info(f"配置的问题数量: {questions_per_type}")
    if args.qa_only:
        logger.info("⚠️ 仅问答生成模式：将跳过概念提取步骤")
    
    # 处理输入
    if os.path.isdir(args.input):
        pipeline.process_directory(args.input)
    else:
        pipeline.process_single_file(args.input)

# 使用示例
def example_usage():
    """使用示例"""
    # 使用默认配置创建 Pipeline（无需提供 API 密钥）
    pipeline = IntegratedPipeline(
        output_dir="./example_output"
    )
    
    # 处理单个文件
    # results = pipeline.process_single_file("example.pdf")
    
    # 或处理整个目录
    # results = pipeline.process_directory("./documents/")

def quick_start():
    """快速开始示例"""
    logger.info("🚀 快速开始示例")
    logger.info("=" * 50)
    
    # 使用默认配置
    pipeline = IntegratedPipeline()
    
    logger.info("✅ Pipeline 初始化完成，使用默认配置:")
    logger.info(f"   - 模型: {DEFAULT_CONFIG['llm_model']}")
    logger.info(f"   - API URL: {DEFAULT_CONFIG['BASE_URL']}")
    logger.info("现在可以开始处理文档了！")

if __name__ == "__main__":
    add_cli()