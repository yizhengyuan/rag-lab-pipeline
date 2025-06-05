"""
SimpleQA History数据集 Pipeline测试脚本 - 修复版 + QA生成
==========================================================

修复内容：
- 修复文档加载问题
- 改为手动读取markdown文件
- 添加更详细的错误处理
- 整合问答生成功能：概念提取 -> 证据提取 -> 问答对生成
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import traceback
import uuid
from PyPDF2 import PdfReader  # 🆕 添加PDF处理支持

# 导入pipeline相关模块
from pipeline_new import ImprovedConceptBasedPipeline
from llama_index.core import Document

# 🆕 导入问答生成模块
from data_generate_0526 import SimpleDataGenerator, FileProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simpleqa_test_fixed.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleQATestSuiteWithQAGeneration:
    """SimpleQA数据集测试套件 - 修复版 + QA生成功能 + PDF支持"""
    
    def __init__(self, 
                 data_dir: str = "simpleqa_histroy_md",
                 output_dir: str = "simpleqa_test_results_fixed",
                 config_path: str = None,
                 enable_qa_generation: bool = True,
                 qa_model_name: str = "gpt-4o-mini",
                 questions_per_type: Dict[str, int] = None,
                 # 🆕 新增参数
                 input_file: str = None):
        """
        初始化测试套件
        
        Args:
            data_dir: SimpleQA数据集目录
            output_dir: 测试结果输出目录
            config_path: Pipeline配置文件路径
            enable_qa_generation: 是否启用问答生成
            qa_model_name: 问答生成使用的模型
            questions_per_type: 每种类型问题的数量
            input_file: 🆕 单个输入文件路径 (支持PDF/TXT/MD)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        self.enable_qa_generation = enable_qa_generation
        self.input_file = Path(input_file) if input_file else None  # 🆕
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化pipeline
        if config_path:
            self.pipeline = ImprovedConceptBasedPipeline(config_path=config_path)
        else:
            self.pipeline = ImprovedConceptBasedPipeline()
        
        # 🆕 初始化问答生成器
        if self.enable_qa_generation:
            self.questions_per_type = questions_per_type or {
                "remember": 2,
                "understand": 2,
                "apply": 1,
                "analyze": 1,
                "evaluate": 1,
                "create": 1
            }
            self.qa_generator = SimpleDataGenerator(
                model_name=qa_model_name,
                questions_per_type=self.questions_per_type
            )
            logger.info(f"✅ 问答生成器已启用: {qa_model_name}")
            logger.info(f"   问题配置: {self.questions_per_type}")
        else:
            self.qa_generator = None
            logger.info("❌ 问答生成器已禁用")
        
        # 测试统计
        self.test_stats = {
            "total_questions": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "total_documents": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            # 🆕 QA生成统计
            "total_qa_pairs": 0,
            "qa_generation_time": 0.0,
            "qa_generation_errors": 0
        }
        
        # 🆕 确定处理模式
        if self.input_file and self.input_file.exists():
            logger.info(f"📄 单文件处理模式: {self.input_file}")
            logger.info(f"输出目录: {self.output_dir}")
        else:
            logger.info(f"📁 批量处理模式: {self.data_dir}")
            logger.info(f"输出目录: {self.output_dir}")
    
    def get_available_questions(self) -> List[str]:
        """获取所有可用的问题文件夹"""
        questions = []
        if self.data_dir.exists():
            for item in self.data_dir.iterdir():
                if item.is_dir() and item.name.startswith("question_"):
                    questions.append(item.name)
        return sorted(questions)
    
    def load_question_documents(self, question_folder: str) -> List[Document]:
        """
        加载指定问题文件夹中的所有文档 - 修复版
        
        Args:
            question_folder: 问题文件夹名称 (如 'question_100')
            
        Returns:
            Document列表
        """
        question_path = self.data_dir / question_folder
        if not question_path.exists():
            raise ValueError(f"问题文件夹不存在: {question_path}")
        
        documents = []
        
        # 手动遍历并加载markdown文件
        md_files = list(question_path.glob("*.md"))
        logger.info(f"在 {question_folder} 中找到 {len(md_files)} 个markdown文件")
        
        for md_file in md_files:
            try:
                # 直接读取文件内容
                with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                
                # 检查内容是否为空
                if not content:
                    logger.warning(f"文件 {md_file.name} 内容为空，跳过")
                    continue
                
                # 创建Document对象
                doc = Document(
                    text=content,
                    metadata={
                        "file_name": md_file.name,
                        "file_path": str(md_file),
                        "question_folder": question_folder,
                        "source_dataset": "simpleqa_history",
                        "file_size": len(content)
                    }
                )
                documents.append(doc)
                logger.debug(f"✅ 成功加载文件: {md_file.name} ({len(content)} 字符)")
                
            except Exception as e:
                logger.warning(f"❌ 加载文件失败 {md_file.name}: {e}")
                continue
        
        if not documents:
            logger.warning(f"⚠️ 问题 {question_folder} 未加载到任何有效文档")
        else:
            total_chars = sum(len(doc.text) for doc in documents)
            logger.info(f"✅ 问题 {question_folder}: 成功加载 {len(documents)} 个文档，总计 {total_chars} 字符")
        
        return documents
    
    def extract_evidence_texts(self, pipeline_results: Dict[str, Any]) -> List[str]:
        """
        🆕 从Pipeline结果中提取Evidence文本
        
        Args:
            pipeline_results: Pipeline处理结果
            
        Returns:
            Evidence文本列表
        """
        logger.info("📝 提取Evidence文本用于QA生成...")
        
        evidence_texts = []
        evidence_nodes = pipeline_results.get("evidence_nodes", [])
        
        for evidence_node in evidence_nodes:
            try:
                # 获取evidence文本
                evidence_text = evidence_node.text
                if evidence_text and len(evidence_text.strip()) > 20:  # 最少20字符
                    evidence_texts.append(evidence_text.strip())
                    logger.debug(f"提取Evidence: {evidence_text[:100]}...")
            except Exception as e:
                logger.warning(f"提取Evidence文本失败: {e}")
        
        logger.info(f"✅ 提取了 {len(evidence_texts)} 个Evidence文本")
        return evidence_texts
    
    def generate_qa_pairs(self, evidence_texts: List[str], question_folder: str) -> List[Dict[str, Any]]:
        """
        🆕 基于Evidence文本生成问答对
        
        Args:
            evidence_texts: Evidence文本列表
            question_folder: 问题文件夹名称
            
        Returns:
            问答对列表
        """
        if not self.enable_qa_generation or not evidence_texts:
            return []
        
        logger.info("❓ 基于Evidence生成问答对...")
        
        all_qa_pairs = []
        qa_start_time = time.time()
        
        for i, evidence_text in enumerate(evidence_texts):
            try:
                logger.info(f"   处理Evidence {i+1}/{len(evidence_texts)} ({len(evidence_text)} 字符)...")
                
                # 为每个Evidence生成问答对
                qa_pairs = self.qa_generator.generate_qa_pairs_from_text(evidence_text)
                
                # 为每个问答对添加来源信息
                for qa_pair in qa_pairs:
                    qa_pair["evidence_source"] = f"{question_folder}_evidence_{i}"
                    qa_pair["evidence_text"] = evidence_text[:200] + "..." if len(evidence_text) > 200 else evidence_text
                    qa_pair["question_folder"] = question_folder
                    qa_pair["generation_timestamp"] = datetime.now().isoformat()
                
                all_qa_pairs.extend(qa_pairs)
                logger.info(f"   从Evidence {i+1} 生成了 {len(qa_pairs)} 个问答对")
                
            except Exception as e:
                logger.error(f"从Evidence {i+1} 生成问答对失败: {e}")
                self.test_stats["qa_generation_errors"] += 1
                continue
        
        qa_time = time.time() - qa_start_time
        self.test_stats["total_qa_pairs"] += len(all_qa_pairs)
        self.test_stats["qa_generation_time"] += qa_time
        
        logger.info(f"✅ 生成 {len(all_qa_pairs)} 个问答对，耗时: {qa_time:.2f}秒")
        return all_qa_pairs
    
    def create_training_data(self, qa_pairs: List[Dict[str, Any]], question_folder: str, 
                            pipeline_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        🆕 创建训练数据格式 - 符合标准格式
        
        Args:
            qa_pairs: 问答对列表
            question_folder: 问题文件夹名称
            pipeline_results: Pipeline处理结果（用于提取Context）
            
        Returns:
            训练数据列表
        """
        training_data = []
        
        # 获取pipeline相关数据
        chunk_nodes = pipeline_results.get("chunk_nodes", [])
        evidence_nodes = pipeline_results.get("evidence_nodes", [])
        concept_to_chunks = pipeline_results.get("concept_to_chunks", {})
        
        for i, qa_pair in enumerate(qa_pairs):
            try:
                # 生成唯一ID
                unique_id = uuid.uuid4().hex
                
                # 构建Context - 基于Evidence和相关的Chunks
                context = self._build_context_for_qa(
                    qa_pair, 
                    chunk_nodes, 
                    evidence_nodes, 
                    question_folder, 
                    i
                )
                
                # 创建标准格式的训练数据项
                training_item = {
                    "_id": unique_id,
                    "Question": qa_pair["question"],
                    "Answer": qa_pair["answer"],
                    "Type": qa_pair["type"],
                    "Difficulty": qa_pair["difficulty"],
                    "Domain": getattr(self.qa_generator, 'domain', 'general'),
                    "Rationale": qa_pair.get("rationale", ""),
                    "Context": context
                }
                
                training_data.append(training_item)
                
            except Exception as e:
                logger.warning(f"创建训练数据项 {i} 失败: {e}")
                continue
        
        logger.info(f"✅ 创建了 {len(training_data)} 个标准格式训练数据项")
        return training_data
    
    def _build_context_for_qa(self, qa_pair: Dict[str, Any], chunk_nodes: List, 
                             evidence_nodes: List, question_folder: str, qa_index: int) -> List[Dict[str, Any]]:
        """
        🆕 为问答对构建Context数组
        
        Args:
            qa_pair: 问答对信息
            chunk_nodes: 文档块节点列表
            evidence_nodes: 证据节点列表
            question_folder: 问题文件夹名称
            qa_index: 问答对索引
            
        Returns:
            Context数组
        """
        context = []
        chunk_id_counter = 0
        doc_id = f"{question_folder}_doc_{qa_index}"
        
        # 1. 处理与当前QA相关的Evidence作为支持性chunk
        qa_evidence_source = qa_pair.get("evidence_source", "")
        related_evidence = None
        
        for evidence_node in evidence_nodes:
            if qa_evidence_source in str(evidence_node.metadata.get("source_chunks", [])):
                related_evidence = evidence_node
                break
        
        if related_evidence:
            # 添加支持性Evidence chunk
            chunk_id = f"chunk_{chunk_id_counter}"
            evidence_keywords = self._extract_keywords_from_text(
                qa_pair["question"], 
                related_evidence.text
            )
            
            context.append({
                "chunk_id": chunk_id,
                "content": related_evidence.text,
                "type": "fully_support",
                "reason": "Evidence extracted by pipeline that directly supports the question",
                "keywords": evidence_keywords,
                "DocId": doc_id
            })
            chunk_id_counter += 1
        
        # 2. 添加相关的文档chunks作为部分支持
        max_support_chunks = 2
        added_support_chunks = 0
        
        for chunk_node in chunk_nodes[:5]:  # 限制检查前5个chunk
            if added_support_chunks >= max_support_chunks:
                break
            
            # 检查chunk是否与问题相关
            if self._is_chunk_relevant_to_question(qa_pair["question"], chunk_node.text):
                chunk_id = f"chunk_{chunk_id_counter}"
                chunk_keywords = self._extract_keywords_from_text(
                    qa_pair["question"], 
                    chunk_node.text
                )
                
                context.append({
                    "chunk_id": chunk_id,
                    "content": chunk_node.text,
                    "type": "partial_support",
                    "reason": "Document chunk with relevant information to the question topic",
                    "keywords": chunk_keywords,
                    "DocId": doc_id
                })
                chunk_id_counter += 1
                added_support_chunks += 1
        
        # 3. 添加干扰性chunks（confused和irrelevant）
        # 从剩余chunks中选择一些作为干扰项
        remaining_chunks = chunk_nodes[5:8]  # 选择中间的一些chunk作为干扰
        
        for i, chunk_node in enumerate(remaining_chunks):
            chunk_id = f"chunk_{chunk_id_counter}"
            
            # 交替添加confused和irrelevant类型
            if i % 2 == 0:
                chunk_type = "confused"
                reason = "Contains similar terms but doesn't directly answer the question"
            else:
                chunk_type = "irrelevant"
                reason = "Contains unrelated information to the question"
            
            context.append({
                "chunk_id": chunk_id,
                "content": chunk_node.text,
                "type": chunk_type,
                "reason": reason,
                "keywords": [],  # 干扰chunk通常不提供关键词
                "DocId": doc_id
            })
            chunk_id_counter += 1
        
        return context
    
    def _extract_keywords_from_text(self, question: str, text: str) -> List[str]:
        """
        🆕 从文本中提取与问题相关的关键词
        
        Args:
            question: 问题文本
            text: 要提取关键词的文本
            
        Returns:
            关键词列表
        """
        try:
            # 简化的关键词提取逻辑
            # 可以集成到QA生成器中使用更复杂的方法
            
            # 移除常见停用词
            stop_words = set(['的', '是', '在', '和', '与', '或', '但', '而且', '这', '那', '一个', '我们', '他们'])
            
            # 从问题中提取关键词作为参考
            question_words = set()
            for word in question.replace('？', '').replace('?', '').split():
                if len(word) > 1 and word not in stop_words:
                    question_words.add(word)
            
            # 从文本中找到与问题相关的词汇
            keywords = []
            for word in text.split():
                if (len(word) > 1 and 
                    word not in stop_words and 
                    (word in question_words or any(qw in word for qw in question_words))):
                    keywords.append(word)
            
            # 去重并限制数量
            keywords = list(set(keywords))[:5]
            
            return keywords
            
        except Exception as e:
            logger.debug(f"提取关键词失败: {e}")
            return []
    
    def _is_chunk_relevant_to_question(self, question: str, chunk_text: str) -> bool:
        """
        🆕 判断chunk是否与问题相关
        
        Args:
            question: 问题文本
            chunk_text: chunk文本
            
        Returns:
            是否相关
        """
        try:
            # 简化的相关性判断
            question_words = set(question.lower().split())
            chunk_words = set(chunk_text.lower().split())
            
            # 计算词汇重叠度
            overlap = len(question_words.intersection(chunk_words))
            relevance_score = overlap / max(len(question_words), 1)
            
            # 设置阈值
            return relevance_score > 0.1
            
        except Exception as e:
            logger.debug(f"相关性判断失败: {e}")
            return False
    
    def load_single_file_as_document(self, file_path: Path) -> List[Document]:
        """
        🆕 加载单个文件为Document对象
        
        Args:
            file_path: 文件路径
            
        Returns:
            Document列表
        """
        logger.info(f"📄 加载单个文件: {file_path}")
        
        if not file_path.exists():
            raise ValueError(f"文件不存在: {file_path}")
        
        try:
            # 使用FileProcessor提取文本
            text_content = FileProcessor.extract_text(str(file_path))
            
            if not text_content or len(text_content.strip()) < 50:
                raise ValueError(f"文件内容太少或为空: {len(text_content)} 字符")
            
            # 创建Document对象
            doc = Document(
                text=text_content,
                metadata={
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_path.suffix.lower(),
                    "file_size": file_path.stat().st_size,
                    "text_length": len(text_content),
                    "source_dataset": "single_file_input",
                    "processing_timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ 成功加载文件: {file_path.name}")
            logger.info(f"   文件大小: {file_path.stat().st_size / 1024:.1f} KB")
            logger.info(f"   文本长度: {len(text_content)} 字符")
            logger.info(f"   文本预览: {text_content[:200]}...")
            
            return [doc]
            
        except Exception as e:
            logger.error(f"❌ 加载文件失败 {file_path}: {e}")
            raise e
    
    def test_single_file(self, file_path: Path = None) -> Dict[str, Any]:
        """
        🆕 测试单个文件 - 支持PDF/TXT/MD
        
        Args:
            file_path: 文件路径，如果为None则使用初始化时的input_file
            
        Returns:
            测试结果字典
        """
        target_file = file_path or self.input_file
        if not target_file:
            raise ValueError("未指定输入文件")
        
        file_name = target_file.stem  # 文件名（不含扩展名）
        logger.info(f"🧪 开始测试文件: {target_file}")
        logger.info("=" * 60)
        
        start_time = time.time()
        test_result = {
            "input_file": str(target_file),
            "file_name": file_name,
            "status": "unknown",
            "start_time": datetime.now().isoformat(),
            "documents_count": 0,
            "processing_time": 0.0,
            "pipeline_results": {},
            "pipeline_stats": {},
            "qa_results": {},
            "error_message": None
        }
        
        try:
            # 1. 加载文件
            logger.info("📄 步骤1: 加载文件...")
            documents = self.load_single_file_as_document(target_file)
            test_result["documents_count"] = len(documents)
            
            if not documents:
                raise ValueError(f"文件 {target_file} 加载失败")
            
            # 记录文档信息
            test_result["document_details"] = [
                {
                    "index": i,
                    "text_length": len(doc.text),
                    "file_type": doc.metadata.get("file_type", "unknown"),
                    "file_size": doc.metadata.get("file_size", 0),
                    "text_preview": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
                }
                for i, doc in enumerate(documents)
            ]
            
            logger.info(f"   ✅ 成功加载 {len(documents)} 个文档")
            
            # 2. 重置Pipeline状态
            self.pipeline.reset_pipeline()
            
            # 3. 运行Pipeline
            logger.info("🔬 步骤2: 运行概念提取和证据提取Pipeline...")
            pipeline_start = time.time()
            
            pipeline_results = self.pipeline.run_pipeline(documents)
            
            pipeline_time = time.time() - pipeline_start
            logger.info(f"   ✅ Pipeline运行完成，耗时: {pipeline_time:.2f}秒")
            
            # 4. 获取统计信息
            pipeline_stats = self.pipeline.get_pipeline_statistics()
            
            # 5. 🆕 QA生成步骤
            qa_pairs = []
            training_data = []
            if self.enable_qa_generation:
                logger.info("❓ 步骤3: 生成问答对...")
                
                # 提取Evidence文本
                evidence_texts = self.extract_evidence_texts(pipeline_results)
                
                if evidence_texts:
                    # 生成问答对
                    qa_pairs = self.generate_qa_pairs(evidence_texts, file_name)
                    
                    # 创建训练数据
                    if qa_pairs:
                        training_data = self.create_training_data(qa_pairs, file_name, pipeline_results)
                else:
                    logger.warning("⚠️ 没有提取到Evidence，跳过QA生成")
            
            # 6. 记录结果
            test_result.update({
                "status": "success",
                "processing_time": time.time() - start_time,
                "pipeline_processing_time": pipeline_time,
                "pipeline_results": {
                    "chunk_count": len(pipeline_results["chunk_nodes"]),
                    "concept_count": len(pipeline_results["concept_nodes"]),
                    "evidence_count": len(pipeline_results["evidence_nodes"]),
                    "concept_to_chunks_mapping": len(pipeline_results.get("concept_to_chunks", {}))
                },
                "pipeline_stats": pipeline_stats,
                # 🆕 QA结果
                "qa_results": {
                    "qa_pairs_count": len(qa_pairs),
                    "training_data_count": len(training_data),
                    "qa_generation_enabled": self.enable_qa_generation,
                    "questions_config": self.questions_per_type if self.enable_qa_generation else None
                }
            })
            
            # 7. 保存详细结果
            self._save_file_results(file_name, pipeline_results, test_result, qa_pairs, training_data)
            
            logger.info(f"✅ 文件 {target_file.name} 测试成功")
            logger.info(f"   📊 Pipeline结果: {test_result['pipeline_results']}")
            if self.enable_qa_generation:
                logger.info(f"   📊 QA结果: {test_result['qa_results']}")
            
            self.test_stats["successful_tests"] += 1
            
        except Exception as e:
            error_msg = f"测试失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            test_result.update({
                "status": "failed",
                "processing_time": time.time() - start_time,
                "error_message": error_msg,
                "traceback": traceback.format_exc()
            })
            
            self.test_stats["failed_tests"] += 1
        
        test_result["end_time"] = datetime.now().isoformat()
        self.test_stats["total_questions"] += 1
        self.test_stats["total_documents"] += test_result["documents_count"]
        self.test_stats["total_processing_time"] += test_result["processing_time"]
        
        return test_result
    
    def test_single_question(self, question_folder: str) -> Dict[str, Any]:
        """
        测试单个问题 - 修复版 + QA生成
        
        Args:
            question_folder: 问题文件夹名称
            
        Returns:
            测试结果字典
        """
        logger.info(f"🧪 开始测试问题: {question_folder}")
        logger.info("=" * 60)
        
        start_time = time.time()
        test_result = {
            "question_folder": question_folder,
            "status": "unknown",
            "start_time": datetime.now().isoformat(),
            "documents_count": 0,
            "processing_time": 0.0,
            "pipeline_results": {},
            "pipeline_stats": {},
            "qa_results": {},  # 🆕 QA生成结果
            "error_message": None
        }
        
        try:
            # 1. 加载文档
            logger.info("📄 步骤1: 加载文档...")
            documents = self.load_question_documents(question_folder)
            test_result["documents_count"] = len(documents)
            
            if not documents:
                raise ValueError(f"问题 {question_folder} 没有加载到任何文档")
            
            # 记录文档信息
            test_result["document_details"] = [
                {
                    "index": i,
                    "text_length": len(doc.text),
                    "source_file": doc.metadata.get("file_name", "unknown"),
                    "text_preview": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
                }
                for i, doc in enumerate(documents)
            ]
            
            logger.info(f"   ✅ 成功加载 {len(documents)} 个文档")
            
            # 2. 重置Pipeline状态
            self.pipeline.reset_pipeline()
            
            # 3. 运行Pipeline
            logger.info("🔬 步骤2: 运行概念提取和证据提取Pipeline...")
            pipeline_start = time.time()
            
            pipeline_results = self.pipeline.run_pipeline(documents)
            
            pipeline_time = time.time() - pipeline_start
            logger.info(f"   ✅ Pipeline运行完成，耗时: {pipeline_time:.2f}秒")
            
            # 4. 获取统计信息
            pipeline_stats = self.pipeline.get_pipeline_statistics()
            
            # 5. 🆕 QA生成步骤
            qa_pairs = []
            training_data = []
            if self.enable_qa_generation:
                logger.info("❓ 步骤3: 生成问答对...")
                
                # 提取Evidence文本
                evidence_texts = self.extract_evidence_texts(pipeline_results)
                
                if evidence_texts:
                    # 生成问答对
                    qa_pairs = self.generate_qa_pairs(evidence_texts, question_folder)
                    
                    # 创建训练数据
                    if qa_pairs:
                        training_data = self.create_training_data(qa_pairs, question_folder, pipeline_results)
                else:
                    logger.warning("⚠️ 没有提取到Evidence，跳过QA生成")
            
            # 6. 记录结果
            test_result.update({
                "status": "success",
                "processing_time": time.time() - start_time,
                "pipeline_processing_time": pipeline_time,
                "pipeline_results": {
                    "chunk_count": len(pipeline_results["chunk_nodes"]),
                    "concept_count": len(pipeline_results["concept_nodes"]),
                    "evidence_count": len(pipeline_results["evidence_nodes"]),
                    "concept_to_chunks_mapping": len(pipeline_results.get("concept_to_chunks", {}))
                },
                "pipeline_stats": pipeline_stats,
                # 🆕 QA结果
                "qa_results": {
                    "qa_pairs_count": len(qa_pairs),
                    "training_data_count": len(training_data),
                    "qa_generation_enabled": self.enable_qa_generation,
                    "questions_config": self.questions_per_type if self.enable_qa_generation else None
                }
            })
            
            # 7. 保存详细结果
            self._save_question_results(question_folder, pipeline_results, test_result, qa_pairs, training_data)
            
            logger.info(f"✅ 问题 {question_folder} 测试成功")
            logger.info(f"   📊 Pipeline结果: {test_result['pipeline_results']}")
            if self.enable_qa_generation:
                logger.info(f"   📊 QA结果: {test_result['qa_results']}")
            
            self.test_stats["successful_tests"] += 1
            
        except Exception as e:
            error_msg = f"测试失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            test_result.update({
                "status": "failed",
                "processing_time": time.time() - start_time,
                "error_message": error_msg,
                "traceback": traceback.format_exc()
            })
            
            self.test_stats["failed_tests"] += 1
        
        test_result["end_time"] = datetime.now().isoformat()
        self.test_stats["total_questions"] += 1
        self.test_stats["total_documents"] += test_result["documents_count"]
        self.test_stats["total_processing_time"] += test_result["processing_time"]
        
        return test_result
    
    def test_multiple_questions(self, question_list: List[str]) -> Dict[str, Any]:
        """
        测试多个问题
        
        Args:
            question_list: 问题文件夹名称列表
            
        Returns:
            批量测试结果
        """
        logger.info(f"🚀 开始批量测试 {len(question_list)} 个问题")
        logger.info("=" * 80)
        
        batch_start_time = time.time()
        batch_results = {
            "batch_info": {
                "start_time": datetime.now().isoformat(),
                "question_count": len(question_list),
                "questions": question_list,
                "qa_generation_enabled": self.enable_qa_generation,
                "questions_config": self.questions_per_type if self.enable_qa_generation else None
            },
            "individual_results": [],
            "batch_summary": {},
            "pipeline_config": self.pipeline.get_pipeline_statistics()["config_summary"]
        }
        
        # 逐个测试问题
        for i, question_folder in enumerate(question_list, 1):
            logger.info(f"\n{'='*20} 测试进度: {i}/{len(question_list)} {'='*20}")
            
            result = self.test_single_question(question_folder)
            batch_results["individual_results"].append(result)
            
            # 每3个问题保存一次中间结果
            if i % 3 == 0 or i == len(question_list):
                self._save_batch_intermediate_results(batch_results, i)
        
        # 计算批量统计
        batch_time = time.time() - batch_start_time
        self.test_stats["average_processing_time"] = (
            self.test_stats["total_processing_time"] / max(self.test_stats["total_questions"], 1)
        )
        
        batch_results["batch_summary"] = {
            "end_time": datetime.now().isoformat(),
            "total_batch_time": batch_time,
            "statistics": self.test_stats.copy(),
            "success_rate": (
                self.test_stats["successful_tests"] / max(self.test_stats["total_questions"], 1) * 100
            )
        }
        
        # 保存最终结果
        self._save_batch_final_results(batch_results)
        
        logger.info(f"\n🎉 批量测试完成!")
        logger.info(f"📊 统计信息:")
        logger.info(f"   - 总问题数: {self.test_stats['total_questions']}")
        logger.info(f"   - 成功: {self.test_stats['successful_tests']}")
        logger.info(f"   - 失败: {self.test_stats['failed_tests']}")
        logger.info(f"   - 成功率: {batch_results['batch_summary']['success_rate']:.1f}%")
        logger.info(f"   - 总处理时间: {batch_time:.2f}秒")
        logger.info(f"   - 平均处理时间: {self.test_stats['average_processing_time']:.2f}秒/问题")
        
        # 🆕 QA生成统计
        if self.enable_qa_generation:
            logger.info(f"   - 生成问答对: {self.test_stats['total_qa_pairs']} 个")
            logger.info(f"   - QA生成时间: {self.test_stats['qa_generation_time']:.2f}秒")
            logger.info(f"   - QA生成错误: {self.test_stats['qa_generation_errors']} 次")
        
        return batch_results
    
    def run_quick_test(self) -> Dict[str, Any]:
        """
        快速测试 - 选择一个代表性问题
        """
        representative_questions = [
            "question_150",  # 书籍相关 (文学历史) - 文档较小，测试友好
        ]
        
        # 检查问题是否存在
        available_questions = self.get_available_questions()
        test_questions = [q for q in representative_questions if q in available_questions]
        
        if not test_questions:
            logger.warning("没有找到代表性问题，使用第一个可用问题")
            test_questions = available_questions[:1]  # 修改为只取第一个
        
        logger.info(f"🎯 快速测试选择的问题: {test_questions}")
        return self.test_multiple_questions(test_questions)
    
    def run_full_test(self) -> Dict[str, Any]:
        """
        完整测试 - 测试所有问题
        """
        all_questions = self.get_available_questions()
        logger.info(f"🔥 完整测试，将测试所有 {len(all_questions)} 个问题")
        return self.test_multiple_questions(all_questions)
    
    def _save_question_results(self, question_folder: str, pipeline_results: Dict, test_result: Dict, 
                              qa_pairs: List[Dict[str, Any]], training_data: List[Dict[str, Any]]):
        """保存单个问题的详细结果 - 包含QA数据"""
        question_output_dir = self.output_dir / question_folder
        question_output_dir.mkdir(exist_ok=True)
        
        # 保存pipeline结果
        pipeline_file = question_output_dir / "pipeline_results.json"
        self.pipeline.save_results(pipeline_results, str(pipeline_file))
        
        # 保存测试元信息
        test_info_file = question_output_dir / "test_info.json"
        with open(test_info_file, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, ensure_ascii=False, indent=2)
        
        # 🆕 保存问答对
        if qa_pairs:
            qa_file = question_output_dir / "qa_pairs.json"
            with open(qa_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 问答对已保存: {qa_file}")
        
        # 🆕 保存训练数据
        if training_data:
            training_file = question_output_dir / "training_data.jsonl"
            with open(training_file, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"💾 训练数据已保存: {training_file}")
    
    def _save_batch_intermediate_results(self, batch_results: Dict, current_index: int):
        """保存批量测试的中间结果"""
        intermediate_file = self.output_dir / f"batch_intermediate_{current_index}.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 已保存中间结果: {intermediate_file}")
    
    def _save_batch_final_results(self, batch_results: Dict):
        """保存批量测试的最终结果 - 包含QA统计"""
        final_file = self.output_dir / "batch_final_results.json"
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        # 🆕 合并所有训练数据
        if self.enable_qa_generation:
            self._combine_all_training_data()
        
        # 生成简化的统计报告
        summary_file = self.output_dir / "test_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("SimpleQA History数据集 Pipeline测试报告 (修复版 + QA生成)\n")
            f.write("=" * 65 + "\n\n")
            f.write(f"测试时间: {batch_results['batch_info']['start_time']}\n")
            f.write(f"测试问题数: {batch_results['batch_info']['question_count']}\n")
            f.write(f"成功: {self.test_stats['successful_tests']}\n")
            f.write(f"失败: {self.test_stats['failed_tests']}\n")
            f.write(f"成功率: {batch_results['batch_summary']['success_rate']:.1f}%\n")
            f.write(f"总处理时间: {batch_results['batch_summary']['total_batch_time']:.2f}秒\n")
            f.write(f"平均处理时间: {self.test_stats['average_processing_time']:.2f}秒/问题\n")
            f.write(f"总文档数: {self.test_stats['total_documents']}\n")
            
            # 🆕 QA生成统计
            if self.enable_qa_generation:
                f.write(f"\nQA生成统计:\n")
                f.write(f"启用状态: 是\n")
                f.write(f"生成问答对总数: {self.test_stats['total_qa_pairs']}\n")
                f.write(f"QA生成总时间: {self.test_stats['qa_generation_time']:.2f}秒\n")
                f.write(f"QA生成错误数: {self.test_stats['qa_generation_errors']}\n")
                f.write(f"问题类型配置: {self.questions_per_type}\n")
            else:
                f.write(f"\nQA生成统计: 未启用\n")
            
            f.write("\n详细结果:\n")
            f.write("-" * 50 + "\n")
            for result in batch_results["individual_results"]:
                status_emoji = "✅" if result["status"] == "success" else "❌"
                f.write(f"{status_emoji} {result['question_folder']}: "
                       f"{result['documents_count']}文档, "
                       f"{result['processing_time']:.2f}秒")
                
                if result["status"] == "success":
                    pipeline_res = result["pipeline_results"]
                    qa_res = result.get("qa_results", {})
                    f.write(f"\n   📊 Pipeline: Chunks:{pipeline_res['chunk_count']}, "
                           f"Concepts:{pipeline_res['concept_count']}, "
                           f"Evidence:{pipeline_res['evidence_count']}")
                    if qa_res.get("qa_generation_enabled"):
                        f.write(f"\n   📊 QA生成: {qa_res['qa_pairs_count']} 个问答对")
                    f.write(f"\n")
                elif result.get("error_message"):
                    f.write(f"\n   ❌ 错误: {result['error_message']}\n")
        
        logger.info(f"💾 已保存最终结果: {final_file}")
        logger.info(f"📋 已生成测试报告: {summary_file}")
    
    def _combine_all_training_data(self):
        """🆕 合并所有问题的训练数据"""
        combined_file = self.output_dir / "combined_training_data.jsonl"
        total_count = 0
        
        with open(combined_file, 'w', encoding='utf-8') as outfile:
            for question_dir in self.output_dir.iterdir():
                if question_dir.is_dir() and question_dir.name.startswith("question_"):
                    training_file = question_dir / "training_data.jsonl"
                    if training_file.exists():
                        with open(training_file, 'r', encoding='utf-8') as infile:
                            for line in infile:
                                outfile.write(line)
                                total_count += 1
        
        logger.info(f"💾 合并训练数据完成: {combined_file} ({total_count} 条记录)")
    
    def _save_file_results(self, file_name: str, pipeline_results: Dict, test_result: Dict, 
                          qa_pairs: List[Dict[str, Any]], training_data: List[Dict[str, Any]]):
        """🆕 保存单个文件的详细结果"""
        file_output_dir = self.output_dir / f"file_{file_name}"
        file_output_dir.mkdir(exist_ok=True)
        
        # 保存pipeline结果
        pipeline_file = file_output_dir / "pipeline_results.json"
        self.pipeline.save_results(pipeline_results, str(pipeline_file))
        
        # 保存测试元信息
        test_info_file = file_output_dir / "test_info.json"
        with open(test_info_file, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, ensure_ascii=False, indent=2)
        
        # 🆕 保存问答对
        if qa_pairs:
            qa_file = file_output_dir / "qa_pairs.json"
            with open(qa_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 问答对已保存: {qa_file}")
        
        # 🆕 保存训练数据
        if training_data:
            training_file = file_output_dir / "training_data.jsonl"
            with open(training_file, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"💾 训练数据已保存: {training_file}")
            
            # 🆕 也保存为完整的训练数据集
            complete_training_file = self.output_dir / f"{file_name}_training_data.jsonl"
            with open(complete_training_file, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"💾 完整训练数据已保存: {complete_training_file}")

def main():
    """主函数 - 提供交互式测试选择"""
    print("🧪 SimpleQA History数据集 Pipeline测试工具 (修复版 + QA生成 + PDF支持)")
    print("=" * 70)
    
    # 🆕 询问输入模式
    print("选择输入模式:")
    print("1. 单个文件处理 (PDF/TXT/MD)")
    print("2. 批量数据集处理 (simpleqa_histroy_md)")
    
    input_mode = input("\n请选择 (1-2): ").strip()
    
    input_file = None
    if input_mode == "1":
        # 单文件模式
        file_path = input("请输入文件路径 (例: attention is all you need.pdf): ").strip()
        if not file_path:
            print("❌ 未提供文件路径")
            return
        
        input_file = Path(file_path)
        if not input_file.exists():
            print(f"❌ 文件不存在: {input_file}")
            return
        
        print(f"📄 将处理文件: {input_file}")
    
    # 询问是否启用QA生成
    enable_qa = input("是否启用问答生成功能? (y/N): ").strip().lower() == 'y'
    
    qa_config = {}
    if enable_qa:
        print("\n配置问答生成参数:")
        print("1. 使用默认配置 (每种类型1-2个问题)")
        print("2. 自定义配置")
        
        config_choice = input("请选择 (1-2): ").strip()
        
        if config_choice == "2":
            print("\n请输入每种类型的问题数量:")
            qa_config = {
                "remember": int(input("Remember类型 (默认2): ") or "2"),
                "understand": int(input("Understand类型 (默认2): ") or "2"),
                "apply": int(input("Apply类型 (默认1): ") or "1"),
                "analyze": int(input("Analyze类型 (默认1): ") or "1"),
                "evaluate": int(input("Evaluate类型 (默认1): ") or "1"),
                "create": int(input("Create类型 (默认1): ") or "1")
            }
        
        qa_model = input("QA生成模型 (默认gpt-4o-mini): ").strip() or "gpt-4o-mini"
    else:
        qa_model = "gpt-4o-mini"
    
    # 初始化测试套件
    test_suite = SimpleQATestSuiteWithQAGeneration(
        enable_qa_generation=enable_qa,
        qa_model_name=qa_model,
        questions_per_type=qa_config if qa_config else None,
        input_file=str(input_file) if input_file else None  # 🆕
    )
    
    try:
        if input_mode == "1":
            # 🆕 单文件处理
            print(f"\n📄 开始处理文件: {input_file.name}")
            result = test_suite.test_single_file()
            
            print(f"\n结果: {result['status']}")
            if result['status'] == 'success':
                print(f"   处理时间: {result['processing_time']:.2f}秒")
                print(f"   Pipeline结果: {result['pipeline_results']}")
                if enable_qa:
                    print(f"   QA结果: {result['qa_results']}")
                print(f"\n🎉 处理完成! 结果已保存到: {test_suite.output_dir}")
                if enable_qa:
                    print("📋 训练数据文件:")
                    print(f"   - 主文件: {test_suite.output_dir}/{input_file.stem}_training_data.jsonl")
                    print(f"   - 详细目录: {test_suite.output_dir}/file_{input_file.stem}/")
            else:
                print(f"   错误: {result.get('error_message', '未知错误')}")
                
        else:
            # 原有的批量处理逻辑
            # 检查数据是否存在
            available_questions = test_suite.get_available_questions()
            if not available_questions:
                print("❌ 错误: 未找到simpleqa_histroy_md数据集")
                return
            
            print(f"📊 发现 {len(available_questions)} 个问题可供测试")
            print(f"问题范围: {available_questions[0]} 到 {available_questions[-1]}")
            
            # 原有的测试选项逻辑...
            print("\n选择测试模式:")
            print("1. 快速测试 (1个代表性问题)")
            print("2. 自定义测试 (指定问题)")
            print("3. 完整测试 (所有问题)")
            print("4. 单个问题测试")
            
            choice = input("\n请输入选择 (1-4): ").strip()
            
            if choice == "1":
                print("\n🎯 开始快速测试...")
                results = test_suite.run_quick_test()
            # ... 其他选项的处理逻辑保持不变
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 处理被用户中断")
    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {str(e)}")
        logger.error(f"主程序错误: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 