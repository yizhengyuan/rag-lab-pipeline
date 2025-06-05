"""
SimpleQA History数据集 Pipeline测试脚本 - 修复版
==============================================

修复内容：
- 修复文档加载问题
- 改为手动读取markdown文件
- 添加更详细的错误处理
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import traceback

# 导入pipeline相关模块
from pipeline_new import ImprovedConceptBasedPipeline
from llama_index.core import Document

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

class SimpleQATestSuiteFixed:
    """SimpleQA数据集测试套件 - 修复版"""
    
    def __init__(self, 
                 data_dir: str = "simpleqa_histroy_md",
                 output_dir: str = "simpleqa_test_results_fixed",
                 config_path: str = None):
        """
        初始化测试套件
        
        Args:
            data_dir: SimpleQA数据集目录
            output_dir: 测试结果输出目录
            config_path: Pipeline配置文件路径
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化pipeline
        if config_path:
            self.pipeline = ImprovedConceptBasedPipeline(config_path=config_path)
        else:
            # 使用默认配置
            self.pipeline = ImprovedConceptBasedPipeline()
        
        # 测试统计
        self.test_stats = {
            "total_questions": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "total_documents": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0
        }
        
        logger.info(f"初始化SimpleQA测试套件 (修复版)")
        logger.info(f"数据目录: {self.data_dir}")
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
    
    def test_single_question(self, question_folder: str) -> Dict[str, Any]:
        """
        测试单个问题
        
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
            logger.info("🔬 步骤2: 运行Pipeline...")
            pipeline_start = time.time()
            
            pipeline_results = self.pipeline.run_pipeline(documents)
            
            pipeline_time = time.time() - pipeline_start
            logger.info(f"   ✅ Pipeline运行完成，耗时: {pipeline_time:.2f}秒")
            
            # 4. 获取统计信息
            pipeline_stats = self.pipeline.get_pipeline_statistics()
            
            # 5. 记录结果
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
                "pipeline_stats": pipeline_stats
            })
            
            # 6. 保存详细结果
            self._save_question_results(question_folder, pipeline_results, test_result)
            
            logger.info(f"✅ 问题 {question_folder} 测试成功")
            logger.info(f"   📊 结果: {test_result['pipeline_results']}")
            
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
                "questions": question_list
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
    
    def _save_question_results(self, question_folder: str, pipeline_results: Dict, test_result: Dict):
        """保存单个问题的详细结果"""
        question_output_dir = self.output_dir / question_folder
        question_output_dir.mkdir(exist_ok=True)
        
        # 保存pipeline结果
        pipeline_file = question_output_dir / "pipeline_results.json"
        self.pipeline.save_results(pipeline_results, str(pipeline_file))
        
        # 保存测试元信息
        test_info_file = question_output_dir / "test_info.json"
        with open(test_info_file, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, ensure_ascii=False, indent=2)
    
    def _save_batch_intermediate_results(self, batch_results: Dict, current_index: int):
        """保存批量测试的中间结果"""
        intermediate_file = self.output_dir / f"batch_intermediate_{current_index}.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 已保存中间结果: {intermediate_file}")
    
    def _save_batch_final_results(self, batch_results: Dict):
        """保存批量测试的最终结果"""
        final_file = self.output_dir / "batch_final_results.json"
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        # 生成简化的统计报告
        summary_file = self.output_dir / "test_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("SimpleQA History数据集 Pipeline测试报告 (修复版)\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"测试时间: {batch_results['batch_info']['start_time']}\n")
            f.write(f"测试问题数: {batch_results['batch_info']['question_count']}\n")
            f.write(f"成功: {self.test_stats['successful_tests']}\n")
            f.write(f"失败: {self.test_stats['failed_tests']}\n")
            f.write(f"成功率: {batch_results['batch_summary']['success_rate']:.1f}%\n")
            f.write(f"总处理时间: {batch_results['batch_summary']['total_batch_time']:.2f}秒\n")
            f.write(f"平均处理时间: {self.test_stats['average_processing_time']:.2f}秒/问题\n")
            f.write(f"总文档数: {self.test_stats['total_documents']}\n")
            
            f.write("\n详细结果:\n")
            f.write("-" * 40 + "\n")
            for result in batch_results["individual_results"]:
                status_emoji = "✅" if result["status"] == "success" else "❌"
                f.write(f"{status_emoji} {result['question_folder']}: "
                       f"{result['documents_count']}文档, "
                       f"{result['processing_time']:.2f}秒\n")
                if result["status"] == "success":
                    pipeline_res = result["pipeline_results"]
                    f.write(f"   📊 Chunks: {pipeline_res['chunk_count']}, "
                           f"Concepts: {pipeline_res['concept_count']}, "
                           f"Evidence: {pipeline_res['evidence_count']}\n")
                elif result.get("error_message"):
                    f.write(f"   ❌ 错误: {result['error_message']}\n")
        
        logger.info(f"💾 已保存最终结果: {final_file}")
        logger.info(f"📋 已生成测试报告: {summary_file}")

def main():
    """主函数 - 提供交互式测试选择"""
    print("🧪 SimpleQA History数据集 Pipeline测试工具 (修复版)")
    print("=" * 55)
    
    # 初始化测试套件
    test_suite = SimpleQATestSuiteFixed()
    
    # 检查数据是否存在
    available_questions = test_suite.get_available_questions()
    if not available_questions:
        print("❌ 错误: 未找到simpleqa_histroy_md数据集")
        return
    
    print(f"📊 发现 {len(available_questions)} 个问题可供测试")
    print(f"问题范围: {available_questions[0]} 到 {available_questions[-1]}")
    
    # 测试选项
    print("\n选择测试模式:")
    print("1. 快速测试 (1个代表性问题)")
    print("2. 自定义测试 (指定问题)")
    print("3. 完整测试 (所有问题)")
    print("4. 单个问题测试")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    try:
        if choice == "1":
            print("\n🎯 开始快速测试...")
            results = test_suite.run_quick_test()
            
        elif choice == "2":
            print(f"\n可用问题: {available_questions[:10]}... (显示前10个)")
            questions_input = input("请输入问题文件夹名称 (用逗号分隔): ").strip()
            question_list = [q.strip() for q in questions_input.split(",") if q.strip()]
            
            if question_list:
                print(f"\n🔧 自定义测试: {question_list}")
                results = test_suite.test_multiple_questions(question_list)
            else:
                print("❌ 未提供有效的问题列表")
                return
                
        elif choice == "3":
            confirm = input("\n⚠️  完整测试将需要较长时间，确认继续? (y/N): ").strip().lower()
            if confirm == 'y':
                print("\n🔥 开始完整测试...")
                results = test_suite.run_full_test()
            else:
                print("取消完整测试")
                return
                
        elif choice == "4":
            print(f"\n可用问题: {available_questions[:10]}... (显示前10个)")
            question = input("请输入单个问题文件夹名称: ").strip()
            
            if question in available_questions:
                print(f"\n🔍 单个问题测试: {question}")
                result = test_suite.test_single_question(question)
                print(f"\n结果: {result['status']}")
                if result['status'] == 'success':
                    print(f"   处理时间: {result['processing_time']:.2f}秒")
                    print(f"   Pipeline结果: {result['pipeline_results']}")
                else:
                    print(f"   错误: {result.get('error_message', '未知错误')}")
            else:
                print(f"❌ 问题 {question} 不存在")
                return
        else:
            print("❌ 无效选择")
            return
            
        print(f"\n🎉 测试完成! 结果已保存到: {test_suite.output_dir}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        logger.error(f"主程序错误: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 