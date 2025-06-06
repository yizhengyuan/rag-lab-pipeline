"""
实验输出管理器
================

统一管理实验输出文件夹和文件命名，解决根目录文件混乱的问题。

功能：
1. 创建基于时间戳的实验文件夹
2. 管理步骤输出文件的命名和路径
3. 保存实验元数据和配置信息
4. 提供实验结果的统一格式化
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from .helpers import FileHelper

logger = logging.getLogger(__name__)

class ExperimentManager:
    """实验输出管理器"""
    
    # 步骤信息映射
    STEP_INFO = {
        1: {"name": "vectorization", "description": "文档加载与向量化存储"},
        2: {"name": "chunking", "description": "文档分块与概念提取"},
        3: {"name": "retrieval", "description": "概念检索与相似性计算"},
        4: {"name": "reranking", "description": "检索结果重排序"},
        5: {"name": "answer_generation", "description": "答案生成与推理"},
        6: {"name": "evaluation", "description": "结果评估与质量检查"},
        7: {"name": "debugging", "description": "调试分析与问题诊断"},
        8: {"name": "final_report", "description": "最终报告与结果汇总"},
    }
    
    def __init__(self, 
                 base_dir: str = "experiments",
                 experiment_name: Optional[str] = None,
                 input_file: Optional[str] = None):
        """
        初始化实验管理器
        
        Args:
            base_dir: 实验基础目录
            experiment_name: 实验名称（可选，将自动生成）
            input_file: 输入文件路径（用于生成实验名称）
        """
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成实验名称
        if experiment_name:
            self.experiment_name = experiment_name
        else:
            self.experiment_name = self._generate_experiment_name(input_file)
        
        # 创建实验目录
        self.experiment_dir = self.base_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化实验元数据
        self.metadata = {
            "experiment_id": self.experiment_name,
            "timestamp": self.timestamp,
            "created_at": datetime.now().isoformat(),
            "input_file": str(input_file) if input_file else None,
            "base_directory": str(self.base_dir),
            "experiment_directory": str(self.experiment_dir),
            "steps_completed": [],
            "total_processing_time": 0.0,
            "status": "initialized"
        }
        
        logger.info(f"🧪 实验管理器初始化完成")
        logger.info(f"   实验名称: {self.experiment_name}")
        logger.info(f"   实验目录: {self.experiment_dir}")
        
        # 保存初始元数据
        self._save_metadata()
        
    def _generate_experiment_name(self, input_file: Optional[str] = None) -> str:
        """
        生成实验名称
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            str: 生成的实验名称
        """
        base_name = f"exp_{self.timestamp}"
        
        if input_file:
            # 从输入文件提取简短的名称
            file_path = Path(input_file)
            file_name = file_path.stem  # 不含扩展名的文件名
            
            # 清理文件名（移除特殊字符，限制长度）
            clean_name = "".join(c for c in file_name if c.isalnum() or c in "-_")[:20]
            if clean_name:
                base_name = f"{self.timestamp}_{clean_name}"
        
        return base_name
    
    def get_step_output_path(self, step_num: int, file_format: str = "txt") -> Path:
        """
        获取步骤输出文件路径
        
        Args:
            step_num: 步骤编号（1-8）
            file_format: 文件格式（txt, json, md等）
            
        Returns:
            Path: 输出文件路径
        """
        if step_num not in self.STEP_INFO:
            raise ValueError(f"无效的步骤编号: {step_num}")
        
        step_info = self.STEP_INFO[step_num]
        filename = f"step{step_num}_{step_info['name']}.{file_format}"
        return self.experiment_dir / filename
    
    def save_step_result(self, 
                        step_num: int, 
                        result: Dict[str, Any], 
                        save_formats: list = None) -> Dict[str, Path]:
        """
        保存步骤结果
        
        Args:
            step_num: 步骤编号
            result: 步骤结果字典
            save_formats: 保存格式列表，默认为 ['txt', 'json']
            
        Returns:
            Dict[str, Path]: 保存的文件路径字典
        """
        if save_formats is None:
            save_formats = ['txt', 'json']
        
        saved_files = {}
        
        # 更新步骤结果中的元数据
        result.update({
            "experiment_id": self.experiment_name,
            "step_number": step_num,
            "step_name": self.STEP_INFO[step_num]["name"],
            "step_description": self.STEP_INFO[step_num]["description"],
            "output_timestamp": datetime.now().isoformat()
        })
        
        # 保存不同格式
        for file_format in save_formats:
            try:
                if file_format == 'txt':
                    file_path = self._save_step_txt(step_num, result)
                elif file_format == 'json':
                    file_path = self._save_step_json(step_num, result)
                elif file_format == 'md':
                    file_path = self._save_step_md(step_num, result)
                else:
                    logger.warning(f"不支持的文件格式: {file_format}")
                    continue
                
                saved_files[file_format] = file_path
                logger.info(f"✅ 步骤{step_num}结果已保存: {file_path}")
                
            except Exception as e:
                logger.error(f"❌ 保存步骤{step_num}的{file_format}格式失败: {e}")
        
        # 更新实验元数据
        self._update_step_metadata(step_num, result, saved_files)
        
        return saved_files
    
    def _save_step_txt(self, step_num: int, result: Dict[str, Any]) -> Path:
        """保存步骤结果为TXT格式"""
        file_path = self.get_step_output_path(step_num, "txt")
        step_info = self.STEP_INFO[step_num]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"步骤{step_num}: {step_info['description']}\n")
            f.write("=" * 80 + "\n")
            f.write(f"实验ID: {self.experiment_name}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理状态: {'✅ 成功' if result.get('success') else '❌ 失败'}\n")
            f.write(f"处理时间: {result.get('processing_time', 0):.2f} 秒\n")
            f.write("=" * 80 + "\n\n")
            
            # 根据步骤类型生成具体内容
            if step_num == 1:
                self._write_step1_content(f, result)
            elif step_num == 2:
                self._write_step2_content(f, result)
            else:
                self._write_generic_content(f, result)
            
            # 添加机器可读的JSON数据
            f.write("\n" + "=" * 80 + "\n")
            f.write("# 机器可读数据 (请勿手动修改)\n")
            f.write("# JSON_DATA_START\n")
            
            try:
                # 创建可序列化的结果副本
                serializable_result = self._make_serializable(result)
                f.write(json.dumps(serializable_result, ensure_ascii=False, indent=2))
            except Exception as e:
                logger.warning(f"JSON序列化失败: {e}")
                f.write(json.dumps({"error": "序列化失败", "step": step_num}, ensure_ascii=False))
            
            f.write("\n# JSON_DATA_END\n")
        
        return file_path
    
    def _save_step_json(self, step_num: int, result: Dict[str, Any]) -> Path:
        """保存步骤结果为JSON格式"""
        file_path = self.get_step_output_path(step_num, "json")
        
        # 创建可序列化的结果
        serializable_result = self._make_serializable(result)
        
        FileHelper.save_json(serializable_result, str(file_path))
        return file_path
    
    def _save_step_md(self, step_num: int, result: Dict[str, Any]) -> Path:
        """保存步骤结果为Markdown格式"""
        file_path = self.get_step_output_path(step_num, "md")
        step_info = self.STEP_INFO[step_num]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# 步骤{step_num}: {step_info['description']}\n\n")
            f.write(f"**实验ID**: {self.experiment_name}\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**处理状态**: {'✅ 成功' if result.get('success') else '❌ 失败'}\n")
            f.write(f"**处理时间**: {result.get('processing_time', 0):.2f} 秒\n\n")
            
            # 添加统计信息
            if 'statistics' in result:
                f.write("## 统计信息\n\n")
                stats = result['statistics']
                for key, value in stats.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            # 添加错误信息（如果有）
            if not result.get('success') and 'error' in result:
                f.write("## 错误信息\n\n")
                f.write(f"```\n{result['error']}\n```\n\n")
        
        return file_path
    
    def _write_step1_content(self, f, result: Dict[str, Any]):
        """写入步骤1（向量化）的具体内容"""
        if result.get("success"):
            # 文档基本信息
            doc_data = result.get("document", {})
            if hasattr(doc_data, 'metadata'):
                metadata = doc_data.metadata
                text_content = doc_data.text
            elif isinstance(doc_data, dict):
                metadata = doc_data.get("metadata", {})
                text_content = doc_data.get("text", "")
            else:
                metadata = {}
                text_content = str(doc_data) if doc_data else ""
            
            f.write(f"📊 文档基本信息:\n")
            f.write(f"- 文件名: {metadata.get('file_name', '未知')}\n")
            f.write(f"- 文件类型: {metadata.get('file_type', '未知')}\n")
            f.write(f"- 文件大小: {metadata.get('file_size', 0) / 1024:.2f} KB\n")
            f.write(f"- 文本长度: {len(text_content):,} 字符\n\n")
            
            # 分块信息
            chunks = result.get("chunk_nodes", [])
            f.write(f"📄 文档分块信息:\n")
            f.write(f"- 总分块数: {len(chunks)}\n")
            
            if chunks:
                chunk_lengths = [len(getattr(chunk, 'text', '')) for chunk in chunks]
                f.write(f"- 平均分块长度: {sum(chunk_lengths) / len(chunk_lengths):.1f} 字符\n")
                f.write(f"- 最短分块: {min(chunk_lengths)} 字符\n")
                f.write(f"- 最长分块: {max(chunk_lengths)} 字符\n\n")
            
            # 向量化信息
            vector_info = result.get("vector_info", {})
            f.write(f"🗃️ 向量化存储信息:\n")
            f.write(f"- 向量数据库类型: {vector_info.get('store_type', '未知')}\n")
            f.write(f"- 存储目录: {vector_info.get('persist_directory', '未知')}\n")
            f.write(f"- 向量维度: {vector_info.get('dimension', '未知')}\n")
            f.write(f"- 向量化节点数: {vector_info.get('vectorized_nodes', 0)}\n")
            f.write(f"- 存储大小: {vector_info.get('storage_size_mb', 0):.2f} MB\n")
        
        else:
            f.write(f"❌ 错误信息: {result.get('error', '未知错误')}\n")
    
    def _write_step2_content(self, f, result: Dict[str, Any]):
        """写入步骤2（分块）的具体内容"""
        if result.get("success"):
            stats = result.get("statistics", {})
            
            f.write(f"📊 分块统计信息:\n")
            f.write(f"- 总分块数: {stats.get('total_chunks', 0)}\n")
            f.write(f"- 平均分块长度: {stats.get('avg_chunk_length', 0):.1f} 字符\n")
            f.write(f"- 总概念数: {stats.get('total_concepts', 0)}\n")
            f.write(f"- 唯一概念数: {stats.get('unique_concepts', 0)}\n")
            f.write(f"- 平均每分块概念数: {stats.get('avg_concepts_per_chunk', 0):.1f}\n\n")
        else:
            f.write(f"❌ 错误信息: {result.get('error', '未知错误')}\n")
    
    def _write_generic_content(self, f, result: Dict[str, Any]):
        """写入通用步骤内容"""
        if result.get("success"):
            f.write("✅ 步骤执行成功\n\n")
            
            # 写入统计信息
            if 'statistics' in result:
                f.write("📊 统计信息:\n")
                for key, value in result['statistics'].items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")
        else:
            f.write(f"❌ 错误信息: {result.get('error', '未知错误')}\n")
    
    def _make_serializable(self, obj) -> Any:
        """将对象转换为可序列化的格式"""
        if hasattr(obj, '__dict__'):
            # 处理有属性的对象
            if hasattr(obj, 'text') and hasattr(obj, 'metadata'):
                # TextNode或Document对象
                return {
                    "text": obj.text,
                    "metadata": dict(obj.metadata) if hasattr(obj, 'metadata') else {},
                    "node_id": getattr(obj, 'node_id', None)
                }
            else:
                return self._make_serializable(obj.__dict__)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # 对于其他类型，尝试转换为字符串
            return str(obj)
    
    def _update_step_metadata(self, step_num: int, result: Dict[str, Any], saved_files: Dict[str, Path]):
        """更新步骤元数据"""
        step_meta = {
            "step_number": step_num,
            "step_name": self.STEP_INFO[step_num]["name"],
            "step_description": self.STEP_INFO[step_num]["description"],
            "success": result.get("success", False),
            "processing_time": result.get("processing_time", 0),
            "timestamp": datetime.now().isoformat(),
            "output_files": {fmt: str(path) for fmt, path in saved_files.items()}
        }
        
        # 添加到已完成步骤列表
        if step_num not in self.metadata["steps_completed"]:
            self.metadata["steps_completed"].append(step_num)
        
        # 更新总处理时间
        self.metadata["total_processing_time"] += result.get("processing_time", 0)
        
        # 更新状态
        if result.get("success"):
            self.metadata["status"] = f"step_{step_num}_completed"
        else:
            self.metadata["status"] = f"step_{step_num}_failed"
        
        # 保存步骤特定的元数据
        step_key = f"step_{step_num}"
        if "steps" not in self.metadata:
            self.metadata["steps"] = {}
        self.metadata["steps"][step_key] = step_meta
        
        # 保存更新的元数据
        self._save_metadata()
    
    def _save_metadata(self):
        """保存实验元数据"""
        metadata_path = self.experiment_dir / "experiment_info.json"
        FileHelper.save_json(self.metadata, str(metadata_path))
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """获取实验摘要"""
        return {
            "experiment_id": self.experiment_name,
            "experiment_directory": str(self.experiment_dir),
            "steps_completed": len(self.metadata["steps_completed"]),
            "total_steps": 8,
            "total_processing_time": self.metadata["total_processing_time"],
            "status": self.metadata["status"],
            "created_at": self.metadata["created_at"],
            "input_file": self.metadata["input_file"]
        }
    
    def list_output_files(self) -> Dict[str, Any]:
        """列出所有输出文件"""
        files = {}
        for step_num in range(1, 9):
            step_files = {}
            for ext in ['txt', 'json', 'md']:
                file_path = self.get_step_output_path(step_num, ext)
                if file_path.exists():
                    step_files[ext] = str(file_path)
            
            if step_files:
                files[f"step_{step_num}"] = step_files
        
        return files
    
    def cleanup_experiment(self, confirm: bool = False):
        """清理实验文件（谨慎使用）"""
        if not confirm:
            logger.warning("清理实验需要确认，请设置 confirm=True")
            return
        
        if self.experiment_dir.exists():
            import shutil
            shutil.rmtree(self.experiment_dir)
            logger.info(f"🗑️ 实验目录已清理: {self.experiment_dir}")
        else:
            logger.info("实验目录不存在，无需清理")
    
    @classmethod
    def load_experiment(cls, experiment_dir: str) -> 'ExperimentManager':
        """加载现有实验"""
        experiment_path = Path(experiment_dir)
        metadata_path = experiment_path / "experiment_info.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"实验元数据文件不存在: {metadata_path}")
        
        metadata = FileHelper.load_json(str(metadata_path))
        
        # 创建实验管理器实例
        manager = cls.__new__(cls)  # 跳过 __init__
        manager.experiment_dir = experiment_path
        manager.base_dir = experiment_path.parent
        manager.experiment_name = experiment_path.name
        manager.metadata = metadata
        
        logger.info(f"📂 已加载实验: {manager.experiment_name}")
        return manager
    
    @staticmethod
    def list_experiments(base_dir: str = "experiments") -> list:
        """列出所有实验"""
        base_path = Path(base_dir)
        if not base_path.exists():
            return []
        
        experiments = []
        for exp_dir in base_path.iterdir():
            if exp_dir.is_dir():
                metadata_path = exp_dir / "experiment_info.json"
                if metadata_path.exists():
                    try:
                        metadata = FileHelper.load_json(str(metadata_path))
                        experiments.append({
                            "name": exp_dir.name,
                            "path": str(exp_dir),
                            "created_at": metadata.get("created_at"),
                            "status": metadata.get("status"),
                            "steps_completed": len(metadata.get("steps_completed", [])),
                            "input_file": metadata.get("input_file")
                        })
                    except:
                        # 跳过无效的实验目录
                        continue
        
        # 按创建时间排序
        experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return experiments


def create_experiment_manager(input_file: str, 
                            base_dir: str = "experiments",
                            experiment_name: Optional[str] = None) -> ExperimentManager:
    """
    便利函数：创建实验管理器
    
    Args:
        input_file: 输入文件路径
        base_dir: 实验基础目录
        experiment_name: 实验名称（可选）
        
    Returns:
        ExperimentManager: 实验管理器实例
    """
    return ExperimentManager(
        base_dir=base_dir,
        experiment_name=experiment_name,
        input_file=input_file
    )


def load_latest_experiment(base_dir: str = "experiments") -> Optional[ExperimentManager]:
    """
    便利函数：加载最新的实验
    
    Args:
        base_dir: 实验基础目录
        
    Returns:
        Optional[ExperimentManager]: 实验管理器实例或None
    """
    experiments = ExperimentManager.list_experiments(base_dir)
    if experiments:
        latest_exp = experiments[0]  # 已按时间排序
        return ExperimentManager.load_experiment(latest_exp["path"])
    return None 