"""
配置管理模块
"""

import os
import yaml
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

logger = logging.getLogger(__name__)

class LlamaIndexConfig:
    """LlamaIndex 配置管理器"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """初始化配置"""
        # 默认配置
        self.default_config = {
            "api": {
                "openai_key": "sk-zk2884399e3bbb43b998bd31be7b517f82f67bb0e95df2a1",
                "base_url": "https://api.zhizengzeng.com/v1/",
                "model": "gpt-4o-mini",
                "embedding_model": "text-embedding-ada-002",
                "temperature": 0.1,
                "max_tokens": 4000,
                "timeout": 60
            },
            "chunking": {
                "buffer_size": 1,
                "breakpoint_percentile_threshold": 95,
                "chunk_size": 1024,
                "chunk_overlap": 200
            },
            "concepts": {
                "similarity_threshold": 0.7,
                "max_concepts": 10,
                "concepts_per_chunk": 5,
                "min_concept_length": 10,
                "max_concept_length": 500
            },
            "retrieval": {
                "top_k": 5,
                "similarity_cutoff": 0.7,
                "rerank": True,
                "alpha": 0.5  # 混合检索权重
            },
            "evidence": {
                "min_length": 20,
                "max_length": 200,
                "min_relevance_score": 0.5,
                "max_evidence_per_concept": 10
            },
            "vector_store": {
                "type": "simple",  # simple, chroma, pinecone, qdrant
                "persist_directory": "./vector_db",
                "collection_name": "concepts",
                "dimension": 1536  # OpenAI embedding dimension
            },
            "qa_generation": {
                "questions_per_concept": 3,
                "difficulty_levels": ["easy", "medium", "hard"],
                "question_types": ["factual", "analytical", "creative"],
                "min_answer_length": 50,
                "max_answer_length": 300
            },
            "output": {
                "save_intermediate": True,
                "output_format": "json",  # json, txt, yaml
                "output_directory": "./output",
                "filename_template": "{timestamp}_{document_name}_{stage}"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "./logs/pipeline.log"
            }
        }
        
        # 合并用户配置
        if config_dict:
            self.config = self._merge_configs(self.default_config, config_dict)
        else:
            self.config = self.default_config.copy()
        
        # 设置属性
        self._setup_properties()
        
        # 设置环境变量
        self._setup_environment()
        
        # 配置日志
        self._setup_logging()
        
        # 配置 LlamaIndex
        self.setup_llamaindex_settings()
    
    def _setup_properties(self):
        """设置配置属性"""
        # API 配置
        api_config = self.config["api"]
        self.openai_api_key = api_config["openai_key"]
        self.base_url = api_config["base_url"]
        self.model_name = api_config["model"]
        self.embedding_model = api_config["embedding_model"]
        self.temperature = api_config["temperature"]
        self.max_tokens = api_config["max_tokens"]
        self.timeout = api_config["timeout"]
        
        # 其他配置
        self.chunking_config = self.config["chunking"]
        self.concepts_config = self.config["concepts"]
        self.retrieval_config = self.config["retrieval"]
        self.evidence_config = self.config["evidence"]
        self.vector_store_config = self.config["vector_store"]
        self.qa_config = self.config["qa_generation"]
        self.output_config = self.config["output"]
    
    def _setup_environment(self):
        """设置环境变量"""
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.base_url:
            os.environ["OPENAI_API_BASE"] = self.base_url
    
    def _setup_logging(self):
        """配置日志"""
        log_config = self.config["logging"]
        
        # 创建日志目录
        if "file" in log_config:
            log_file = Path(log_config["file"])
            log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 配置日志级别
        log_level = getattr(logging, log_config["level"].upper(), logging.INFO)
        
        # 配置日志格式
        logging.basicConfig(
            level=log_level,
            format=log_config["format"],
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_config.get("file", "pipeline.log"), encoding='utf-8')
            ] if "file" in log_config else [logging.StreamHandler()]
        )
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """递归合并配置字典"""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def setup_llamaindex_settings(self):
        """配置 LlamaIndex 全局设置"""
        try:
            # 配置 LLM
            llm_kwargs = {
                "model": self.model_name,
                "api_key": self.openai_api_key,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout
            }
            
            if self.base_url:
                llm_kwargs["api_base"] = self.base_url
            
            Settings.llm = OpenAI(**llm_kwargs)
            
            # 配置嵌入模型
            embed_kwargs = {
                "model": self.embedding_model,
                "api_key": self.openai_api_key
            }
            
            if self.base_url:
                embed_kwargs["api_base"] = self.base_url
            
            Settings.embed_model = OpenAIEmbedding(**embed_kwargs)
            
            # 设置其他全局配置
            Settings.chunk_size = self.chunking_config["chunk_size"]
            Settings.chunk_overlap = self.chunking_config["chunk_overlap"]
            
            logger.info("LlamaIndex 配置成功")
            
        except Exception as e:
            logger.error(f"LlamaIndex 配置失败: {e}")
            raise e
    
    def get(self, key_path: str, default=None):
        """
        获取配置值
        
        Args:
            key_path: 配置路径，如 'api.model' 或 'chunking.buffer_size'
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any):
        """
        设置配置值
        
        Args:
            key_path: 配置路径
            value: 新值
        """
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 验证必需的配置项
            required_keys = [
                "api.openai_key",
                "api.model",
                "api.embedding_model"
            ]
            
            for key in required_keys:
                if not self.get(key):
                    logger.error(f"缺少必需的配置项: {key}")
                    return False
            
            # 验证数值范围
            if not (0 <= self.temperature <= 2):
                logger.error("temperature 必须在 0-2 之间")
                return False
            
            if not (0 < self.concepts_config["similarity_threshold"] <= 1):
                logger.error("concepts.similarity_threshold 必须在 0-1 之间")
                return False
            
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def save_to_yaml(self, filepath: str):
        """保存配置到 YAML 文件"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise e
    
    def to_dict(self) -> Dict[str, Any]:
        """返回配置字典"""
        return self.config.copy()
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"LlamaIndexConfig(model={self.model_name}, base_url={self.base_url})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()

def load_config_from_yaml(config_path: str = "config.yml") -> LlamaIndexConfig:
    """
    从 YAML 文件加载配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        LlamaIndexConfig 实例
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            logger.info(f"从 {config_path} 加载配置成功")
            return LlamaIndexConfig(config_dict)
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
            return LlamaIndexConfig()
    else:
        logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
        return LlamaIndexConfig()

def create_default_config_file(filepath: str = "config.yml"):
    """
    创建默认配置文件
    
    Args:
        filepath: 配置文件路径
    """
    config = LlamaIndexConfig()
    config.save_to_yaml(filepath)
    logger.info(f"默认配置文件已创建: {filepath}")

# 全局配置实例
_global_config: Optional[LlamaIndexConfig] = None

def get_global_config() -> LlamaIndexConfig:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = load_config_from_yaml()
    return _global_config

def set_global_config(config: LlamaIndexConfig):
    """设置全局配置实例"""
    global _global_config
    _global_config = config 