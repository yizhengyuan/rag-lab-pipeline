"""
配置加载器模块
兼容性包装器，用于加载和管理项目配置
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigLoader:
    """配置加载器 - 兼容性包装器"""
    
    def __init__(self, config_data: Dict[str, Any] = None):
        """
        初始化配置加载器
        
        Args:
            config_data: 配置数据字典
        """
        self.config_data = config_data or {}
    
    @classmethod
    def load_config(cls, config_path: str = "config.yml") -> 'ConfigLoader':
        """
        从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            ConfigLoader: 配置加载器实例
        """
        config_data = {}
        
        # 尝试从根目录加载 config.yml
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
                logger.info(f"✅ 成功从 {config_path} 加载配置")
            except Exception as e:
                logger.warning(f"⚠️ 加载配置文件失败: {e}")
        
        # 尝试从 config/ 目录加载
        elif os.path.exists("config/config.yml"):
            try:
                with open("config/config.yml", 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
                logger.info("✅ 成功从 config/config.yml 加载配置")
            except Exception as e:
                logger.warning(f"⚠️ 加载 config/config.yml 失败: {e}")
        
        else:
            logger.warning(f"⚠️ 配置文件 {config_path} 不存在，使用默认配置")
        
        # 如果仍然没有配置，使用默认配置
        if not config_data:
            config_data = cls._get_default_config()
            logger.info("🔧 使用默认配置")
        else:
            # 🔑 新增：转换配置格式以兼容 SemanticChunker
            config_data = cls._convert_config_format(config_data)
        
        return cls(config_data)
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 🔑 修正：使用api.前缀以匹配SemanticChunker的期望格式
            "api": {
                "openai_key": "sk-zk2884399e3bbb43b998bd31be7b517f82f67bb0e95df2a1",
                "model": "gpt-4o-mini",
                "embedding_model": "text-embedding-ada-002",
                "temperature": 0.1,
                "base_url": "https://api.zhizengzeng.com/v1/"
            },
            # 🔑 同时保留openai.前缀以兼容现有配置文件
            "openai": {
                "api_key": "sk-zk2884399e3bbb43b998bd31be7b517f82f67bb0e95df2a1",
                "model": "gpt-4o-mini",
                "embedding_model": "text-embedding-ada-002",
                "temperature": 0.1,
                "base_url": "https://api.zhizengzeng.com/v1/"
            },
            "vector_store": {
                "type": "chroma",
                "persist_directory": "./vector_db",
                "collection_name": "concept_pipeline",
                "dimension": 1536,
                "enable_embedding_cache": True,
                "embedding_cache_dir": "./embedding_cache",
                "cache_expiry_days": 30
            },
            # 🔑 修正：使用chunking.前缀
            "chunking": {
                "buffer_size": 1,
                "breakpoint_percentile_threshold": 95
            },
            # 🔑 修正：使用concepts.前缀
            "concepts": {
                "concepts_per_chunk": 5
            },
            "semantic_chunking": {
                "buffer_size": 1,
                "breakpoint_percentile_threshold": 95
            },
            "concept_extraction": {
                "concepts_per_chunk": 5
            },
            "retrieval": {
                "top_k": 5
            },
            "logging": {
                "level": "INFO"
            }
        }
    
    @staticmethod
    def _convert_config_format(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🔑 新增：转换配置格式以兼容不同的模块期望
        
        Args:
            config_data: 原始配置数据
            
        Returns:
            Dict[str, Any]: 转换后的配置数据
        """
        converted = config_data.copy()
        
        # 如果存在 openai 配置但没有 api 配置，则创建 api 配置
        if "openai" in converted and "api" not in converted:
            openai_config = converted["openai"]
            converted["api"] = {
                "openai_key": openai_config.get("api_key"),
                "model": openai_config.get("model"),
                "embedding_model": openai_config.get("embedding_model"), 
                "temperature": openai_config.get("temperature"),
                "base_url": openai_config.get("base_url")
            }
            logger.info("🔄 已将 openai.* 配置转换为 api.* 格式")
        
        # 如果存在 semantic_chunking 但没有 chunking，创建 chunking 配置
        if "semantic_chunking" in converted and "chunking" not in converted:
            converted["chunking"] = converted["semantic_chunking"].copy()
            logger.info("🔄 已将 semantic_chunking.* 配置转换为 chunking.* 格式")
        
        # 如果存在 concept_extraction 但没有 concepts，创建 concepts 配置  
        if "concept_extraction" in converted and "concepts" not in converted:
            converted["concepts"] = converted["concept_extraction"].copy()
            logger.info("🔄 已将 concept_extraction.* 配置转换为 concepts.* 格式")
        
        return converted
    
    def get(self, key_path: str, default=None):
        """
        获取配置值，支持点号路径
        
        Args:
            key_path: 配置路径，如 'vector_store.type' 或 'openai.model'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config_data
        
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
        config = self.config_data
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """返回配置字典"""
        return self.config_data.copy()
    
    def save_to_file(self, filepath: str):
        """保存配置到文件"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise e
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        try:
            # 检查必需的配置项 - 优先检查api.前缀格式
            required_keys = [
                "api.openai_key",
                "api.model", 
                "vector_store.type"
            ]
            
            for key in required_keys:
                if not self.get(key):
                    # 如果api.格式不存在，尝试openai.格式作为备选
                    if key.startswith("api."):
                        fallback_key = key.replace("api.", "openai.")
                        if self.get(fallback_key):
                            logger.info(f"✅ 使用备选配置格式: {fallback_key}")
                            continue
                    
                    logger.error(f"❌ 缺少必需的配置项: {key}")
                    return False
            
            # 检查向量数据库类型
            vector_type = self.get("vector_store.type")
            if vector_type not in ["simple", "chroma", "pinecone", "qdrant"]:
                logger.warning(f"⚠️ 未知的向量数据库类型: {vector_type}")
            
            logger.info("✅ 配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 配置验证失败: {e}")
            return False
    
    def update_from_dict(self, update_dict: Dict[str, Any]):
        """从字典更新配置"""
        def merge_dict(base: Dict, update: Dict) -> Dict:
            """递归合并字典"""
            result = base.copy()
            for key, value in update.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
        
        self.config_data = merge_dict(self.config_data, update_dict)
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        获取配置的某个部分
        
        Args:
            section_name: 部分名称，如 'openai' 或 'vector_store'
            
        Returns:
            该部分的配置字典
        """
        return self.config_data.get(section_name, {})
    
    def __str__(self) -> str:
        """字符串表示"""
        vector_type = self.get("vector_store.type", "unknown")
        model = self.get("openai.model", "unknown")
        return f"ConfigLoader(vector_store={vector_type}, model={model})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()

# 便利函数
def load_config(config_path: str = "config.yml") -> ConfigLoader:
    """
    便利函数：加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ConfigLoader: 配置加载器实例
    """
    return ConfigLoader.load_config(config_path)

def create_default_config(filepath: str = "config.yml"):
    """
    创建默认配置文件
    
    Args:
        filepath: 配置文件路径
    """
    config = ConfigLoader()
    config.save_to_file(filepath)
    logger.info(f"✅ 默认配置文件已创建: {filepath}")

# 全局配置实例
_global_config: Optional[ConfigLoader] = None

def get_global_config() -> ConfigLoader:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config

def set_global_config(config: ConfigLoader):
    """设置全局配置实例"""
    global _global_config
    _global_config = config 