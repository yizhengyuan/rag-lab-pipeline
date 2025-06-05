"""
辅助工具模块
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class FileHelper:
    """文件操作辅助类"""
    
    @staticmethod
    def ensure_directory(directory: str) -> None:
        """确保目录存在"""
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str) -> None:
        """保存JSON文件"""
        FileHelper.ensure_directory(os.path.dirname(filepath))
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存JSON文件: {filepath}")
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """加载JSON文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"文件不存在: {filepath}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return {}
    
    @staticmethod
    def save_text(text: str, filepath: str) -> None:
        """保存文本文件"""
        FileHelper.ensure_directory(os.path.dirname(filepath))
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"已保存文本文件: {filepath}")

class TextProcessor:
    """文本处理辅助类"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本"""
        # 移除多余的空白字符
        text = ' '.join(text.split())
        # 移除特殊字符（可根据需要调整）
        return text.strip()
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """将文本分割为句子"""
        import re
        # 简单的句子分割（可以使用更复杂的NLP库）
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """计算文本相似度（简单版本）"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """提取关键词（简单版本）"""
        import re
        from collections import Counter
        
        # 简单的关键词提取
        words = re.findall(r'\b\w+\b', text.lower())
        # 过滤停用词（简化版）
        stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '而', '了', '着', '过'}
        words = [w for w in words if w not in stop_words and len(w) > 1]
        
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(top_k)] 