"""
验证器模块
"""

import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ConceptValidator:
    """概念验证器"""
    
    @staticmethod
    def validate_documents(documents: List) -> bool:
        """验证文档列表是否有效"""
        if not documents:
            logger.error("文档列表为空")
            return False
        
        # 检查每个文档是否有text属性
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'text') or not doc.text.strip():
                logger.error(f"文档 {i} 没有有效的文本内容")
                return False
        
        logger.info(f"文档验证通过: {len(documents)} 个文档")
        return True
    
    @staticmethod
    def validate_concept_format(concept: Dict[str, Any]) -> bool:
        """验证概念格式是否正确"""
        required_fields = ['name', 'definition']
        return all(field in concept for field in required_fields)
    
    @staticmethod
    def validate_concepts_list(concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证并过滤概念列表"""
        valid_concepts = []
        for concept in concepts:
            if ConceptValidator.validate_concept_format(concept):
                valid_concepts.append(concept)
            else:
                logger.warning(f"无效的概念格式: {concept}")
        return valid_concepts
    
    @staticmethod
    def validate_json_response(response: str) -> Dict[str, Any]:
        """验证JSON响应格式"""
        try:
            data = json.loads(response)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return {}
    
    @staticmethod
    def validate_qa_pair(qa_pair: Dict[str, Any]) -> bool:
        """验证问答对格式"""
        required_fields = ['question', 'answer']
        return all(field in qa_pair for field in required_fields) 