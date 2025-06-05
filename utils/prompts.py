"""
提示词模板模块
"""

class PromptTemplates:
    """提示词模板集合"""
    
    CONCEPT_EXTRACTION = """
    请从以下文本中提取关键概念。对于每个概念，请提供：
    1. 概念名称
    2. 概念定义
    3. 相关上下文
    
    文本内容：
    {text}
    
    请以JSON格式返回，格式如下：
    {{
        "concepts": [
            {{
                "name": "概念名称",
                "definition": "概念定义",
                "context": "相关上下文"
            }}
        ]
    }}
    """
    
    CONCEPT_MERGE = """
    请合并以下相似的概念，去除重复内容：
    
    概念列表：
    {concepts}
    
    请返回合并后的概念列表，保持JSON格式。
    """
    
    EVIDENCE_EXTRACTION = """
    基于以下概念，从文本中提取支持证据：
    
    概念：{concept}
    文本：{text}
    
    请提取相关的证据片段，并评估其相关性（0-1分）。
    """
    
    QA_GENERATION = """
    基于以下概念和证据，生成问答对：
    
    概念：{concept}
    证据：{evidence}
    
    请生成3-5个不同难度的问答对。
    """ 