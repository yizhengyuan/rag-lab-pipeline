"""
简化版数据生成脚本
直接基于文档内容生成QA对，不使用schema
"""

import os
import json
import time
import random
from typing import Dict, List, Union, Any
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import re
import traceback
import uuid

# 加载环境变量
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY', "sk-zk2884399e3bbb43b998bd31be7b517f82f67bb0e95df2a1")
base_url = os.getenv('BASE_URL', "https://api.zhizengzeng.com/v1/")
default_model = os.getenv('DEFAULT_MODEL', "gpt-4o-mini")
max_tokens = int(os.getenv('MAX_TOKENS', "1024"))
temperature = float(os.getenv('TEMPERATURE', "0.1"))
output_dir = os.getenv('OUTPUT_DIR', "./output")
qa_pairs_per_file = int(os.getenv('QA_PAIRS_PER_FILE', "5"))
max_workers = int(os.getenv('MAX_WORKERS', "4"))
max_retries = int(os.getenv('MAX_RETRIES', "3"))
cognitive_levels = ["remember", "understand", "apply", "analyze", "evaluate", "create"]

# 可配置的每种类型问题数量
QUESTIONS_PER_TYPE = {
    "remember": 2,
    "understand": 2, 
    "apply": 1,
    "analyze": 1,
    "evaluate": 1,
    "create": 1
}

# Prompt模板
prompt_domain_system = 'You are an expert in content classification.'
prompt_domain_user = '''Classify the domain of the following document content. Choose from: medical, legal, scientific, technical, business, educational, general.

Document content (first 3000 characters):
{text}

Return only the domain name in lowercase.'''

prompt_qa_system = 'You are an expert in creating diverse question-answer pairs for the {domain} domain following Bloom\'s Taxonomy cognitive levels.'
prompt_qa_user = '''Based on the following document content, first analyze what types of questions are appropriate for this content, then generate questions accordingly.

Step 1: Analyze the document and determine which cognitive levels from Bloom's Taxonomy are suitable for this content:

1. Remember: Basic factual questions that test recall of information (suitable when document contains specific facts, dates, names, definitions)
2. Understand: Questions that ask to explain concepts or mechanisms (suitable when document explains processes, concepts, or relationships)  
3. Apply: Questions that require applying knowledge to new situations (suitable when document provides methods, procedures, or principles that can be applied)
4. Analyze: Questions that involve comparing, contrasting, or breaking down information (suitable when document contains multiple elements that can be compared or analyzed)
5. Evaluate: Questions that require making judgments or decisions based on criteria (suitable when document provides enough context for evaluation or contains controversial topics)
6. Create: Questions that involve synthesizing or designing new solutions (suitable when document provides foundational knowledge that can be used to create something new)

Step 2: From the suitable cognitive levels you identified, generate questions following this priority and distribution:
{questions_per_type_str}

IMPORTANT: Only generate questions for the cognitive levels that are actually suitable for this document content. If a cognitive level is not suitable, skip it entirely. Focus on the types that best match the document's content and information depth.

Document content:
{document_content}

Format your response as a JSON object with this structure:
{{
  "suitable_types": ["list of cognitive levels that are actually suitable for this document"],
  "questions": [
    {{
      "question": "The question text",
      "type": "cognitive level (must be from suitable_types)",
      "difficulty": "easy/medium/hard",
      "rationale": "brief explanation of why this question type is suitable for this content"
    }}
  ]
}}

Return only valid JSON.'''

prompt_extract_chunks_system = 'You are an expert in extracting relevant information from documents for the {domain} domain.'
prompt_extract_chunks_user = '''You are an expert document analyst. Extract the most relevant paragraphs from the document that answer the following question:

Question: {question}

Original document:
{text}

For each segment, classify it into one of these categories:
1. FULLY_SUPPORT: Text from a single segment directly answers the question. The answer can be extracted verbatim.
2. PARTIAL_SUPPORT: Multiple segments or additional knowledge/calculation is needed to derive the answer.

For each extracted segment, provide:
- The exact text from the document
- The type (fully_support or partial_support)
- A brief reason explaining why you classified it this way
- Key terms/concepts that connect this segment to the question

Format your response as a JSON array with objects containing "content", "type", "reason", and "keywords" fields.
'''

prompt_irrelevant_chunks_user = '''Create distractor text segments that appear to come from the same document but are either:

1. CONFUSED: Text mentions similar concepts but doesn't help answer the question or provides contradictory information.
2. IRRELEVANT: Text is completely off-topic with no information value for the question.

Question: {question}

Please create:
- 2 "confused" distractors (text related to the question's topic but not answering it)
- 2 "irrelevant" distractors (text completely unrelated to the question's topic)

Format your response as a JSON array with objects containing:
- "content": The distractor text
- "type": Either "confused" or "irrelevant"  
- "reason": Brief explanation of why this is a distractor
- "keywords": [] (empty array for distractors)
'''

prompt_extract_keywords_user = '''Extract 5 important keywords or key phrases from the following text segments that are relevant to the question:

Question: {question}

Text segments:
{text_segments}

Return the keywords as a JSON array of strings. Keywords should be short (1-3 words), specific, and represent the most important concepts in the text related to the question.'''

# 新增：文本适用性分析的prompt模板
prompt_analyze_suitability_system = 'You are an expert in educational content analysis and Bloom\'s Taxonomy.'
prompt_analyze_suitability_user = '''Analyze the following document content and determine which cognitive levels from Bloom's Taxonomy are suitable for creating questions based on this content.

Document content:
{document_content}

For each cognitive level, evaluate whether the document contains sufficient information to create meaningful questions:

1. Remember: Does the document contain specific facts, dates, names, definitions, or concrete information that can be recalled?
2. Understand: Does the document explain concepts, processes, mechanisms, or relationships that can be comprehended and explained?
3. Apply: Does the document provide methods, procedures, principles, or knowledge that can be applied to new situations?
4. Analyze: Does the document contain multiple elements, components, or aspects that can be compared, contrasted, or broken down?
5. Evaluate: Does the document provide enough context for making judgments, assessments, or decisions based on criteria?
6. Create: Does the document provide foundational knowledge that can be used to synthesize or design new solutions?

Format your response as a JSON object:
{{
  "suitable_types": [
    {{
      "type": "cognitive_level_name",
      "suitable": true/false,
      "reason": "explanation of why this type is or isn't suitable",
      "confidence": "high/medium/low"
    }}
  ],
  "recommended_distribution": {{
    "type_name": number_of_questions_recommended
  }}
}}

Return only valid JSON.'''

# 新增：专门生成问题的prompt模板
prompt_generate_questions_system = 'You are an expert in creating educational questions for the {domain} domain following Bloom\'s Taxonomy.'
prompt_generate_questions_user = '''Based on the document content and the specified cognitive types, generate {total_questions} questions with the following distribution:
{questions_distribution}

Document content:
{document_content}

For each question, ensure it matches the specified cognitive level:
- Remember: Ask for specific facts, dates, names, or definitions from the document
- Understand: Ask to explain concepts, processes, or relationships described in the document
- Apply: Ask to apply methods or principles from the document to new scenarios
- Analyze: Ask to compare, contrast, or break down elements from the document
- Evaluate: Ask to make judgments or assessments based on criteria from the document
- Create: Ask to synthesize or design something new using knowledge from the document

Format your response as a JSON array:
[
  {{
    "question": "The question text",
    "type": "cognitive_level",
    "difficulty": "easy/medium/hard",
    "rationale": "brief explanation of why this question fits this cognitive level"
  }}
]

Return only valid JSON.'''

# 新增：专门生成答案的prompt模板
prompt_generate_answer_system = 'You are an expert in providing comprehensive answers based on document content for the {domain} domain.'
prompt_generate_answer_user = '''Based on the following document content, provide a detailed and accurate answer to the question.

Question: {question}
Question Type: {question_type}
Question Difficulty: {difficulty}

Document content:
{document_content}

Guidelines for answering:
- For Remember questions: Provide specific facts, dates, names, or definitions directly from the document
- For Understand questions: Explain concepts, processes, or relationships described in the document
- For Apply questions: Show how to apply methods or principles to the given scenario
- For Analyze questions: Break down, compare, or contrast elements as requested
- For Evaluate questions: Make judgments based on criteria and evidence from the document
- For Create questions: Synthesize information to design or propose new solutions

Provide a clear, comprehensive answer based solely on the information available in the document.'''

# ====================== 文件处理模块 ======================
class FileProcessor:
    """Process different types of files and extract text content"""
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """Extract text based on file type"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            text = FileProcessor.extract_from_pdf(file_path)
        elif ext == '.txt':
            text = FileProcessor.extract_from_txt(file_path)
        elif ext == '.md':
            text = FileProcessor.extract_from_txt(file_path)
        elif ext in ['.json', '.jsonl']:
            text = FileProcessor.extract_from_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return FileProcessor.text_preprocessing(text)
    
    @staticmethod
    def text_preprocessing(text: str) -> str:
        """Perform basic text preprocessing to clean the text"""
        if not text:
            return ""
        
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.replace('\t', ' ')
        text = re.sub(r' +', ' ', text)
        lines = [line.strip() for line in text.split('\n')]
        
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            if not line:
                if not prev_empty:
                    cleaned_lines.append('')
                    prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Fix encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        text = text.replace('â€"', '-')
        
        if text.startswith('\ufeff'):
            text = text[1:]
            
        return text.strip()
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def extract_from_txt(file_path: str) -> str:
        """Extract content from text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    @staticmethod
    def extract_from_json(file_path: str) -> str:
        """Extract content from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            if file_path.endswith('.jsonl'):
                text = ""
                for line in file:
                    json_obj = json.loads(line)
                    text += json.dumps(json_obj, ensure_ascii=False) + "\n"
                return text
            else:
                data = json.load(file)
                return json.dumps(data, ensure_ascii=False, indent=2)

# ====================== 简化数据生成模块 ======================
class SimpleDataGenerator:
    """简化的数据生成器，直接基于文档内容"""
    
    def __init__(self, model_name: str = default_model, questions_per_type: Dict[str, int] = None):
        self.model_name = model_name
        self.client = self._init_openai_client()
        self.domain = "general"
        self.questions_per_type = questions_per_type or QUESTIONS_PER_TYPE
    
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        if base_url:
            return openai.OpenAI(api_key=openai_api_key, base_url=base_url)
        else:
            return openai.OpenAI(api_key=openai_api_key)
    
    def detect_domain(self, text: str) -> str:
        """检测文档领域"""
        try:
            user_prompt = prompt_domain_user.format(text=text[:3000])
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt_domain_system},
                    {"role": "user", "content": user_prompt}
                ]
            ).choices[0].message.content.strip().lower()
            
            # 确保返回的是有效的领域
            valid_domains = ["medical", "legal", "scientific", "technical", "business", "educational", "general"]
            if response in valid_domains:
                return response
            else:
                return "general"
        except Exception as e:
            print(f"Error detecting domain: {e}")
            return "general"
    
    def analyze_content_suitability(self, text: str) -> Dict:
        """步骤1: 分析文本内容适合哪些认知层次类型"""
        try:
            # 截断文本以避免token限制
            truncated_text = text[:15000]
            if len(text) > 15000:
                last_paragraph = truncated_text.rfind('\n\n')
                if last_paragraph > 10000:
                    truncated_text = truncated_text[:last_paragraph]
                truncated_text += "... (text continues)"
            
            user_prompt = prompt_analyze_suitability_user.format(document_content=truncated_text)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt_analyze_suitability_system},
                    {"role": "user", "content": user_prompt}
                ]
            ).choices[0].message.content
            
            # 清理响应
            json_str = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_str:
                response = json_str.group(1).strip()
            elif response.startswith('```') and response.endswith('```'):
                response = response[3:-3].strip()
            
            # 提取JSON部分
            first_brace = response.find('{')
            last_brace = response.rfind('}')
            if first_brace != -1 and last_brace != -1:
                response = response[first_brace:last_brace+1]
            
            try:
                result = json.loads(response)
                
                # 提取适合的类型
                suitable_types = []
                for type_info in result.get("suitable_types", []):
                    if type_info.get("suitable", False):
                        suitable_types.append(type_info["type"])
                
                # 获取推荐分布
                recommended_distribution = result.get("recommended_distribution", {})
                
                print(f"Content analysis result:")
                print(f"  Suitable types: {suitable_types}")
                print(f"  Recommended distribution: {recommended_distribution}")
                
                return {
                    "suitable_types": suitable_types,
                    "recommended_distribution": recommended_distribution,
                    "analysis_details": result.get("suitable_types", [])
                }
                
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"原始响应: {response[:200]}...")
                # 返回默认分析结果
                return {
                    "suitable_types": ["remember", "understand"],
                    "recommended_distribution": {"remember": 2, "understand": 2},
                    "analysis_details": []
                }
                
        except Exception as e:
            print(f"分析内容适用性时出错: {str(e)}")
            return {
                "suitable_types": ["remember", "understand"],
                "recommended_distribution": {"remember": 2, "understand": 2},
                "analysis_details": []
            }
    
    def generate_questions_by_types(self, text: str, suitable_types: List[str], questions_distribution: Dict[str, int]) -> List[Dict]:
        """步骤2: 基于适合的类型生成问题"""
        try:
            # 截断文本以避免token限制
            truncated_text = text[:15000]
            if len(text) > 15000:
                last_paragraph = truncated_text.rfind('\n\n')
                if last_paragraph > 10000:
                    truncated_text = truncated_text[:last_paragraph]
                truncated_text += "... (text continues)"
            
            # 计算总问题数
            total_questions = sum(questions_distribution.values())
            
            # 格式化问题分布
            distribution_str = ""
            for q_type, count in questions_distribution.items():
                if count > 0 and q_type in suitable_types:
                    distribution_str += f"- {q_type.title()}: {count} questions\n"
            
            system_prompt = prompt_generate_questions_system.format(domain=self.domain)
            user_prompt = prompt_generate_questions_user.format(
                total_questions=total_questions,
                questions_distribution=distribution_str,
                document_content=truncated_text
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            ).choices[0].message.content
            
            # 清理响应
            json_str = re.search(r'```json(.*?)```', response, re.DOTALL)
            if json_str:
                response = json_str.group(1).strip()
            elif response.startswith('```') and response.endswith('```'):
                response = response[3:-3].strip()
            
            # 提取JSON部分
            first_bracket = response.find('[')
            last_bracket = response.rfind(']')
            if first_bracket != -1 and last_bracket != -1:
                response = response[first_bracket:last_bracket+1]
            
            try:
                questions = json.loads(response)
                
                if not isinstance(questions, list):
                    questions = [questions] if isinstance(questions, dict) else []
                
                # 验证和清理问题
                for i, question in enumerate(questions):
                    if not isinstance(question, dict):
                        questions[i] = {"question": str(question), "type": "remember", "difficulty": "medium"}
                        continue
                    
                    if "question" not in question:
                        question["question"] = f"What is the main topic discussed in this document?"
                    if "type" not in question:
                        question["type"] = "remember"
                    if "difficulty" not in question:
                        question["difficulty"] = "medium"
                    
                    # 确保type是适合的认知层次
                    if question["type"] not in suitable_types:
                        question["type"] = suitable_types[0] if suitable_types else "remember"
                
                print(f"Generated {len(questions)} questions:")
                for q in questions:
                    print(f"  - {q['type']}: {q['question'][:50]}...")
                
                return questions
                
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"原始响应: {response[:200]}...")
                # 创建默认问题
                return [
                    {
                        "question": "What is this document about?",
                        "type": suitable_types[0] if suitable_types else "remember",
                        "difficulty": "easy"
                    }
                ]
                
        except Exception as e:
            print(f"生成问题时出错: {str(e)}")
            return [
                {
                    "question": "What is this document about?",
                    "type": suitable_types[0] if suitable_types else "remember",
                    "difficulty": "easy"
                }
            ]
    
    def generate_answer_for_question(self, question: str, question_type: str, difficulty: str, text: str) -> str:
        """步骤3: 基于问题和原文生成答案"""
        try:
            # 截断文本以避免token限制
            truncated_text = text[:20000]
            if len(text) > 20000:
                last_paragraph = truncated_text.rfind('\n\n')
                if last_paragraph > 15000:
                    truncated_text = truncated_text[:last_paragraph]
                truncated_text += "... (text continues)"
            
            system_prompt = prompt_generate_answer_system.format(domain=self.domain)
            user_prompt = prompt_generate_answer_user.format(
                question=question,
                question_type=question_type,
                difficulty=difficulty,
                document_content=truncated_text
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            ).choices[0].message.content
            
            return response.strip()
            
        except Exception as e:
            print(f"生成答案时出错: {str(e)}")
            return "Unable to provide an answer due to processing error."
    
    def generate_qa_pairs_from_text(self, text: str) -> List[Dict]:
        """主流程：三步法生成问答对"""
        try:
            # 检测领域
            self.domain = self.detect_domain(text)
            print(f"Detected domain: {self.domain}")
            
            # 步骤1: 分析内容适用性
            suitability_analysis = self.analyze_content_suitability(text)
            suitable_types = suitability_analysis["suitable_types"]
            recommended_distribution = suitability_analysis["recommended_distribution"]
            
            # 结合用户配置和推荐分布
            final_distribution = {}
            for q_type in suitable_types:
                if q_type in self.questions_per_type:
                    # 使用用户配置的数量，但不超过推荐数量的2倍
                    user_count = self.questions_per_type[q_type]
                    recommended_count = recommended_distribution.get(q_type, user_count)
                    final_distribution[q_type] = min(user_count, max(recommended_count * 2, 1))
                else:
                    final_distribution[q_type] = recommended_distribution.get(q_type, 1)
            
            print(f"Final question distribution: {final_distribution}")
            
            # 步骤2: 生成问题
            questions = self.generate_questions_by_types(text, suitable_types, final_distribution)
            
            # 步骤3: 为每个问题生成答案
            qa_pairs = []
            for question_data in questions:
                question = question_data["question"]
                question_type = question_data["type"]
                difficulty = question_data["difficulty"]
                
                answer = self.generate_answer_for_question(question, question_type, difficulty, text)
                
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "type": question_type,
                    "difficulty": difficulty,
                    "rationale": question_data.get("rationale", "")
                })
            
            print(f"Generated {len(qa_pairs)} complete QA pairs")
            return qa_pairs
            
        except Exception as e:
            print(f"生成QA对时出错: {str(e)}")
            return [
                {
                    "question": "What is this document about?",
                    "answer": "This document discusses various topics that require further analysis.",
                    "type": "remember",
                    "difficulty": "easy",
                    "rationale": "Default question due to processing error"
                }
            ]
    
    def extract_text_chunks(self, text: str, question: str) -> List[Dict]:
        """从文档中提取相关文本块"""
        try:
            # 处理文本以避免token限制
            truncated_text = text[:20000]
            if len(text) > 20000:
                last_paragraph = truncated_text.rfind('\n\n')
                if last_paragraph > 15000:
                    truncated_text = truncated_text[:last_paragraph]
                truncated_text += "... (text continues)"
            
            # 转义花括号
            processed_text = truncated_text.replace("{", "{{").replace("}", "}}")
            escaped_question = question.replace("{", "{{").replace("}", "}}")
            
            system_prompt = prompt_extract_chunks_system.format(domain=self.domain)
            user_prompt = prompt_extract_chunks_user.replace("{question}", escaped_question)
            user_prompt = user_prompt.replace("{text}", processed_text)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            ).choices[0].message.content
            
            # 清理响应
            response = response.strip()
            json_str = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
            if json_str:
                response = json_str.group(1).strip()
            
            first_bracket = response.find('[')
            last_bracket = response.rfind(']')
            if first_bracket != -1 and last_bracket != -1:
                response = response[first_bracket:last_bracket+1]
            
            try:
                extracted_chunks = json.loads(response)
                if isinstance(extracted_chunks, list):
                    # 确保每个chunk都有必要的字段
                    for chunk in extracted_chunks:
                        if "content" not in chunk:
                            chunk["content"] = ""
                        if "type" not in chunk:
                            chunk["type"] = "partial_support"
                        if "reason" not in chunk:
                            chunk["reason"] = "Automatically added reason"
                        if "keywords" not in chunk:
                            chunk["keywords"] = []
                    return extracted_chunks
                else:
                    return [extracted_chunks] if extracted_chunks else []
            except json.JSONDecodeError:
                print("Warning: Failed to parse extracted chunks")
                return []
                
        except Exception as e:
            print(f"提取文本块时出错: {str(e)}")
            return []
    
    def generate_irrelevant_chunks(self, question: str) -> List[Dict]:
        """生成干扰文本块"""
        try:
            escaped_question = question.replace("{", "{{").replace("}", "}}")
            user_prompt = prompt_irrelevant_chunks_user.replace("{question}", escaped_question)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are an expert in creating distractor content for the {self.domain} domain."},
                    {"role": "user", "content": user_prompt}
                ]
            ).choices[0].message.content
            
            # 清理响应
            response = response.strip()
            json_str = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
            if json_str:
                response = json_str.group(1).strip()
            
            first_bracket = response.find('[')
            last_bracket = response.rfind(']')
            if first_bracket != -1 and last_bracket != -1:
                response = response[first_bracket:last_bracket+1]
            
            try:
                distractor_chunks = json.loads(response)
                return distractor_chunks if isinstance(distractor_chunks, list) else []
            except json.JSONDecodeError:
                # 返回默认干扰块
                return [
                    {"content": "This information is not directly relevant to the question.", "type": "irrelevant", "reason": "Default irrelevant content", "keywords": []},
                    {"content": "This text contains information that may look related but has factual errors.", "type": "confused", "reason": "Default confused content", "keywords": []}
                ]
                
        except Exception as e:
            print(f"生成干扰块时出错: {str(e)}")
            return []
    
    def extract_keywords_for_chunk(self, question: str, chunk_text: str) -> List[str]:
        """为文本块提取关键词"""
        if not chunk_text:
            return []
        
        try:
            user_prompt = prompt_extract_keywords_user.format(
                question=question,
                text_segments=chunk_text
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in extracting relevant keywords from text."},
                    {"role": "user", "content": user_prompt}
                ]
            ).choices[0].message.content
            
            # 清理响应
            response = response.strip()
            json_str = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
            if json_str:
                response = json_str.group(1).strip()
            
            first_bracket = response.find('[')
            last_bracket = response.rfind(']')
            if first_bracket != -1 and last_bracket != -1:
                response = response[first_bracket:last_bracket+1]
            
            try:
                keywords = json.loads(response)
                return keywords if isinstance(keywords, list) else []
            except:
                # 尝试简单的文本解析
                if ',' in response:
                    return [k.strip().strip('"\'') for k in response.split(',')]
                return []
                
        except Exception as e:
            print(f"提取关键词时出错: {str(e)}")
            return []
    
    def create_training_data(self, qa_pair: Dict, text: str, file_name: str) -> Dict:
        """创建单个训练数据项"""
        try:
            question = qa_pair["question"]
            answer = qa_pair["answer"]
            question_type = qa_pair["type"]
            difficulty = qa_pair["difficulty"]
            
            # 提取相关文本块
            relevant_chunks = self.extract_text_chunks(text, question)
            
            # 生成干扰块
            distractor_chunks = self.generate_irrelevant_chunks(question)
            
            # 获取DocId
            doc_id = os.path.splitext(file_name)[0]
            
            # 为相关块分配ID和关键词
            chunk_id_counter = 0
            consolidated_context = []
            
            # 处理相关块
            for chunk in relevant_chunks:
                chunk_id = f"chunk_{chunk_id_counter}"
                chunk_keywords = self.extract_keywords_for_chunk(question, chunk["content"])
                consolidated_context.append({
                    "chunk_id": chunk_id,
                    "content": chunk["content"],
                    "type": chunk["type"].lower(),
                    "reason": chunk.get("reason", "Extracted from document"),
                    "keywords": chunk_keywords,
                    "DocId": doc_id
                })
                chunk_id_counter += 1
            
            # 处理干扰块
            for chunk in distractor_chunks:
                chunk_id = f"chunk_{chunk_id_counter}"
                consolidated_context.append({
                    "chunk_id": chunk_id,
                    "content": chunk["content"],
                    "type": chunk["type"].lower(),
                    "reason": chunk.get("reason", f"This is a {chunk['type'].lower()} distractor"),
                    "keywords": [],
                    "DocId": doc_id
                })
                chunk_id_counter += 1
            
            # 生成唯一ID
            unique_id = uuid.uuid4().hex
            
            # 构建训练数据
            training_data = {
                "_id": unique_id,
                "Question": question,
                "Answer": answer,
                "Type": question_type,
                "Difficulty": difficulty,
                "Domain": self.domain,
                "Rationale": qa_pair.get("rationale", ""),
                "Context": consolidated_context
            }
            
            return training_data
            
        except Exception as e:
            print(f"创建训练数据时出错: {str(e)}")
            return None

# ====================== 主流程控制 ======================
class SimpleDataPipeline:
    """简化的数据处理流水线"""
    
    def __init__(self, model_name: str = default_model, output_dir: str = output_dir, questions_per_type: Dict[str, int] = None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.file_processor = FileProcessor()
        self.data_generator = SimpleDataGenerator(model_name, questions_per_type)
        self.ensure_output_dirs()
    
    def ensure_output_dirs(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def process_file(self, file_path: str):
        """处理单个文件"""
        try:
            print(f"Processing file: {file_path}")
            
            # 提取文本
            text = self.file_processor.extract_text(file_path)
            file_name = os.path.basename(file_path)
            
            print(f"Text extracted: {len(text)} characters")
            
            # 使用新的三步法生成问答对
            qa_pairs = self.data_generator.generate_qa_pairs_from_text(text)
            
            # 为每个问答对创建训练数据
            training_data = []
            for qa_pair in qa_pairs:
                training_item = self.data_generator.create_training_data(qa_pair, text, file_name)
                if training_item:
                    training_data.append(training_item)
            
            # 保存训练数据
            output_path = os.path.join(self.output_dir, f"{os.path.splitext(file_name)[0]}_training.jsonl")
            with open(output_path, 'w', encoding='utf-8') as f:
                for data in training_data:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            print(f"Generated {len(training_data)} training examples from {file_path}")
            return training_data
            
        except Exception as e:
            print(f"处理文件时出错 {file_path}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return []
    
    def process_directory(self, dir_path: str, file_extensions: List[str] = ['.pdf', '.txt', '.md', '.json', '.jsonl']):
        """处理目录中的所有文件"""
        all_training_data = []
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        training_data = self.process_file(file_path)
                        all_training_data.extend(training_data)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        # 合并所有训练数据
        combined_path = os.path.join(self.output_dir, "combined_training_data.jsonl")
        with open(combined_path, 'w', encoding='utf-8') as f:
            for data in all_training_data:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        print(f"Total generated training examples: {len(all_training_data)}")
        return combined_path

# ====================== 命令行接口 ======================
def add_cli():
    """添加命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='简化版训练数据生成')
    
    parser.add_argument('--input', '-i', required=True, help='输入文件或目录路径')
    parser.add_argument('--output', '-o', default='./output', help='输出目录路径')
    parser.add_argument('--model', '-m', default=default_model, help='使用的模型名称')
    
    # 添加每种类型问题数量的参数
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
    
    pipeline = SimpleDataPipeline(
        model_name=args.model, 
        output_dir=args.output, 
        questions_per_type=questions_per_type
    )
    
    print(f"配置的问题数量: {questions_per_type}")
    
    if os.path.isdir(args.input):
        pipeline.process_directory(args.input)
    else:
        pipeline.process_file(args.input)

if __name__ == "__main__":
    add_cli()