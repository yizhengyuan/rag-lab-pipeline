"""
基于已有chunking结果的概念合并演示
====================================

这个脚本读取已有的chunking结果，模拟概念合并过程，
并将结果导出为可读的txt文件
"""

import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any
from collections import Counter

def demo_concept_merge_from_chunking():
    """基于已有的chunking结果演示概念合并"""
    
    # 读取chunking结果
    chunking_dir = "./chunking_export/chunks"
    output_dir = "./concept_merge_demo"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "concepts"), exist_ok=True)
    
    print("🔄 开始基于已有chunking结果的概念合并演示...")
    print(f"📁 输入目录: {chunking_dir}")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 80)
    
    # 1. 收集所有概念
    print("📊 步骤1: 收集所有chunk中的概念...")
    all_concepts, concept_to_chunks = collect_concepts_from_files(chunking_dir)
    print(f"   ✅ 总共收集到 {len(all_concepts)} 个概念")
    
    # 2. 概念频率分析和合并
    print("🔗 步骤2: 分析概念频率和相似性...")
    merged_concepts = simulate_concept_merging(all_concepts, concept_to_chunks)
    print(f"   ✅ 合并后得到 {len(merged_concepts)} 个概念")
    
    # 3. 生成统计信息
    print("📈 步骤3: 生成统计信息...")
    stats = generate_merge_statistics(all_concepts, merged_concepts, concept_to_chunks)
    print(f"   ✅ 统计完成")
    
    # 4. 导出总览文件
    print("💾 步骤4: 导出概念合并总览...")
    overview_path = export_demo_overview(merged_concepts, stats, output_dir)
    print(f"   ✅ 总览文件: {overview_path}")
    
    # 5. 导出详细概念文件
    print("📄 步骤5: 导出详细概念文件...")
    concept_files = export_demo_concepts(merged_concepts, concept_to_chunks, chunking_dir, output_dir)
    print(f"   ✅ 已导出 {len(concept_files)} 个概念文件")
    
    # 6. 导出JSON数据
    print("📋 步骤6: 导出JSON数据...")
    json_path = export_demo_json(merged_concepts, stats, output_dir)
    print(f"   ✅ JSON文件: {json_path}")
    
    # 7. 生成报告
    print("📖 步骤7: 生成详细报告...")
    report_path = generate_demo_report(stats, overview_path, concept_files, json_path, output_dir)
    print(f"   ✅ 报告文件: {report_path}")
    
    print(f"\n✅ 概念合并演示完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 总览文件: {overview_path}")
    print(f"📋 详细报告: {report_path}")
    print(f"\n📈 合并统计:")
    print(f"   - 原始概念数: {stats['original_concepts_count']}")
    print(f"   - 合并后概念数: {stats['merged_concepts_count']}")
    print(f"   - 合并减少率: {stats['reduction_ratio']:.1%}")
    print(f"   - 高频概念数: {stats['high_frequency_concepts']}")

def collect_concepts_from_files(chunking_dir: str) -> tuple:
    """从chunking文件中收集概念"""
    all_concepts = []
    concept_to_chunks = {}
    
    # 扫描所有chunk文件
    if not os.path.exists(chunking_dir):
        print(f"❌ 目录不存在: {chunking_dir}")
        return [], {}
    
    for filename in os.listdir(chunking_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(chunking_dir, filename)
            chunk_id = filename.replace('.txt', '')
            
            # 读取文件并提取概念
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                concepts = extract_concepts_from_content(content)
                
                for concept in concepts:
                    all_concepts.append(concept)
                    if concept not in concept_to_chunks:
                        concept_to_chunks[concept] = []
                    concept_to_chunks[concept].append(chunk_id)
    
    return all_concepts, concept_to_chunks

def extract_concepts_from_content(content: str) -> List[str]:
    """从chunk文件内容中提取概念"""
    concepts = []
    
    # 查找概念部分
    concept_section = False
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if '🧠 提取的概念' in line:
            concept_section = True
            continue
        elif concept_section and line.startswith('='):
            break
        elif concept_section and line:
            # 提取概念（去掉序号）
            match = re.match(r'\s*\d+\.\s*(.+)', line)
            if match:
                concept = match.group(1).strip()
                if concept:
                    concepts.append(concept)
    
    return concepts

def simulate_concept_merging(all_concepts: List[str], concept_to_chunks: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """模拟概念合并过程"""
    
    # 统计概念频率
    concept_freq = Counter(all_concepts)
    
    # 按频率和相似性进行概念分组和合并
    merged_concepts = []
    processed_concepts = set()
    
    # 根据频率排序
    sorted_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)
    
    concept_id = 0
    for concept, frequency in sorted_concepts:
        if concept in processed_concepts:
            continue
        
        # 寻找相似概念进行合并
        similar_concepts = find_similar_concepts(concept, concept_freq, processed_concepts)
        processed_concepts.update(similar_concepts)
        
        # 创建合并后的概念
        merged_concept = {
            "id": f"merged_concept_{concept_id:03d}",
            "name": concept,  # 使用频率最高的作为代表
            "text": concept,
            "category": classify_concept(concept),
            "confidence_score": calculate_confidence(concept, frequency, len(all_concepts)),
            "frequency": frequency,
            "source_concepts": similar_concepts,
            "source_chunks": list(set().union(*[concept_to_chunks.get(c, []) for c in similar_concepts])),
            "keywords": extract_keywords(concept)
        }
        
        merged_concepts.append(merged_concept)
        concept_id += 1
    
    return merged_concepts

def find_similar_concepts(target_concept: str, concept_freq: Counter, processed: set) -> List[str]:
    """寻找相似的概念"""
    similar = [target_concept]
    target_lower = target_concept.lower()
    
    for concept, freq in concept_freq.items():
        if concept in processed or concept == target_concept:
            continue
        
        concept_lower = concept.lower()
        
        # 简单的相似性判断
        if (target_lower in concept_lower or concept_lower in target_lower or
            calculate_word_overlap(target_lower, concept_lower) > 0.5):
            similar.append(concept)
    
    return similar

def calculate_word_overlap(concept1: str, concept2: str) -> float:
    """计算词汇重叠度"""
    words1 = set(concept1.split())
    words2 = set(concept2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def classify_concept(concept: str) -> str:
    """简单的概念分类"""
    concept_lower = concept.lower()
    
    if any(word in concept_lower for word in ['模型', '架构', '网络', 'model', 'architecture', 'network']):
        return "模型架构"
    elif any(word in concept_lower for word in ['机制', '方法', '算法', 'mechanism', 'method', 'algorithm']):
        return "方法技术"
    elif any(word in concept_lower for word in ['任务', '应用', 'task', 'application']):
        return "应用任务"
    elif any(word in concept_lower for word in ['评估', '指标', '分数', 'evaluation', 'metric', 'score']):
        return "评估指标"
    elif any(word in concept_lower for word in ['数据', '语料', 'data', 'dataset']):
        return "数据资源"
    else:
        return "其他"

def calculate_confidence(concept: str, frequency: int, total_concepts: int) -> float:
    """计算概念置信度"""
    freq_score = frequency / total_concepts
    length_score = min(len(concept.split()) / 5, 1.0)  # 偏好多词概念
    return (freq_score + length_score) / 2

def extract_keywords(concept: str) -> List[str]:
    """提取概念关键词"""
    # 简单的关键词提取
    words = concept.split()
    
    # 过滤停用词
    stop_words = {'的', '和', '与', '或', '是', '有', '在', 'for', 'and', 'or', 'the', 'a', 'an', 'of', 'to', 'in'}
    keywords = [word for word in words if word.lower() not in stop_words and len(word) > 1]
    
    return keywords[:5]  # 最多5个关键词

def generate_merge_statistics(all_concepts: List[str], merged_concepts: List[Dict], concept_to_chunks: Dict) -> Dict[str, Any]:
    """生成合并统计信息"""
    original_count = len(all_concepts)
    merged_count = len(merged_concepts)
    reduction_ratio = (original_count - merged_count) / original_count if original_count > 0 else 0
    
    # 置信度统计
    confidence_scores = [concept['confidence_score'] for concept in merged_concepts]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    # 类别统计
    categories = [concept['category'] for concept in merged_concepts]
    category_counts = Counter(categories)
    
    # 高频概念统计（频率>=2）
    high_freq_concepts = len([c for c in merged_concepts if c['frequency'] >= 2])
    
    return {
        "original_concepts_count": original_count,
        "merged_concepts_count": merged_count,
        "reduction_ratio": reduction_ratio,
        "avg_confidence": avg_confidence,
        "max_confidence": max(confidence_scores) if confidence_scores else 0,
        "min_confidence": min(confidence_scores) if confidence_scores else 0,
        "category_distribution": dict(category_counts),
        "high_frequency_concepts": high_freq_concepts,
        "total_chunks": len(set().union(*concept_to_chunks.values())) if concept_to_chunks else 0,
        "processing_time": datetime.now().isoformat()
    }

def export_demo_overview(merged_concepts: List[Dict], stats: Dict, output_dir: str) -> str:
    """导出概念合并总览"""
    overview_path = os.path.join(output_dir, "attention_is_all_you_need_concept_merge_overview.txt")
    
    with open(overview_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("概念合并(Concept Merge)结果总览 - 基于chunking结果的演示\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"文档: Attention Is All You Need\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据来源: chunking_export/chunks/ 目录\n")
        f.write(f"处理方式: 演示版概念合并算法\n")
        f.write("=" * 80 + "\n\n")
        
        # 统计信息
        f.write("📊 合并统计信息:\n\n")
        f.write(f"  原始概念总数: {stats['original_concepts_count']}\n")
        f.write(f"  合并后概念数: {stats['merged_concepts_count']}\n")
        f.write(f"  概念减少数量: {stats['original_concepts_count'] - stats['merged_concepts_count']}\n")
        f.write(f"  合并减少率: {stats['reduction_ratio']:.1%}\n")
        f.write(f"  高频概念数: {stats['high_frequency_concepts']} 个（出现频率>=2）\n")
        f.write(f"  平均置信度: {stats['avg_confidence']:.3f}\n")
        f.write(f"  最高置信度: {stats['max_confidence']:.3f}\n")
        f.write(f"  最低置信度: {stats['min_confidence']:.3f}\n\n")
        
        # 类别分布
        f.write("📂 概念类别分布:\n\n")
        for category, count in stats['category_distribution'].items():
            percentage = count / stats['merged_concepts_count'] * 100 if stats['merged_concepts_count'] > 0 else 0
            f.write(f"  {category}: {count} 个概念 ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n\n")
        
        # 合并后的概念列表
        f.write("🧠 合并后的概念列表（按置信度排序）:\n\n")
        
        # 按置信度排序
        sorted_concepts = sorted(merged_concepts, key=lambda x: x['confidence_score'], reverse=True)
        
        for i, concept in enumerate(sorted_concepts, 1):
            f.write(f"{i:3d}. 【{concept['category']}】{concept['name']}\n")
            f.write(f"     置信度: {concept['confidence_score']:.3f} | ")
            f.write(f"频率: {concept['frequency']} | ")
            f.write(f"来源chunks: {len(concept['source_chunks'])} 个\n")
            if concept['keywords']:
                f.write(f"     关键词: {', '.join(concept['keywords'])}\n")
            if len(concept['source_concepts']) > 1:
                f.write(f"     合并了: {len(concept['source_concepts'])} 个相似概念\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("📄 查看详细信息: 请参考 concepts/ 目录下的单独概念文件\n")
        f.write("📋 JSON格式数据: 请参考对应的 _merge_data.json 文件\n")
        f.write("💡 这是基于已有chunking结果的演示版concept merge效果\n")
        f.write("=" * 80 + "\n")
    
    return overview_path

def export_demo_concepts(merged_concepts: List[Dict], concept_to_chunks: Dict, 
                        chunking_dir: str, output_dir: str) -> List[str]:
    """导出详细概念文件"""
    concept_files = []
    concepts_dir = os.path.join(output_dir, "concepts")
    
    # 读取chunk文本内容
    chunk_texts = {}
    for filename in os.listdir(chunking_dir):
        if filename.endswith('.txt'):
            chunk_id = filename.replace('.txt', '')
            with open(os.path.join(chunking_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                # 提取主要文本内容
                text_start = content.find("📄 Chunk 完整文本内容:")
                if text_start != -1:
                    text_content = content[text_start:]
                    text_end = text_content.find("\n\n============================================================")
                    if text_end != -1:
                        chunk_texts[chunk_id] = text_content[len("📄 Chunk 完整文本内容:"):text_end].strip()
                    else:
                        chunk_texts[chunk_id] = text_content[len("📄 Chunk 完整文本内容:"):].strip()
    
    for i, concept in enumerate(merged_concepts):
        # 创建安全的文件名
        safe_name = "".join(c for c in concept['name'] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')[:30]  # 限制长度
        
        concept_filename = f"concept_{i:03d}_{safe_name}.txt"
        concept_path = os.path.join(concepts_dir, concept_filename)
        
        with open(concept_path, 'w', encoding='utf-8') as f:
            # 头部信息
            f.write("=" * 80 + "\n")
            f.write("概念合并(Concept Merge)详细结果 - 演示版\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"概念编号: {i+1}\n")
            f.write(f"概念ID: {concept['id']}\n")
            f.write(f"概念名称: {concept['name']}\n")
            f.write(f"概念类别: {concept['category']}\n")
            f.write(f"置信度分数: {concept['confidence_score']:.3f}\n")
            f.write(f"出现频率: {concept['frequency']} 次\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # 概念详细信息
            f.write("🧠 概念详细信息:\n\n")
            f.write(f"概念文本: {concept['text']}\n\n")
            
            # 关键词
            if concept['keywords']:
                f.write(f"🔑 关键词:\n")
                for j, keyword in enumerate(concept['keywords'], 1):
                    f.write(f"  {j}. {keyword}\n")
                f.write("\n")
            
            # 合并信息
            if len(concept['source_concepts']) > 1:
                f.write(f"🔗 概念合并信息:\n")
                f.write(f"  本概念由以下 {len(concept['source_concepts'])} 个相似概念合并而成:\n")
                for j, source in enumerate(concept['source_concepts'], 1):
                    f.write(f"    {j}. {source}\n")
                f.write("\n")
            
            # 来源chunk信息
            f.write(f"📄 来源chunk信息 (共{len(concept['source_chunks'])}个):\n\n")
            for j, chunk_id in enumerate(concept['source_chunks'], 1):
                f.write(f"  {j}. Chunk ID: {chunk_id}\n")
                if chunk_id in chunk_texts:
                    preview = chunk_texts[chunk_id][:300] + "..." if len(chunk_texts[chunk_id]) > 300 else chunk_texts[chunk_id]
                    f.write(f"     文本预览: {preview}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("💡 这是基于chunking结果的演示版concept merge\n")
            f.write("💡 实际的概念合并会使用更复杂的语义相似度计算\n")
            f.write("=" * 80 + "\n")
        
        concept_files.append(concept_path)
    
    return concept_files

def export_demo_json(merged_concepts: List[Dict], stats: Dict, output_dir: str) -> str:
    """导出JSON格式数据"""
    json_path = os.path.join(output_dir, "attention_is_all_you_need_concept_merge_data.json")
    
    json_data = {
        "metadata": {
            "document_name": "Attention Is All You Need",
            "processing_time": datetime.now().isoformat(),
            "processor": "演示版概念合并算法",
            "source": "chunking_export/chunks/",
            "statistics": stats
        },
        "merged_concepts": merged_concepts
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    return json_path

def generate_demo_report(stats: Dict, overview_path: str, concept_files: List[str], 
                        json_path: str, output_dir: str) -> str:
    """生成详细报告"""
    report_path = os.path.join(output_dir, "attention_is_all_you_need_concept_merge_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 概念合并(Concept Merge)演示报告\n\n")
        f.write(f"**文档**: Attention Is All You Need\n")
        f.write(f"**处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**演示说明**: 基于已有chunking结果的概念合并演示\n\n")
        
        f.write(f"## 📊 合并效果统计\n\n")
        f.write(f"| 指标 | 数值 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| 原始概念总数 | {stats['original_concepts_count']} |\n")
        f.write(f"| 合并后概念数 | {stats['merged_concepts_count']} |\n")
        f.write(f"| 合并减少率 | {stats['reduction_ratio']:.1%} |\n")
        f.write(f"| 高频概念数 | {stats['high_frequency_concepts']} |\n")
        f.write(f"| 平均置信度 | {stats['avg_confidence']:.3f} |\n\n")
        
        f.write(f"## 📂 概念类别分布\n\n")
        for category, count in stats['category_distribution'].items():
            percentage = count / stats['merged_concepts_count'] * 100 if stats['merged_concepts_count'] > 0 else 0
            f.write(f"- **{category}**: {count} 个概念 ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write(f"## 📁 生成的文件\n\n")
        f.write(f"- **总览文件**: `{os.path.basename(overview_path)}`\n")
        f.write(f"- **JSON数据**: `{os.path.basename(json_path)}`\n")
        f.write(f"- **详细概念文件**: {len(concept_files)} 个 (在 `concepts/` 目录)\n\n")
        
        f.write(f"## 🎯 演示说明\n\n")
        f.write(f"这是一个基于已有chunking结果的概念合并演示，展示了concept merge功能的效果：\n\n")
        f.write(f"1. **概念收集**: 从 `chunking_export/chunks/` 目录读取所有概念\n")
        f.write(f"2. **频率分析**: 统计每个概念的出现频率\n")
        f.write(f"3. **相似性检测**: 基于词汇重叠度检测相似概念\n")
        f.write(f"4. **概念合并**: 将相似概念合并为更通用的表述\n")
        f.write(f"5. **分类标注**: 为每个概念分配类别标签\n")
        f.write(f"6. **置信度计算**: 基于频率和其他因素计算置信度\n\n")
        
        f.write(f"## 💡 实际应用价值\n\n")
        f.write(f"通过概念合并，我们将 {stats['original_concepts_count']} 个原始概念精炼为 {stats['merged_concepts_count']} 个高质量概念，")
        f.write(f"减少了 {stats['reduction_ratio']:.1%} 的冗余，同时保持了概念的完整性和可用性。\n\n")
        f.write(f"这些合并后的概念可以用于：\n")
        f.write(f"- 📖 文档知识图谱构建\n")
        f.write(f"- 🔍 智能检索系统\n")
        f.write(f"- 📝 自动摘要生成\n")
        f.write(f"- 🎓 学习内容推荐\n")
    
    return report_path

if __name__ == "__main__":
    demo_concept_merge_from_chunking() 