#!/usr/bin/env python3
"""
测试两个新数据集的pipeline功能并评估生成数据质量
"""

import sys
import os
import logging
import json
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def analyze_qa_quality(qa_pairs: List[Dict], file_name: str) -> Dict[str, Any]:
    """分析问答对的质量"""
    logger.info(f"📊 分析 {file_name} 的问答对质量...")
    
    if not qa_pairs:
        return {
            "total_pairs": 0,
            "quality_score": 0,
            "issues": ["没有生成问答对"],
            "recommendations": ["检查数据生成配置和API连接"]
        }
    
    quality_analysis = {
        "total_pairs": len(qa_pairs),
        "question_types": {},
        "difficulty_levels": {},
        "quality_issues": [],
        "good_examples": [],
        "quality_score": 0
    }
    
    for i, qa in enumerate(qa_pairs):
        # 统计问题类型
        q_type = qa.get('type', 'unknown')
        quality_analysis["question_types"][q_type] = quality_analysis["question_types"].get(q_type, 0) + 1
        
        # 统计难度级别
        difficulty = qa.get('difficulty', 'unknown')
        quality_analysis["difficulty_levels"][difficulty] = quality_analysis["difficulty_levels"].get(difficulty, 0) + 1
        
        # 质量检查
        question = qa.get('question', '')
        answer = qa.get('answer', '')
        
        # 检查是否有处理错误
        if 'Unable to provide' in answer or 'processing error' in answer:
            quality_analysis["quality_issues"].append(f"问答对{i+1}: 生成失败 - {answer}")
        elif len(question) < 10:
            quality_analysis["quality_issues"].append(f"问答对{i+1}: 问题过短")
        elif len(answer) < 20:
            quality_analysis["quality_issues"].append(f"问答对{i+1}: 答案过短")
        else:
            quality_analysis["good_examples"].append({
                "question": question,
                "answer": answer[:100] + "..." if len(answer) > 100 else answer,
                "type": q_type,
                "difficulty": difficulty
            })
    
    # 计算质量得分
    total_pairs = len(qa_pairs)
    good_pairs = len(quality_analysis["good_examples"])
    quality_analysis["quality_score"] = (good_pairs / total_pairs * 100) if total_pairs > 0 else 0
    
    return quality_analysis

def test_dataset(file_path: str, output_suffix: str) -> Dict[str, Any]:
    """测试单个数据集"""
    logger.info(f"🧪 测试数据集: {file_path}")
    
    try:
        from integrated_pipeline_new import ModularIntegratedPipeline
        
        # 创建pipeline，增加问题数量以获得更好的测试结果
        pipeline = ModularIntegratedPipeline(
            config_path="config.yml",
            output_dir=f"./test_quality_{output_suffix}",
            questions_per_type={
                "remember": 2,
                "understand": 2,
                "apply": 1,
                "analyze": 1,
                "evaluate": 0,  # 暂时减少以避免API问题
                "create": 0     # 暂时减少以避免API问题
            }
        )
        
        # 处理文档
        results = pipeline.process_single_file(file_path, save_intermediate=True)
        
        if not results:
            logger.error(f"❌ {file_path} 处理失败")
            return {"success": False, "error": "处理失败"}
        
        # 分析结果质量
        qa_analysis = analyze_qa_quality(results.get('qa_pairs', []), file_path)
        
        # 构建测试结果
        test_result = {
            "success": True,
            "file_path": file_path,
            "processing_stats": {
                "text_length": results.get('file_info', {}).get('text_length', 0),
                "concepts_count": len(results.get('concept_results', {}).get('concept_nodes', [])),
                "evidence_count": len(results.get('concept_results', {}).get('evidence_nodes', [])),
                "qa_pairs_count": len(results.get('qa_pairs', [])),
                "training_data_count": len(results.get('training_data', []))
            },
            "qa_quality_analysis": qa_analysis
        }
        
        return test_result
        
    except Exception as e:
        logger.error(f"❌ 测试 {file_path} 时发生异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

def compare_datasets(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """比较两个数据集的处理结果"""
    logger.info("📊 比较数据集处理结果...")
    
    comparison = {
        "dataset_comparison": [],
        "overall_summary": {},
        "recommendations": []
    }
    
    for result in results:
        if not result.get('success'):
            continue
            
        dataset_summary = {
            "file": result['file_path'],
            "text_length": result['processing_stats']['text_length'],
            "qa_quality_score": result['qa_quality_analysis']['quality_score'],
            "total_qa_pairs": result['qa_quality_analysis']['total_pairs'],
            "good_examples_count": len(result['qa_quality_analysis']['good_examples'])
        }
        comparison["dataset_comparison"].append(dataset_summary)
    
    # 总结
    if comparison["dataset_comparison"]:
        avg_quality = sum(ds['qa_quality_score'] for ds in comparison["dataset_comparison"]) / len(comparison["dataset_comparison"])
        total_qa_pairs = sum(ds['total_qa_pairs'] for ds in comparison["dataset_comparison"])
        
        comparison["overall_summary"] = {
            "average_quality_score": avg_quality,
            "total_qa_pairs_generated": total_qa_pairs,
            "datasets_processed": len(comparison["dataset_comparison"])
        }
        
        # 推荐
        if avg_quality < 50:
            comparison["recommendations"].append("质量得分较低，建议优化API配置和提示词")
        if total_qa_pairs < 4:
            comparison["recommendations"].append("生成的问答对数量较少，建议增加每种类型的问题数量")
        
        comparison["recommendations"].append("考虑增加更多类型的认知层次问题")
        comparison["recommendations"].append("优化概念提取以获得更好的上下文信息")
    
    return comparison

def main():
    """主测试函数"""
    logger.info("🚀 开始测试两个新数据集的质量...")
    
    datasets = [
        ("en.wikipedia.org_wiki_IEEE_Frank_Rosenblatt_Award.md", "ieee_award"),
        ("ieeexplore.ieee.org_author_37271220500.md", "ieee_author")
    ]
    
    results = []
    
    for file_path, suffix in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 测试数据集: {file_path}")
        logger.info(f"{'='*60}")
        
        if not os.path.exists(file_path):
            logger.error(f"❌ 文件不存在: {file_path}")
            continue
            
        result = test_dataset(file_path, suffix)
        results.append(result)
        
        # 显示单个数据集结果
        if result.get('success'):
            stats = result['processing_stats']
            qa_analysis = result['qa_quality_analysis']
            
            logger.info(f"✅ 处理成功!")
            logger.info(f"   📄 文本长度: {stats['text_length']} 字符")
            logger.info(f"   🧠 概念数量: {stats['concepts_count']}")
            logger.info(f"   📝 问答对数量: {stats['qa_pairs_count']}")
            logger.info(f"   🏆 质量得分: {qa_analysis['quality_score']:.1f}%")
            
            if qa_analysis['good_examples']:
                logger.info(f"   📋 优质示例:")
                for i, example in enumerate(qa_analysis['good_examples'][:2]):
                    logger.info(f"      {i+1}. [{example['type']}] {example['question']}")
                    logger.info(f"         答案: {example['answer']}")
            
            if qa_analysis['quality_issues']:
                logger.info(f"   ⚠️  质量问题:")
                for issue in qa_analysis['quality_issues'][:3]:
                    logger.info(f"      - {issue}")
        else:
            logger.error(f"❌ 处理失败: {result.get('error', 'Unknown error')}")
    
    # 比较结果
    logger.info(f"\n{'='*60}")
    logger.info("📊 数据集比较分析")
    logger.info(f"{'='*60}")
    
    comparison = compare_datasets(results)
    
    if comparison["dataset_comparison"]:
        logger.info("📈 数据集对比:")
        for ds in comparison["dataset_comparison"]:
            logger.info(f"   {ds['file']}:")
            logger.info(f"     - 文本长度: {ds['text_length']} 字符")
            logger.info(f"     - 问答对数量: {ds['total_qa_pairs']}")
            logger.info(f"     - 质量得分: {ds['qa_quality_score']:.1f}%")
        
        summary = comparison["overall_summary"]
        logger.info(f"\n📊 总体统计:")
        logger.info(f"   - 平均质量得分: {summary['average_quality_score']:.1f}%")
        logger.info(f"   - 总问答对数量: {summary['total_qa_pairs_generated']}")
        logger.info(f"   - 处理数据集数量: {summary['datasets_processed']}")
        
        if comparison["recommendations"]:
            logger.info(f"\n💡 改进建议:")
            for rec in comparison["recommendations"]:
                logger.info(f"   - {rec}")
    
    # 保存详细结果
    output_file = "dataset_quality_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "comparison": comparison
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n📁 详细分析结果已保存到: {output_file}")
    
    # 结论
    successful_tests = sum(1 for r in results if r.get('success'))
    total_tests = len(results)
    
    logger.info(f"\n🎯 测试总结: {successful_tests}/{total_tests} 数据集处理成功")
    
    if successful_tests == total_tests and comparison["overall_summary"].get('average_quality_score', 0) > 60:
        logger.info("🎉 pipeline对新数据集的处理质量良好！")
        return True
    elif successful_tests == total_tests:
        logger.info("✅ pipeline能够处理新数据集，但质量有待改进")
        return True
    else:
        logger.error("⚠️  pipeline在处理新数据集时遇到问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 