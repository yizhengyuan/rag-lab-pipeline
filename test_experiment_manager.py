#!/usr/bin/env python3
"""
实验管理器测试和演示脚本
=========================

展示新的实验输出管理功能
"""

import sys
from pathlib import Path
from utils.experiment_manager import ExperimentManager, create_experiment_manager
from utils.helpers import FileHelper

def demo_experiment_manager():
    """演示实验管理器功能"""
    
    print("🧪 实验管理器功能演示")
    print("=" * 50)
    
    # 1. 创建实验管理器
    print("1. 创建实验管理器...")
    manager = create_experiment_manager("attention is all you need.pdf")
    print(f"   ✅ 实验ID: {manager.experiment_name}")
    print(f"   📁 实验目录: {manager.experiment_dir}")
    print()
    
    # 2. 模拟步骤1结果
    print("2. 保存模拟的步骤1结果...")
    step1_result = {
        "success": True,
        "step": 1,
        "step_name": "文件加载与向量化存储",
        "processing_time": 15.6,
        "statistics": {
            "total_chunks": 45,
            "total_concepts": 127,
            "unique_concepts": 89
        },
        "vector_info": {
            "store_type": "chroma",
            "vectorized_nodes": 45,
            "storage_size_mb": 12.3,
            "dimension": 1536
        }
    }
    
    saved_files = manager.save_step_result(1, step1_result, ['txt', 'json', 'md'])
    for format_type, file_path in saved_files.items():
        print(f"   ✅ {format_type.upper()}: {file_path}")
    print()
    
    # 3. 模拟步骤2结果
    print("3. 保存模拟的步骤2结果...")
    step2_result = {
        "success": True,
        "step": 2,
        "step_name": "文档分块与概念提取",
        "processing_time": 8.2,
        "statistics": {
            "total_chunks": 45,
            "avg_chunk_length": 834.2,
            "total_concepts": 127
        }
    }
    
    saved_files = manager.save_step_result(2, step2_result)
    print(f"   ✅ 步骤2结果已保存")
    print()
    
    # 4. 显示实验摘要
    print("4. 实验摘要:")
    summary = manager.get_experiment_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    print()
    
    # 5. 列出输出文件
    print("5. 输出文件列表:")
    files = manager.list_output_files()
    for step, step_files in files.items():
        print(f"   {step}:")
        for format_type, file_path in step_files.items():
            print(f"     - {format_type}: {file_path}")
    print()
    
    # 6. 列出所有实验
    print("6. 所有实验列表:")
    experiments = ExperimentManager.list_experiments()
    for exp in experiments[:3]:  # 只显示前3个
        print(f"   📁 {exp['name']}: {exp['status']} ({exp['steps_completed']} 步完成)")
    print()
    
    print("✅ 实验管理器演示完成！")
    print(f"📁 查看结果: ls {manager.experiment_dir}")

if __name__ == "__main__":
    demo_experiment_manager() 