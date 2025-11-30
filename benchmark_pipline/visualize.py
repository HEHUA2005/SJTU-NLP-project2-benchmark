#!/usr/bin/env python3
"""
可视化模块：生成评分结果的可视化图表
"""

import csv
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置matplotlib使用非交互式后端
matplotlib.use('Agg')

class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    RESET = "\033[0m"


def load_evaluation_results(input_path: Path) -> List[Dict]:
    """加载评分结果"""
    results = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = dict(row)
            for key in ['final_score', 'source_accuracy_score', 'content_accuracy_score',
                       'completeness_score', 'relevance_score']:
                if key in result:
                    result[key] = float(result[key])
            results.append(result)
    return results


def create_visualizations(results: List[Dict], output_dir: Path, split_name: str = None):
    """生成评分可视化图表"""
    if not results:
        print(f"{Colors.YELLOW}没有数据可供可视化{Colors.RESET}")
        return

    print(f"\n{Colors.YELLOW}生成可视化图表...{Colors.RESET}")
    output_dir.mkdir(parents=True, exist_ok=True)

    final_scores = [r["final_score"] for r in results]
    source_scores = [r["source_accuracy_score"] for r in results]
    content_scores = [r["content_accuracy_score"] for r in results]
    completeness_scores = [r["completeness_score"] for r in results]
    relevance_scores = [r["relevance_score"] for r in results]

    avg_scores = {
        'Source Accuracy\n(60%)': np.mean(source_scores),
        'Content Accuracy\n(20%)': np.mean(content_scores),
        'Completeness\n(15%)': np.mean(completeness_scores),
        'Relevance\n(5%)': np.mean(relevance_scores),
        'Final Score': np.mean(final_scores)
    }

    # 文件名前缀
    prefix = f"{split_name}_" if split_name else ""

    # 1. 平均分柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax.bar(range(len(avg_scores)), list(avg_scores.values()), color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(avg_scores)))
    ax.set_xticklabels(list(avg_scores.keys()), fontsize=11)
    ax.set_ylabel('Average Score', fontsize=12)
    title = f'RAG System Evaluation - Average Scores by Dimension'
    if split_name:
        title += f'\n({split_name})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}average_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{Colors.GREEN}  ✓ 平均分柱状图: {output_dir / f'{prefix}average_scores.png'}{Colors.RESET}")

    # 2. 分数分布箱线图
    fig, ax = plt.subplots(figsize=(12, 6))
    box_data = [source_scores, content_scores, completeness_scores, relevance_scores, final_scores]
    bp = ax.boxplot(box_data, labels=['Source Accuracy\n(60%)', 'Content Accuracy\n(20%)', 'Completeness\n(15%)', 'Relevance\n(5%)', 'Final Score'],
                    patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Score', fontsize=12)
    title = 'RAG System Evaluation - Score Distribution'
    if split_name:
        title += f'\n({split_name})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{Colors.GREEN}  ✓ 分数分布箱线图: {output_dir / f'{prefix}score_distribution.png'}{Colors.RESET}")

    # 3. 雷达图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    categories = ['Source Accuracy', 'Content Accuracy', 'Completeness', 'Relevance']
    values = [np.mean(source_scores), np.mean(content_scores),
              np.mean(completeness_scores), np.mean(relevance_scores)]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    ax.plot(angles, values, 'o-', linewidth=2, color='#4ECDC4')
    ax.fill(angles, values, alpha=0.25, color='#4ECDC4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 10)
    title = 'RAG System Evaluation - Radar Chart'
    if split_name:
        title += f'\n({split_name})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{Colors.GREEN}  ✓ 雷达图: {output_dir / f'{prefix}radar_chart.png'}{Colors.RESET}")

    # 4. 最终得分分布直方图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(final_scores, bins=20, color='#98D8C8', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(final_scores), color='red', linestyle='--', linewidth=2, label=f'Average: {np.mean(final_scores):.2f}')
    ax.set_xlabel('Final Score', fontsize=12)
    ax.set_ylabel('Number of Questions', fontsize=12)
    title = 'RAG System Evaluation - Final Score Distribution'
    if split_name:
        title += f'\n({split_name})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}final_score_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{Colors.GREEN}  ✓ 最终得分分布图: {output_dir / f'{prefix}final_score_histogram.png'}{Colors.RESET}")

    # 5. 综合对比图（如果问题数量合适）
    if len(results) <= 100:  # 只在问题数量不太多时生成
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(results))
        width = 0.2
        ax.bar(x - 1.5*width, source_scores, width, label='Source Accuracy', color='#FF6B6B', alpha=0.8)
        ax.bar(x - 0.5*width, content_scores, width, label='Content Accuracy', color='#4ECDC4', alpha=0.8)
        ax.bar(x + 0.5*width, completeness_scores, width, label='Completeness', color='#45B7D1', alpha=0.8)
        ax.bar(x + 1.5*width, relevance_scores, width, label='Relevance', color='#FFA07A', alpha=0.8)
        ax.set_xlabel('Question Index', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        title = 'RAG System Evaluation - Scores Comparison by Question'
        if split_name:
            title += f'\n({split_name})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if len(results) > 20:
            ax.set_xticks(x[::max(1, len(results)//20)])
            ax.set_xticklabels(x[::max(1, len(results)//20)])
        plt.tight_layout()
        plt.savefig(output_dir / f'{prefix}all_questions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{Colors.GREEN}  ✓ 所有问题对比图: {output_dir / f'{prefix}all_questions_comparison.png'}{Colors.RESET}")

    # 6. 权重贡献分析图
    fig, ax = plt.subplots(figsize=(10, 10))
    contributions = {
        'Source Accuracy': np.mean(source_scores) * 0.6,
        'Content Accuracy': np.mean(content_scores) * 0.2,
        'Completeness': np.mean(completeness_scores) * 0.15,
        'Relevance': np.mean(relevance_scores) * 0.05
    }
    labels = list(contributions.keys())
    values = list(contributions.values())
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    title = 'RAG System Evaluation - Weight Contribution'
    if split_name:
        title += f'\n({split_name})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}weight_contribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{Colors.GREEN}  ✓ 权重贡献分析图: {output_dir / f'{prefix}weight_contribution.png'}{Colors.RESET}")

    print(f"{Colors.GREEN}✓ 所有可视化图表已保存到: {output_dir}{Colors.RESET}")


def visualize_results(csv_file: Path, output_dir: Path = None, split_name: str = None):
    """
    为评分结果生成可视化图表

    Args:
        csv_file: 评分结果CSV文件路径
        output_dir: 可视化图表输出目录（默认为CSV文件同目录下的visualizations子目录）
        split_name: split名称（用于图表标题）
    """
    if not csv_file.exists():
        print(f"{Colors.RED}评分结果文件不存在: {csv_file}{Colors.RESET}")
        return

    if output_dir is None:
        output_dir = csv_file.parent / "visualizations"

    try:
        results = load_evaluation_results(csv_file)
        if results:
            create_visualizations(results, output_dir, split_name)
        else:
            print(f"{Colors.YELLOW}没有数据可供可视化{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}生成可视化图表失败: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
