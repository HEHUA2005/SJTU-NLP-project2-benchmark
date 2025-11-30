#!/usr/bin/env python3
"""
Step 4: RAG系统回答问题
独立版本，不依赖benchmark目录
"""

import sys
import csv
import threading
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    RESET = '\033[0m'


def process_single_question(agent, question: Dict, lock: threading.Lock) -> Dict:
    """处理单个问题（用于并发）"""
    query = question.get("query", "").strip()
    if not query:
        return None

    try:
        agent_answer = agent.answer_question(query, chat_history=None)

        if len(agent_answer) > 1000:
            agent_answer = agent_answer[:1000]

        # 保留step3的所有字段，添加agent_answer
        result = dict(question)  # 复制所有原始字段
        result["agent_answer"] = agent_answer
        return result
    except Exception as e:
        result = dict(question)
        result["agent_answer"] = f"[错误] {str(e)}"
        return result


def process_questions_parallel(agent, questions: List[Dict], workers: int = 4) -> List[Dict]:
    """并行处理问题"""
    results = []
    lock = threading.Lock()

    print(f"{Colors.CYAN}使用 {workers} 个并行worker处理问题{Colors.RESET}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_question = {
            executor.submit(process_single_question, agent, q, lock): q
            for q in questions
        }

        with tqdm(total=len(questions), desc="处理问题") as pbar:
            for future in as_completed(future_to_question):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="步骤 4: RAG 系统回答问题")
    parser.add_argument("--input", "-i", required=True, help="输入的benchmark CSV文件")
    parser.add_argument("--output", "-o", required=True, help="输出的CSV文件（包含agent_answer）")
    parser.add_argument("--workers", "-w", type=int, default=4, help="并行worker数量（默认：4）")
    args = parser.parse_args()

    # 导入RAG Agent
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rag_agent import RAGAgent

    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Step 4: RAG Answer Generation{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Workers: {args.workers}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")

    # 初始化agent
    print(f"{Colors.YELLOW}初始化RAG Agent...{Colors.RESET}")
    agent = RAGAgent()
    print(f"{Colors.GREEN}✓ RAG Agent初始化完成{Colors.RESET}\n")

    # 读取问题
    print(f"{Colors.YELLOW}读取问题...{Colors.RESET}")
    questions = []
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)
    print(f"{Colors.GREEN}✓ 读取了 {len(questions)} 个问题{Colors.RESET}\n")

    # 并行处理
    print(f"{Colors.YELLOW}开始处理问题...{Colors.RESET}")
    results = process_questions_parallel(agent, questions, workers=args.workers)

    # 保存结果
    print(f"\n{Colors.YELLOW}保存结果...{Colors.RESET}")
    fieldnames = ["query", "standard_answer", "course", "material", "page_range", "question_type", "agent_answer"]

    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"{Colors.GREEN}✓ 保存了 {len(results)} 个结果到: {args.output}{Colors.RESET}")

    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.GREEN}Step 4 完成!{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")


if __name__ == "__main__":
    main()
