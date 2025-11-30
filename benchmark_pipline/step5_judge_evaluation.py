#!/usr/bin/env python3
"""
Step 5: Judge Model 评分模块
独立版本，不依赖benchmark目录
"""

import json
import csv
import yaml
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    RESET = "\033[0m"


def load_config(config_path: Path) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def build_judge_prompt(
    query: str,
    standard_answer: str,
    rag_answer: str,
    standard_page_range: str,
    standard_material: str,
    config: Dict,
) -> str:
    """构建判官评分提示词"""

    prompt = f"""你是一个专业的RAG系统评估判官。你的任务是对RAG系统的回答进行多维度评分。

## 评分标准

你需要从以下四个维度进行评分（每个维度0-10分）：

1. **来源准确性 (Source Accuracy)** - 权重60%
   - **这是RAG系统检索能力的核心体现，是最重要的评分维度**
   - 检查RAG回答中是否提到了页码、章节、或来源信息
   - 标准答案来源：{standard_material}，页码：{standard_page_range}
   - 评估标准：
     * 明确提到正确页码/章节（如"根据第15页"、"第3章提到"）：8-10分
     * 提到了页码但不完全准确：5-7分
     * 提到了来源但没有具体页码：3-5分
     * 完全没有提到任何来源信息：0-2分

2. **内容准确性 (Content Accuracy)** - 权重20%
   - 评估RAG回答的核心信息是否正确
   - 是否存在事实性错误
   - 是否与标准答案的主要观点一致

3. **完整性 (Completeness)** - 权重15%
   - 评估RAG回答是否涵盖了标准答案的关键信息
   - 是否遗漏重要内容
   - 信息的详细程度是否足够

4. **相关性 (Relevance)** - 权重5%
   - 评估RAG回答是否切题
   - 是否包含无关信息
   - 回答的重点是否正确

## 输入信息

**问题：** {query}

**标准答案：**
{standard_answer}
**标准答案来源：** {standard_material}，页码：{standard_page_range}

**RAG系统回答：**
{rag_answer}

## 输出格式

请以 JSON 格式输出评分结果：

```json
{{
  "source_accuracy_score": 8.5,  // 来源准确性评分（0-10，着重检查回答中是否提到页码/来源）
  "content_accuracy_score": 7.0,  // 内容准确性评分（0-10）
  "completeness_score": 7.5,  // 完整性评分（0-10）
  "relevance_score": 9.0,  // 相关性评分（0-10）
  "final_score": 7.9,  // 最终得分（加权平均：来源60% + 内容20% + 完整15% + 相关5%）
  "source_accuracy_reasoning": "回答中明确提到'根据第15页'，与标准答案页码一致，来源定位准确",
  "content_accuracy_reasoning": "核心信息准确...",
  "completeness_reasoning": "涵盖了主要内容...",
  "relevance_reasoning": "回答切题...",
  "overall_reasoning": "整体回答质量良好，检索能力强，能准确定位来源..."
}}
```

**计算公式**：final_score = source_accuracy * 0.6 + content_accuracy * 0.2 + completeness * 0.15 + relevance * 0.05

**评分要点**：
- 来源准确性占60%权重，是RAG系统最重要的能力指标
- 即使内容正确，如果没有提到来源信息，来源准确性得分也应该很低
- 好的RAG回答应该像："根据课件第15页，毛泽东思想的核心是..."

请只输出 JSON，不要有其他内容。
"""

    return prompt


def call_llm(prompt: str, config: Dict) -> str:
    """调用 LLM API"""
    api_config = config["api"]

    for attempt in range(api_config["max_retries"]):
        try:
            client = OpenAI(
                api_key=api_config["api_key"], base_url=api_config["base_url"]
            )

            response = client.chat.completions.create(
                model=api_config["model_id"],
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的RAG系统评估判官，擅长客观、公正地评估答案质量。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # 使用较低的温度以保证评分的一致性
                max_tokens=api_config["max_tokens"],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.warning(
                f"API 调用失败 (尝试 {attempt + 1}/{api_config['max_retries']}): {e}"
            )
            if attempt == api_config["max_retries"] - 1:
                raise
    return ""


def parse_judge_response(response: str) -> Dict:
    """解析 LLM 返回的评分结果"""
    try:
        # 尝试提取 JSON 部分
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()

        result = json.loads(json_str)

        # 验证必要字段
        required_fields = [
            "source_accuracy_score",
            "content_accuracy_score",
            "completeness_score",
            "relevance_score",
            "final_score",
        ]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"缺少必要字段: {field}")

        return result
    except Exception as e:
        logging.error(f"解析评分结果失败: {e}")
        logging.error(f"响应内容: {response[:500]}")
        # 返回默认评分
        return {
            "source_accuracy_score": 0.0,
            "content_accuracy_score": 0.0,
            "completeness_score": 0.0,
            "relevance_score": 0.0,
            "final_score": 0.0,
            "source_accuracy_reasoning": f"解析失败: {str(e)}",
            "content_accuracy_reasoning": "",
            "completeness_reasoning": "",
            "relevance_reasoning": "",
            "overall_reasoning": "",
        }


def load_rag_results(rag_results_path: Path) -> List[Dict]:
    """加载 step4 的结果（包含标准答案和agent回答）"""
    results = []
    with open(rag_results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def evaluate_single_question(row: Dict, config: Dict) -> Dict:
    """评估单个问题（row包含step4的所有字段）"""
    query = row.get("query", "")
    standard_answer = row.get("standard_answer", "")
    standard_page_range = row.get("page_range", "")
    standard_material = row.get("material", "")
    agent_answer = row.get("agent_answer", "")

    # 构建提示词
    prompt = build_judge_prompt(
        query=query,
        standard_answer=standard_answer,
        rag_answer=agent_answer,
        standard_page_range=standard_page_range,
        standard_material=standard_material,
        config=config,
    )

    # 调用 LLM
    response = call_llm(prompt, config)

    # 解析结果
    judge_result = parse_judge_response(response)

    # 合并结果
    result = {
        "query": query,
        "standard_answer": standard_answer,
        "standard_page_range": standard_page_range,
        "agent_answer": agent_answer,
        "source_accuracy_score": judge_result.get("source_accuracy_score", 0.0),
        "content_accuracy_score": judge_result.get("content_accuracy_score", 0.0),
        "completeness_score": judge_result.get("completeness_score", 0.0),
        "relevance_score": judge_result.get("relevance_score", 0.0),
        "final_score": judge_result.get("final_score", 0.0),
        "source_accuracy_reasoning": judge_result.get("source_accuracy_reasoning", ""),
        "content_accuracy_reasoning": judge_result.get(
            "content_accuracy_reasoning", ""
        ),
        "completeness_reasoning": judge_result.get("completeness_reasoning", ""),
        "relevance_reasoning": judge_result.get("relevance_reasoning", ""),
        "overall_reasoning": judge_result.get("overall_reasoning", ""),
        "course": row.get("course", ""),
        "material": standard_material,
        "question_type": row.get("question_type", ""),
    }

    return result


def evaluate_single_question_wrapper(row: Dict, config: Dict) -> Dict:
    """评估单个问题的包装函数（用于并发处理）"""
    try:
        result = evaluate_single_question(row, config)
        return result
    except Exception as e:
        logging.error(f"评估失败: {e}")
        query = row.get("query", "")[:50]
        print(f"{Colors.RED}✗ 评估失败: {query}... - {e}{Colors.RESET}")
        # 返回一个默认结果，避免丢失这个问题
        return {
            "query": row.get("query", ""),
            "standard_answer": row.get("standard_answer", ""),
            "standard_page_range": row.get("page_range", ""),
            "agent_answer": row.get("agent_answer", ""),
            "source_accuracy_score": 0.0,
            "content_accuracy_score": 0.0,
            "completeness_score": 0.0,
            "relevance_score": 0.0,
            "final_score": 0.0,
            "source_accuracy_reasoning": f"评估失败: {str(e)}",
            "content_accuracy_reasoning": "",
            "completeness_reasoning": "",
            "relevance_reasoning": "",
            "overall_reasoning": "",
            "course": row.get("course", ""),
            "material": row.get("material", ""),
            "question_type": row.get("question_type", ""),
        }


def evaluate_batch(
    rag_results: List[Dict], config: Dict, workers: int = 4
) -> List[Dict]:
    """批量评估问题（支持并行处理）"""
    all_results = []

    print(f"{Colors.CYAN}使用 {workers} 个并行worker进行评估{Colors.RESET}")

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务
        future_to_row = {
            executor.submit(evaluate_single_question_wrapper, row, config): row
            for row in rag_results
        }

        # 使用tqdm显示进度
        with tqdm(total=len(rag_results), desc="评估问题") as pbar:
            for future in as_completed(future_to_row):
                result = future.result()
                all_results.append(result)
                pbar.update(1)

    # 计算平均得分
    avg_score = (
        sum(r["final_score"] for r in all_results) / len(all_results)
        if all_results
        else 0
    )
    print(
        f"{Colors.GREEN}✓ 完成 {len(all_results)} 个问题的评估，平均得分: {avg_score:.2f}{Colors.RESET}"
    )

    return all_results


def save_results(results: List[Dict], output_path: Path, config: Dict):
    """保存评分结果"""
    output_format = config.get("judge_evaluation", {}).get("output", {}).get("format", "csv")
    detailed_reasoning = config.get("judge_evaluation", {}).get("detailed_reasoning", True)

    if output_format == "csv":
        # 确定字段
        fieldnames = [
            "query",
            "standard_answer",
            "standard_page_range",
            "agent_answer",
            "source_accuracy_score",
            "content_accuracy_score",
            "completeness_score",
            "relevance_score",
            "final_score",
            "course",
            "material",
            "question_type",
        ]

        if detailed_reasoning:
            fieldnames.extend(
                [
                    "source_accuracy_reasoning",
                    "content_accuracy_reasoning",
                    "completeness_reasoning",
                    "relevance_reasoning",
                    "overall_reasoning",
                ]
            )

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {field: result.get(field, "") for field in fieldnames}
                writer.writerow(row)

    elif output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{Colors.GREEN}✓ 评分结果已保存到: {output_path}{Colors.RESET}")


def print_summary(results: List[Dict]):
    """打印评估摘要"""
    if not results:
        print(f"{Colors.RED}没有评估结果{Colors.RESET}")
        return

    total = len(results)
    avg_final_score = sum(r["final_score"] for r in results) / total
    avg_source_score = sum(r["source_accuracy_score"] for r in results) / total
    avg_content_score = sum(r["content_accuracy_score"] for r in results) / total
    avg_completeness_score = sum(r["completeness_score"] for r in results) / total
    avg_relevance_score = sum(r["relevance_score"] for r in results) / total

    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}评估摘要{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"总问题数: {total}")
    print(f"\n{Colors.GREEN}平均最终得分: {avg_final_score:.2f}/10{Colors.RESET}")
    print(f"平均来源准确性得分: {avg_source_score:.2f}/10 (权重60%)")
    print(f"平均内容准确性得分: {avg_content_score:.2f}/10 (权重20%)")
    print(f"平均完整性得分: {avg_completeness_score:.2f}/10 (权重15%)")
    print(f"平均相关性得分: {avg_relevance_score:.2f}/10 (权重5%)")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="步骤 5: Judge Model 评分模块")
    parser.add_argument("--config", "-c", required=True, help="配置文件路径")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Step4 输出的 CSV 文件路径（包含标准答案和agent回答）",
    )
    parser.add_argument("--output", "-o", help="输出文件路径（可选，默认根据配置生成）")
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="并行评估的worker数量（默认：4）",
    )
    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"{Colors.RED}配置文件不存在: {config_path}{Colors.RESET}")
        return

    config = load_config(config_path)

    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting judge evaluation process")

    # 解析输入文件路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"{Colors.RED}输入文件不存在: {input_path}{Colors.RESET}")
        return

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        # 自动生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_format = config.get("judge_evaluation", {}).get("output", {}).get("format", "csv")
        output_path = input_path.parent / f"evaluation_{timestamp}.{output_format}"

    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}Step 5: Judge Model Evaluation{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"Config file: {config_path}")
    print(f"Input file (from Step4): {input_path}")
    print(f"Output file: {output_path}")
    print(f"LLM Model: {config['api']['model_id']}")
    print(f"Workers: {args.workers}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")

    # 加载step4的结果
    print(f"{Colors.YELLOW}[1/2] Loading Step4 results...{Colors.RESET}")
    rag_results = load_rag_results(input_path)
    print(
        f"{Colors.GREEN}✓ Loaded {len(rag_results)} questions with agent answers{Colors.RESET}"
    )

    if not rag_results:
        print(f"{Colors.RED}没有数据，请检查输入文件{Colors.RESET}")
        return

    # 批量评估
    print(f"{Colors.YELLOW}[2/2] Evaluating questions...{Colors.RESET}")
    results = evaluate_batch(rag_results, config, workers=args.workers)

    # 保存结果
    save_results(results, output_path, config)

    # 打印摘要
    print_summary(results)

    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.GREEN}Evaluation completed!{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


if __name__ == "__main__":
    main()
