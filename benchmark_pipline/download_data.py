#!/usr/bin/env python3
"""
从Hugging Face Hub下载数据
- 下载PDF文件到主目录的 data/ 下
- 下载QA数据集到 huggingface_pipeline/QA_data/ 下
保持目录结构
"""

from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from datasets import load_dataset
import csv
import sys

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    MAGENTA = '\033[35m'
    RESET = '\033[0m'


# 硬编码仓库信息
PDF_REPO_ID = "HEHUA2005/rag-benchmark-pdf-data"
QA_REPO_ID = "HEHUA2005/rag-benchmark-qa-dataset"
REPO_TYPE = "dataset"

# 所有可用的QA数据集splits
ALL_SPLITS = [
    "Mao_Zedong_Thought",
    "Principles_of_Marxism",
    "Outline_of_Modern_and_Contemporary_Chinese_History",
    "Ideological_Morality_and_Legal_System",
    "An_Introduction_to_Xi_Jinping_Thought_on_Socialism_with_Chinese_Characteristics_for_a_New_Era"
]


def download_pdf_data(output_dir: Path, repo_id: str):
    """从Hugging Face Hub下载PDF数据"""

    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Downloading PDF Data{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"Repository: {repo_id}")
    print(f"Output directory: {output_dir}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 列出仓库中的所有文件
        print(f"{Colors.YELLOW}Fetching file list from repository...{Colors.RESET}")
        files = list_repo_files(repo_id=repo_id, repo_type=REPO_TYPE)

        # 过滤出PDF文件
        pdf_files = [f for f in files if f.endswith('.pdf')]

        if not pdf_files:
            print(f"{Colors.YELLOW}No PDF files found in repository{Colors.RESET}")
            return False

        print(f"{Colors.GREEN}✓ Found {len(pdf_files)} PDF files{Colors.RESET}\n")

        # 下载每个PDF文件
        success_count = 0
        for pdf_file in pdf_files:
            # 检查文件是否已存在
            local_path = output_dir / pdf_file

            if local_path.exists():
                print(f"{Colors.CYAN}Checking {pdf_file}...{Colors.RESET}")
                file_size = local_path.stat().st_size
                if file_size > 0:
                    print(f"{Colors.GREEN}✓ Already exists: {local_path} ({file_size:,} bytes){Colors.RESET}")
                    success_count += 1
                    continue
                else:
                    print(f"{Colors.YELLOW}  File exists but is empty, re-downloading...{Colors.RESET}")

            print(f"{Colors.YELLOW}Downloading {pdf_file}...{Colors.RESET}")

            try:
                # 下载文件
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    repo_type=REPO_TYPE,
                    filename=pdf_file,
                    local_dir=output_dir,
                    local_dir_use_symlinks=False
                )

                file_size = Path(downloaded_path).stat().st_size
                print(f"{Colors.GREEN}✓ Downloaded {pdf_file} ({file_size:,} bytes){Colors.RESET}")
                success_count += 1

            except Exception as e:
                print(f"{Colors.RED}✗ Failed to download {pdf_file}: {e}{Colors.RESET}")

        # 下载README（如果存在）
        if "README.md" in files:
            print(f"\n{Colors.YELLOW}Downloading README.md...{Colors.RESET}")
            try:
                readme_path = hf_hub_download(
                    repo_id=repo_id,
                    repo_type=REPO_TYPE,
                    filename="README.md",
                    local_dir=output_dir,
                    local_dir_use_symlinks=False
                )
                print(f"{Colors.GREEN}✓ Downloaded README.md{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.YELLOW}⚠ Failed to download README.md: {e}{Colors.RESET}")

        # 总结
        print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
        print(f"{Colors.BLUE}PDF Download Summary{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
        print(f"Successfully downloaded: {success_count}/{len(pdf_files)} files")
        print(f"Output directory: {output_dir}")
        print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")

        return success_count == len(pdf_files)

    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.RESET}")
        return False


def download_qa_datasets(qa_data_dir: Path, splits: list, repo_id: str):
    """下载QA数据集到本地QA_data目录"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Downloading QA Datasets{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"Repository: {repo_id}")
    print(f"Output directory: {qa_data_dir}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")

    qa_data_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = {}
    success_count = 0

    for split_name in splits:
        csv_file = qa_data_dir / f"{split_name}.csv"

        # 检查文件是否已存在
        if csv_file.exists():
            print(f"\n{Colors.CYAN}Checking {split_name}...{Colors.RESET}")
            try:
                # 验证文件是否有效
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    row_count = sum(1 for _ in reader)

                if row_count > 0:
                    downloaded_files[split_name] = csv_file
                    print(f"{Colors.GREEN}✓ Already exists: {csv_file} ({row_count} questions){Colors.RESET}")
                    success_count += 1
                    continue
                else:
                    print(f"{Colors.YELLOW}  File exists but is empty, re-downloading...{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.YELLOW}  File exists but is invalid ({e}), re-downloading...{Colors.RESET}")

        # 下载数据集
        print(f"\n{Colors.YELLOW}Downloading {split_name}...{Colors.RESET}")
        try:
            dataset = load_dataset(repo_id, split=split_name)

            # 保存为CSV
            with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                if len(dataset) > 0:
                    fieldnames = list(dataset[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for row in dataset:
                        writer.writerow(row)

            downloaded_files[split_name] = csv_file
            print(f"{Colors.GREEN}✓ Downloaded {len(dataset)} questions to {csv_file}{Colors.RESET}")
            success_count += 1

        except Exception as e:
            print(f"{Colors.RED}✗ Failed to download {split_name}: {e}{Colors.RESET}")

    # 总结
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}QA Dataset Download Summary{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"Successfully downloaded: {success_count}/{len(splits)} datasets")
    print(f"Output directory: {qa_data_dir}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")

    return downloaded_files, success_count == len(splits)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download data from Hugging Face Hub")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        help="PDF output directory (default: ../data)"
    )
    parser.add_argument(
        "--qa-dir",
        type=str,
        help="QA dataset output directory (default: ./QA_data)"
    )
    parser.add_argument(
        "--download",
        type=str,
        choices=["all", "pdf", "qa"],
        default="all",
        help="What to download: all, pdf, or qa (default: all)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        help=f"QA dataset splits to download (default: all). Available: {', '.join(ALL_SPLITS)}"
    )

    args = parser.parse_args()

    # 确定PDF输出目录
    if args.pdf_dir:
        pdf_output_dir = Path(args.pdf_dir)
    else:
        # 默认路径：相对于脚本位置的 ../data
        script_dir = Path(__file__).parent
        pdf_output_dir = script_dir.parent / "data"

    # 确定QA数据集输出目录
    if args.qa_dir:
        qa_output_dir = Path(args.qa_dir)
    else:
        # 默认路径：相对于脚本位置的 ./QA_data
        script_dir = Path(__file__).parent
        qa_output_dir = script_dir / "QA_data"

    # 确定要下载的QA splits
    if args.splits:
        splits_to_download = args.splits
        # 验证splits
        invalid_splits = [s for s in splits_to_download if s not in ALL_SPLITS]
        if invalid_splits:
            print(f"{Colors.RED}Invalid split names: {', '.join(invalid_splits)}{Colors.RESET}")
            print(f"Available splits: {', '.join(ALL_SPLITS)}")
            sys.exit(1)
    else:
        splits_to_download = ALL_SPLITS

    # 下载
    pdf_success = True
    qa_success = True

    if args.download in ["all", "pdf"]:
        pdf_success = download_pdf_data(pdf_output_dir, PDF_REPO_ID)

    if args.download in ["all", "qa"]:
        _, qa_success = download_qa_datasets(qa_output_dir, splits_to_download, QA_REPO_ID)

    # 总结
    print(f"\n{Colors.MAGENTA}{'='*60}{Colors.RESET}")
    print(f"{Colors.MAGENTA}Overall Summary{Colors.RESET}")
    print(f"{Colors.MAGENTA}{'='*60}{Colors.RESET}")

    if args.download in ["all", "pdf"]:
        pdf_status = f"{Colors.GREEN}✓ SUCCESS{Colors.RESET}" if pdf_success else f"{Colors.RED}✗ FAILED{Colors.RESET}"
        print(f"PDF Data: {pdf_status}")
        if pdf_success:
            print(f"  Location: {pdf_output_dir}")

    if args.download in ["all", "qa"]:
        qa_status = f"{Colors.GREEN}✓ SUCCESS{Colors.RESET}" if qa_success else f"{Colors.RED}✗ FAILED{Colors.RESET}"
        print(f"QA Datasets: {qa_status}")
        if qa_success:
            print(f"  Location: {qa_output_dir}")

    print(f"{Colors.MAGENTA}{'='*60}{Colors.RESET}\n")

    if pdf_success and qa_success:
        print(f"{Colors.GREEN}All data downloaded successfully!{Colors.RESET}\n")
        sys.exit(0)
    else:
        print(f"{Colors.RED}Some downloads failed.{Colors.RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
