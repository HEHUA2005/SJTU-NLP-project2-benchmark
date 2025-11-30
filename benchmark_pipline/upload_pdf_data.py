#!/usr/bin/env python3
"""
上传PDF数据到Hugging Face Hub
将 benchmark/pdf_data/ 下的PDF文件上传到 Hugging Face
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo
import sys

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    RESET = '\033[0m'


# 硬编码仓库信息
REPO_ID = "HEHUA2005/rag-benchmark-pdf-data"
REPO_TYPE = "dataset"


def upload_pdf_data(pdf_data_dir: Path, repo_id: str):
    """上传PDF数据到Hugging Face Hub"""

    if not pdf_data_dir.exists():
        print(f"{Colors.RED}PDF data directory not found: {pdf_data_dir}{Colors.RESET}")
        return False

    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Uploading PDF Data to Hugging Face Hub{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"Source directory: {pdf_data_dir}")
    print(f"Repository: {repo_id}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")

    # 初始化 Hugging Face API
    api = HfApi()

    # 创建仓库（如果不存在）
    try:
        print(f"{Colors.YELLOW}Creating repository (if not exists)...{Colors.RESET}")
        create_repo(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            exist_ok=True
        )
        print(f"{Colors.GREEN}✓ Repository ready{Colors.RESET}\n")
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to create repository: {e}{Colors.RESET}")
        return False

    # 收集所有PDF文件
    pdf_files = list(pdf_data_dir.glob("*/*.pdf"))

    if not pdf_files:
        print(f"{Colors.YELLOW}No PDF files found in {pdf_data_dir}{Colors.RESET}")
        return False

    print(f"{Colors.CYAN}Found {len(pdf_files)} PDF files{Colors.RESET}\n")

    # 上传每个PDF文件
    success_count = 0
    for pdf_file in pdf_files:
        # 获取相对路径（保持目录结构）
        relative_path = pdf_file.relative_to(pdf_data_dir)

        print(f"{Colors.YELLOW}Uploading {relative_path}...{Colors.RESET}")

        try:
            api.upload_file(
                path_or_fileobj=str(pdf_file),
                path_in_repo=str(relative_path),
                repo_id=repo_id,
                repo_type=REPO_TYPE,
            )
            print(f"{Colors.GREEN}✓ Uploaded {relative_path}{Colors.RESET}")
            success_count += 1
        except Exception as e:
            print(f"{Colors.RED}✗ Failed to upload {relative_path}: {e}{Colors.RESET}")

    # 上传README
    readme_file = pdf_data_dir / "README.md"
    if readme_file.exists():
        print(f"\n{Colors.YELLOW}Uploading README.md...{Colors.RESET}")
        try:
            api.upload_file(
                path_or_fileobj=str(readme_file),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type=REPO_TYPE,
            )
            print(f"{Colors.GREEN}✓ Uploaded README.md{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Failed to upload README.md: {e}{Colors.RESET}")

    # 总结
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Upload Summary{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"Successfully uploaded: {success_count}/{len(pdf_files)} files")
    print(f"Repository URL: https://huggingface.co/datasets/{repo_id}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")

    return success_count == len(pdf_files)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Upload PDF data to Hugging Face Hub")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        help="Path to PDF data directory (default: ../benchmark/pdf_data)"
    )

    args = parser.parse_args()

    # 确定PDF数据目录
    if args.pdf_dir:
        pdf_data_dir = Path(args.pdf_dir)
    else:
        # 默认路径：相对于脚本位置的 ../benchmark/pdf_data
        script_dir = Path(__file__).parent
        pdf_data_dir = script_dir.parent / "benchmark" / "pdf_data"

    # 上传
    success = upload_pdf_data(pdf_data_dir, REPO_ID)

    if success:
        print(f"{Colors.GREEN}All files uploaded successfully!{Colors.RESET}")
        sys.exit(0)
    else:
        print(f"{Colors.RED}Some files failed to upload.{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
