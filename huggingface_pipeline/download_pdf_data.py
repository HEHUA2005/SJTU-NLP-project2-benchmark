#!/usr/bin/env python3
"""
从Hugging Face Hub下载PDF数据
下载PDF文件到主目录的 data/ 下，保持目录结构
"""

from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
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


def download_pdf_data(output_dir: Path, repo_id: str):
    """从Hugging Face Hub下载PDF数据"""

    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}Downloading PDF Data from Hugging Face Hub{Colors.RESET}")
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
        print(f"{Colors.BLUE}Download Summary{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
        print(f"Successfully downloaded: {success_count}/{len(pdf_files)} files")
        print(f"Output directory: {output_dir}")
        print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")

        return success_count == len(pdf_files)

    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.RESET}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download PDF data from Hugging Face Hub")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: ../data)"
    )

    args = parser.parse_args()

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 默认路径：相对于脚本位置的 ../data
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "data"

    # 下载
    success = download_pdf_data(output_dir, REPO_ID)

    if success:
        print(f"{Colors.GREEN}All files downloaded successfully!{Colors.RESET}")
        print(f"\n{Colors.CYAN}You can now use the PDF files in: {output_dir}{Colors.RESET}\n")
        sys.exit(0)
    else:
        print(f"{Colors.RED}Some files failed to download.{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
