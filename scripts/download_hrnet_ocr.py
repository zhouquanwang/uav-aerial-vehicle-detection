#!/usr/bin/env python3
"""
下载 HRNet-OCR Cityscapes 道路分割模型权重

来源: HRNet/HRNet-Semantic-Segmentation (GitHub)
模型: HRNetV2-W48 + OCR, Cityscapes val mIoU 81.6%
输出: models/hrnet_ocr_cs_8162_torch11.pth (~170 MB)
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

# 官方权重下载地址
MODEL_URL = (
    "https://github.com/hsfzxjy/models.storage/releases/download/"
    "HRNet-OCR/hrnet_ocr_cs_8162_torch11.pth"
)

# 备选: 百度网盘 (提取码 fa6i)
# "https://pan.baidu.com/s/1BGNt4Xmx3yfXUS8yjde0Qh"


def download(url: str, dest: Path):
    """下载文件并显示进度"""
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(count, block_size, total):
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            print(f"\r  下载中... {pct}%", end="", flush=True)

    print(f"  下载: {Path(url).name}")
    urlretrieve(url, str(dest), _progress)
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"\n  ✓ 完成 ({size_mb:.1f} MB) -> {dest}")


def main():
    parser = argparse.ArgumentParser(description="下载 HRNet-OCR 道路分割权重")
    parser.add_argument("--url", default=MODEL_URL, help="下载地址")
    parser.add_argument("--output", type=Path, default=MODEL_DIR, help="输出目录")
    args = parser.parse_args()

    filename = args.url.split("/")[-1]
    dest = args.output / filename

    if dest.exists():
        print(f"  已存在: {dest} ({dest.stat().st_size/1024/1024:.1f} MB)")
        return 0

    print("HRNet-OCR Cityscapes 权重下载")
    print(f"  模型: HRNetV2-W48 + OCR")
    print(f"  数据: Cityscapes train, val mIoU=81.6%")
    print(f"  路径: {dest}")
    print()

    try:
        download(args.url, dest)
    except Exception as e:
        print(f"\n  ✗ 下载失败: {e}", file=sys.stderr)
        print("  请检查网络或手动下载后放入 models/ 目录", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
