#!/usr/bin/env python3
"""
UAV 检测系统 — Windows 安装包构建脚本
========================================
打包流程:
  python build_installer.py          → 使用 PyInstaller 生成 dist 目录
  python build_installer.py --nsis   → 额外用 Inno Setup 生成 .exe 安装程序

输出:
  dist/UAV检测系统/          → PyInstaller 打包目录（可直接运行）
  installer/UAV检测系统_Setup.exe → Inno Setup 安装程序（可选）
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# ─── 路径定义 ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTALLER_DIR = Path(__file__).resolve().parent
SPEC_FILE = INSTALLER_DIR / "uav_detection.spec"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
OUTPUT_NAME = "UAV检测系统"


def run(cmd: list[str], cwd: Path = None, desc: str = "") -> int:
    """运行命令并实时输出"""
    if desc:
        print(f"\n{'=' * 60}")
        print(f"  {desc}")
        print(f"{'=' * 60}")
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  ✗ 失败 (exit code {result.returncode})", file=sys.stderr)
    else:
        print(f"  ✓ 完成")
    return result.returncode


def clean_build():
    """清理旧的构建产物"""
    for d in [BUILD_DIR, DIST_DIR]:
        if d.exists():
            print(f"  清理: {d}")
            shutil.rmtree(d)
    # 清理 spec 产生的缓存
    for p in PROJECT_ROOT.glob("*.spec"):
        if p != SPEC_FILE:
            p.unlink()


def build_pyinstaller() -> int:
    """使用 PyInstaller 打包"""
    ret = run(
        [
            str(PROJECT_ROOT / ".venv" / "Scripts" / "pyinstaller.exe"),
            "--distpath", str(DIST_DIR),
            "--workpath", str(BUILD_DIR),
            "--noconfirm",
            "--clean",
            str(SPEC_FILE),
        ],
        cwd=PROJECT_ROOT,
        desc="PyInstaller 打包中 (预计 5-15 分钟)...",
    )
    return ret


def create_outputs_folder():
    """在打包输出目录创建 outputs 占位文件夹"""
    outputs = DIST_DIR / OUTPUT_NAME / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / ".gitkeep").write_text("")
    print(f"  创建: {outputs}")


def create_launcher():
    """在 dist 根目录创建启动批处理"""
    bat = DIST_DIR / "启动UAV检测系统.bat"
    bat.write_text(
        f'@echo off\n'
        f'echo 启动无人机航拍视频检测系统...\n'
        f'start "" "{OUTPUT_NAME}\\{OUTPUT_NAME}.exe"\n',
        encoding="gbk",
    )
    print(f"  创建启动脚本: {bat.name}")


def print_summary():
    """打印构建摘要"""
    exe = DIST_DIR / OUTPUT_NAME / f"{OUTPUT_NAME}.exe"
    if exe.exists():
        size_mb = sum(f.stat().st_size for f in exe.parent.rglob("*")) / 1024 / 1024
        print(f"\n{'=' * 60}")
        print(f"  ✓ 打包完成!")
        print(f"  输出目录: {DIST_DIR / OUTPUT_NAME}")
        print(f"  可执行文件: {exe}")
        print(f"  总大小: {size_mb:.0f} MB")
        print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")
        print(f"\n  使用方法:")
        print(f"    1. 打开 {DIST_DIR}")
        print(f"    2. 双击 '启动UAV检测系统.bat' 或进入 'UAV检测系统' 目录运行 .exe")
        print(f"    3. 首次启动加载模型可能需要 5-30 秒")
        print()
    else:
        print(f"\n  ✗ 打包似乎未成功，请检查上方错误信息")


def main():
    parser = argparse.ArgumentParser(description="UAV 检测系统安装包构建工具")
    parser.add_argument("--clean", action="store_true", help="构建前清理旧产物")
    parser.add_argument("--nsis", action="store_true", help="同时生成 NSIS/Inno Setup 安装包")
    args = parser.parse_args()

    print("=" * 60)
    print("  UAV 航拍视频车辆检测系统 — 安装包构建工具")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 前置检查
    if not SPEC_FILE.exists():
        print(f"✗ spec 文件不存在: {SPEC_FILE}")
        return 1

    venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        print(f"✗ 虚拟环境不存在，请先运行 scripts/setup_venv.ps1")
        return 1

    # 步骤 1：清理
    if args.clean:
        clean_build()

    # 步骤 2：PyInstaller 打包
    ret = build_pyinstaller()
    if ret != 0:
        print("\n✗ PyInstaller 打包失败，请检查错误信息")
        return ret

    # 步骤 3：输出目录整理
    create_outputs_folder()
    create_launcher()

    # 步骤 4：打印结果
    print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
