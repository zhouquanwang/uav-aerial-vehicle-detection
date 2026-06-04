"""
资源路径解析 — 兼容开发环境和 PyInstaller 打包环境
"""

from pathlib import Path
import sys


def get_project_root() -> Path:
    """获取项目根目录（开发模式）或 exe 所在目录（打包模式）"""
    if getattr(sys, "frozen", False):
        # PyInstaller 打包后：exe 的父目录就是应用根目录
        return Path(sys.executable).parent
    else:
        # 开发模式：src/../..
        return Path(__file__).resolve().parent.parent


def get_resource_path(relative_path: str) -> Path:
    """
    获取资源文件的绝对路径。
    开发模式 → 项目根/relative_path
    打包模式 → exe 所在目录/relative_path
    """
    return get_project_root() / relative_path
