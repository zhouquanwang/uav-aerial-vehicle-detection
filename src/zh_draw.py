"""中文视频标注：使用支持 Unicode 的字体绘制标签。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import supervision as sv

# Windows 常见中文字体（按优先级）
_WINDOWS_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),   # 微软雅黑
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),  # 黑体
    Path(r"C:\Windows\Fonts\simsun.ttc"),  # 宋体
)

# Linux 常见路径
_LINUX_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
)

# 缓存已加载的字体
_font_cache = {}


def resolve_chinese_font_path(configured: str | Path | None = None) -> Path:
    """
    解析用于绘制中文标签的字体文件路径。
    优先使用配置项；否则在系统字体目录中查找。
    """
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            path = (project_root / path).resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Configured font not found: {path}")

    candidates = (
        _WINDOWS_FONT_CANDIDATES
        if sys.platform == "win32"
        else _LINUX_FONT_CANDIDATES
    )
    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        "No Chinese font found. Set font_path in configs/inference.yaml "
        "(e.g. C:/Windows/Fonts/msyh.ttc)."
    )


def _get_pil_font(font_path: str, size: int = 16) -> ImageFont.FreeTypeFont:
    """获取缓存的 PIL 字体对象"""
    key = (font_path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(font_path, size)
    return _font_cache[key]


def put_chinese_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 16,
    color: Tuple[int, int, int] = (0, 0, 0),
    font_path: str | None = None,
) -> None:
    """
    在 OpenCV 帧上绘制中文文本（使用 PIL + TrueType 字体）。

    Args:
        frame: BGR 格式的 OpenCV 图像（原地修改）
        text: 要绘制的文本（支持中文）
        position: (x, y) 左上角坐标
        font_size: 字体大小
        color: BGR 颜色
        font_path: 字体路径，None 则自动查找
    """
    if font_path is None:
        font_path = str(resolve_chinese_font_path())
    font = _get_pil_font(font_path, font_size)

    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    # PIL 用 RGB 颜色
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)

    # RGB -> BGR 写回
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def get_chinese_text_size(
    text: str,
    font_size: int = 16,
    font_path: str | None = None,
) -> Tuple[int, int]:
    """获取中文文本的像素尺寸 (width, height)"""
    if font_path is None:
        font_path = str(resolve_chinese_font_path())
    font = _get_pil_font(font_path, font_size)

    # 用 PIL 测量
    from PIL import ImageDraw
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def create_chinese_label_annotator(
    *,
    font_path: str | Path | None = None,
    font_size: int = 18,
) -> sv.RichLabelAnnotator:
    """创建可显示中文的 RichLabelAnnotator（PIL + TrueType）。"""
    resolved = resolve_chinese_font_path(font_path)
    return sv.RichLabelAnnotator(
        font_path=str(resolved),
        font_size=font_size,
        text_padding=4,
        smart_position=True,
    )
