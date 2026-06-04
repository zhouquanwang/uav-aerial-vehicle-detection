# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for UAV Aerial Vehicle Detection System
生成 Windows 安装包目录
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(SPECPATH).parent  # installer 目录所在的项目根

# ─── 收集数据文件 ───
datas = []

# 模型文件
models_dir = PROJECT_ROOT / "models"
for pt_file in models_dir.glob("*.pt"):
    datas.append((str(pt_file), "models"))

# 配置文件
configs_dir = PROJECT_ROOT / "configs"
datas.append((str(configs_dir / "inference.yaml"), "configs"))

# 流量统计线配置（如果存在）
traffic_lines_dir = configs_dir / "traffic_lines"
if traffic_lines_dir.is_dir():
    for yaml_file in traffic_lines_dir.glob("*.yaml"):
        datas.append((str(yaml_file), "configs/traffic_lines"))

# ─── 隐藏导入（PyTorch 相关包不会被自动发现）───
hiddenimports = [
    # PyTorch
    "torch", "torchvision", "torchvision.transforms",
    "torchvision.ops", "torchvision.io",
    "torch._C", "torch._VF",
    "torch.cuda", "torch.cuda.amp",
    # Ultralytics / YOLO
    "ultralytics", "ultralytics.nn", "ultralytics.utils",
    "ultralytics.engine", "ultralytics.data",
    "ultralytics.models", "ultralytics.models.yolo",
    "ultralytics.cfg",
    # SAHI
    "sahi", "sahi.auto_model", "sahi.predict",
    "sahi.models", "sahi.utils",
    # Supervision
    "supervision", "supervision.tracker",
    "supervision.annotators",
    # OpenCV
    "cv2", "cv2.data",
    # 其他
    "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont",
    "numpy", "numpy.core", "numpy.linalg",
    "yaml", "openpyxl", "openpyxl.utils",
    "seaborn", "matplotlib",
    # 本项目模块
    "src.class_labels", "src.device_util",
    "src.zh_draw", "src.traffic_counter",
    "src.predict_video",
]

# ─── 排除不需要的模块（减小体积）───
excludes = [
    "tkinter.test",
    "matplotlib.tests", "numpy.tests",
    "torch.testing", "torch.testing._internal",
    "IPython", "jupyter", "notebook",
    "pytest", "setuptools", "pip",
]

a = Analysis(
    [str(PROJECT_ROOT / "src" / "ui_uav_main.py")],
    pathex=[str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="UAV检测系统",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="UAV检测系统",
)
