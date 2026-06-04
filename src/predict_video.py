"""
航拍视频车辆检测 — 主推理脚本

功能：读取 MP4，用 WALDO 模型 + SAHI 切片检测，输出带检测框的视频。

推荐运行方式（在项目根目录）：
  .\\.venv\\Scripts\\python.exe -m src.predict_video
  python predict.py
"""

# 启用「类型注解」的向前兼容写法（Python 3.7+ 常用，可忽略细节）
from __future__ import annotations

# ---------- 标准库：与系统、命令行、路径相关 ----------
import argparse      # 解析命令行参数，例如 --input xxx.mp4
import subprocess    # 调用外部进程，用于切换到正确的 Python 虚拟环境
import sys           # 访问当前 Python 解释器信息（如 sys.executable、sys.argv）
import threading     # 用于检测停止信号
from pathlib import Path   # 跨平台处理文件/文件夹路径
from typing import Any     # 类型标注：表示「任意类型」

# 项目根目录 = 本文件所在目录的上一级（src 的父目录）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 运行本程序必须已安装的第三方包名（cv2 即 opencv-python）
_REQUIRED = ("cv2", "numpy", "yaml", "sahi", "supervision")


def _project_venv_python() -> Path | None:
    """返回本项目 .venv 里 python 可执行文件的路径；不存在则返回 None。"""
    # 依次检查：项目内 .venv → 上级目录 .venv
    for _root in [PROJECT_ROOT, PROJECT_ROOT.parent]:
        if sys.platform == "win32":  # Windows 系统
            candidate = _root / ".venv" / "Scripts" / "python.exe"
        else:  # Linux / macOS
            candidate = _root / ".venv" / "bin" / "python"
        if candidate.is_file():
            return candidate
    return None


def _missing_modules() -> list[str]:
    """检查哪些依赖包没装上，返回缺失包名列表（空列表表示全部已安装）。"""
    missing: list[str] = []  # 用来收集缺失的包名
    for name in _REQUIRED:   # 逐个尝试导入
        try:
            # opencv 导入名是 cv2，其它包名与 pip 包名一致
            __import__(name if name != "cv2" else "cv2")
        except ModuleNotFoundError:  # 导入失败说明没安装
            missing.append(name)
    return missing


def _ensure_runtime() -> None:
    """
    确保用「装齐依赖的 Python」运行。
    若当前解释器缺包，且项目下有 .venv，则自动用 .venv 里的 Python 重新启动本程序。
    """
    missing = _missing_modules()  # 先看缺什么包
    if not missing:  # 什么都不缺，直接继续执行后面代码
        return

    venv_py = _project_venv_python()  # 找项目专用虚拟环境里的 Python
    if venv_py is None:  # 没有 .venv，只能报错并提示用户安装
        _die_missing_deps(missing, venv_available=False)
        return

    try:
        # 当前正在用的 python.exe 的绝对路径
        current = Path(sys.executable).resolve()
        # 项目 .venv 里的 python.exe 的绝对路径
        target = venv_py.resolve()
    except OSError:  # 极少数情况下 resolve 失败，用未 resolve 的路径比较
        current = Path(sys.executable)
        target = venv_py

    if current == target:
        # 已经在用项目 .venv，但依赖仍缺失 → 说明没执行过 setup_venv
        _die_missing_deps(missing, venv_available=True)
        return

    # 在标准错误流打印提示（红色/黄色取决于终端，不影响程序逻辑）
    print(
        f"Current Python lacks: {', '.join(missing)}\n"
        f"  {sys.executable}\n"
        f"Re-launching with project venv:\n  {venv_py}",
        file=sys.stderr,
    )
    # 组装新命令：用项目 venv 的 Python，以模块方式运行本脚本，并带上原有命令行参数
    cmd = [str(venv_py), "-m", "src.predict_video", *sys.argv[1:]]
    # 用新进程执行，cwd 设为项目根，退出码传给当前进程
    raise SystemExit(subprocess.call(cmd, cwd=str(PROJECT_ROOT)))


def _die_missing_deps(missing: list[str], *, venv_available: bool) -> None:
    """依赖缺失且无法自动修复时，打印中文/英文安装说明并退出。"""
    lines = [
        "Missing Python packages: " + ", ".join(missing),
        f"Project root: {PROJECT_ROOT}",
        "",
        "Install / use the project virtualenv:",
        r"  cd f:\cursor\uav-aerial-vehicle-detection",
        r"  .\scripts\setup_venv.ps1",
        r"  .\.venv\Scripts\Activate.ps1",
        r"  python -m src.predict_video",
        "",
        "Or from repo root:",
        r"  python predict.py",
    ]
    if venv_available:  # .venv 存在但包没装全
        venv_py = _project_venv_python()
        lines.extend(
            [
                "",
                "Project .venv exists but packages are not installed. Run setup_venv.ps1.",
                f"  {venv_py}",
            ]
        )
    sys.stderr.write("\n".join(lines) + "\n")  # 输出说明文字
    raise SystemExit(1)  # 以错误码 1 结束程序


# ---------- 第三方库：视频、数值、配置、检测 ----------
# 把项目根目录加入模块搜索路径，这样才能写 from src.device_util import ...
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import cv2  # OpenCV：读视频、写视频
import numpy as np  # 数值数组，存放检测框坐标等
import yaml  # 读取 configs/inference.yaml 配置文件
# sahi / supervision 在 run_inference 内部惰性导入（避免被 UI import 时立即报错）


# ---------- 模型配置 ----------
MODEL_OPTIONS = {
    1: {"name": "WALDO-YOLOv8m", "type": "yolov8", "path": "models/WALDO30_yolov8m_640x640 (1).pt"},
    2: {"name": "RT-DETR", "type": "rtdetr", "path": "models/4epoch-rtdetr-best.pt"},
    3: {"name": "YOLOv8l", "type": "yolov8", "path": "models/4epoch-yolov8l-best.pt"},
}

# 本项目工具：根据配置选择 CPU/GPU，并打印设备信息
from src.class_labels import (
    MODEL_CLASS_CONFIGS,
    DEFAULT_TARGET_CLASSES,
    label_for_class,
    parse_class_labels_zh,
)
from src.device_util import device_status_line, resolve_device
from src.zh_draw import create_chinese_label_annotator, resolve_chinese_font_path
from src.cross_section_counter import CrossSectionCounter, CrossSectionLine
from src.auto_traffic_counter import AutoTrafficCounter


def _resolve_path(path: str | Path, root: Path) -> Path:
    """
    把配置里的相对路径转成绝对路径。
    若已是绝对路径（如 D:\\videos\\a.mp4）则原样返回。
    """
    p = Path(path)
    return p if p.is_absolute() else (root / p).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    """读取 YAML 配置文件，返回字典（键值对）。"""
    with config_path.open(encoding="utf-8") as f:  # 以 UTF-8 打开文件
        data = yaml.safe_load(f) or {}  # 解析 YAML；空文件则得到 {}
    if not isinstance(data, dict):  # 配置必须是「字典」结构
        raise ValueError(f"Invalid config: {config_path}")
    return data


def should_run_detection(frame_index: int, detect_skip_interval: int) -> bool:
    """
    判断是否对本帧做检测。

    detect_skip_interval=0：不跳帧，每帧检测。
    detect_skip_interval=N (N>=1)：每检测 1 帧后跳过 N 帧，即仅在
    frame_index % (N + 1) == 0 时检测（如 N=1 检测 0,2,4…）。
    """
    if detect_skip_interval < 0:
        raise ValueError("detect_skip_interval must be >= 0")
    if detect_skip_interval == 0:
        return True
    return frame_index % (detect_skip_interval + 1) == 0


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    """计算两个 xyxy 格式检测框的交并比（IoU）。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iom_xyxy(a: list[float], b: list[float]) -> float:
    """
    计算两个 xyxy 框的 IoM（Intersection over Minimum area）。

    IoM = 交集面积 / min(框A面积, 框B面积)
    与 IoU 不同，IoM 专门衡量「小框被大框包含」的程度：
    若小框几乎完全落在大框内部，则 IoM 接近 1.0，即使 IoU 很低。
    这是解决「大框套小框但 IoU 不足」场景的关键辅助指标。
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / min(area_a, area_b)


def _nms_object_predictions(
    predictions: list,
    iou_threshold: float = 0.5,
    iom_threshold: float = 0.75,
) -> list:
    """
    对 SAHI ObjectPrediction 列表做基于 IoU + IoM 的非极大值抑制（NMS）。

    背景：SAHI 切片检测时，相邻切片存在重叠区域，同一目标可能被多个切片
    分别检出，甚至可能因各切片的局部特征差异而被赋予不同类别，导致视频上
    同一辆车出现两个不同颜色的框与标签。

    优化记录（2025-05-22）：
    - 初始版本仅使用 IoU 做 NMS，但存在缺陷：当一个框被另一个大框完全或
      大部分包含时（如小框面积 0.2，大框面积 1.0，小框有 0.18 在大框内），
      IoU = 0.18 / (1.0 + 0.2 - 0.18) ≈ 0.16，远低于 0.5 阈值，导致同一目标
      的两个框都被保留，视频上仍出现重复检测。
    - 解决方案：引入 IoM（Intersection over Minimum area）作为辅助指标。
      IoM = 交集 / min(面积A, 面积B)。对于上述包含场景，IoM = 0.18 / 0.2 = 0.9，
      远高于阈值，可有效抑制小框，仅保留置信度更高的框。
    - 抑制条件改为：IoU >= iou_threshold **或** IoM >= iom_threshold 时丢弃低分框。
    - 该操作在「目标类别过滤」之后、「构建 sv.Detections」之前执行，
      不影响模型原始输出，只过滤后处理阶段的冗余框。
    - 采用 class-agnostic NMS（不区分类别），专门抑制「同一目标被检测成
      两种不同类别」的情况。

    Args:
        predictions: SAHI ObjectPrediction 对象列表（已过滤过目标类别）。
        iou_threshold: IoU 阈值，超过此值视为同一目标，默认 0.5。
        iom_threshold: IoM 阈值，超过此值视为同一目标（解决包含关系），默认 0.75。

    Returns:
        去重后的 ObjectPrediction 列表。
    """
    if not predictions:
        return []

    # 按置信度降序排序（分数高的排在前面优先保留）
    sorted_preds = sorted(predictions, key=lambda p: p.score.value, reverse=True)
    keep: list = []
    while sorted_preds:
        current = sorted_preds.pop(0)
        keep.append(current)
        current_box = current.bbox.to_xyxy()
        # 移除与 current IoU 或 IoM 过高的剩余框（class-agnostic：不区分类别）
        sorted_preds = [
            p for p in sorted_preds
            if (
                _iou_xyxy(current_box, p.bbox.to_xyxy()) < iou_threshold
                and _iom_xyxy(current_box, p.bbox.to_xyxy()) < iom_threshold
            )
        ]
    return keep


def annotate_frame(
    frame: np.ndarray,
    *,
    detection_model: AutoDetectionModel,
    model_name: str,  # 模型名称
    target_set: set[int],
    class_labels_zh: dict[int, str],
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    box_annotator: sv.BoxCornerAnnotator,
    label_annotator: sv.RichLabelAnnotator,
    add_model_watermark: bool = True,  # 是否添加模型水印
    class_stats: dict[int, int] | None = None,  # 可选：累计各类别检测数
    show_labels: bool = True,  # 是否绘制车型标签（类别、编号、置信度）
) -> np.ndarray:
    """对单帧做 SAHI 切片检测并画框；无目标时返回原帧副本。"""
    # 惰性导入（避免模块级导入时报错）
    from sahi.predict import get_sliced_prediction
    import supervision as sv

    result = get_sliced_prediction(
        image=frame,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    object_predictions = [
        pred
        for pred in result.object_prediction_list
        if pred.category.id in target_set
    ]

    # ═══════ 2025-05-22 优化：NMS 去重 ═══════
    # SAHI 切片重叠会导致同一目标被多个切片分别检出，甚至被赋予不同类别。
    # 在构建 sv.Detections 之前做一次 class-agnostic NMS，抑制冗余框。
    object_predictions = _nms_object_predictions(object_predictions, iou_threshold=0.5)

    xyxy: list[list[float]] = []
    confidences: list[float] = []
    class_ids: list[int] = []
    display_names: list[str] = []

    for pred in object_predictions:
        xyxy.append(list(pred.bbox.to_xyxy()))
        confidences.append(pred.score.value)
        cid = pred.category.id
        class_ids.append(cid)
        display_names.append(
            label_for_class(cid, class_labels_zh, fallback_name=pred.category.name)
        )

    if not xyxy:
        return frame.copy()

    # 累积类别统计
    if class_stats is not None:
        for cid in class_ids:
            class_stats[cid] = class_stats.get(cid, 0) + 1

    detections = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )
    annotated = frame.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    if show_labels:
        labels = [
            f"{name} {conf:.2f}"
            for name, conf in zip(display_names, confidences)
        ]
        annotated = label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

    # 添加模型水印标记
    if add_model_watermark:
        import os
        font_file = resolve_chinese_font_path(None)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"[Model: {model_name}]"
        # 获取文本尺寸
        (text_w, text_h), baseline = cv2.getTextSize(text, font, 0.7, 2)
        # 水印位置：右上角
        x = annotated.shape[1] - text_w - 15
        y = text_h + 15
        # 半透明背景
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x - 10, y - text_h - 5), (x + text_w + 10, y + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        # 绘制文字
        cv2.putText(annotated, text, (x, y), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return annotated


def run_inference(
    *,
    model_select: int,             # 模型选择：1=WALDO, 2=RT-DETR, 3=YOLOv8l
    model_path: Path,              # 权重 .pt 文件路径
    model_type: str,               # 模型类型：yolov8 或 rtdetr
    model_name: str,                # 模型显示名称
    input_video: Path,             # 输入 MP4 路径
    output_video: Path,            # 输出 MP4 路径
    device: str,                   # 推理设备，如 cpu、cuda:0
    confidence_threshold: float,   # 置信度阈值，低于此值的框会被丢弃
    target_classes: list[int],     # 只保留这些类别 ID
    class_labels_zh: dict[int, str],  # 类别 ID → 视频上显示的中文名
    slice_height: int,             # SAHI 切片高度（像素）
    slice_width: int,              # SAHI 切片宽度（像素）
    overlap_height_ratio: float,   # 切片垂直方向重叠比例（减少边界漏检）
    overlap_width_ratio: float,    # 切片水平方向重叠比例
    detect_skip_interval: int = 0,  # 跳帧检测间隔，0=不跳帧
    max_frames: int | None = None, # 最多输出多少帧；None 表示处理完整视频
    font_path: str | Path | None = None,  # 中文字体路径，None 则自动查找系统字体
    label_font_size: int = 18,  # 中文标签字号
    start_frame: int = 0,  # 起始帧
    end_frame: int | None = None,  # 结束帧，None 表示处理到末尾
    progress_callback=None,  # 可选回调(current_frame, total_frames, annotated_frame, detect_count, stats, traffic_stats)
    log_callback=None,  # 可选回调(message)
    enable_tracking: bool = False,  # 是否启用 ByteTrack 轨迹跟踪
    count_lines: list[CrossSectionLine] | None = None,  # 截面流量统计线配置
    show_labels: bool = True,  # 是否绘制车型标签（类别、编号、置信度）
    use_sahi: bool = True,  # 是否启用 SAHI 切片检测
    write_output: bool = True,  # 是否输出检测视频文件
    enable_road_detect: bool = False,  # 是否启用车道边界检测
    road_model_type: str = "hsv",  # 道路边界检测模型: hsv / segformer / hrnet_ocr
    enable_auto_traffic: bool = False,  # 是否启用自动流量统计（轨迹聚类）
    stop_event: threading.Event | None = None,  # 外部停止信号
) -> dict:
    """
    核心推理循环：逐帧读视频 → 按间隔检测/画框 → 写入输出视频（帧数与输入一致）。
    返回值：写入输出视频的帧数。
    """
    # ═══════ 惰性导入 sahi / supervision / time ═══════
    import time
    # 支持自动发现 .venv 的 site-packages（解决 IDE 未使用 .venv 的情况）
    _sahi_ok = False
    for _attempt in range(2):
        try:
            from sahi.auto_model import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            import supervision as sv
            _sahi_ok = True
            break
        except ImportError:
            if _attempt == 0:
                # 尝试从多个可能的位置加载 .venv site-packages
                import site as _site
                for _root in [PROJECT_ROOT, PROJECT_ROOT.parent, PROJECT_ROOT.parent.parent]:
                    for _venv_name in ['.venv', 'venv', '.env', 'env']:
                        _site_pkg = _root / _venv_name / 'Lib' / 'site-packages'
                        if _site_pkg.is_dir() and str(_site_pkg) not in sys.path:
                            sys.path.insert(0, str(_site_pkg))
            else:
                raise

    if detect_skip_interval < 0:
        raise ValueError("detect_skip_interval must be >= 0")
    # ---------- 1. 检查输入文件 ----------
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    print(f"使用模型 [{model_select}]: {model_name} ({model_type}) - {model_path.name}")

    # 确保输出目录存在（例如 outputs/），parents=True 表示可创建多级目录
    output_video.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 2. 加载检测模型（SAHI 包装的 YOLO/RT-DETR）----------
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,                    # yolov8 或 rtdetr
        model_path=str(model_path),               # 权重文件路径（需转成字符串）
        confidence_threshold=confidence_threshold,  # 模型内部置信度过滤
        device=device,                            # CPU 或 GPU
    )

    # ---------- 3. 打开输入视频 ----------
    cap = cv2.VideoCapture(str(input_video))  # 创建视频读取对象
    if not cap.isOpened():  # 打开失败（路径错、格式不支持等）
        raise RuntimeError(f"Cannot open video: {input_video}")

    # 读取视频属性，用于创建同样分辨率和帧率的输出视频
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 帧宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧高度
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # 帧率；读不到则默认 25

    # 输出视频（仅在需要时创建）
    out = None
    if write_output:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {output_video}")
    else:
        print("视频输出: 已关闭（仅统计模式）")

    # 设置起终点范围
    actual_start = max(0, min(start_frame, total_video_frames - 1))
    actual_end = end_frame if end_frame is not None else total_video_frames - 1
    actual_end = min(actual_end, total_video_frames - 1)
    total_to_process = actual_end - actual_start + 1

    if actual_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)
        if log_callback:
            log_callback(f"从第 {actual_start} 帧开始处理")

    # ---------- 4. 准备画框工具与类别过滤集合 ----------
    target_set = set(target_classes)  # 用 set 加快「类别是否在目标列表」判断
    box_annotator = sv.BoxCornerAnnotator(thickness=2)  # 角点式矩形框
    # OpenCV 默认字体不支持中文，使用 RichLabelAnnotator + 系统中文字体
    font_file = resolve_chinese_font_path(font_path)
    label_annotator = create_chinese_label_annotator(
        font_path=font_file,
        font_size=label_font_size,
    )
    print(f"中文标注字体: {font_file}")

    # 初始化轨迹跟踪器（可选）
    _tracker = None
    _smoother = None
    if enable_tracking:
        from supervision import ByteTrack, DetectionsSmoother
        _tracker = ByteTrack(frame_rate=fps)
        _smoother = DetectionsSmoother()
        print("轨迹跟踪: 已启用 (ByteTrack + DetectionsSmoother)")

    # 初始化流量统计器
    _traffic_counter = None
    if count_lines and enable_tracking:
        _traffic_counter = CrossSectionCounter(count_lines)
        print(f"流量统计: 已启用，共 {len(count_lines)} 条截面线")
    elif count_lines and not enable_tracking:
        print("流量统计: 需要启用轨迹跟踪，当前未启用，忽略截面线")

    # 初始化道路边界检测器
    _road_segmenter = None
    if enable_road_detect:
        try:
            from src.road_segmentation import RoadSegmenter
            _road_segmenter = RoadSegmenter(model_type=road_model_type)
            print(f"车道边界检测: 已启用 ({road_model_type})")
        except Exception as e:
            print(f"车道边界检测: 初始化失败 ({e})，已禁用")

    # 初始化自动流量统计器
    _auto_traffic_counter = None
    if enable_auto_traffic:
        _auto_traffic_counter = AutoTrafficCounter(
            min_traj_len=5, min_displacement=30, angle_threshold_deg=30, min_samples=3,
        )
        print(f"自动流量统计: 已启用（基于轨迹方向聚类）")


    frame_count = 0       # 已写入输出视频的帧数（含跳帧未标注帧）
    detect_count = 0      # 实际跑过模型的帧数
    frame_index = actual_start  # 当前处理的帧序号（从 start_frame 开始）
    last_log_time = time.time()
    class_stats: dict[int, int] = {}  # 各类别检测总数

    if detect_skip_interval == 0:
        print("跳帧检测: 关闭（每帧检测）")
    else:
        period = detect_skip_interval + 1
        print(
            f"跳帧检测: 间隔 {detect_skip_interval} "
            f"(每 {period} 帧检测 1 次，其余帧原样输出不标注)"
        )

    while cap.isOpened():
        # 检查外部停止信号
        if stop_event and stop_event.is_set():
            print("\n[停止] 收到停止信号，正在退出...")
            if log_callback:
                log_callback("收到停止信号，检测已中止")
            break

        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_count >= max_frames:
            break
        if frame_index > actual_end:
            break

        if should_run_detection(frame_index, detect_skip_interval):
            # ── 检测帧 ──
            # 非SAHI模式：切片尺寸设为帧尺寸（等效于整帧推理）
            _eff_slice_h = slice_height if use_sahi else height
            _eff_slice_w = slice_width if use_sahi else width
            _eff_overlap_h = overlap_height_ratio if use_sahi else 0.0
            _eff_overlap_w = overlap_width_ratio if use_sahi else 0.0

            if enable_tracking:
                # 跟踪模式：获取 sv.Detections 后更新跟踪器再标注
                from supervision import Detections
                from sahi.predict import get_sliced_prediction as _gsp
                _result = _gsp(
                    image=frame, detection_model=detection_model,
                    slice_height=_eff_slice_h, slice_width=_eff_slice_w,
                    overlap_height_ratio=_eff_overlap_h,
                    overlap_width_ratio=_eff_overlap_w,
                )
                # 先过滤目标类别，再做 NMS 去重（解决切片重叠导致的多类别重复检测）
                _filtered_preds = [
                    _pred for _pred in _result.object_prediction_list
                    if _pred.category.id in target_set
                ]
                _filtered_preds = _nms_object_predictions(_filtered_preds, iou_threshold=0.5)
                _xyxy, _conf, _cids, _names = [], [], [], []
                for _pred in _filtered_preds:
                    _xyxy.append(list(_pred.bbox.to_xyxy()))
                    _conf.append(_pred.score.value)
                    _cids.append(_pred.category.id)
                    _names.append(label_for_class(
                        _pred.category.id, class_labels_zh,
                        fallback_name=_pred.category.name))
                if _xyxy:
                    _dets = Detections(
                        xyxy=np.array(_xyxy, dtype=np.float32),
                        confidence=np.array(_conf, dtype=np.float32),
                        class_id=np.array(_cids, dtype=int),
                    )
                    # 更新跟踪器和平滑器
                    _dets = _tracker.update_with_detections(_dets)
                    _dets = _smoother.update_with_detections(_dets)
                    # 累积统计
                    if class_stats is not None:
                        for _cid in _cids:
                            class_stats[_cid] = class_stats.get(_cid, 0) + 1
                    # 流量统计更新（截面线）
                    if _traffic_counter is not None and _dets.tracker_id is not None:
                        for _ti in range(len(_dets.xyxy)):
                            _tid = int(_dets.tracker_id[_ti])
                            _tcid = int(_dets.class_id[_ti])
                            _traffic_counter.update(_tid, _tcid, _dets.xyxy[_ti])
                    # 自动流量统计更新（轨迹记录）
                    if _auto_traffic_counter is not None and _dets.tracker_id is not None:
                        for _ti in range(len(_dets.xyxy)):
                            _tid = int(_dets.tracker_id[_ti])
                            _tcid = int(_dets.class_id[_ti])
                            _auto_traffic_counter.record(_tid, _tcid, _dets.xyxy[_ti], frame_index)
                    # 标注（含跟踪 ID）
                    output_frame = frame.copy()
                    output_frame = box_annotator.annotate(scene=output_frame, detections=_dets)
                    if show_labels:
                        _labels = []
                        for _i in range(len(_dets.xyxy)):
                            _clsid = _dets.class_id[_i]
                            _label = f"{label_for_class(_clsid, class_labels_zh, fallback_name='?')} {_dets.confidence[_i]:.2f}"
                            if _dets.tracker_id is not None:
                                _label = f"ID {_dets.tracker_id[_i]} {_label}"
                            _labels.append(_label)
                        output_frame = label_annotator.annotate(scene=output_frame, detections=_dets, labels=_labels)
                    # 叠加流量统计线
                    if _traffic_counter is not None:
                        output_frame = _traffic_counter.draw_on_frame(output_frame)
                else:
                    output_frame = frame.copy()
                    if _traffic_counter is not None:
                        output_frame = _traffic_counter.draw_on_frame(output_frame)
            else:
                # 普通检测模式
                output_frame = annotate_frame(
                    frame,
                    detection_model=detection_model,
                    model_name=model_name,
                    target_set=target_set,
                    class_labels_zh=class_labels_zh,
                    slice_height=_eff_slice_h,
                    slice_width=_eff_slice_w,
                    overlap_height_ratio=_eff_overlap_h,
                    overlap_width_ratio=_eff_overlap_w,
                    box_annotator=box_annotator,
                    label_annotator=label_annotator,
                    add_model_watermark=True,
                    class_stats=class_stats,
                    show_labels=show_labels,
                )
            detect_count += 1
            status = "detect"
        else:
            output_frame = frame.copy()
            status = "skip"

        # 叠加道路边界掩码（若启用）
        if _road_segmenter is not None:
            try:
                _mask = _road_segmenter.predict(output_frame)
                output_frame = _road_segmenter.get_mask_overlay(output_frame, _mask, alpha=0.2)
            except Exception:
                pass  # 静默失败，不中断检测

        if out is not None:
            out.write(output_frame)
        frame_count += 1
        frame_index += 1

        # 调用进度回调（UI 更新，频率降低到 500ms 以减少 CPU 开销）
        now = time.time()
        if progress_callback and now - last_log_time > 0.5:
            traffic_stats = _traffic_counter.get_stats() if _traffic_counter else None
            progress_callback(frame_index, total_to_process, output_frame, detect_count, class_stats, traffic_stats)
            last_log_time = now

        # 控制台输出（不干预 UI）
        print(
            f"Frame {frame_count} [{status}] | detected {detect_count}",
            end="\r",
            flush=True,
        )

    cap.release()
    if out is not None:
        out.release()
    print(
        f"\nDone. output {frame_count} frames "
        f"(model runs on {detect_count} frames) -> {output_video}"
    )
    if log_callback:
        log_callback(f"检测完成! 共处理 {frame_count} 帧，检测 {detect_count} 帧")

    # 自动流量统计后处理
    auto_flows = None
    if _auto_traffic_counter is not None:
        try:
            auto_flows = _auto_traffic_counter.analyze(
                image_width=width, image_height=height,
            )
            flow_count = len(auto_flows)
            total_vehicles = sum(f.vehicle_count for f in auto_flows)
            print(f"自动流量统计: 发现 {flow_count} 个方向，共 {total_vehicles} 辆车")
            if log_callback:
                log_callback(f"自动流量统计: 发现 {flow_count} 个方向，共 {total_vehicles} 辆车")
        except Exception as e:
            print(f"自动流量统计: 分析失败 ({e})")

    result = {
        "frame_count": frame_count,
        "detect_count": detect_count,
        "output_video": str(output_video),
        "traffic_stats": _traffic_counter.get_stats() if _traffic_counter else None,
        "auto_flows": auto_flows,
        "auto_traffic_counter": _auto_traffic_counter,
    }
    return result


def main() -> None:
    """命令行入口：解析参数 → 读配置 → 调用 run_inference。"""
    # 创建参数解析器，description 会显示在 python -m src.predict_video --help 里
    parser = argparse.ArgumentParser(
        description="UAV Vehicle Detection - YOLOv8/RT-DETR + SAHI sliced inference on aerial MP4"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "inference.yaml",  # 默认配置文件
        help="YAML config path",
    )
    parser.add_argument("--input", type=Path, default=None, help="Override input video")
    parser.add_argument("--output", type=Path, default=None, help="Override output video")
    parser.add_argument("--model", type=Path, default=None, help="Override model .pt")
    parser.add_argument(
        "--model-select",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Model selection: 1=WALDO-YOLOv8m, 2=RT-DETR, 3=YOLOv8l",
    )
    parser.add_argument("--device", type=str, default=None, help="auto | cpu | cuda:0")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only first N frames (smoke test)",
    )
    parser.add_argument(
        "--detect-skip-interval",
        type=int,
        default=None,
        help="0=every frame; N>=1 skip N frames between detections (output keeps all frames)",
    )
    args = parser.parse_args()  # 解析用户在命令行传入的参数

    cfg = load_config(args.config.resolve())  # 读取 YAML 配置
    root = PROJECT_ROOT  # 相对路径都相对于项目根目录

    # 模型选择逻辑：命令行参数优先，否则使用配置文件
    model_select = args.model_select
    if model_select is None:
        model_select = cfg.get("model_select", 1)

    if model_select not in MODEL_OPTIONS:
        raise ValueError(f"Invalid model_select: {model_select}. Must be 1, 2, or 3.")

    model_config = MODEL_OPTIONS[model_select]

    # 命令行 --model 参数优先于配置文件
    if args.model:
        model_path = _resolve_path(args.model, root)
        model_type = "yolov8"  # 默认类型
        model_name = args.model.stem
    else:
        # 使用模型配置
        model_path_str = cfg.get("models", {}).get(model_select, model_config["path"])
        model_path = _resolve_path(model_path_str, root)
        model_type = model_config["type"]
        model_name = model_config["name"]
    input_video = _resolve_path(args.input or cfg["input_video"], root)

    # 输出视频路径：命令行指定 > 根据模型名称自动生成
    if args.output:
        output_video = _resolve_path(args.output, root)
    else:
        # 默认输出目录为 outputs/
        output_dir = root / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        # 输出文件名格式：predict_[模型名称].mp4
        output_filename = f"predict_{model_name}.mp4"
        output_video = output_dir / output_filename

    device = resolve_device(args.device or cfg.get("device", "auto"))  # auto → GPU 或 CPU
    print(device_status_line(device))  # 打印当前使用的设备信息

    # 命令行 --max-frames 优先；否则读配置 max_frames（null 表示处理全片）
    max_frames = args.max_frames
    if max_frames is None and "max_frames" in cfg:
        cfg_max = cfg["max_frames"]
        if cfg_max is not None:
            max_frames = int(cfg_max)

    # 根据模型选择加载对应的类别配置（模型默认配置）
    model_class_config = MODEL_CLASS_CONFIGS[model_select]
    
    # 如果配置文件没有明确指定类别，则使用模型默认配置
    # 这样确保不同模型使用正确的类别映射
    target_classes = list(cfg.get("target_classes")) if "target_classes" in cfg else model_class_config["target_classes"]
    
    cfg_labels = cfg.get("class_labels_zh")
    class_labels_zh = parse_class_labels_zh(cfg_labels) if cfg_labels else model_class_config["labels"]

    print(f"检测类别: {target_classes}")
    print(f"类别标签: {class_labels_zh}")

    # 调用核心推理函数，cfg.get(key, 默认值) 防止配置项缺失
    run_inference(
        model_select=model_select,
        model_path=model_path,
        model_type=model_type,
        model_name=model_name,
        input_video=input_video,
        output_video=output_video,
        device=device,
        confidence_threshold=float(cfg.get("confidence_threshold", 0.2)),
        target_classes=target_classes,
        class_labels_zh=class_labels_zh,
        slice_height=int(cfg.get("slice_height", 640)),
        slice_width=int(cfg.get("slice_width", 640)),
        overlap_height_ratio=float(cfg.get("overlap_height_ratio", 0.1)),
        overlap_width_ratio=float(cfg.get("overlap_width_ratio", 0.1)),
        detect_skip_interval=int(
            args.detect_skip_interval
            if args.detect_skip_interval is not None
            else cfg.get("detect_skip_interval", 0)
        ),
        max_frames=max_frames,
        font_path=cfg.get("font_path"),
        label_font_size=int(cfg.get("label_font_size", 18)),
    )


# 仅当「直接运行本文件」或「python -m src.predict_video」时执行 main
# 若被其它模块 import，不会自动跑推理
if __name__ == "__main__":
    # 在导入 cv2 等大库之前，先完成上面的环境检查（可能触发进程重启）
    # 仅在 CLI 模式运行时检查，import 时不触发
    _ensure_runtime()
    main()
