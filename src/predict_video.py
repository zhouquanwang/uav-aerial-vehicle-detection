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
from pathlib import Path   # 跨平台处理文件/文件夹路径
from typing import Any     # 类型标注：表示「任意类型」

# 项目根目录 = 本文件所在目录的上一级（src 的父目录）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 运行本程序必须已安装的第三方包名（sahi 已改为延迟导入，不在此检查）
_REQUIRED = ("cv2", "numpy", "yaml", "supervision")


def _project_venv_python() -> Path | None:
    """返回本项目 .venv 里 python 可执行文件的路径；不存在则返回 None。"""
    if sys.platform == "win32":  # Windows 系统
        # 例如：uav-aerial-vehicle-detection\.venv\Scripts\python.exe
        candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:  # Linux / macOS
        candidate = PROJECT_ROOT / ".venv" / "bin" / "python"
    # 只有文件真实存在才返回，否则 None
    return candidate if candidate.is_file() else None


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


# 注：_ensure_runtime() 移到 __main__ 块内调用
# 避免 UI 等模块 import 时意外触发进程重启和 sahi 检查

# 把项目根目录加入模块搜索路径，这样才能写 from src.device_util import ...
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------- 第三方库：视频、数值、配置、检测 ----------
import cv2  # OpenCV：读视频、写视频
import numpy as np  # 数值数组，存放检测框坐标等
import yaml  # 读取 configs/inference.yaml 配置文件
import supervision as sv  # 画检测框、写标签（基于 OpenCV）


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


def annotate_frame(
    frame: np.ndarray,
    *,
    detection_model: Any,  # SAHI AutoDetectionModel（延迟导入，在此用 Any 占位）
    model_name: str,
    target_set: set[int],
    class_labels_zh: dict[int, str],
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    box_annotator: sv.BoxCornerAnnotator,
    label_annotator: sv.RichLabelAnnotator,
    add_model_watermark: bool = True,
) -> np.ndarray:
    """对单帧做 SAHI 切片检测并画框；无目标时返回原帧副本。"""
    from sahi.predict import get_sliced_prediction
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

    detections = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )
    labels = [
        f"{name} {conf:.2f}"
        for name, conf in zip(display_names, confidences)
    ]
    annotated = frame.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
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


# ═══════════ IoU + IoM 双阈值 NMS ═══════════
def dual_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.5,
    iom_threshold: float = 0.75,
) -> np.ndarray:
    """
    IoU + IoM 双阈值非极大抑制，解决 SAHI 切片重叠导致的冗余框。

    策略：
      1. 按置信度降序排列
      2. 对每对 Box：(A=高置信度保留框，B=低置信度候选框)
         - IoU(A,B) >= iou_threshold → 抑制 B（常规重叠）
         - IoM(A,B) >= iom_threshold → 抑制 B（大框套小框，IoU不足但IoM能捕获）
      3. 返回保留框的索引

    Args:
        boxes:  (N, 4) xyxy 格式
        scores: (N,) 置信度
        class_ids: (N,) 类别ID
        iou_threshold: IoU 阈值（默认 0.5）
        iom_threshold: IoM 阈值（默认 0.75）

    Returns:
        保留框的索引数组
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    # 计算所有框的面积
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按置信度降序
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        # 计算 order[0] 与剩余框的交集
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # IoU = inter / (area_i + area_j - inter)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        # IoM = inter / min(area_i, area_j)
        iom = inter / (np.minimum(areas[i], areas[order[1:]]) + 1e-6)

        # 双阈值抑制
        suppress = (ovr >= iou_threshold) | (iom >= iom_threshold)

        keep_idx = np.where(~suppress)[0]
        order = order[keep_idx + 1]

    return np.array(keep, dtype=int)


# ═══════════ 轨迹实时绘制 ═══════════

def _draw_trajectories_on_frame(frame: np.ndarray, counter) -> None:
    """在当前帧上绘制所有已记录的车辆轨迹（RGB 彩线 + 方向箭头）。"""
    try:
        from src.auto_traffic_counter import get_flow_color
    except ImportError:
        return


    trajectories = counter.trajectories
    color_idx = 0
    for traj in trajectories:
        if traj.length < 2:
            continue
        color = get_flow_color(color_idx)
        color_idx += 1
        pts = np.array([[int(p.x), int(p.y)] for p in traj.points], dtype=np.int32)
        cv2.polylines(frame, [pts], False, color, thickness=1, lineType=cv2.LINE_AA)
        # 方向箭头（末端）
        if len(pts) >= 2:
            last, prev = pts[-1], pts[-2]
            cv2.arrowedLine(frame, tuple(prev), tuple(last), color, thickness=1, tipLength=0.3)


def run_inference(
    *,
    model_select: int,
    model_path: Path,
    model_type: str,
    model_name: str,
    input_video: Path,
    output_video: Path,
    device: str,
    confidence_threshold: float,
    target_classes: list[int],
    class_labels_zh: dict[int, str],
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    detect_skip_interval: int = 0,
    start_frame: int = 0,
    end_frame: int | None = None,
    max_frames: int | None = None,
    font_path: str | Path | None = None,
    label_font_size: int = 18,
    enable_tracking: bool = False,
    count_lines: list | None = None,
    show_labels: bool = True,
    use_sahi: bool = True,
    write_output: bool = True,
    enable_detection: bool = True,
    enable_road_detect: bool = False,
    road_model_type: str = "hsv",
    enable_auto_traffic: bool = False,
    show_trajectory: bool = False,
    stop_event: Any = None,
    log_callback: Any = None,
    progress_callback: Any = None,
) -> dict:
    """核心推理循环：支持 ByteTrack、截面线流量统计、自动轨迹聚类。"""

    import threading
    import time

    def _log(msg: str):
        if log_callback: log_callback(msg)
        else: print(msg)

    # ── 1. 检查输入 ──
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    output_video.parent.mkdir(parents=True, exist_ok=True)

    # ── 2. 加载检测模型 ──
    if use_sahi:
        try:
            from sahi.auto_model import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
        except ModuleNotFoundError:
            import subprocess as _sp; import sys as _sys
            _log("SAHI 未安装，正在自动安装...")
            _sp.check_call([_sys.executable, "-m", "pip", "install", "sahi", "-q"])
            from sahi.auto_model import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            _log("SAHI 安装完成")
        detection_model = AutoDetectionModel.from_pretrained(
            model_type=model_type, model_path=str(model_path),
            confidence_threshold=confidence_threshold, device=device,
        )
        _log(f"模型已加载 (SAHI): {model_name}")
    else:
        from ultralytics import YOLO
        detection_model = YOLO(str(model_path))
        _log(f"模型已加载 (直接): {model_name}")

    # ── 3. 打开视频 ──
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None or end_frame <= 0:
        end_frame = total_frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    if write_output:
        out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {output_video}")

    # ── 4. 标注器 + 字体 ──
    target_set = set(target_classes)
    box_annotator = sv.BoxCornerAnnotator(thickness=2)
    font_file = resolve_chinese_font_path(font_path)
    label_annotator = create_chinese_label_annotator(font_path=font_file, font_size=label_font_size)

    # ── 5. ByteTrack 跟踪 ──
    tracker = None
    if enable_tracking:
        tracker = sv.ByteTrack()
        _log("ByteTrack: 已启用")

    # ── 6. 截面线流量统计 ──
    traffic_counter = None
    if count_lines:
        from src.cross_section_counter import CrossSectionCounter
        traffic_counter = CrossSectionCounter(count_lines)
        _log(f"截面线流量统计: 已加载 {len(count_lines)} 条截面线")

    # ── 7. 自动流量统计 ──
    auto_traffic_counter = None
    if enable_auto_traffic:
        from src.auto_traffic_counter import AutoTrafficCounter
        auto_traffic_counter = AutoTrafficCounter(min_traj_len=5, min_displacement=30)
        _log("自动流量统计: 已启用")

    # ── 8. 路面分割 ──
    road_segmenter = None
    if enable_road_detect:
        try:
            from src.road_segmentation import RoadSegmenter
            road_segmenter = RoadSegmenter(method=road_model_type)
            _log(f"道路边界检测: {road_model_type} 已启用")
        except Exception as e:
            _log(f"道路边界检测: 初始化失败 ({e})")

    # ── 9. 主循环 ──
    frame_count = 0
    detect_count = 0
    detect_start_time = time.time()

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame

    while cap.isOpened():
        if stop_event and stop_event.is_set():
            _log("检测被用户停止")
            break

        ret, frame = cap.read()
        if not ret: break
        if frame_index >= end_frame: break

        # 跳帧判断 + 目标检测总开关
        do_detect = enable_detection and should_run_detection(frame_index - start_frame, detect_skip_interval)

        if do_detect:
            if use_sahi:
                # ── SAHI 切片路径 ──
                _annotated = annotate_frame(
                    frame, detection_model=detection_model, model_name=model_name,
                    target_set=target_set, class_labels_zh=class_labels_zh,
                    slice_height=slice_height, slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio,
                    box_annotator=box_annotator, label_annotator=label_annotator,
                    add_model_watermark=True,
                )
                raw_result = get_sliced_prediction(
                    image=frame, detection_model=detection_model,
                    slice_height=slice_height, slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio,
                )
                object_predictions = [
                    pred for pred in raw_result.object_prediction_list
                    if pred.category.id in target_set
                ]
                xyxy_list = [[float(v) for v in pred.bbox.to_xyxy()] for pred in object_predictions]
                confidences_list = [pred.score.value for pred in object_predictions]
                class_ids_list = [pred.category.id for pred in object_predictions]
            else:
                # ── 直接推理路径（不使用SAHI）──
                results = detection_model(frame, verbose=False)
                xyxy_list, confidences_list, class_ids_list = [], [], []
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        cid = int(boxes.cls[i].item())
                        if cid not in target_set:
                            continue
                        conf = float(boxes.conf[i].item())
                        if conf < confidence_threshold:
                            continue
                        xyxy = boxes.xyxy[i].tolist()
                        xyxy_list.append(xyxy)
                        confidences_list.append(conf)
                        class_ids_list.append(cid)

                # 非SAHI标注
                _annotated = frame.copy()
                if xyxy_list:
                    disp_names = [
                        f"{class_labels_zh.get(cid, f'cls{cid}')} {conf:.2f}"
                        for cid, conf in zip(class_ids_list, confidences_list)
                    ]
                    _annotated = box_annotator.annotate(
                        scene=_annotated,
                        detections=sv.Detections(
                            xyxy=np.array(xyxy_list, dtype=np.float32),
                            confidence=np.array(confidences_list, dtype=np.float32),
                            class_id=np.array(class_ids_list, dtype=int),
                        ))
                    _annotated = label_annotator.annotate(
                        scene=_annotated,
                        detections=sv.Detections(
                            xyxy=np.array(xyxy_list, dtype=np.float32),
                            confidence=np.array(confidences_list, dtype=np.float32),
                            class_id=np.array(class_ids_list, dtype=int),
                        ),
                        labels=disp_names,
                    )
                # 水印
                text = f"[Model: {model_name}]"
                (tw, th), bs = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                xw = _annotated.shape[1] - tw - 15; yw = th + 15
                ovl = _annotated.copy()
                cv2.rectangle(ovl, (xw-10, yw-th-5), (xw+tw+10, yw+5), (0,0,0), -1)
                cv2.addWeighted(ovl, 0.6, _annotated, 0.4, 0, _annotated)
                cv2.putText(_annotated, text, (xw, yw), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

            detections = sv.Detections(
                xyxy=np.array(xyxy_list, dtype=np.float32) if xyxy_list else np.empty((0,4)),
                confidence=np.array(confidences_list, dtype=np.float32) if xyxy_list else np.empty((0,)),
                class_id=np.array(class_ids_list, dtype=int) if xyxy_list else np.empty((0,)),
            )

            # IoU + IoM 双阈值 NMS：去除 SAHI 切片重叠冗余框
            if len(detections) > 1:
                keep_idx = dual_nms(
                    detections.xyxy,
                    detections.confidence,
                    detections.class_id,
                )
                detections = detections[keep_idx]

            # ByteTrack
            if tracker is not None and len(detections) > 0:
                detections = tracker.update_with_detections(detections)

            # 截面线流量统计
            if traffic_counter is not None and detections.tracker_id is not None:
                for ti in range(len(detections.xyxy)):
                    if detections.tracker_id[ti] is not None:
                        traffic_counter.update(
                            int(detections.tracker_id[ti]),
                            int(detections.class_id[ti]),
                            detections.xyxy[ti],
                        )

            # 自动流量轨迹记录
            if auto_traffic_counter is not None and detections.tracker_id is not None:
                for ti in range(len(detections.xyxy)):
                    if detections.tracker_id[ti] is not None:
                        auto_traffic_counter.record(
                            int(detections.tracker_id[ti]),
                            int(detections.class_id[ti]),
                            detections.xyxy[ti],
                            frame_index,
                        )

            # 标注
            labels = [
                f"{class_labels_zh.get(int(detections.class_id[i]), f'cls{int(detections.class_id[i])}')} {detections.confidence[i]:.2f}"
                for i in range(len(detections))
            ] if len(detections) > 0 else []

            if show_labels:
                _annotated = box_annotator.annotate(scene=_annotated, detections=detections)
                _annotated = label_annotator.annotate(scene=_annotated, detections=detections, labels=labels)

            # 道路边界叠加
            if road_segmenter is not None:
                try:
                    mask = road_segmenter.segment(frame)
                    if mask is not None:
                        mask_resized = cv2.resize(mask, (_annotated.shape[1], _annotated.shape[0]))
                        overlay = _annotated.copy()
                        overlay[mask_resized > 0] = (0, 255, 0)
                        _annotated = cv2.addWeighted(overlay, 0.3, _annotated, 0.7, 0)
                except Exception:
                    pass

            detect_count += 1
        else:
            _annotated = frame.copy()

        # 实时绘制轨迹
        if show_trajectory and auto_traffic_counter is not None:
            _draw_trajectories_on_frame(_annotated, auto_traffic_counter)

        if out is not None:
            out.write(_annotated)
        frame_count += 1
        frame_index += 1

        # 进度回调（传递实时流量统计）
        _traffic_snapshot = traffic_counter.get_stats() if traffic_counter else {}
        if progress_callback:
            progress_callback(frame_count, end_frame - start_frame, _annotated, detect_count, {}, _traffic_snapshot)

    # ── 10. 后处理 ──
    cap.release()
    if out is not None:
        out.release()

    _log(f"检测完成! 共处理 {frame_count} 帧，检测 {detect_count} 帧")

    # 自动流量统计
    auto_flows = None
    if auto_traffic_counter is not None:
        try:
            auto_flows = auto_traffic_counter.analyze(image_width=width, image_height=height)
            flow_count = len(auto_flows)
            total_vehicles = sum(f.vehicle_count for f in auto_flows)
            _log(f"自动流量统计: 发现 {flow_count} 个方向，共 {total_vehicles} 辆车")
        except Exception as e:
            _log(f"自动流量统计: 分析失败 ({e})")

    result = {
        "frame_count": frame_count,
        "detect_count": detect_count,
        "output_video": str(output_video),
        "traffic_stats": traffic_counter.get_stats() if traffic_counter else None,
        "auto_flows": auto_flows,
        "auto_traffic_counter": auto_traffic_counter,
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
    _ensure_runtime()  # CLI 入口：检查依赖，缺失则自动切换到 venv
    main()
