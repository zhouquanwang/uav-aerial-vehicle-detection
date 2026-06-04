"""类别 ID → 视频标注用中文名称。"""

from __future__ import annotations

from typing import Any

# ========== 模型 1: WALDO30 YOLOv8m ==========
WALDO_TARGET_CLASSES: list[int] = [0, 1, 5, 6, 7, 8, 9, 11]
WALDO_CLASS_LABELS_ZH: dict[int, str] = {
    0: "小汽车",      # LightVehicle
    1: "行人",        # Person
    5: "非机动车",    # Bike
    6: "集装箱车",    # Container
    7: "货车",        # Truck
    8: "罐车",        # Gastank
    9: "工程车",      # Digger
    11: "大客车",     # Bus
}

# ========== 模型 2/3: VisDrone (RT-DETR / YOLOv8l) ==========
VISDRONE_TARGET_CLASSES: list[int] = [0, 2, 3, 4, 5, 8, 9]
VISDRONE_CLASS_LABELS_ZH: dict[int, str] = {
    0: "行人",        # pedestrian
    2: "自行车",      # bicycle
    3: "汽车",        # car
    4: "面包车",      # van
    5: "卡车",        # truck
    8: "公交车",      # bus
    9: "摩托车",      # motor
}

# ========== 默认值（兼容旧配置）==========
DEFAULT_TARGET_CLASSES: list[int] = WALDO_TARGET_CLASSES
DEFAULT_CLASS_LABELS_ZH: dict[int, str] = WALDO_CLASS_LABELS_ZH

# ========== 模型类别配置映射 ==========
MODEL_CLASS_CONFIGS = {
    1: {"target_classes": WALDO_TARGET_CLASSES, "labels": WALDO_CLASS_LABELS_ZH},
    2: {"target_classes": VISDRONE_TARGET_CLASSES, "labels": VISDRONE_CLASS_LABELS_ZH},
    3: {"target_classes": VISDRONE_TARGET_CLASSES, "labels": VISDRONE_CLASS_LABELS_ZH},
}


def parse_class_labels_zh(raw: Any) -> dict[int, str]:
    """从 YAML 的 class_labels_zh 段解析为 {id: 中文名}。"""
    if not raw:
        return dict(DEFAULT_CLASS_LABELS_ZH)
    if not isinstance(raw, dict):
        raise ValueError("class_labels_zh must be a mapping of id -> name")
    return {int(k): str(v) for k, v in raw.items()}


def label_for_class(
    class_id: int,
    class_labels_zh: dict[int, str],
    *,
    fallback_name: str | None = None,
) -> str:
    """取中文标注名；未配置时回退到模型英文名。"""
    if class_id in class_labels_zh:
        return class_labels_zh[class_id]
    if fallback_name:
        return fallback_name
    return f"class_{class_id}"
