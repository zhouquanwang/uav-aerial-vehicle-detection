"""
道路分割模块

支持的模型:
  - hsv: 基于 HSV 色彩空间的快速道路检测（默认，无需额外依赖）
  - segformer: SegFormer B0 Cityscapes (Hugging Face, 需 pip install transformers)
  - hrnet_ocr: HRNet-W48-OCR (MMSeg, 需手动下载权重)

Cityscapes 道路相关类别:
  class 0: road, class 1: sidewalk

用途:
  为透视校正提供道路区域掩码，解决拍摄角度抖动导致的统计线偏移问题
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class RoadSegmenter:
    """
    道路分割器

    使用方式:
        seg = RoadSegmenter(model_type="hsv")
        mask = seg.predict(frame)  # 二值掩码: 255=道路, 0=非道路
        overlay = seg.get_mask_overlay(frame, mask)  # 可视化
    """

    MODEL_TYPES = ("hsv", "segformer", "hrnet_ocr")

    def __init__(self, model_type: str = "hsv", device: str = "cpu",
                 hrnet_weight: Optional[str | Path] = None):
        """
        Args:
            model_type: "hsv" | "segformer" | "hrnet_ocr"
            device: "cpu" | "cuda"
            hrnet_weight: HRNet-OCR 权重 .pth 路径 (默认: models/hrnet_ocr_cs_8162_torch11.pth)
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"不支持的模型: {model_type}，可选 {self.MODEL_TYPES}")

        self.model_type = model_type
        self.device = device
        self.model = None
        self._input_size = (1024, 512)

        print(f"[道路分割] 模型: {model_type} (device={device})")
        t0 = time.time()

        if model_type == "segformer":
            self._init_segformer()
        elif model_type == "hrnet_ocr":
            self._init_hrnet_ocr(hrnet_weight)
        # hsv 无需初始化

        if self.model:
            self.model.eval()
            if hasattr(self.model, 'to'):
                self.model.to(device)
        print(f"[道路分割] 就绪 ({time.time() - t0:.1f}s)")

    # ════════════════════════════════════════
    # 模型初始化
    # ════════════════════════════════════════

    def _init_segformer(self):
        try:
            from transformers import SegformerForSemanticSegmentation
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
            )
        except ImportError:
            raise ImportError("pip install transformers")

    def _init_hrnet_ocr(self, weight_path: Optional[str | Path] = None):
        """加载 HRNet-W48-OCR + Cityscapes 权重"""
        try:
            from mmseg.apis import init_model
        except ImportError:
            raise ImportError("pip install mmcv mmsegmentation")

        # 解析权重路径 (默认: models/hrnet_ocr_cs_8162_torch11.pth)
        if weight_path is None:
            default = Path(__file__).resolve().parent.parent / "models" / "hrnet_ocr_cs_8162_torch11.pth"
            weight_path = default if default.exists() else None

        if weight_path and Path(weight_path).exists():
            print(f"  加载权重: {weight_path}")
            ckpt = str(weight_path)
        else:
            ckpt = None
            print("  [WARNING] 未找到 HRNet-OCR 权重，仅加载模型结构")

        self.model = init_model(
            "ocrnet_hr48_512x1024_160k_cityscapes",
            checkpoint=ckpt,
            device="cpu",
        )

    # ════════════════════════════════════════
    # 推理
    # ════════════════════════════════════════

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        道路分割 → 二值掩码

        Args:
            frame: BGR (H, W, 3)

        Returns:
            mask: uint8 (H, W), 255=道路, 0=非道路
        """
        if self.model_type == "hsv":
            return self._predict_hsv(frame)
        elif self.model_type == "segformer":
            return self._predict_segformer(frame)
        elif self.model_type == "hrnet_ocr":
            return self._predict_hrnet_ocr(frame)
        raise ValueError(f"未知模型: {self.model_type}")

    # ─── HSV 颜色检测 ───

    def _predict_hsv(self, frame: np.ndarray) -> np.ndarray:
        """
        基于 HSV 色彩空间的航拍道路检测。

        原理: 航拍视角下，道路通常是灰色区域（低饱和度）。
        步骤:
          1. HSV 转换 → 低饱和度 + 中等明度阈值
          2. 形态学闭运算连接碎片
          3. 保留最大连通域
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 低饱和度 = 灰色（道路）
        s_low = hsv[:, :, 1] < 60
        # 排除过暗（阴影）和过亮（白车）
        v_mid = (hsv[:, :, 2] > 50) & (hsv[:, :, 2] < 220)

        road_mask = (s_low & v_mid).astype(np.uint8) * 255

        # 形态学闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

        # 保留最大连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask, connectivity=8)
        if num_labels > 1:
            largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            road_mask = ((labels == largest) * 255).astype(np.uint8)

        return road_mask

    # ─── SegFormer ───

    def _predict_segformer(self, frame: np.ndarray) -> np.ndarray:
        import torch
        from transformers import SegformerImageProcessor

        processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits, size=frame.shape[:2], mode="bilinear", align_corners=False
            )
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Cityscapes class 0=road, 1=sidewalk
        road_mask = np.isin(pred, [0, 1])
        return (road_mask * 255).astype(np.uint8)

    # ─── HRNet-OCR ───

    def _predict_hrnet_ocr(self, frame: np.ndarray) -> np.ndarray:
        from mmseg.apis import inference_model

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = inference_model(self.model, rgb)
        seg_map = result.pred_sem_seg.data.cpu().numpy().squeeze()

        mask = np.isin(seg_map, [0, 1])  # road + sidewalk
        return (mask * 255).astype(np.uint8)

    # ════════════════════════════════════════
    # 可视化
    # ════════════════════════════════════════

    def get_mask_overlay(self, frame: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        """将掩码叠加到帧上（绿色=道路）"""
        overlay = frame.copy()
        color = np.array([0, 255, 0], dtype=np.float32)
        road_area = mask == 255
        overlay[road_area] = (
            frame[road_area].astype(np.float32) * (1 - alpha) + color * alpha
        ).astype(np.uint8)

        # 轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        return overlay


# ════════════════════════════════════════════
# 测试入口
# ════════════════════════════════════════════

def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="测试图片")
    parser.add_argument("--model", default="hsv", choices=RoadSegmenter.MODEL_TYPES)
    parser.add_argument("--output", type=Path, help="输出路径")
    args = parser.parse_args()

    seg = RoadSegmenter(model_type=args.model)

    if args.image:
        frame = cv2.imread(str(args.image))
    else:
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

    t0 = time.time()
    mask = seg.predict(frame)
    dt = (time.time() - t0) * 1000

    pct = (mask == 255).sum() / mask.size * 100
    print(f"  {frame.shape[1]}x{frame.shape[0]} -> {dt:.0f}ms, 道路={pct:.1f}%")

    overlay = seg.get_mask_overlay(frame, mask)
    out = args.output or PROJECT_ROOT / "outputs" / "road_seg_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay)
    print(f"  -> {out}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    import sys
    sys.exit(test())
