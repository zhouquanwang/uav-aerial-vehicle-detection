# 无人机航拍视频车辆检测

UAV Aerial Video Vehicle Detection — 工程目录（初始化阶段，尚未开发业务代码）。

## 项目目标

- 从无人机航拍视频中检测、定位车辆
- 支持离线视频批处理与后续可扩展的实时流处理
- 面向俯视/斜视航拍场景的小目标与密集车流

## 当前状态

**推理测试版**：WALDO + SAHI 切片检测 MP4（见 `src/predict_video.py`）。训练与 UI 尚未实现。

## 规划模块（待开发）

| 模块 | 说明 |
|------|------|
| 视频输入 | 读取 MP4/AVI 等航拍素材，抽帧与预处理 |
| 车辆检测 | 基于深度学习的目标检测（如 YOLO 系列） |
| 后处理 | NMS、置信度过滤、可选跟踪与轨迹输出 |
| 结果导出 | 标注视频、JSON/CSV 统计、可视化预览 |

## 目录结构

```
uav-aerial-vehicle-detection/
├── configs/          # 运行时与训练配置（YAML 等）
├── data/             # 原始视频、标注数据集（不提交大文件）
├── docs/             # 设计说明、数据集说明、实验记录
├── models/           # 训练权重与导出模型
├── outputs/          # 推理结果、日志、导出视频
├── scripts/          # 数据下载、格式转换等辅助脚本
└── src/              # 应用与推理源码（待实现）
```

## 推理测试版快速开始

依赖版本对齐 `yolo_drone_env_linux.yml`（Python 3.10+，本机可用 3.11）。

```powershell
cd f:\cursor\uav-aerial-vehicle-detection
.\scripts\setup_venv.ps1          # 创建 .venv 并安装 GPU 依赖
.\.venv\Scripts\Activate.ps1
python scripts\smoke_test.py      # 加载模型 + 3 帧合成视频切片推理

# 将航拍 MP4 放到 data\input.mp4（或改 configs\inference.yaml）
python -m src.predict_video
# 或在任意目录（会自动切到本项目 .venv）：
python f:\cursor\uav-aerial-vehicle-detection\predict.py
# 或指定路径：
python -m src.predict_video --input data\your.mp4 --output outputs\your_out.mp4
```

- 默认模型：`models/WALDO30_yolov8m_640x640 (1).pt`
- 默认只检测 WALDO 类别 **0（LightVehicle）**
- 配置：`configs/inference.yaml`

## 环境约定

- 虚拟环境：项目根目录 `.venv`
- `requirements-gpu.txt` / `requirements-cpu.txt`：PyTorch 安装顺序说明
- `requirements.txt`：sahi、supervision、ultralytics 等推理依赖

## 文档

- [检测程序逻辑说明](docs/检测程序逻辑说明.md) — 推理流程、跳帧、SAHI 切片、配置项说明

## 相关资源

- [VisDrone](http://aiskyeye.com/) — 常见航拍检测数据集
- 同 workspace：`aerial-traffic-system` — 航拍交通视觉分析（检测、跟踪等）
