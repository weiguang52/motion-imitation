# Motion Imitation

本项目将普通视频或摄像头视频流中的人体动作恢复为三维人体运动，并进一步重定向为双足人形机器人的关节角序列。核心流程由 YOLO 人体跟踪、GVHMR 三维动作恢复、SMPL/SMPL-X 人体模型以及 Pink/Pinocchio 逆运动学组成。

GitHub 仓库：<https://github.com/weiguang52/motion-imitation>

## 功能

- 从 MP4、AVI、MOV 等视频中恢复连续三维人体动作。
- 多人画面中只保留位于画面中心、身体较完整且轨迹稳定的主人物。
- 将 SMPL 人体动作重定向为项目机器人模型的 28 维 DOF 序列。
- 支持只导出 IK 前的 SMPL 关节点 `.npy`，方便下游自行重定向。
- 提供文件批处理、REST API、LLM Tool API、WebSocket 摄像头流和 ZMQ 视频流入口。
- 提供机器人动作插值、碰撞感知过渡和可视化工具。

## 多人主目标选择

多人视频不会逐帧选择“当前最中心的人”，而是先用 YOLO 建立人物轨迹，再对整段轨迹评分，因此多人交叉时不容易频繁切换动作源。默认权重为：

| 指标 | 权重 | 含义 |
| --- | ---: | --- |
| 画面中心度 | 45% | 人物检测框中心距离画面中心越近越好 |
| 身体完整度 | 25% | 检测框远离四个画面边缘，减少头、手、脚被裁切 |
| 轨迹连续性 | 20% | 在视频片段内持续出现且跟踪稳定 |
| 可见面积 | 10% | 在不压过中心度的前提下，避免选择过小的远处人物 |

实现位于：

- `GVHMR/hmr4d/utils/preproc/person_selector.py`
- `GVHMR/hmr4d/utils/preproc/tracker.py`

当前身体完整度主要根据检测框是否被画面边缘裁切估计。如果人物被桌子或另一个人从画面内部遮挡，后续可以再叠加关键点置信度判断。

## 处理流程

```text
视频文件 / JPEG 视频流
        ↓
YOLO 多人检测与轨迹跟踪
        ↓
中心、完整、连续的主人物轨迹
        ↓
ViTPose + HMR2 特征
        ↓
GVHMR 三维 SMPL/SMPL-X 动作
        ↓
原始 3D 关节点 .npy（可选）
        ↓
Pink / Pinocchio IK
        ↓
机器人 28-DOF 动作序列
```

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `server_side/` | GVHMR 常驻模型、REST/WebSocket/ZMQ 服务与客户端 |
| `GVHMR/` | GVHMR 源码、人体检测、姿态估计和主人物选择 |
| `ik_redirection_npy.py` | SMPL 动作到机器人关节角的 IK 重定向 |
| `easy_MotionInterpolator.py` | 两段机器人动作的平滑、碰撞感知过渡 |
| `data/urdf/`、`data/meshes/` | 机器人 URDF 和网格 |
| `inputs/test_mp4/` | 单人和多人测试视频 |
| `SMPLSim-master/` | SMPLSim 相关源码 |
| `smplx-master/` | SMPL-X Python 源码 |
| `output/`、`raw_motion_npy/` | 本地运行输出，不提交 Git |

## 系统要求

- Linux
- Python 3.10
- NVIDIA GPU；主流程当前固定使用 CUDA
- 与 PyTorch 匹配的 CUDA 环境
- FFmpeg
- Git

建议至少预留约 15GB 磁盘空间，用于环境、模型权重和中间结果。首次启动需要加载多个模型，等待几十秒属于正常现象。

## 克隆

DPVO 作为子模块保存。推荐递归克隆：

```bash
git clone --recursive https://github.com/weiguang52/motion-imitation.git
cd motion-imitation
```

如果已经普通克隆：

```bash
git submodule update --init --recursive
```

DPVO 默认没有参与快速推理路径；不需要移动相机视觉里程计时，可以暂不安装它的编译依赖。

## Python 环境

仓库提供当前开发环境的 Conda 描述：

```bash
conda env create -f environment.yml
conda activate IKRetargeting
pip install -e ./GVHMR
```

运行服务前：

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/GVHMR:${PYTHONPATH}"
```

注意：

- `requirements-ik.txt` 和 `requirements-vis.txt` 是历史开发环境快照，不是完全可移植的最小依赖清单。
- `requirements-ik.txt` 中包含旧机器的可编辑安装绝对路径，不能在新机器上原样复现。
- PyTorch、TorchVision、CUDA 和 NumPy 必须保持二进制兼容。若出现 `_ARRAY_API not found` 或 “compiled using NumPy 1.x” 等提示，应按照当前 PyTorch 版本安装兼容的 NumPy，而不是忽略警告。
- GVHMR 的基础安装说明见 [`GVHMR/docs/INSTALL.md`](GVHMR/docs/INSTALL.md)。

## 未上传到 GitHub 的文件

模型权重、授权受限的人体模型、运行输出和缓存均被 `.gitignore` 排除。本机这些内容包含重复副本时约占 11GB；按唯一模型计算约 6.1GB。

### 模型与授权文件

| 文件或目录 | 本机约大小 | 用途 | 是否必需 |
| --- | ---: | --- | --- |
| `inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt` | 163MB | GVHMR 主模型 | 必需 |
| `inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt` | 2.71GB | HMR2 图像特征 | 必需 |
| `inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth` | 2.55GB | 二维人体关键点 | 必需 |
| `inputs/checkpoints/yolo/yolov8x.pt` | 137MB | 人体检测与跟踪 | 必需 |
| `inputs/checkpoints/dpvo/dpvo.pth` | 14MB | DPVO 相机运动 | 可选 |
| `inputs/checkpoints/body_models/smplx/SMPLX_*.npz` | 合计约 326MB | SMPL-X 人体模型 | 必需 |
| `inputs/checkpoints/body_models/smpl/SMPL_*.pkl` | 依下载版本而定 | SMPL 人体模型 | 必需 |
| `data/smpl/SMPL_NEUTRAL.pkl` | 247MB | 服务端 SMPL 关节点生成 | 必需 |

GVHMR、HMR2、ViTPose、YOLO 和 DPVO 权重下载入口及目录说明见 [`GVHMR/docs/INSTALL.md`](GVHMR/docs/INSTALL.md)。SMPL 与 SMPL-X 需要分别在其官方网站注册并同意许可后下载，不能直接随本仓库重新分发。

代码同时存在仓库根目录相对路径和 `GVHMR` 项目根路径。为避免复制大型权重，推荐只保存一份：

```bash
mkdir -p inputs/checkpoints
mkdir -p GVHMR/inputs
ln -s ../../inputs/checkpoints GVHMR/inputs/checkpoints
```

如果 `GVHMR/inputs/checkpoints` 已存在，应先确认其中是否有唯一文件；不要直接删除有用权重。完成符号链接后，模型目录应类似：

```text
inputs/checkpoints/
├── body_models/
│   ├── smpl/
│   │   ├── SMPL_NEUTRAL.pkl
│   │   ├── SMPL_MALE.pkl
│   │   └── SMPL_FEMALE.pkl
│   └── smplx/
│       ├── SMPLX_NEUTRAL.npz
│       ├── SMPLX_MALE.npz
│       └── SMPLX_FEMALE.npz
├── dpvo/dpvo.pth
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
└── yolo/yolov8x.pt
```

`server_side/run.py` 还会从 `data/smpl/SMPL_NEUTRAL.pkl` 加载 neutral SMPL。可以使用符号链接避免另一份复制：

```bash
mkdir -p data/smpl
ln -s ../../inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl \
  data/smpl/SMPL_NEUTRAL.pkl
```

### 其他未上传内容

- `output/`：GIF、PNG 等历史可视化结果。
- `raw_motion_npy/`、`outputs/`：视频流或批处理生成的动作文件。
- `data/output/`：IK 和插值结果。
- `.vscode/`、`__pycache__/`、`.pytest_cache/`：编辑器和 Python 缓存。
- `GVHMR/.git-upstream/`：本机保留的原始 GVHMR Git 元数据备份。
- 根目录下载压缩包：源码已经展开，因此不重复提交 ZIP。

这些文件没有从本机删除，只是不进入 Git 历史。

## 使用方法

所有命令默认从仓库根目录执行，并先激活环境：

```bash
conda activate IKRetargeting
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/GVHMR:${PYTHONPATH}"
```

### 1. 视频导出原始三维动作

该模式只运行到 GVHMR/SMPL 关节点，不执行机器人 IK：

```bash
python server_side/run.py \
  --input inputs/test_mp4/imitation_multiple_people_test.mp4 \
  --raw-motion-only \
  --raw-motion-output-dir raw_motion_npy \
  --raw-motion-coord tw
```

输出文件形如：

```text
raw_motion_npy/action_imitation_1780000000000.npy
```

坐标系可选值：

- `tw`: recommended for tw_retargeting; pelvis-relative and robot-facing.
- `ik_input` or `h1`: legacy H1 orientation, pelvis-relative.
- `smpl`：原始 SMPL 坐标，不归零。
- `v3`：旧版 Z-up 调试格式。

#### 可视化导出的 `.npy`

`server_side/npyvisual.py` renders the first 29 joints of an exported NPY.
Set its `path` variable before running it; `tw` mode includes the robot-facing alignment.
Legacy `ik_input` and `h1` files retain their original orientation.

```bash
python server_side/npyvisual.py
```

脚本兼容 `(T, J, 3)`、`(J, 3, T)` 和 `(1, J, 3, T)`，也支持包含 `motion`
字段的 object dict；关节数应为 22 或 29。默认渲染全部可用关节，将
`USE_22_ONLY` 设为 `True` 可只显示前 22 个 SMPL 关节。

结果写入 `output/`：

```text
output/<输入文件名>.gif
output/<输入文件名>.png
```

### 2. 启动完整动作模仿服务

完整模式会执行 GVHMR 和机器人 IK，返回 28-DOF 序列：

```bash
python server_side/action_imitation_server.py --host 0.0.0.0 --port 8003
```

服务入口：

| 地址 | 用途 |
| --- | --- |
| `GET /tools` | 返回 LLM Tool JSON Schema |
| `POST /tool/call` | 执行 LLM 工具调用 |
| `POST /generate` | 输入服务器本地视频路径 |
| `POST /action_imitation/session/start` | Agent 在视频到达前登记 `robot_id + task_id` |
| `WS /ws/stream` | 发送 JPEG 视频帧 |
| `GET /docs` | FastAPI 交互文档 |

处理文件：

```bash
curl -X POST http://127.0.0.1:8003/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "video_path": "/absolute/path/to/video.mp4",
    "static_cam": true
  }'
```

`video_path` 是服务端机器上的绝对路径。固定机位使用 `"static_cam": true`；手持或移动相机使用 `false`。

### 3. 启动只保存原始动作的服务

```bash
python server_side/action_imitation_server.py \
  --host 0.0.0.0 \
  --port 8003 \
  --raw-motion-only \
  --raw-motion-output-dir raw_motion_npy \
  --raw-motion-coord tw
```

启动较慢。出现“模型加载完成”和服务监听信息后才表示可以接收请求。

### 4. 用本地视频测试 WebSocket 流

先启动服务，再在另一个终端运行：

```bash
python server_side/test_stream.py \
  --video inputs/test_mp4/imitation_multiple_people_test.mp4 \
  --fps 30 \
  --chunk_sec 4 \
  --server localhost:8003
```

服务端每累计 `chunk_sec` 秒帧数处理一次。较短 chunk 延迟更低，但时序稳定性通常也更弱。

### 5. 推送真实摄像头

```bash
python server_side/camera_client.py \
  --camera 0 \
  --fps 30 \
  --chunk_sec 4 \
  --server localhost:8003 \
  --static_cam
```

移动摄像机时移除 `--static_cam`。客户端发送 JPEG 二进制帧，结束时发送文本 `__END__`。

### 6. ZMQ 视频流

默认 ZMQ topic 为 `robot_robot_001_action_imitation_camera_left`，默认连接 `tcp://127.0.0.1:5558`：

```bash
ZMQ_STREAM_ENABLED=1 \
ZMQ_STREAM_URL=tcp://127.0.0.1:5558 \
ZMQ_STREAM_TOPIC=robot_robot_001_action_imitation_camera_left \
python server_side/action_imitation_server.py \
  --raw-motion-only \
  --raw-motion-output-dir raw_motion_npy \
  --raw-motion-coord tw
```

Agent 必须在发送该任务的视频帧前登记 session；重复登记相同
`robot_id + task_id` 是幂等的，不同 `task_id` 与当前录制冲突时返回 HTTP 409：

```bash
curl -X POST http://127.0.0.1:8003/action_imitation/session/start \
  -H 'Content-Type: application/json' \
  -d '{
    "robot_id": "robot_001",
    "task_id": "action_imitation_example",
    "source_turn_id": "turn-example",
    "source_request_id": "request-example",
    "function_call_id": "call-example"
  }'
```

未登记 session 的 ZMQ 帧会被忽略。录制 idle flush 时，服务会固定本次
session 身份；最终回调包含 `robot_id`、`task_id`、绝对 `output_path` 和
文件 `sha256`，不会被后续任务覆盖。

登记接口的主要响应：

- 缺少非空 `robot_id` 或 `task_id`：HTTP 400，
  `reason=missing_robot_id_or_task_id`
- 首次登记：HTTP 200，`state=armed`
- 重复登记同一 `robot_id + task_id`：HTTP 200，
  `reason=already_registered`，且不会重置已收到的帧或已有状态
- 同一机器人仍有其他 active task：HTTP 409，
  `reason=active_session_conflict`，并返回 `active_task_id` 和
  `requested_task_id`

session 的内部状态依次为 `armed -> recording -> processing`。合并与回调成功后
进入 `completed`；没有可合并分块或处理失败时进入 `processing_failed`；回调重试
仍失败时保留合并文件路径与 SHA-256，并进入 `notify_pending`。旧任务进入
`processing` 后会从 active 槽位原子摘除，因此同一机器人可以登记下一任务，而旧任务
仍使用自己的身份快照完成合并和回调。

idle flush 只合并已经达到 `ZMQ_STREAM_CHUNK_SEC` 并完成推理的分块。末尾不足一个
chunk 的帧会被丢弃，并记录 `session_tail_dropped` 日志；没有 armed session 的帧会
记录 `frame_ignored`，不会进入动作处理。

支持的主要环境变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `ZMQ_STREAM_ENABLED` | `0` | 是否启用 ZMQ SUB |
| `ZMQ_STREAM_URL` | `tcp://127.0.0.1:5558` | 视频桥输出地址 |
| `ZMQ_STREAM_TOPIC` | `robot_robot_001_action_imitation_camera_left` | 订阅 topic |
| `ZMQ_STREAM_FPS` | `30` | 输入帧率 |
| `ZMQ_STREAM_CHUNK_SEC` | `4` | 单次推理片段秒数 |
| `ZMQ_STREAM_STATIC_CAM` | `0` | 是否固定机位 |
| `ZMQ_STREAM_MAX_CHUNKS` | `0` | 最大 chunk 数，0 表示不限 |
| `ZMQ_STREAM_IDLE_FLUSH_SEC` | `6.0` | 无新帧后合并本次会话的等待时间 |
| `AGENT_ACTION_RESULT_URL` | `http://127.0.0.1:8007/action_imitation/result` | 动作结果回调地址 |
| `AGENT_ACTION_RESULT_TIMEOUT_SEC` | `5` | Agent 结果回调超时秒数 |
| `AGENT_ACTION_RESULT_RETRIES` | `3` | Agent 结果回调最大尝试次数（HTTP 4xx 不重试） |

更完整的现有集成命令见 [`server_side/READEME.md`](server_side/READEME.md)。其中 `robot_ai` 和模拟边缘发布器属于同一工作区的其他项目，并不包含在本仓库内。

### 7. 单独运行 IK 示例

仓库自带 `data/input_for_ik/batch_size=18.npy` 示例：

```bash
python ik_redirection_npy.py
```

结果保存到：

```text
data/output/result_for_ik.pkl
```

### 8. 动作过渡与插值

```bash
python easy_MotionInterpolator.py
```

默认读取：

- `data/input_for_interpolation/right-back.pkl`
- `data/input_for_interpolation/right-front.pkl`

并写入 `data/output/result_for_interpolation.pkl`。插值器会统一帧率，并在直接过渡有碰撞风险时加入中继姿态。

## 测试

多人主目标选择测试不需要加载 YOLO 或 GPU：

```bash
python -m unittest GVHMR/tools/unitest/test_person_selector.py -v
```

语法检查：

```bash
python -m py_compile \
  GVHMR/hmr4d/utils/preproc/person_selector.py \
  GVHMR/hmr4d/utils/preproc/tracker.py \
  GVHMR/tools/unitest/test_person_selector.py
```

完整端到端测试需要 CUDA、所有权重和 SMPL/SMPL-X 文件。

## 常见问题

### 启动时报模型文件不存在

检查 `inputs/checkpoints` 是否完整，并确认 `GVHMR/inputs/checkpoints` 指向同一目录。再检查 `data/smpl/SMPL_NEUTRAL.pkl`。

### 多人时选错人物

尽量让动作示范者位于画面中心并完整入镜。调整权重时修改 `DEFAULT_SELECTION_WEIGHTS`，随后运行多人选择测试。需要识别画面内部遮挡时，应增加关键点置信度或人体分割指标。

### 服务启动后很久没有监听端口

服务在创建 FastAPI 应用前会加载所有模型。观察日志，只有模型加载完成后才会开始监听端口。

### GitHub 中为什么没有权重

HMR2 和 ViTPose 单文件均超过 GitHub 普通 Git 的 100MB 限制；SMPL/SMPL-X 还受各自许可约束。不要使用 `git add -f` 把它们提交，也不要把运行输出加入 Git。

## 第三方项目与许可

本仓库包含或引用 GVHMR、SMPLSim、SMPL-X 和 DPVO。请保留各目录中的 LICENSE，并分别遵守模型、代码和数据集的原始许可。本仓库不授予重新分发 SMPL/SMPL-X 模型文件的权利。

## Native tw_retargeting NPY bridge

The raw-motion exporter writes C-contiguous little-endian `float32` NPY. It reads
video FPS from OpenCV and resamples joint positions to 20 Hz by default, matching
the native `tw_retargeting` input contract. A clip must contain at least nine
exported frames. Set `--raw-motion-target-fps 0` to keep the source frame rate.

```bash
python server_side/run.py --input inputs/test_mp4/your_video.mp4 \
  --raw-motion-only --raw-motion-coord tw \
  --raw-motion-target-fps 20 --raw-motion-extended \
  --raw-motion-output-dir raw_motion_npy
```

The ordinary export has shape `(T,29,3)`. `--raw-motion-extended` writes
`(T,43,3)` while retaining the original 29 joint positions without reordering:

| Rows | Data |
| --- | --- |
| 0:29 | SMPL joint positions, pelvis-relative with `tw` |
| 29:32 | Left ankle world rotation matrix, three rows |
| 32:35 | Right ankle world rotation matrix, three rows |
| 35:38 | Left wrist world rotation matrix, three rows |
| 38:41 | Right wrist world rotation matrix, three rows |
| 41 | Left hand `[openness, detection_confidence, 0]` |
| 42 | Right hand `[openness, detection_confidence, 0]` |

The `tw` mode applies a 180-degree SMPL Y-up yaw before the native (z,x,y)
permutation, aligning human forward/left axes with the fixed-base robot URDF.
Use `ik_input` only for the legacy H1 orientation. Existing NPY files made
with `ik_input` need this alignment before tw_retargeting.

Rotation matrices are derived from the SMPL kinematic chain and use the same
source axes as the exported joints. Hand openness is estimated from visible
2D finger landmarks with MediaPipe: 0 means closed, 1 means open. If a hand is
not detected, openness is 0.5 and confidence is 0. These hand estimates are
not SMPL predictions or reliable grasp measurements. Install `mediapipe==0.10.14`
to enable the extended export. The native retargeter reads only the first 22
joint rows, so the appended rows do not affect its output. The existing
`server_side/npyvisual.py` renders the first 29 joints of an exported NPY.
Set its `path` variable before running it; `tw` mode includes the robot-facing alignment.
Legacy `ik_input` and `h1` files retain their original orientation.
