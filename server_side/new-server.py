"""
new-server.py - 支持视频流输入的GVHMR服务，可作为大模型工具调用

相比 server.py 的主要改进：
  1. WebSocket 视频流接口：客户端逐帧推送，服务端缓冲 4s 后触发处理
  2. LLM 工具调用接口：Pydantic 定义输入结构，/tools 返回 JSON Schema 供大模型使用
  3. 保留原有的文件路径 REST 接口（/generate）向后兼容

最小时间单位：4 秒
  - 处理耗时约 2.25s（线性外推：4/8 * 4.5s）
  - 流水线下有效延迟 ≈ 4s（采集第N+1块时处理第N块）
  - 4s@30fps=120帧，GVHMR 时序平滑效果稳定
  - 若对延迟不敏感，改为 8s 质量更佳

依赖安装：
  pip install opencv-python-headless

运行方式：
  python new-server.py --port 8003

大模型调用流程：
  1. GET /tools          → 获取工具的 JSON Schema，注入到 LLM system prompt 或 tools 字段
  2. LLM 决策后返回 JSON → POST /tool/call 执行，Pydantic 校验后推理
"""

import asyncio
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from run import GVHMRSystem

# ============================================================
# 全局配置
# ============================================================
CHUNK_DURATION_SEC = 4   # 最小处理单位（秒）
DEFAULT_FPS = 30         # 默认摄像头帧率

# ============================================================
# 模型加载（全局单例，常驻显存）
# ============================================================
print("正在启动服务，请稍候（正在加载模型）...")
try:
    gvhmr_sys = GVHMRSystem(device='cuda')
    print("模型加载完成，服务准备就绪！")
except Exception as e:
    print(f"严重错误：模型加载失败 - {e}")
    sys.exit(1)

# 用于在 asyncio 中跑同步阻塞的推理任务
_executor = ThreadPoolExecutor(max_workers=1)


# ============================================================
# LLM 工具输入结构定义（Pydantic + Field 描述）
# Field 的 description 会被自动提取进 JSON Schema，直接喂给大模型
# ============================================================

class ToolName(str, Enum):
    PROCESS_VIDEO = "process_video_to_robot_motion"
    START_STREAM  = "start_camera_stream"

class ProcessVideoInput(BaseModel):
    """输入一段视频文件，输出机器人关节角度序列（DOF）"""
    video_path: str = Field(
        ...,
        description="视频文件的绝对路径，支持 .mp4 / .avi / .mov 等格式"
    )
    static_cam: bool = Field(
        default=False,
        description="相机是否固定不动。固定机位填 true，手持或移动相机填 false"
    )

class StartStreamInput(BaseModel):
    """启动摄像头视频流处理，每隔 chunk_sec 秒返回一次机器人关节角度（DOF）"""
    camera_index: int = Field(
        default=0,
        description="摄像头编号，通常内置摄像头为 0，外接为 1、2……"
    )
    fps: int = Field(
        default=30,
        ge=1, le=60,
        description="摄像头帧率，通常为 30"
    )
    chunk_sec: int = Field(
        default=CHUNK_DURATION_SEC,
        ge=2, le=16,
        description=(
            f"每隔多少秒触发一次推理，推荐 {CHUNK_DURATION_SEC} 秒。"
            "越小延迟越低但动作质量略降，越大质量越好但延迟增加。"
        )
    )
    static_cam: bool = Field(
        default=False,
        description="相机是否固定不动"
    )
    max_chunks: int = Field(
        default=0,
        ge=0,
        description="最多处理几个 chunk 后自动停止，0 表示持续运行直到客户端断开"
    )


# --- 生成给 LLM 的工具 Schema（对应 OpenAI tools 格式）---
def generate_tool_schemas() -> list[dict]:
    return [
        {
            "name": ToolName.PROCESS_VIDEO,
            "description": (
                "将包含人体运动的视频文件转换为机器人关节角度序列（DOF）。"
                "流程：GVHMR 提取3D人体姿态 → 逆运动学映射到机器人电机角度。"
                "返回时序 DOF 数组（shape: [frames, n_joints]）和帧率。"
                "适用于已有录像文件的场景，如模仿学习数据采集、离线动作重定向。"
                f"处理速度约为视频时长的 0.56 倍（8s视频约 4.5s 处理完）。"
            ),
            "parameters": ProcessVideoInput.model_json_schema()
        },
        {
            "name": ToolName.START_STREAM,
            "description": (
                "启动摄像头实时视频流处理，每隔 chunk_sec 秒输出一次机器人关节角度（DOF）。"
                "适用于实时遥操作、在线运动捕捉场景。"
                f"推荐 chunk_sec={CHUNK_DURATION_SEC}，此时稳态延迟约 {CHUNK_DURATION_SEC}s，"
                "推理耗时约 2.25s（流水线并行，不阻塞采集）。"
                "调用后服务端通过 WebSocket /ws/stream 推送结果，需同步建立 WS 连接。"
            ),
            "parameters": StartStreamInput.model_json_schema()
        }
    ]


# ============================================================
# 核心推理函数（同步，在线程池中调用）
# ============================================================
def _run_gvhmr(video_path: str, static_cam: bool = False) -> dict:
    result = gvhmr_sys.process_video(
        video_path_str=video_path,
        static_cam=static_cam,
        verbose=False
    )
    if result is None:
        raise RuntimeError("GVHMR optimization returned None")
    dof = result["dof"].tolist()
    fps = result.get("fps", 90)
    return {
        "status": "success",
        "dof": dof,
        "fps": fps,
        "total_frames": len(dof),
        "duration_sec": round(len(dof) / fps, 2) if fps else 0,
        "dof_shape": [len(dof), len(dof[0]) if dof else 0]
    }


def _frames_to_tmp_video(frames: list[np.ndarray], fps: int) -> str:
    h, w = frames[0].shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return tmp.name


async def _process_frames_async(frames: list[np.ndarray], fps: int, static_cam: bool) -> dict:
    loop = asyncio.get_event_loop()
    tmp_path = await loop.run_in_executor(_executor, _frames_to_tmp_video, frames, fps)
    try:
        result = await loop.run_in_executor(_executor, _run_gvhmr, tmp_path, static_cam)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return result


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="GVHMR Robot Motion API",
    version="2.0",
    description="将人体运动视频转换为机器人关节角度(DOF)序列，支持大模型工具调用"
)


# ============================================================
# 1. 工具 Schema 查询接口（LLM 启动时调用一次，获取可用工具列表）
# ============================================================
@app.get(
    "/tools",
    summary="获取所有工具的 JSON Schema",
    description="大模型在初始化时调用此接口，将返回的 schema 注入到 tools/system prompt 中"
)
def get_tools():
    return {"tools": generate_tool_schemas()}


# ============================================================
# 2. LLM 工具调用执行接口
#    LLM 决策后将工具名 + 参数以 JSON 发到此接口
#    Pydantic 负责校验，保证数据安全
# ============================================================
class ToolCallRequest(BaseModel):
    name: ToolName = Field(..., description="工具名称")
    parameters: dict   = Field(..., description="工具参数，格式见 /tools 返回的 schema")

@app.post(
    "/tool/call",
    summary="执行大模型选定的工具",
    description=(
        "接收大模型输出的工具调用指令（name + parameters），"
        "校验参数后执行推理，返回结果。"
    )
)
async def tool_call(request: ToolCallRequest):
    loop = asyncio.get_event_loop()

    # --- 工具：处理视频文件 ---
    if request.name == ToolName.PROCESS_VIDEO:
        try:
            inp = ProcessVideoInput.model_validate(request.parameters)
        except Exception as e:
            raise HTTPException(422, f"参数校验失败: {e}")

        if not os.path.exists(inp.video_path):
            raise HTTPException(400, f"Video file not found: {inp.video_path}")

        try:
            result = await loop.run_in_executor(
                _executor, _run_gvhmr, inp.video_path, inp.static_cam
            )
            return result
        except Exception as e:
            raise HTTPException(500, str(e))

    # --- 工具：启动摄像头流（通知客户端去建 WS 连接）---
    elif request.name == ToolName.START_STREAM:
        try:
            inp = StartStreamInput.model_validate(request.parameters)
        except Exception as e:
            raise HTTPException(422, f"参数校验失败: {e}")

        return {
            "status": "ready",
            "message": "请建立 WebSocket 连接到 /ws/stream 开始推流",
            "ws_endpoint": "/ws/stream",
            "stream_config": {
                "fps": inp.fps,
                "chunk_sec": inp.chunk_sec,
                "static_cam": inp.static_cam,
                "max_chunks": inp.max_chunks
            }
        }


# ============================================================
# 3. 原有文件路径接口（向后兼容）
# ============================================================
class VideoRequest(BaseModel):
    video_path: str
    static_cam: bool = False

@app.post("/generate", summary="[兼容] 输入视频文件路径，返回完整 DOF 序列")
async def generate_from_file(request: VideoRequest):
    if not os.path.exists(request.video_path):
        raise HTTPException(400, f"Video file not found: {request.video_path}")
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            _executor, _run_gvhmr, request.video_path, request.static_cam
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================
# 4. WebSocket 视频流接口
# ============================================================
@app.websocket("/ws/stream")
async def video_stream_ws(websocket: WebSocket):
    """
    WebSocket 视频流接口（推荐用于相机实时输入）

    === 客户端协议 ===
    步骤1  发送 JSON 配置（文本帧），字段同 StartStreamInput:
           {"fps": 30, "chunk_sec": 4, "static_cam": false, "max_chunks": 0}

    步骤2  循环发送视频帧（二进制帧）:
           每帧为 JPEG 编码后的原始字节
           cv2.imencode('.jpg', frame)[1].tobytes()

    步骤3  发送结束信号（文本帧）:
           "__END__"

    === 服务端响应 ===
    每积累 chunk_sec 秒的帧后触发推理，返回（文本帧）:
    {"status": "chunk", "chunk_id": 0, "dof": [[...], ...], "fps": 90,
     "total_frames": 360, "duration_sec": 4.0}

    流结束后:
    {"status": "done", "total_chunks": N}

    错误时:
    {"status": "error", "chunk_id": N, "detail": "..."}
    """
    await websocket.accept()

    # 接收并校验配置（复用 StartStreamInput）
    try:
        config_raw = await websocket.receive_text()
        config = StartStreamInput.model_validate(json.loads(config_raw))
    except Exception as e:
        await websocket.send_text(json.dumps({"status": "error", "detail": f"Invalid config: {e}"}))
        await websocket.close()
        return

    fps = config.fps
    frames_per_chunk = fps * config.chunk_sec
    static_cam = config.static_cam
    max_chunks = config.max_chunks  # 0 = 不限

    frame_buffer: list[np.ndarray] = []
    chunk_id = 0
    pending_task: asyncio.Task | None = None

    async def process_chunk(frames: list[np.ndarray], cid: int):
        try:
            result = await _process_frames_async(frames, fps, static_cam)
            await websocket.send_text(json.dumps({
                "status": "chunk",
                "chunk_id": cid,
                **result
            }))
        except Exception as e:
            await websocket.send_text(json.dumps({
                "status": "error",
                "chunk_id": cid,
                "detail": str(e)
            }))

    try:
        while True:
            # 达到 max_chunks 后主动结束
            if max_chunks > 0 and chunk_id >= max_chunks:
                break

            msg = await websocket.receive()
            if msg["type"] == "websocket.receive":
                if "text" in msg and msg["text"] == "__END__":
                    break
                if not msg.get("bytes"):
                    continue

            raw_bytes = msg.get("bytes")
            if not raw_bytes:
                continue

            img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frame_buffer.append(frame)

            if len(frame_buffer) >= frames_per_chunk:
                chunk_frames = frame_buffer[:frames_per_chunk]
                frame_buffer = frame_buffer[frames_per_chunk:]
                if pending_task is not None:
                    await pending_task
                pending_task = asyncio.create_task(process_chunk(chunk_frames, chunk_id))
                chunk_id += 1

        if pending_task is not None:
            await pending_task

        # 末尾剩余帧（≥ 1s 才处理）
        if len(frame_buffer) >= fps:
            await process_chunk(frame_buffer, chunk_id)
            chunk_id += 1

        await websocket.send_text(json.dumps({"status": "done", "total_chunks": chunk_id}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"status": "error", "detail": str(e)}))
        except Exception:
            pass


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GVHMR Robot Motion Server v2")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()

    print(f"服务启动，监听 {args.host}:{args.port}")
    print(f"  工具 Schema:  GET  http://{args.host}:{args.port}/tools")
    print(f"  工具调用:     POST http://{args.host}:{args.port}/tool/call")
    print(f"  文件接口:     POST http://{args.host}:{args.port}/generate")
    print(f"  视频流:       WS   ws://{args.host}:{args.port}/ws/stream")
    print(f"  API 文档:     http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)
