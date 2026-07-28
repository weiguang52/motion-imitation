import asyncio
import contextlib
import hashlib
import json
import os
import sys
import time
import tempfile
import argparse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from action_imitation_session import (
    ActionImitationSessionRegistry,
    register_session_request,
)
from run import GVHMRSystem

# ============================================================
# 全局配置
# ============================================================
CHUNK_DURATION_SEC = 4   # 最小处理单位（秒）
DEFAULT_FPS = 30         # 默认摄像头帧率

# 先解析本文件需要用到的参数；parse_known_args 不会影响 uvicorn / 其他参数。
_PRE_PARSER = argparse.ArgumentParser(add_help=False)
_PRE_PARSER.add_argument(
    "--raw-motion-only",
    action="store_true",
    help="只保存重定向前的 SMPL 关节动作帧 .npy，不运行 H1 IK",
)
_PRE_PARSER.add_argument(
    "--raw-motion-output-dir",
    default=os.getenv("RAW_MOTION_OUTPUT_DIR", "outputs/raw_motion_npy"),
    help="raw motion .npy 输出目录，文件名自动使用时间戳",
)
_PRE_PARSER.add_argument(
    "--raw-motion-coord",
    choices=["ik_input", "h1", "smpl", "v3"],
    default=os.getenv("RAW_MOTION_COORD", "ik_input"),
    help="raw motion 坐标系：ik_input/h1=推荐，保存为可视化 rot/H1PinkSolver 的输入并 pelvis 归零；smpl=原始 SMPL 不归零；v3=旧 V3/Z-up 调试格式",
)
_PRE_ARGS, _UNKNOWN_ARGS = _PRE_PARSER.parse_known_args()
RAW_MOTION_ONLY = _PRE_ARGS.raw_motion_only
RAW_MOTION_OUTPUT_DIR = _PRE_ARGS.raw_motion_output_dir
RAW_MOTION_COORD = _PRE_ARGS.raw_motion_coord

# ZMQ 视频流入口配置。默认关闭，避免影响原有 REST / WebSocket 行为。
def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

ZMQ_STREAM_ENABLED = _env_bool("ZMQ_STREAM_ENABLED", False)
ZMQ_STREAM_URL = os.getenv("ZMQ_STREAM_URL", "tcp://127.0.0.1:5558")
ZMQ_STREAM_TOPIC = os.getenv(
    "ZMQ_STREAM_TOPIC",
    "robot_robot_001_action_imitation_camera_left",
)
ZMQ_STREAM_FPS = int(os.getenv("ZMQ_STREAM_FPS", str(DEFAULT_FPS)))
ZMQ_STREAM_CHUNK_SEC = int(os.getenv("ZMQ_STREAM_CHUNK_SEC", str(CHUNK_DURATION_SEC)))
ZMQ_STREAM_STATIC_CAM = _env_bool("ZMQ_STREAM_STATIC_CAM", False)
ZMQ_STREAM_MAX_CHUNKS = int(os.getenv("ZMQ_STREAM_MAX_CHUNKS", "0"))  # 0 = 不限
# 一次模仿的视频被切成多个 chunk_sec 分块，每块单独出 npy。端侧一个 task_id 只等一个
# artifact，所以按「视频帧空闲」判定本次录制结束：连续 IDLE_FLUSH_SEC 秒没有新帧 ->
# 把本次累计的分块按时间轴拼成一个完整 npy，只 notify 一次。
ZMQ_STREAM_IDLE_FLUSH_SEC = float(os.getenv("ZMQ_STREAM_IDLE_FLUSH_SEC", "6.0"))

# 生成 npy 后回调 Agent 触发下传的配置。默认打开；URL 指向 Agent 的 localhost HTTP 入口。
AGENT_ACTION_RESULT_URL = os.getenv(
    "AGENT_ACTION_RESULT_URL", "http://127.0.0.1:8007/action_imitation/result"
).strip()
AGENT_ACTION_RESULT_TIMEOUT_SEC = float(
    os.getenv("AGENT_ACTION_RESULT_TIMEOUT_SEC", "5")
)
AGENT_ACTION_RESULT_RETRIES = max(
    1, int(os.getenv("AGENT_ACTION_RESULT_RETRIES", "3"))
)

action_imitation_sessions = ActionImitationSessionRegistry()


def _parse_robot_id_from_topic(topic: str) -> str:
    """从视频 topic 解析 robot_id。

    topic 形如 ``robot_{robot_id}_action_imitation_camera_{camera}``，
    例如 ``robot_robot_001_action_imitation_camera_left`` -> ``robot_001``。
    """
    s = str(topic or "")
    prefix = "robot_"
    marker = "_action_imitation_camera_"
    if s.startswith(prefix) and marker in s:
        return s[len(prefix):s.index(marker)]
    return ""


def _calculate_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _post_action_result_notify(
    robot_id: str, task_id: str, output_path: str, sha256: str
) -> bool:
    """Post an exact session identity and artifact metadata to Agent."""
    url = AGENT_ACTION_RESULT_URL
    if not url or not robot_id or not task_id or not output_path or not sha256:
        return False
    body = json.dumps({
        "robot_id": robot_id,
        "task_id": task_id,
        "output_path": output_path,
        "sha256": sha256,
    }).encode("utf-8")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    for attempt in range(1, AGENT_ACTION_RESULT_RETRIES + 1):
        if attempt == 2:
            time.sleep(0.1)
        elif attempt > 2:
            time.sleep(0.5)
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        started = time.monotonic()
        try:
            with opener.open(request, timeout=AGENT_ACTION_RESULT_TIMEOUT_SEC) as resp:
                response_body = resp.read().decode("utf-8", errors="replace")
                http_status = getattr(resp, "status", None)
            print(json.dumps({
                "source": "zmq_stream", "status": "notify_agent",
                "robot_id": robot_id, "task_id": task_id,
                "output_path": output_path, "sha256": sha256,
                "http_status": http_status, "response_body": response_body,
                "attempt": attempt,
                "latency_ms": round((time.monotonic() - started) * 1000),
            }, ensure_ascii=False))
            return True
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            print(json.dumps({
                "source": "zmq_stream", "status": "notify_agent_failed",
                "robot_id": robot_id, "task_id": task_id,
                "output_path": output_path, "sha256": sha256,
                "http_status": exc.code, "response_body": response_body,
                "attempt": attempt, "error_type": type(exc).__name__,
                "detail": str(exc),
            }, ensure_ascii=False))
            # Contract/conflict failures need operator attention, not blind retries.
            if 400 <= exc.code < 500:
                break
        except Exception as exc:
            print(json.dumps({
                "source": "zmq_stream", "status": "notify_agent_failed",
                "robot_id": robot_id, "task_id": task_id,
                "output_path": output_path, "sha256": sha256,
                "http_status": None, "response_body": None,
                "attempt": attempt, "error_type": type(exc).__name__,
                "detail": str(exc),
            }, ensure_ascii=False))
    return False


def _combine_and_notify_session(paths: list, robot_id: str, task_id: str) -> None:
    """一次录制结束时：把累计的分块 npy 按时间轴(axis=0)拼成一个完整 npy，只 notify 一次。

    同步函数，供 executor 调用。npy 为 [T, J, C] float32；沿 T 拼接得到完整动作序列。
    注意：各分块由 GVHMR 独立估计，拼接处可能有轻微不连续（首版可接受，后续可在
    GVHMR 侧对全段视频重估或做边界平滑）。
    """
    if not paths or not robot_id:
        return
    try:
        import numpy as _np
        arrays = [_np.load(p) for p in paths]
        combined = _np.concatenate(arrays, axis=0)
        ts_ms = int(time.time() * 1000)
        out_path = os.path.abspath(os.path.join(
            RAW_MOTION_OUTPUT_DIR, f"action_imitation_{ts_ms}_combined.npy"
        ))
        _np.save(out_path, combined)
        sha256 = _calculate_sha256(out_path)
        print(json.dumps({
            "source": "zmq_stream", "status": "session_combined",
            "robot_id": robot_id, "task_id": task_id,
            "chunks": len(paths), "combined_shape": list(combined.shape),
            "combined_path": out_path, "sha256": sha256,
        }, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({
            "source": "zmq_stream", "status": "session_combine_failed",
            "chunks": len(paths), "error_type": type(e).__name__, "detail": str(e),
        }, ensure_ascii=False))
        action_imitation_sessions.mark_state(robot_id, task_id, "processing_failed")
        return
    if not task_id:
        print(json.dumps({
            "source": "zmq_stream", "status": "notify_agent_skipped",
            "reason": "missing_task_id", "robot_id": robot_id,
            "output_path": out_path, "sha256": sha256,
        }, ensure_ascii=False))
        return
    action_imitation_sessions.record_result(
        robot_id,
        task_id,
        combined_path=out_path,
        sha256=sha256,
    )
    notified = _post_action_result_notify(robot_id, task_id, out_path, sha256)
    action_imitation_sessions.mark_state(
        robot_id, task_id, "completed" if notified else "notify_pending"
    )

# ============================================================
# 模型加载（全局单例，常驻显存）
# ============================================================
print("正在启动服务，请稍候（正在加载模型）...")
try:
    gvhmr_sys = GVHMRSystem(device='cuda', load_h1_optimizer=not RAW_MOTION_ONLY)
    print("模型加载完成，服务准备就绪！")
    if RAW_MOTION_ONLY:
        print(f"Raw motion only: enabled, output_dir={RAW_MOTION_OUTPUT_DIR}, coord={RAW_MOTION_COORD}")
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
    """输入一段视频文件；raw-motion-only 模式下输出重定向前动作帧 .npy，否则输出机器人 DOF。"""
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


class ActionImitationSessionStart(BaseModel):
    robot_id: str | None = ""
    task_id: str | None = ""
    source_turn_id: str | None = ""
    source_request_id: str | None = ""
    function_call_id: str | None = ""


# --- 生成给 LLM 的工具 Schema（对应 OpenAI tools 格式）---
def generate_tool_schemas() -> list[dict]:
    return [
        {
            "name": ToolName.PROCESS_VIDEO,
            "description": (
                "处理包含人体运动的视频文件。raw-motion-only 模式下，"
                "只运行 GVHMR 并保存重定向前的 SMPL 关节动作帧 .npy；"
                "普通模式下继续执行 H1 IK 并返回机器人关节角度序列（DOF）。"
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


def _run_raw_motion(
    video_path: str,
    static_cam: bool = False,
    output_dir: str | None = None,
    raw_motion_coord: str | None = None,
) -> dict:
    result = gvhmr_sys.process_video_raw_motion(
        video_path_str=video_path,
        output_dir=output_dir or RAW_MOTION_OUTPUT_DIR,
        static_cam=static_cam,
        verbose=False,
        raw_motion_coord=raw_motion_coord or RAW_MOTION_COORD,
    )
    if result is None:
        raise RuntimeError("GVHMR raw motion export returned None")
    return result


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


async def _process_frames_raw_async(frames: list[np.ndarray], fps: int, static_cam: bool) -> dict:
    loop = asyncio.get_event_loop()
    tmp_path = await loop.run_in_executor(_executor, _frames_to_tmp_video, frames, fps)
    try:
        result = await loop.run_in_executor(
            _executor,
            _run_raw_motion,
            tmp_path,
            static_cam,
            RAW_MOTION_OUTPUT_DIR,
            RAW_MOTION_COORD,
        )
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
# Agent -> GVHMR 动作模仿 session 登记
# ============================================================
@app.post("/action_imitation/session/start")
async def start_action_imitation_session(request: ActionImitationSessionStart):
    http_status, response, session = register_session_request(
        action_imitation_sessions,
        robot_id=request.robot_id,
        task_id=request.task_id,
        source_turn_id=request.source_turn_id,
        source_request_id=request.source_request_id,
        function_call_id=request.function_call_id,
    )
    if http_status != 200:
        return JSONResponse(status_code=http_status, content=response)

    assert session is not None
    print(json.dumps({
        "source": "action_imitation_session",
        "status": "session_registered",
        "robot_id": response["robot_id"],
        "task_id": response["task_id"],
        "state": response["state"],
        "session_state": session.state,
        "reason": response.get("reason", ""),
        "source_turn_id": session.source_turn_id,
        "source_request_id": session.source_request_id,
        "function_call_id": session.function_call_id,
    }, ensure_ascii=False))
    return response


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
            if RAW_MOTION_ONLY:
                result = await loop.run_in_executor(
                    _executor, _run_raw_motion, inp.video_path, inp.static_cam, RAW_MOTION_OUTPUT_DIR, RAW_MOTION_COORD
                )
                return result

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

@app.post("/generate", summary="[兼容] 输入视频文件路径；raw-motion-only 模式下保存 .npy，否则返回完整 DOF 序列")
async def generate_from_file(request: VideoRequest):
    if not os.path.exists(request.video_path):
        raise HTTPException(400, f"Video file not found: {request.video_path}")
    loop = asyncio.get_event_loop()
    try:
        if RAW_MOTION_ONLY:
            return await loop.run_in_executor(
                _executor, _run_raw_motion, request.video_path, request.static_cam, RAW_MOTION_OUTPUT_DIR, RAW_MOTION_COORD
            )
        return await loop.run_in_executor(
            _executor, _run_gvhmr, request.video_path, request.static_cam
        )
    except Exception as e:
        raise HTTPException(500, str(e))



# ============================================================
# 4. ZMQ SUB 视频流入口
# ============================================================
async def _handle_zmq_chunk(frames: list[np.ndarray], cid: int):
    """处理一个 ZMQ 视频块。这里不返回给客户端，只打印摘要；后续可在此处接机器人控制器。"""
    try:
        if RAW_MOTION_ONLY:
            result = await _process_frames_raw_async(
                frames,
                ZMQ_STREAM_FPS,
                ZMQ_STREAM_STATIC_CAM,
            )
            print(json.dumps({
                "source": "zmq_stream",
                "status": "raw_motion_saved",
                "chunk_id": cid,
                "npy_path": result.get("npy_path"),
                "motion_shape": result.get("motion_shape"),
                "fps": result.get("fps"),
                "total_frames": result.get("total_frames"),
                "format": result.get("format"),
                "coord": result.get("coord"),
            }, ensure_ascii=False))

            # 不再逐块 notify：返回本块 npy 路径，由 consumer 累计到本次录制会话；
            # 录制结束(idle-flush)时把整段拼成一个 npy 只 notify 一次。
            return result.get("npy_path")

        result = await _process_frames_async(
            frames,
            ZMQ_STREAM_FPS,
            ZMQ_STREAM_STATIC_CAM,
        )
        print(json.dumps({
            "source": "zmq_stream",
            "status": "chunk",
            "chunk_id": cid,
            "dof_shape": result.get("dof_shape"),
            "fps": result.get("fps"),
            "total_frames": result.get("total_frames"),
            "duration_sec": result.get("duration_sec"),
        }, ensure_ascii=False))

        # 如果后续要把结果发给真实规控/机器人控制器，就从这里接：
        # dof = result["dof"]
        # publish_to_robot_controller(dof)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({
            "source": "zmq_stream",
            "status": "error",
            "chunk_id": cid,
            "error_type": type(e).__name__,
            "detail": str(e),
        }, ensure_ascii=False))


async def zmq_video_stream_consumer():
    """
    从 Agent video_bridge 订阅 JPEG 视频帧。

    支持两种 multipart 格式：
      [topic, jpeg_bytes]
      [topic, timestamp, jpeg_bytes]
    """
    try:
        import zmq
        import zmq.asyncio
    except Exception as e:
        print(f"ZMQ 视频流入口启动失败：请先安装 pyzmq。detail={e}")
        return

    ctx = zmq.asyncio.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_STREAM_URL)
    sub.setsockopt_string(zmq.SUBSCRIBE, ZMQ_STREAM_TOPIC)

    frames_per_chunk = ZMQ_STREAM_FPS * ZMQ_STREAM_CHUNK_SEC
    frame_buffer: list[np.ndarray] = []
    chunk_id = 0
    pending_task: asyncio.Task | None = None
    session_npy_paths: list[str] = []  # 本次录制累计的分块 npy 路径
    recording_robot_id = ""

    async def _collect(task):
        """等一个 chunk 任务完成，把它产出的 npy 路径收进本次会话。"""
        if task is None:
            return
        try:
            npy = await task
        except Exception:
            npy = None
        if npy:
            session_npy_paths.append(str(npy))

    async def _do_flush():
        """录制结束：收尾 pending chunk -> 拼接本次所有分块 -> 只 notify 一次。"""
        nonlocal pending_task, recording_robot_id, frame_buffer
        if not recording_robot_id:
            return

        # Before yielding to a long-running inference task, detach an immutable
        # identity snapshot. A newly armed task cannot overwrite this callback.
        processing_session = action_imitation_sessions.begin_processing(
            recording_robot_id
        )
        flushed_robot_id = recording_robot_id
        recording_robot_id = ""
        task_to_collect = pending_task
        pending_task = None
        paths = list(session_npy_paths)
        session_npy_paths.clear()
        dropped_tail_frames = len(frame_buffer)
        frame_buffer.clear()
        await _collect(task_to_collect)
        if session_npy_paths:
            paths.extend(session_npy_paths)
            session_npy_paths.clear()

        if processing_session is None:
            print(json.dumps({
                "source": "zmq_stream", "status": "session_flush_skipped",
                "reason": "no_active_session", "robot_id": flushed_robot_id,
                "chunks": len(paths),
            }, ensure_ascii=False))
            return
        if not paths:
            print(json.dumps({
                "source": "zmq_stream", "status": "notify_agent_skipped",
                "reason": "no_session_artifacts",
                "robot_id": processing_session.robot_id,
                "task_id": processing_session.task_id,
                "frame_count": processing_session.frame_count,
                "dropped_tail_frames": dropped_tail_frames,
            }, ensure_ascii=False))
            action_imitation_sessions.mark_state(
                processing_session.robot_id,
                processing_session.task_id,
                "processing_failed",
            )
            return
        if dropped_tail_frames:
            print(json.dumps({
                "source": "zmq_stream", "status": "session_tail_dropped",
                "robot_id": processing_session.robot_id,
                "task_id": processing_session.task_id,
                "frames": dropped_tail_frames,
                "reason": "shorter_than_chunk",
            }, ensure_ascii=False))
        await asyncio.get_event_loop().run_in_executor(
            None,
            _combine_and_notify_session,
            paths,
            processing_session.robot_id,
            processing_session.task_id,
        )

    print(
        "ZMQ 视频流入口已启动："
        f"url={ZMQ_STREAM_URL}, topic={ZMQ_STREAM_TOPIC}, "
        f"fps={ZMQ_STREAM_FPS}, chunk_sec={ZMQ_STREAM_CHUNK_SEC}, "
        f"static_cam={ZMQ_STREAM_STATIC_CAM}, max_chunks={ZMQ_STREAM_MAX_CHUNKS}, "
        f"idle_flush_sec={ZMQ_STREAM_IDLE_FLUSH_SEC}"
    )

    try:
        while True:
            if ZMQ_STREAM_MAX_CHUNKS > 0 and chunk_id >= ZMQ_STREAM_MAX_CHUNKS:
                print(f"ZMQ 视频流入口达到 max_chunks={ZMQ_STREAM_MAX_CHUNKS}，停止接收。")
                break

            # 帧空闲 IDLE_FLUSH_SEC 秒 -> 认为本次录制结束 -> 合并+通知一次。
            try:
                parts = await asyncio.wait_for(
                    sub.recv_multipart(), timeout=ZMQ_STREAM_IDLE_FLUSH_SEC
                )
            except asyncio.TimeoutError:
                await _do_flush()
                continue

            if len(parts) == 2:
                topic, raw_bytes = parts
            elif len(parts) == 3:
                topic, _timestamp, raw_bytes = parts
            else:
                print(f"ZMQ 视频流入口忽略非法 multipart，parts={len(parts)}")
                continue

            topic_str = topic.decode("utf-8", errors="ignore") if isinstance(topic, bytes) else str(topic)
            if topic_str != ZMQ_STREAM_TOPIC:
                continue

            robot_id = _parse_robot_id_from_topic(topic_str)
            if recording_robot_id and recording_robot_id != robot_id:
                # Defensive for future wildcard subscriptions. The current
                # deployment subscribes one exact topic.
                await _do_flush()

            img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                print("ZMQ 视频流入口忽略无法解码的 JPEG 帧。")
                continue

            session = action_imitation_sessions.note_frame(robot_id)
            if session is None:
                print(json.dumps({
                    "source": "zmq_stream", "status": "frame_ignored",
                    "reason": "no_armed_session", "robot_id": robot_id,
                    "topic": topic_str,
                }, ensure_ascii=False))
                continue
            recording_robot_id = robot_id
            frame_buffer.append(frame)

            if len(frame_buffer) >= frames_per_chunk:
                chunk_frames = frame_buffer[:frames_per_chunk]
                frame_buffer = frame_buffer[frames_per_chunk:]

                # 保持和原 WebSocket 入口一致：同一时间只跑一个 GVHMR 任务。
                # 等上一块完成时顺便把它的 npy 收进本次会话。
                await _collect(pending_task)

                pending_task = asyncio.create_task(_handle_zmq_chunk(chunk_frames, chunk_id))
                chunk_id += 1

    except asyncio.CancelledError:
        print("ZMQ 视频流入口正在关闭。")
        # 关闭前把手上未 flush 的一段也合并下发，避免丢尾。
        with contextlib.suppress(Exception):
            await _do_flush()
        raise
    except Exception as e:
        print(f"ZMQ 视频流入口异常退出：{e}")
    finally:
        with contextlib.suppress(Exception):
            await _do_flush()
        sub.close(0)


@app.on_event("startup")
async def startup_zmq_stream():
    if not ZMQ_STREAM_ENABLED:
        print("ZMQ 视频流入口未启用。设置 ZMQ_STREAM_ENABLED=1 可开启。")
        return
    app.state.zmq_stream_task = asyncio.create_task(zmq_video_stream_consumer())


@app.on_event("shutdown")
async def shutdown_zmq_stream():
    task = getattr(app.state, "zmq_stream_task", None)
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ============================================================
# 5. WebSocket 视频流接口
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
            if RAW_MOTION_ONLY:
                result = await _process_frames_raw_async(frames, fps, static_cam)
                await websocket.send_text(json.dumps({
                    "status": "raw_motion_saved",
                    "chunk_id": cid,
                    **result,
                }))
                return

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
    parser.add_argument("--raw-motion-only", action="store_true", default=RAW_MOTION_ONLY, help="只保存重定向前的 SMPL 关节动作帧 .npy，不运行 H1 IK")
    parser.add_argument("--raw-motion-output-dir", default=RAW_MOTION_OUTPUT_DIR, help="raw motion .npy 输出目录，文件名自动使用时间戳")
    parser.add_argument("--raw-motion-coord", choices=["ik_input", "h1", "smpl", "v3"], default=RAW_MOTION_COORD, help="raw motion 坐标系：ik_input/h1=推荐，保存为可视化 rot/H1PinkSolver 的输入并 pelvis 归零；smpl=原始 SMPL 不归零；v3=旧 V3/Z-up 调试格式")
    args = parser.parse_args()

    print(f"服务启动，监听 {args.host}:{args.port}")
    print(f"  工具 Schema:  GET  http://{args.host}:{args.port}/tools")
    print(f"  工具调用:     POST http://{args.host}:{args.port}/tool/call")
    print(f"  文件接口:     POST http://{args.host}:{args.port}/generate")
    print(f"  Session登记:  POST http://{args.host}:{args.port}/action_imitation/session/start")
    print(f"  视频流:       WS   ws://{args.host}:{args.port}/ws/stream")
    print(f"  ZMQ视频流:    {'已启用' if ZMQ_STREAM_ENABLED else '未启用'} {ZMQ_STREAM_URL} topic={ZMQ_STREAM_TOPIC}")
    print(f"  Raw Motion:   {'已启用' if RAW_MOTION_ONLY else '未启用'} output_dir={RAW_MOTION_OUTPUT_DIR} coord={RAW_MOTION_COORD}")
    print(f"  API 文档:     http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)
