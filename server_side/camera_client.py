"""
camera_client.py - 读取真实摄像头，推流到 new-server.py 的 /ws/stream 接口

用法：
  # 使用默认摄像头（index=0）
  python camera_client.py

  # 指定摄像头和参数
  python camera_client.py --camera 0 --fps 30 --chunk_sec 4

  # 指定服务器地址
  python camera_client.py --server 192.168.1.100:8003
"""

import argparse
import asyncio
import json
import signal
import sys

import cv2
import websockets

# 用于优雅退出
_stop = False

def _on_signal(sig, frame):
    global _stop
    print("\n收到退出信号，正在停止...")
    _stop = True


async def run_camera_stream(camera_index: int, fps: int, chunk_sec: int,
                             server: str, static_cam: bool):
    url = f"ws://{server}/ws/stream"
    print(f"连接服务器 {url} ...")

    async with websockets.connect(url, max_size=10 * 1024 * 1024) as ws:
        # 发送配置
        config = {
            "fps": fps,
            "chunk_sec": chunk_sec,
            "static_cam": static_cam,
            "max_chunks": 0  # 持续运行
        }
        await ws.send(json.dumps(config))
        print(f"配置已发送: {config}")

        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 index={camera_index}")
            return

        # 设置摄像头帧率和分辨率
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"摄像头已打开：index={camera_index}, 实际帧率={actual_fps}")
        print(f"每 {chunk_sec}s 触发一次推理，按 Ctrl+C 停止\n")

        sent = 0
        try:
            while not _stop:
                ret, frame = cap.read()
                if not ret:
                    print("警告：读取摄像头帧失败，跳过")
                    await asyncio.sleep(0.01)
                    continue

                # 编码为 JPEG 字节推送
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                await ws.send(buf.tobytes())
                sent += 1

                if sent % fps == 0:
                    print(f"已推送 {sent} 帧（{sent // fps}s）...")

                # 非阻塞检查是否有推理结果返回
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.001))
                    _handle_result(msg)
                except asyncio.TimeoutError:
                    pass

                # 控制帧率（避免推太快撑爆缓冲）
                await asyncio.sleep(1.0 / fps)

        finally:
            cap.release()
            print(f"\n摄像头已关闭，共推送 {sent} 帧")

            # 发结束信号，等最后结果
            try:
                await ws.send("__END__")
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                    _handle_result(msg)
                    if msg["status"] in ("done", "error"):
                        break
            except Exception:
                pass


def _handle_result(msg: dict):
    status = msg.get("status")
    if status == "chunk":
        dof = msg.get("dof", [])
        print(f"[推理结果] chunk_id={msg['chunk_id']} | "
              f"帧数={msg.get('total_frames')} | "
              f"时长={msg.get('duration_sec')}s | "
              f"fps={msg.get('fps')} | "
              f"DOF shape={msg.get('dof_shape')}")
        # 这里可以把 dof 发给机器人控制器
        # 例如：publish_to_ros(dof)
    elif status == "done":
        print(f"[完成] 共处理 {msg.get('total_chunks')} 个 chunk")
    elif status == "error":
        print(f"[错误] {msg.get('detail')}")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _on_signal)

    parser = argparse.ArgumentParser(description="摄像头推流客户端")
    parser.add_argument("--camera",     type=int,   default=0,              help="摄像头编号（默认0）")
    parser.add_argument("--fps",        type=int,   default=30,             help="帧率（默认30）")
    parser.add_argument("--chunk_sec",  type=int,   default=4,              help="每个chunk秒数（默认4）")
    parser.add_argument("--server",     default="localhost:8003",           help="服务地址（默认localhost:8003）")
    parser.add_argument("--static_cam", action="store_true",                help="相机固定不动时加此参数")
    args = parser.parse_args()

    asyncio.run(run_camera_stream(
        camera_index=args.camera,
        fps=args.fps,
        chunk_sec=args.chunk_sec,
        server=args.server,
        static_cam=args.static_cam
    ))
