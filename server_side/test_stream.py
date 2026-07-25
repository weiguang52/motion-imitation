"""
test_stream.py - 用本地视频文件模拟摄像头推流，测试 /ws/stream 接口

用法：
  python test_stream.py --video /data/demo.mp4
  python test_stream.py --video /data/demo.mp4 --fps 30 --chunk_sec 4
"""

import argparse
import asyncio
import json

import cv2
import websockets


async def test_stream(video_path: str, fps: int, chunk_sec: int, server: str):
    url = f"ws://{server}/ws/stream"
    print(f"连接到 {url} ...")

    async with websockets.connect(url) as ws:
        # 发送配置
        config = {"fps": fps, "chunk_sec": chunk_sec, "static_cam": False, "max_chunks": 0}
        await ws.send(json.dumps(config))
        print(f"配置已发送: {config}")

        # 用视频文件模拟推流
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return

        sent = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, buf = cv2.imencode('.jpg', frame)
            await ws.send(buf.tobytes())
            sent += 1
            if sent % fps == 0:
                print(f"已发送 {sent} 帧（{sent // fps}s）...")

            # 非阻塞检查是否有结果返回
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.001))
                _print_result(msg)
                if msg["status"] in ("done", "error"):
                    return
            except asyncio.TimeoutError:
                pass

        cap.release()
        print(f"视频发送完毕，共 {sent} 帧，等待最终结果...")
        await ws.send("__END__")

        # 接收剩余结果
        while True:
            msg = json.loads(await ws.recv())
            _print_result(msg)
            if msg["status"] in ("done", "error"):
                break


def _print_result(msg: dict):
    status = msg.get("status")
    if status == "chunk":
        print(f"[chunk {msg['chunk_id']}] dof_shape={msg.get('dof_shape')}, "
              f"fps={msg.get('fps')}, duration={msg.get('duration_sec')}s")
    elif status == "done":
        print(f"[done] 共处理 {msg.get('total_chunks')} 个 chunk")
    elif status == "error":
        print(f"[error] chunk_id={msg.get('chunk_id')}, detail={msg.get('detail')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True,       help="视频文件路径")
    parser.add_argument("--fps",        type=int, default=30, help="帧率（默认30）")
    parser.add_argument("--chunk_sec",  type=int, default=4,  help="每个chunk秒数（默认4）")
    parser.add_argument("--server",     default="localhost:8003", help="服务地址（默认localhost:8003）")
    args = parser.parse_args()

    asyncio.run(test_stream(args.video, args.fps, args.chunk_sec, args.server))
