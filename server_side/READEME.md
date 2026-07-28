cd /home/wd/Agent_memory_patch/IK_Retargeting

conda activate IKRetargeting
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/wd/Agent_memory_patch/IK_Retargeting/GVHMR:$PYTHONPATH

ZMQ_STREAM_ENABLED=1 python server_side/action_imitation_server.py \
  --raw-motion-only \
  --raw-motion-output-dir /home/wd/Agent_memory_patch/IK_Retargeting/raw_motion_npy \
  --raw-motion-coord ik_input

启动服务有点慢，大概得等几十秒直到输出为：ZMQ 视频流入口已启动。说明动作模仿节点成功启动。

视频推流前，Agent 需要用端侧 accepted 结果中的 task_id 登记 session：

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

成功后返回 `state=armed`。相同 `robot_id + task_id` 可安全重试；同一机器人的
另一个 active task 返回 HTTP 409。没有 armed session 的视频帧不会进入动作处理。


另外两个模拟节点：

cd /home/wd/Agent_memory_patch

IMITATION_STATE_PATH=./data/imitation_session_state.json
VIDEO_BRIDGE_PUBLIC_HOST=127.0.0.1
VIDEO_BRIDGE_ZMQ_IN_PORT=5557
VIDEO_BRIDGE_ZMQ_OUT_PORT=5558
VIDEO_BRIDGE_OUTPUT_ROOT=./data/action_imitation

python3 -m robot_ai.capabilities.imitation.video_bridge \
  --in-port 5557 \
  --out-port 5558 \
  --output-root ./data/action_imitation

python3 scripts/action_imitation/mock_edge_video_publisher.py \
  --url tcp://127.0.0.1:5557 \
  --topic robot_robot_001_action_imitation_camera_left \
  --video /home/wd/Agent_memory_patch/IK_Retargeting/GVHMR/docs/example_video/test0712.mp4 \
  --fps 30 \
  --seconds 4 \
  --repeat 1
