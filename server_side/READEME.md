cd /home/wd/Agent_memory_patch/IK_Retargeting

conda activate IKRetargeting
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/wd/Agent_memory_patch/IK_Retargeting/GVHMR:$PYTHONPATH

ZMQ_STREAM_ENABLED=1 python server_side/action_imitation_server.py \
  --raw-motion-only \
  --raw-motion-output-dir /home/wd/Agent_memory_patch/IK_Retargeting/raw_motion_npy \
  --raw-motion-coord ik_input

启动服务有点慢，大概得等几十秒直到输出为：ZMQ 视频流入口已启动。说明动作模仿节点成功启动。


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