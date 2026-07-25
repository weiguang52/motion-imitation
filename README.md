# Motion Imitation

Video-to-robot motion imitation based on GVHMR, SMPL/SMPL-X and inverse
kinematics retargeting.

## Repository layout

- `server_side/`: HTTP, WebSocket and ZMQ video-stream services.
- `GVHMR/`: video human-motion recovery and multi-person tracking.
- `ik_redirection_npy.py`: inverse-kinematics retargeting.
- `data/`: robot meshes, URDF files and IK input samples.
- `inputs/test_mp4/`: local video samples, including a multi-person test.

The tracker selects one primary person for each video chunk using the person's
distance from the image centre, boundary completeness, track continuity and
visible size. This prevents a larger person near the edge from replacing the
central, fully visible motion demonstrator.

## Model files

Pretrained checkpoints and licensed SMPL assets are intentionally not stored
in Git because several files are larger than GitHub's 100 MB file limit.
Follow [`GVHMR/docs/INSTALL.md`](GVHMR/docs/INSTALL.md) to download the GVHMR,
HMR2, ViTPose, YOLO and body-model assets.

The server startup commands are documented in
[`server_side/READEME.md`](server_side/READEME.md).

Clone with submodules so that DPVO is available:

```bash
git clone --recursive https://github.com/weiguang52/motion-imitation.git
```
