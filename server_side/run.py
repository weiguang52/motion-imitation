import cv2
import torch
import pytorch_lightning as pl
import numpy as np
# Legacy SMPL pickle files use chumpy, which imports removed NumPy scalar aliases.
for _alias, _scalar in {"bool": bool, "int": int, "float": float, "complex": complex,
                        "object": object, "unicode": str, "str": str}.items():
    np.__dict__.setdefault(_alias, _scalar)
import argparse
import sys
import os
import time
import threading
from pathlib import Path
from tqdm import tqdm
import joblib
from torch.cuda.amp import autocast
# Hydra / Config
import hydra
from hydra import initialize_config_module, compose
from hydra.core.global_hydra import GlobalHydra
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Pytorch3D
from pytorch3d.transforms import quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
import smplx
try:
    from .raw_motion_export import (resample_values, smpl_global_rotations,
                                    resample_rotations, estimate_hand_openness, pack_extended, align_tw_joint_positions, align_tw_rotation_matrices)
except ImportError:
    from raw_motion_export import (resample_values, smpl_global_rotations,
                                   resample_rotations, estimate_hand_openness, pack_extended, align_tw_joint_positions, align_tw_rotation_matrices)
# GVHMR Imports
from hmr4d.utils.pylogger import Log
from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
from hmr4d.utils.preproc.vitfeat_extractor import get_batch
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel, apply_T_on_points, compute_T_ayfz2ay
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from einops import einsum
CURRENT_DIR = Path(__file__).parent.absolute()
try:
    from pathlib import Path
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = CURRENT_DIR.parent

    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    from ik_redirection_npy import H1PinkSolver

    Log.info(f" Successfully imported H1PinkSolver from {ROOT_DIR / 'ik_redirection_npy.py'}")
except ImportError as e:
    Log.error(f" Could not import H1Optimizer. Check path! Error: {e}")
    sys.exit(1)

CRF = 23

class GVHMRSystem:
    def __init__(self, device='cuda', load_h1_optimizer=True):
        """
        [冷启动阶段]
        只执行一次：加载模型权重、初始化预处理工具、准备 Hydra 环境。
        """
        Log.info("========================================")
        Log.info(" [Cold Start] Initializing GVHMR System...")
        Log.info("========================================")
        self.device = device
        
        # 1. 初始化 Hydra (Global)
        GlobalHydra.instance().clear()
        register_store_gvhmr()
        # 这里我们要初始化 config module，后续 compose 时会用到
        # 注意：version_base 根据你的 hydra 版本可能需要调整，原代码是 1.3
        self.hydra_context = initialize_config_module(version_base="1.3", config_module="hmr4d.configs")
        self.hydra_context.__enter__() # 手动进入上下文

        # 2. 预加载基础配置 (用于加载模型架构参数)
        # 我们先加载一个 dummy 配置来获取 checkpoint 路径和模型结构
        # 具体的 output_root 等参数可以在处理视频时覆盖
        base_cfg = compose(config_name="demo", overrides=["video_name=init", "static_cam=True"])
        self.ckpt_path = base_cfg.ckpt_path
        self.model_cfg = base_cfg.model
        
        # 3. 加载 GVHMR 主模型
        Log.info(">>> Loading GVHMR Model...")
        self.model: DemoPL = hydra.utils.instantiate(self.model_cfg, _recursive_=False)
        self.model.load_pretrained_model(self.ckpt_path)
        self.model = self.model.eval().to(self.device)

        # 4. 加载预处理工具 (最耗时的部分之一，现在只需加载一次)
        Log.info(">>> Loading Preprocessing Tools (Tracker, VitPose, Extractor)...")
        self.tracker = Tracker()
        self.vitpose_extractor = VitPoseExtractor()
        self.feature_extractor = Extractor()
        self.feature_extractor.extractor = self.feature_extractor.extractor.float().eval()
        if hasattr(torch, 'compile'):
            Log.info(">>> Compiling GVHMR model for faster inference...")
            self.model = torch.compile(self.model)

        # 缓存渲染器用的 SMPL 模型 (避免重复加载)
        self.smplx = make_smplx("supermotion").to(self.device)
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").to(self.device)
        self.faces_smpl = make_smplx("smpl").faces
        self.J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").to(self.device)
        
        Log.info(" [System Ready] Waiting for inputs...")
        Log.info("========================================")

        self.h1_pinksolver = None
        if load_h1_optimizer:
            urdf_path = Path("data/Assembly/Assembly.SLDASM.urdf")
            if not urdf_path.exists():
                Log.error(f" URDF not found at {urdf_path}")
            else:
                Log.info(f" Loading H1 Optimizer with URDF: {urdf_path}")
                self.h1_pinksolver = H1PinkSolver(str(urdf_path))
        else:
            Log.info(" H1 Optimizer loading skipped (raw motion output mode).")

        self.smpl_model = smplx.create(
            model_path='data',
            model_type='smpl',
            gender='neutral',
            use_hands=False,
            use_face_contour=False,
            use_feet_keypoints=False,
            batch_size=1
        ).to(self.device)
        Log.info(" [System Ready] Waiting for inputs...")

    def __del__(self):
        # 清理 Hydra 上下文
        if hasattr(self, 'hydra_context'):
            self.hydra_context.__exit__(None, None, None)

    def process_video(self, video_path_str, output_root=None, static_cam=False, f_mm=None, verbose=False):
        """
        [热运行阶段]
        处理单个视频：生成特定配置、预处理、推理、导出。
        """
        pipeline_start = time.time()
        video_path = Path(video_path_str)
        if not video_path.exists():
            Log.error(f"Video not found: {video_path}")
            return

        video_name = video_path.stem
        Log.info(f"\n>>> Processing: {video_name}")

        # 1. 动态生成当前视频的 Config
        overrides = [
            f"video_name={video_name}",
            f"static_cam={static_cam}",
            f"verbose={verbose}",
            f"use_dpvo={False}", # 默认不开 DPVO，按需调整
        ]
        if f_mm is not None:
            overrides.append(f"f_mm={f_mm}")
        if output_root is not None:
            overrides.append(f"output_root={output_root}")
        
        # 使用之前 init 好的环境 compose 新配置
        cfg = compose(config_name="demo", overrides=overrides)
        cfg.video_path = str(video_path.resolve())
       
        # 2. 运行预处理 (使用 self.xxx 复用工具)
        t0 = time.time()
        cache = self._run_preprocess_efficient(cfg)
        Log.info(f"[TIMING] Preprocess total: {time.time()-t0:.3f}s")

        # 3. 准备数据字典
        data = self._load_data_dict(cfg, cache)

        # --- B. Inference ---
        t0 = time.time()
        with torch.no_grad():
            data_gpu = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            pred = self.model.predict(data_gpu, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
        Log.info(f"[TIMING] GVHMR infer:  {time.time()-t0:.3f}s")

        # --- C. Export & Run H1 Optimization ---
        t0 = time.time()
        final_result = self._run_h1_optimization(pred, cfg)
        Log.info(f"[TIMING] H1 IK:        {time.time()-t0:.3f}s")

        pipeline_end = time.time()
        duration = pipeline_end - pipeline_start
        Log.info(f">>> Finished: {video_name} |  Total Time: {duration:.4f}s")
        return final_result

    def _make_smpl_joints_from_pose(self, global_orient, body_pose):
        """
        根据 SMPL axis-angle pose 直接生成关节坐标。
        不在这里做坐标系旋转；调用方决定是否对 global_orient 做 V3 修正。
        """
        body_pose_np = body_pose[:, :63].detach().cpu().numpy().astype(np.float64)
        global_orient_np = global_orient.detach().cpu().numpy().astype(np.float64)

        if body_pose_np.ndim == 3:
            body_pose_np = body_pose_np.reshape(body_pose_np.shape[0], -1)

        poses_raw_np = np.concatenate([global_orient_np, body_pose_np], axis=1)
        T = poses_raw_np.shape[0]
        zeros_padding = np.zeros((T, 6), dtype=np.float64)
        pose_aa_final = np.concatenate([poses_raw_np, zeros_padding], axis=-1)

        with torch.no_grad():
            output = self.smpl_model(
                body_pose=torch.from_numpy(pose_aa_final[:, 3:72]).float().to(self.device),
                global_orient=torch.from_numpy(pose_aa_final[:, :3]).float().to(self.device),
            )

        return output.joints[:, :29, :].detach().cpu().numpy().astype(np.float32)

    def _gvhmr_to_v3_global_orient(self, global_orient, device):
        """
        保留原 H1 IK 路径里的 V3 修正：
        GVHMR/Y-up -> Mujoco-like/Z-up + flip。
        只给 retargeting 旧路径使用；raw npy 默认不要用它。
        """
        base_rot = axis_angle_to_matrix(
            torch.tensor([1.0, 0.0, 0.0], device=device) * (-torch.pi / 2.0)
        ).unsqueeze(0)
        flip_rot = axis_angle_to_matrix(
            torch.tensor([1.0, 0.0, 0.0], device=device) * torch.pi
        ).unsqueeze(0)
        C = torch.matmul(flip_rot, base_rot)

        R_gvhmr = axis_angle_to_matrix(global_orient)
        R_mujoco = torch.matmul(C, R_gvhmr)
        return matrix_to_axis_angle(R_mujoco)

    def _extract_pre_retarget_joints(self, pred, coord="ik_input"):
        """
        从 GVHMR 预测结果中提取“重定向前”的 SMPL 人体关节动作帧。

        返回:
            joints: [T, 29, 3]，float32，未经过 H1 IK。

        coord 约定：
            "ik_input" / "h1":
                给动作模仿/IK 输入使用。保存 GVHMR/SMPL 原始轴系下的 joints，
                只做 pelvis 归零。下游可视化或 H1PinkSolver._preprocess 里继续执行：
                    rot = R.from_quat([0.5, 0.5, 0.5, 0.5]).as_matrix()
                    seq = seq @ rot.T
                也就是说：这个 .npy 是“等待 rot 的输入”，不是已经 rot 之后的 Z-up 数据。

            "smpl":
                调试用。保存 GVHMR/SMPL 原始轴系下的 joints，不做 pelvis 归零。

            "v3":
                调试用。使用旧 V3 global_orient 修正后再生成 joints，并做 pelvis 归零。
                注意：如果你的可视化代码还会执行 quat rot，就不要用这个模式。

            "retarget":
                只给原 H1 IK 路径内部使用。等价于旧代码传给 H1PinkSolver.process_motion
                的 joints，不做 pelvis 归零。
        """
        smpl_params = pred['smpl_params_global']
        global_orient = smpl_params['global_orient']
        body_pose = smpl_params['body_pose']
        transl = smpl_params['transl']
        device = transl.device

        # 默认 raw 导出：不做 V3 修正，保留为 H1PinkSolver / 可视化 rot 的输入。
        if coord in ("ik_input", "h1", "smpl"):
            joints = self._make_smpl_joints_from_pose(global_orient, body_pose)

            if coord == "smpl":
                return joints.astype(np.float32)

            joints = joints - joints[:, 0:1, :]
            return joints.astype(np.float32)

        # 旧调试路径：先做 V3，再生成 joints。
        if coord in ("v3", "retarget"):
            global_orient_v3 = self._gvhmr_to_v3_global_orient(global_orient, device)
            joints = self._make_smpl_joints_from_pose(global_orient_v3, body_pose)

            if coord == "retarget":
                return joints.astype(np.float32)

            joints = joints - joints[:, 0:1, :]
            return joints.astype(np.float32)

        raise ValueError(
            f"Unsupported raw motion coord: {coord}. "
            "Use 'ik_input', 'h1', 'smpl', 'v3', or 'retarget'."
        )

    def process_video_raw_motion(
        self,
        video_path_str,
        output_dir,
        static_cam=False,
        f_mm=None,
        verbose=False,
        raw_motion_coord="tw",
        raw_motion_target_fps=20.0,
        raw_motion_extended=False,
    ):
        """
        只运行到 GVHMR -> SMPL joints，不做 H1 IK 重定向。
        将重定向前的人体动作帧保存为 action_imitation_13位毫秒时间戳.npy。
        """
        pipeline_start = time.time()
        video_path = Path(video_path_str)
        if not video_path.exists():
            Log.error(f"Video not found: {video_path}")
            return None

        video_name = video_path.stem
        Log.info(f"\n>>> Processing raw motion: {video_name}")

        overrides = [
            f"video_name={video_name}",
            f"static_cam={static_cam}",
            f"verbose={verbose}",
            f"use_dpvo={False}",
        ]
        if f_mm is not None:
            overrides.append(f"f_mm={f_mm}")

        cfg = compose(config_name="demo", overrides=overrides)
        cfg.video_path = str(video_path.resolve())

        t0 = time.time()
        cache = self._run_preprocess_efficient(cfg)
        Log.info(f"[TIMING] Preprocess total: {time.time()-t0:.3f}s")

        data = self._load_data_dict(cfg, cache)

        t0 = time.time()
        with torch.no_grad():
            data_gpu = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            pred = self.model.predict(data_gpu, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
        Log.info(f"[TIMING] GVHMR infer:  {time.time()-t0:.3f}s")

        joints = self._extract_pre_retarget_joints(pred, coord="ik_input" if raw_motion_coord == "tw" else raw_motion_coord)
        capture = cv2.VideoCapture(str(video_path))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        target_fps = float(raw_motion_target_fps or source_fps)
        if not np.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("raw_motion_target_fps must be positive")
        if raw_motion_extended:
            params = pred["smpl_params_global"]
            global_orient = params["global_orient"]
            if raw_motion_coord == "v3":
                global_orient = self._gvhmr_to_v3_global_orient(global_orient, global_orient.device)
            rotations = smpl_global_rotations(
                global_orient.detach().cpu().numpy(),
                params["body_pose"].detach().cpu().numpy(),
                self.smpl_model.parents.detach().cpu().numpy(),
            )
            rotations = resample_rotations(rotations, source_fps, target_fps)
            if raw_motion_coord == "tw":
                rotations = align_tw_rotation_matrices(rotations)
            hands = estimate_hand_openness(str(video_path), len(joints), source_fps, target_fps)
        joints = resample_values(joints, source_fps, target_fps)
        if raw_motion_coord == "tw":
            joints = align_tw_joint_positions(joints)
        if len(joints) < 9:
            raise ValueError("tw_retargeting requires at least 9 frames at the export frame rate")
        if raw_motion_extended:
            joints = pack_extended(joints, rotations, hands)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp_ms = int(time.time() * 1000)
        output_path = output_dir / f"action_imitation_{timestamp_ms}.npy"
        np.save(output_path, joints)

        duration = time.time() - pipeline_start
        Log.info(
            f"[Raw Motion] Saved: {output_path} | shape={list(joints.shape)} | "
            f"Total Time: {duration:.4f}s"
        )
        return {
            "status": "success",
            "npy_path": str(output_path),
            "filename": output_path.name,
            "timestamp_ms": timestamp_ms,
            "motion_shape": list(joints.shape),
            "fps": target_fps,
            "source_fps": source_fps,
            "extended": raw_motion_extended,
            "total_frames": int(joints.shape[0]),
            "format": f"smpl_joints_pre_retargeting_float32_T_{joints.shape[1]}_3_coord_{raw_motion_coord}",
            "coord": raw_motion_coord,
        }

    def _run_h1_optimization(self, pred, cfg):
        """
        整合了数据格式化(V3修正)和梯度下降调用
        """
        if self.h1_pinksolver is None:
            raise RuntimeError("H1 optimizer is not loaded. Start without raw-motion-only mode to use retargeting.")

        Log.info("   [H1] Formatting data and running optimization...")
        # H1PinkSolver.process_motion 内部会自行做 SMPL->H1 坐标变换；
        # retargeting 路径保留旧 V3 输入，避免改变原来的机器人重定向行为。
        joints = self._extract_pre_retarget_joints(pred, coord="retarget")
        # result = {
        #     "dof": np.zeros((82,28)),   
        #     "fps": 30,
        #     "input_frames": 82,
        #     "output_frames": 82,
        #     "root_trans_offset": np.zeros((82, 3)),
        #     "root_rot": np.tile(np.array([0, 0, 0, 1]), (82, 1)),
        #     "smpl_joints_target": joints,  # [N, J, 3] 所有关节三维坐标
        #     "h1_joint_pos": np.zeros((82, 1, 3)),  # 占位
        # }
        # print("第0帧 pelvis:", joints[0, 0])
        # print("第0帧 left_hip:", joints[0, 1])
        # print("第0帧 right_hip:", joints[0, 2])
        # print("第0帧 head:", joints[0, 15])
        # print("第0帧 left_ankle:", joints[0, 7])
        # joblib.dump(result, '/home/cxm/GVHMR/outputs/4s/result.pkl')
        # 7. 执行梯度下降
        try:
            Log.info("   [H1] Starting Gradient Descent...")
            optimized_result = self.h1_pinksolver.process_motion(joints)
            return optimized_result
        except Exception as e:
            Log.error(f" [H1] Optimization Failed: {e}")
            import traceback
            traceback.print_exc()
            return None # 失败返回 None

    def _prepare_video_file(self, source_path, cfg):
        if not Path(cfg.video_path).exists() or get_video_lwh(source_path)[0] != get_video_lwh(cfg.video_path)[0]:
            reader = get_video_reader(source_path)
            writer = get_writer(cfg.video_path, fps=30, crf=CRF)
            for img in tqdm(reader, total=get_video_lwh(source_path)[0], desc="Copy Video"):
                writer.write_frame(img)
            writer.close()
            reader.close()

    def _run_preprocess_efficient(self, cfg):
        """
        [优化版] 
        1. SLAM 继续后台 CPU 运行
        2. Tracker 串行 (它是后续的基础)
        3. VitPose 与 Features 任务并行 (利用 GPU 任务间隙)
        """
        cache = {
            "slam_traj": None,
            "vitpose": None,
            "vit_features": None
        }
        
        paths = cfg.paths
        video_path = cfg.video_path
        
        # 1. 启动 SLAM 线程 (CPU 密集型)
        def run_slam_task():
            if not cfg.static_cam:
                Log.info("[Preproc] Running SLAM (Async in Background)...")
                try:
                    simple_vo = SimpleVO(cfg.video_path, scale=0.25, step=16, method="sift", f_mm=cfg.f_mm)
                    vo_results = simple_vo.compute()
                    cache['slam'] = vo_results
                    Log.info("[Preproc]  SLAM Finished (Background).")
                except Exception as e:
                    Log.error(f"SLAM Failed: {e}")

        slam_thread = threading.Thread(target=run_slam_task)
        slam_thread.start()

        # 2. 视频先缩放到半分辨率；get_batch 接收的 bbox 也必须使用半分辨率坐标
        image_scale = 0.5
        t0 = time.time()
        imgs_half = read_video_np(video_path, scale=image_scale)  # (F, H/2, W/2, 3) RGB
        Log.info(f"[TIMING] Video read:   {time.time()-t0:.3f}s")

        # 3. Tracker 接收的是原视频路径，因此返回的是原始分辨率下的 xyxy 坐标
        #    注意：当前 Tracker 仍会自行读取视频，并没有直接复用 imgs_half。
        t0 = time.time()
        bbx_xyxy = self.tracker.get_one_track(video_path).float()

        # get_batch 处理的是 imgs_half，所以先把检测框换算到半分辨率坐标系
        bbx_xyxy_half = bbx_xyxy * image_scale
        bbx_xys_half = get_bbx_xys_from_xyxy(
            bbx_xyxy_half,
            base_enlarge=1.2,
        ).float()
        Log.info(f"[TIMING] Tracker:      {time.time()-t0:.3f}s")

        # 4. imgs_half 已经完成 0.5 缩放，因此 path_type='np' 时 img_ds 保持 1.0，
        #    避免 get_batch 再次缩放。此时输入/输出 bbox 均处于半分辨率坐标系。
        t0 = time.time()
        imgs_shared, bbx_xys_half_processed = get_batch(
            imgs_half,
            bbx_xys_half,
            img_ds=1.0,
            path_type="np",
        )

        # GVHMR 后续的 K_fullimg 使用原始图像尺寸，ViTPose 也需要把关键点映射回
        # 原始图像坐标，因此将 get_batch 输出的 bbox 恢复到原始分辨率坐标系。
        bbx_xys = bbx_xys_half_processed / image_scale
        cache['bbx'] = {"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}
        Log.info(f"[TIMING] get_batch:    {time.time()-t0:.3f}s")

        # 5. VitPose 与 Features 串行 GPU 推理
        t0 = time.time()
        cache['vitpose'] = self.vitpose_extractor.extract(imgs_shared, bbx_xys)
        Log.info(f"[TIMING] VitPose:      {time.time()-t0:.3f}s")

        t0 = time.time()
        cache['features'] = self.feature_extractor.extract_video_features(imgs_shared, bbx_xys)
        Log.info(f"[TIMING] HMR2 Feat:    {time.time()-t0:.3f}s")

        t0 = time.time()
        slam_thread.join()
        Log.info(f"[TIMING] SLAM wait:    {time.time()-t0:.3f}s")
        return cache

    def _load_data_dict(self, cfg, cache):
        # 逻辑与原代码 load_data_dict 一致
        paths = cfg.paths
        length, width, height = get_video_lwh(cfg.video_path)
        if cfg.static_cam:
            R_w2c = torch.eye(3).repeat(length, 1, 1)
        else:
            traj = cache.get('slam')
            # 假设只用 SimpleVO
            R_w2c = torch.from_numpy(traj[:, :3, :3])

        if cfg.f_mm is not None:
            K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
        else:
            K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

        data = {
            "length": torch.tensor(length),
            "bbx_xys": cache['bbx']["bbx_xys"],
            "kp2d": cache['vitpose'],
            "K_fullimg": K_fullimg,
            "cam_angvel": compute_cam_angvel(R_w2c),
            "f_imgseq": cache['features'],
        }
        return data

    def _render_incam(self, cfg):
        # 渲染逻辑封装 (复用 self.smplx 等资源)
        incam_video_path = Path(cfg.paths.incam_video)
        if incam_video_path.exists(): return

        pred = torch.load(cfg.paths.hmr4d_results)
        smplx_out = self.smplx(**to_cuda(pred["smpl_params_incam"]))
        pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smplx_out.vertices])

        video_path = cfg.video_path
        length, width, height = get_video_lwh(video_path)
        K = pred["K_fullimg"][0]
        renderer = Renderer(width, height, device="cuda", faces=self.faces_smpl, K=K)
        reader = get_video_reader(video_path)
        
        writer = get_writer(incam_video_path, fps=30, crf=CRF)
        verts_incam = pred_c_verts
        for i, img_raw in tqdm(enumerate(reader), total=length, desc="Rendering Incam"):
            img = renderer.render_mesh(verts_incam[i].to(self.device), img_raw, [0.8, 0.8, 0.8])
            writer.write_frame(img)
        writer.close()
        reader.close()

    def _render_global(self, cfg):
        global_video_path = Path(cfg.paths.global_video)
        if global_video_path.exists(): return

        pred = torch.load(cfg.paths.hmr4d_results)
        smplx_out = self.smplx(**to_cuda(pred["smpl_params_global"]))
        pred_ay_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smplx_out.vertices])

        # 定义局部辅助函数
        def move_to_start_point_face_z(verts):
            verts = verts.clone()
            offset = einsum(self.J_regressor, verts[0], "j v, v i -> j i")[0]
            offset[1] = verts[:, :, [1]].min()
            verts = verts - offset
            T_ay2ayfz = compute_T_ayfz2ay(einsum(self.J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
            verts = apply_T_on_points(verts, T_ay2ayfz)
            return verts

        verts_glob = move_to_start_point_face_z(pred_ay_verts)
        joints_glob = einsum(self.J_regressor, verts_glob, "j v, l v i -> l j i")
        
        global_R, global_T, global_lights = get_global_cameras_static(
            verts_glob.cpu(), beta=2.0, cam_height_degree=20, target_center_height=1.0,
        )

        length, width, height = get_video_lwh(cfg.video_path)
        _, _, K = create_camera_sensor(width, height, 24)
        renderer = Renderer(width, height, device="cuda", faces=self.faces_smpl, K=K)
        
        scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
        renderer.set_ground(scale * 1.5, cx, cz)
        color = torch.ones(3).float().cuda() * 0.8

        writer = get_writer(global_video_path, fps=30, crf=CRF)
        for i in tqdm(range(length), desc="Rendering Global"):
            cameras = renderer.create_camera(global_R[i], global_T[i])
            img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
            writer.write_frame(img)
        writer.close()

    def _export_for_momask(self, cfg):
        # 原封不动保留你的 V3 导出逻辑
        hmr4d_results_path = Path(cfg.paths.hmr4d_results)
        output_npz_path = Path(cfg.output_dir) / "momask_input.npz"
        
        if not hmr4d_results_path.exists():
            return

        pred = torch.load(hmr4d_results_path, map_location='cpu')
        smpl_params = pred['smpl_params_global']

        global_orient = smpl_params['global_orient']
        body_pose = smpl_params['body_pose']
        transl = smpl_params['transl']
        betas = smpl_params['betas']

        # 1. 基础修正 (-90度 X轴)
        base_correction_angle = -torch.pi / 2.0
        base_correction_axis = torch.tensor([1.0, 0.0, 0.0])
        base_correction_rot_aa = base_correction_axis * base_correction_angle
        C_base = axis_angle_to_matrix(base_correction_rot_aa.unsqueeze(0))

        # 2. 翻转修正 (180度 X轴)
        flip_correction_angle = torch.pi
        flip_correction_axis = torch.tensor([1.0, 0.0, 0.0])
        flip_correction_rot_aa = flip_correction_axis * flip_correction_angle
        C_flip = axis_angle_to_matrix(flip_correction_rot_aa.unsqueeze(0))

        # 3. 组合
        C = torch.matmul(C_flip, C_base)

        transl_mujoco = torch.einsum('ij,nj->ni', C.squeeze(0), transl)
        R_gvhmr = axis_angle_to_matrix(global_orient)
        R_mujoco = torch.matmul(C, R_gvhmr)
        global_orient_mujoco = matrix_to_axis_angle(R_mujoco)

        smpl_body_pose = body_pose[:, :63]
        poses_for_momask = torch.cat([global_orient_mujoco, smpl_body_pose], dim=1)

        output_data = {
            'mocap_framerate': np.array(30, dtype=np.int32),
            'trans': transl_mujoco.numpy(),
            'poses': poses_for_momask.numpy(),
            'betas': betas.mean(dim=0).numpy(),
            'gender': np.array('neutral', dtype=str)
        }
        np.savez(output_npz_path, **output_data)
        Log.info(f"[Export] V3 Momask input saved: {output_npz_path}")

# ==========================================
# 调用示例
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="inputs", help="Video file or directory")
    parser.add_argument("--out", type=str, default="outputs/batch_demo")
    parser.add_argument("--raw-motion-only", action="store_true", help="只输出重定向前的 SMPL 关节动作帧 .npy，不运行 H1 IK")
    parser.add_argument("--raw-motion-output-dir", type=str, default="outputs/raw_motion_npy", help="raw motion .npy 输出目录，文件名自动使用时间戳")
    parser.add_argument("--raw-motion-coord", choices=["tw", "ik_input", "h1", "smpl", "v3"], default="tw", help="tw: aligned for tw_retargeting; ik_input/h1: legacy H1; smpl/v3: debug")
    parser.add_argument("--raw-motion-target-fps", type=float, default=20.0, help="Export FPS; 20 for tw_retargeting, 0 to keep source FPS")
    parser.add_argument("--raw-motion-extended", action="store_true", help="Append ankle/wrist rotation matrices and hand openness")
    args = parser.parse_args()

    # 1. 实例化系统 (此时执行冷启动，加载所有模型，比较慢)
    gvhmr_sys = GVHMRSystem(load_h1_optimizer=not args.raw_motion_only)

    # 2. 准备视频列表
    input_path = Path(args.input)
    if input_path.is_file():
        videos = [input_path]
    else:
        videos = sorted(list(input_path.glob("*.mp4")))

    Log.info(f"Found {len(videos)} videos to process.")

    # 3. 循环处理 (热运行，非常快)
    for vid in videos:
        try:
            if args.raw_motion_only:
                ok = gvhmr_sys.process_video_raw_motion(
                    video_path_str=str(vid),
                    output_dir=args.raw_motion_output_dir,
                    static_cam=False,  # 或 True，根据需要
                    raw_motion_coord=args.raw_motion_coord,
                    raw_motion_target_fps=args.raw_motion_target_fps,
                    raw_motion_extended=args.raw_motion_extended,
                )
                print(ok)
            else:
                # 这里的 output_root 是总目录，内部会根据 video_name 自动创建子目录
                ok = gvhmr_sys.process_video(
                    video_path_str=str(vid),
                    output_root=args.out,
                    static_cam=False,  # 或 True，根据需要
                )
                Path(args.out).mkdir(parents=True, exist_ok=True)
                joblib.dump(ok, str(Path(args.out) / f'{vid.stem}_result.pkl'))
        except Exception as e:
            Log.error(f"Error processing {vid}: {e}")
            import traceback
            traceback.print_exc()
