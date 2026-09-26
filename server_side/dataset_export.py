"""HumanML3D-compatible 22-joint dataset output with separate training labels."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from raw_motion_export import resample_values, resample_rotations, sample_times

SCHEMA_VERSION = "internet_motion_v1"
CONTACT_PARTS = ("left_foot", "right_foot", "left_hand", "right_hand", "pelvis", "left_knee", "right_knee")
SUPPORT_STATES = {"unknown": -1, "none": 0, "left_foot": 1, "right_foot": 2, "both_feet": 3,
                  "chair": 4, "floor_seated": 5, "multi_point": 6}
ACTIVE_SIDES = {"unknown": -1, "none": 0, "left": 1, "right": 2, "both": 3, "alternating": 4}


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _codec(code_dir: str, offsets_path: str):
    import torch
    code = Path(code_dir).resolve()
    offsets = Path(offsets_path).resolve()
    if not (code / "motion_repr_upstream.py").is_file() or not offsets.is_file():
        raise FileNotFoundError("HumanML3D code and target_offsets.npy are required")
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    import motion_repr_upstream as rep
    from paramUtil import t2m_raw_offsets, t2m_kinematic_chain
    rep.l_idx1, rep.l_idx2 = 5, 8
    rep.fid_r, rep.fid_l = [8, 11], [7, 10]
    rep.face_joint_indx = [2, 1, 17, 16]
    rep.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    rep.kinematic_chain = t2m_kinematic_chain
    rep.tgt_offsets = torch.from_numpy(np.load(offsets, allow_pickle=False))
    return rep


def _canonical_heading(joints):
    """Y-up heading rotation for the first frame, after the HumanML X reflection."""
    from scipy.spatial.transform import Rotation
    pose = joints[0]
    across = pose[2] - pose[1] + pose[17] - pose[16]
    across[1] = 0
    norm = np.linalg.norm(across)
    if norm < 1e-5:
        raise ValueError("Initial facing direction is undefined")
    across /= norm
    forward = np.cross([0., 1., 0.], across)
    angle = np.arctan2(forward[0], forward[2])
    return Rotation.from_rotvec([0., -angle, 0.]).as_matrix().astype(np.float32)


def _confidence(kp2d, source_fps, target_fps, count):
    points = _numpy(kp2d)
    if points.ndim != 3 or points.shape[2] < 3 or len(points) == 0:
        return np.full(count, -1, dtype=np.float32)
    score = np.clip(np.nanmean(points[:, :, 2], axis=1), 0, 1)
    times = sample_times(len(points), source_fps, target_fps)[:count]
    indices = np.clip(np.rint(times * source_fps).astype(int), 0, len(score) - 1)
    return score[indices].astype(np.float32)


def build_sample(world_joints, source_fps, target_fps, rotations, hands, kp2d, betas,
                 code_dir, offsets_path):
    """Return (IK joints, HumanML263 vectors, training annotations)."""
    import torch
    world_joints = np.asarray(world_joints, dtype=np.float32)
    if world_joints.ndim != 3 or world_joints.shape[1:] != (22, 3) or len(world_joints) < 10:
        raise ValueError("Expected >=10 world-space SMPL22 frames")
    if not np.isfinite(world_joints).all():
        raise ValueError("Non-finite world joints")
    source = resample_values(world_joints, source_fps, target_fps)
    if len(source) < 10:
        raise ValueError("Need >=10 frames at target FPS")
    # Match the existing AMASS -> HumanML3D conversion in humanml3d_amass.py.
    encoder_input = source.copy()
    encoder_input[:, :, 0] *= -1
    codec = _codec(code_dir, offsets_path)
    vec, _, _, _ = codec.process_file(encoder_input, 0.002)
    vec = np.asarray(vec, dtype=np.float32)
    joints = codec.recover_from_ric(torch.from_numpy(vec).unsqueeze(0), 22).squeeze(0).numpy().astype(np.float32)
    count = len(vec)
    if vec.shape != (count, 263) or joints.shape != (count, 22, 3) or count < 9:
        raise ValueError("Invalid HumanML output shape")
    if not np.isfinite(vec).all() or not np.isfinite(joints).all():
        raise ValueError("Non-finite HumanML output")
    rotations = resample_rotations(np.asarray(rotations), source_fps, target_fps)[:count]
    hands = np.asarray(hands, dtype=np.float32)[:count]
    if rotations.shape != (count, 4, 3, 3) or hands.shape != (count, 2, 2):
        raise ValueError("Extended arrays are not aligned")
    # A reflected coordinate system transforms orientations as F R F.
    reflect = np.diag([-1., 1., 1.]).astype(np.float32)
    heading = _canonical_heading(encoder_input)
    rotations = (heading @ reflect @ rotations @ reflect).astype(np.float32)
    shape = np.asarray(_numpy(betas), dtype=np.float32).reshape(-1, 10)
    if not np.isfinite(shape).all():
        raise ValueError("Invalid SMPL betas")
    # Motion-only suggestions are separated from reviewed ground truth. They
    # cannot identify chair/hand contact or distinguish a resting foot from a
    # load-bearing foot without scene evidence.
    foot_candidate = np.full((count, len(CONTACT_PARTS)), -1, dtype=np.int8)
    foot_speed = np.linalg.norm(np.diff(joints[:, [10, 11]], axis=0, prepend=joints[:1, [10, 11]]), axis=-1)
    foot_height = joints[:, [10, 11], 1]
    near_floor = foot_height < 0.10
    foot_candidate[:, :2][near_floor & (foot_speed < 0.025)] = 1
    foot_candidate[:, :2][foot_height > 0.18] = 0
    pelvis_height = joints[:, 0, 1]
    support_candidate = np.full(count, -1, dtype=np.int8)
    standing = pelvis_height > 0.75
    left = foot_candidate[:, 0] == 1
    right = foot_candidate[:, 1] == 1
    support_candidate[standing & left & right] = SUPPORT_STATES["both_feet"]
    support_candidate[standing & left & ~right] = SUPPORT_STATES["left_foot"]
    support_candidate[standing & right & ~left] = SUPPORT_STATES["right_foot"]
    relative_hands = joints[:, [20, 21]] - joints[:, :1]
    hand_speed = np.linalg.norm(np.diff(relative_hands, axis=0, prepend=relative_hands[:1]), axis=-1)
    active_candidate = np.full(count, -1, dtype=np.int8)
    active_candidate[(hand_speed[:, 0] > 0.035) & (hand_speed[:, 0] > 1.4 * hand_speed[:, 1])] = ACTIVE_SIDES["left"]
    active_candidate[(hand_speed[:, 1] > 0.035) & (hand_speed[:, 1] > 1.4 * hand_speed[:, 0])] = ACTIVE_SIDES["right"]
    active_candidate[(hand_speed[:, 0] > 0.035) & (hand_speed[:, 1] > 0.035) &
                     (active_candidate == -1)] = ACTIVE_SIDES["both"]
    transition_score = np.clip(np.abs(np.diff(pelvis_height, prepend=pelvis_height[:1])) / 0.04, 0, 1).astype(np.float32)
    labels = {
        "contact_candidate": foot_candidate,
        "support_candidate": support_candidate,
        "active_side_candidate": active_candidate,
        "transition_score": transition_score,
        "ankle_wrist_rotation": rotations,
        "hand_openness": hands[:, :, 0],
        "hand_detection_confidence": hands[:, :, 1],
        "contact_label": np.full((count, len(CONTACT_PARTS)), -1, dtype=np.int8),
        "contact_confidence": np.zeros((count, len(CONTACT_PARTS)), dtype=np.float32),
        "support_state": np.full(count, -1, dtype=np.int8),
        "support_confidence": np.zeros(count, dtype=np.float32),
        "confidence_2d_keypoints": _confidence(kp2d, source_fps, target_fps, count),
        "confidence_3d": np.full(count, -1, dtype=np.float32),
        "active_side": np.full(count, -1, dtype=np.int8),
        "smpl_betas": np.median(shape, axis=0).astype(np.float32),
    }
    return joints, vec, labels


def write_sample(output_dir, video_path, joints, vec, labels, source_fps, target_fps,
                 *, source_uri=None, license_id=None, source_object_sha256=None,
                 parent_source_uri=None, source_page=None):
    """Write matching files and provenance to the data disk."""
    video_hash = _sha256(Path(video_path))
    sample_id = hashlib.sha256((SCHEMA_VERSION + ":" + video_hash).encode()).hexdigest()[:24]
    root = Path(output_dir)
    targets = {
        "new_joints": root / "new_joints" / (sample_id + ".npy"),
        "new_joint_vecs": root / "new_joint_vecs" / (sample_id + ".npy"),
        "annotations": root / "annotations" / (sample_id + ".npz"),
        "metadata": root / "metadata" / (sample_id + ".json"),
    }
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("Sample already exists: " + sample_id)
    for path in targets.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(targets["new_joints"], joints, allow_pickle=False)
    np.save(targets["new_joint_vecs"], vec, allow_pickle=False)
    np.savez_compressed(targets["annotations"], **labels)
    manifest = {
        "schema_version": SCHEMA_VERSION, "sample_id": sample_id,
        "processing_video_sha256": video_hash, "source_uri": source_uri,
        "source_object_sha256": source_object_sha256,
        "parent_source_uri": parent_source_uri, "source_page": source_page,
        "source_license": license_id, "source_fps": source_fps,
        "output_fps": target_fps, "frames": len(vec),
        "frame_mapping": "frame i maps to i/output_fps seconds; last source frame dropped for velocity",
        "joint_order": "SMPL22",
        "coordinate_system": "HumanML3D canonical Y-up, floor aligned, first frame faces +Z",
        "arrays": {k: str(v.relative_to(root)) for k, v in targets.items() if k != "metadata"},
        "contact_parts": CONTACT_PARTS,
        "contact_codes": {"unknown": -1, "none": 0, "ground": 1},
        "support_codes": SUPPORT_STATES, "active_side_codes": ACTIVE_SIDES,
        "rotation_parts": ["left_ankle", "right_ankle", "left_wrist", "right_wrist"],
        "rotation_coordinates": "X-reflected source Y-up, then first-heading canonicalized",
        "caption_events": [], "transition_clips": [],
        "review_status": "pending_contact_support_language_review",
        "candidate_definition": "kinematic foot/standing/hand-motion suggestions; not reviewed labels; transition_score is pelvis vertical speed divided by 0.04 m/frame",
        "confidence_definition": "2D mean ViTPose score; 3D -1 means unavailable",
        "smpl_betas_definition": "median per-frame GVHMR SMPL 10D estimate",
    }
    manifest["sha256"] = {k: _sha256(targets[k]) for k in ("new_joints", "new_joint_vecs", "annotations")}
    targets["metadata"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"sample_id": sample_id, "paths": {k: str(v) for k, v in targets.items()}, "frames": len(vec)}
