"""Numeric SMPL joint NPY export shared with native tw_retargeting."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SOURCE_JOINTS = 29
EXTENDED_JOINTS = 43
ROTATION_JOINTS = (7, 8, 20, 21)  # left/right ankle, left/right wrist

def sample_times(frame_count: int, source_fps: float, target_fps: float) -> np.ndarray:
    if frame_count < 1 or source_fps <= 0 or target_fps <= 0:
        raise ValueError('Frame count and frame rates must be positive')
    count = int(np.floor((frame_count - 1) * target_fps / source_fps)) + 1
    return np.arange(count, dtype=np.float64) / target_fps

def resample_values(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] < 1:
        raise ValueError('Motion has no frames')
    if source_fps == target_fps:
        return np.ascontiguousarray(values, dtype=np.float32)
    source_times = np.arange(values.shape[0], dtype=np.float64) / source_fps
    target_times = sample_times(values.shape[0], source_fps, target_fps)
    flat = values.reshape(values.shape[0], -1)
    result = np.stack([np.interp(target_times, source_times, flat[:, i])
                       for i in range(flat.shape[1])], axis=1)
    return np.ascontiguousarray(result.reshape((len(target_times),) + values.shape[1:]), dtype=np.float32)

def smpl_global_rotations(global_orient: np.ndarray, body_pose: np.ndarray,
                          parents: np.ndarray) -> np.ndarray:
    """World rotations for ankle/wrist joints in SMPL joint order."""
    root = np.asarray(global_orient, dtype=np.float64).reshape(-1, 1, 3)
    body = np.asarray(body_pose, dtype=np.float64).reshape(len(root), -1, 3)
    if body.shape[1] < 21:
        raise ValueError('SMPL body pose must contain 21 axis-angle joints')
    local = Rotation.from_rotvec(np.concatenate((root, body[:, :21]), axis=1)
                                 .reshape(-1, 3)).as_matrix().reshape(len(root), 22, 3, 3)
    parents = np.asarray(parents).reshape(-1)
    if len(parents) < 22:
        raise ValueError('SMPL kinematic tree must contain at least 22 joints')
    world = np.empty_like(local)
    world[:, 0] = local[:, 0]
    for joint in range(1, 22):
        parent = int(parents[joint])
        if parent < 0 or parent >= joint:
            raise ValueError(f'Invalid SMPL parent {parent} for joint {joint}')
        world[:, joint] = world[:, parent] @ local[:, joint]
    return world[:, ROTATION_JOINTS]

def resample_rotations(rotations: np.ndarray, source_fps: float,
                       target_fps: float) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.float64)
    if source_fps == target_fps or len(rotations) == 1:
        return rotations.astype(np.float32)
    source_times = np.arange(len(rotations), dtype=np.float64) / source_fps
    target_times = sample_times(len(rotations), source_fps, target_fps)
    result = np.stack([Slerp(source_times, Rotation.from_matrix(rotations[:, joint]))
                       (target_times).as_matrix() for joint in range(4)], axis=1)
    return result.astype(np.float32)

def _finger_openness(landmarks) -> float:
    points = np.array([(point.x, point.y) for point in landmarks.landmark])
    wrist = points[0]
    ratios = []
    for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
        base = np.linalg.norm(points[mcp] - wrist)
        if base > 1e-6:
            ratios.append(np.linalg.norm(points[tip] - wrist) / base)
    if not ratios:
        return 0.5
    return float(np.clip((np.mean(ratios) - 1.2) / 0.8, 0.0, 1.0))

def estimate_hand_openness(video_path: str, frame_count: int, source_fps: float,
                           target_fps: float) -> np.ndarray:
    """Return [left,right] x [openness,confidence]; confidence=0 means unseen."""
    import cv2
    import mediapipe as mp
    times = sample_times(frame_count, source_fps, target_fps)
    result = np.zeros((len(times), 2, 2), dtype=np.float32)
    result[:, :, 0] = 0.5
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f'Cannot read video for hand estimation: {video_path}')
    next_frame = 0
    try:
        with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5) as hands:
            for target, time_sec in enumerate(times):
                wanted = round(time_sec * source_fps)
                frame = None
                ok = False
                while next_frame <= wanted:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    next_frame += 1
                if frame is None or not ok:
                    continue
                detected = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not detected.multi_hand_landmarks:
                    continue
                for landmarks, handedness in zip(detected.multi_hand_landmarks,
                                                 detected.multi_handedness):
                    classification = handedness.classification[0]
                    # MediaPipe labels assume mirrored selfie input; raw video is unmirrored.
                    side = 0 if classification.label == 'Right' else 1
                    confidence = float(classification.score)
                    if confidence > result[target, side, 1]:
                        result[target, side] = (_finger_openness(landmarks), confidence)
    finally:
        capture.release()
    return result

def pack_extended(joints: np.ndarray, rotations: np.ndarray,
                  hand_openness: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    rotations = np.asarray(rotations, dtype=np.float32)
    hand_openness = np.asarray(hand_openness, dtype=np.float32)
    frames = len(joints)
    if joints.shape != (frames, SOURCE_JOINTS, 3):
        raise ValueError('Expected SMPL joints with shape (T,29,3)')
    if rotations.shape != (frames, 4, 3, 3):
        raise ValueError('Expected four 3x3 ankle/wrist rotations per frame')
    if hand_openness.shape != (frames, 2, 2):
        raise ValueError('Expected left/right openness and confidence per frame')
    output = np.zeros((frames, EXTENDED_JOINTS, 3), dtype=np.float32)
    output[:, :SOURCE_JOINTS] = joints
    output[:, 29:41] = rotations.reshape(frames, 12, 3)
    output[:, 41:43, :2] = hand_openness
    if not np.isfinite(output).all():
        raise ValueError('Export contains non-finite values')
    return np.ascontiguousarray(output)


# GVHMR and the robot URDF use opposite horizontal facing/left axes after the
# existing (z, x, y) Y-up -> Z-up permutation in tw_retargeting. A 180-degree
# yaw in the SMPL Y-up frame aligns both the torso forward and left axes.
TW_ALIGNMENT_MATRIX = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)


def align_tw_joint_positions(joints: np.ndarray) -> np.ndarray:
    """Rotate SMPL Y-up positions into tw_retargeting's robot-facing convention."""
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError('Expected joint positions shaped (T,J,3)')
    return np.ascontiguousarray(joints @ TW_ALIGNMENT_MATRIX, dtype=np.float32)


def align_tw_rotation_matrices(rotations: np.ndarray) -> np.ndarray:
    """Apply the same world-frame yaw to the appended ankle/wrist rotations."""
    rotations = np.asarray(rotations, dtype=np.float32)
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError('Expected rotation matrices shaped (T,J,3,3)')
    return np.ascontiguousarray(TW_ALIGNMENT_MATRIX @ rotations, dtype=np.float32)
