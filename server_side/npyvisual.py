import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation as R
import os
from pathlib import Path


# =========================
# 配置
# =========================
path = "/home/wd/Agent_memory_patch/IK_Retargeting/raw_motion_npy/action_imitation_1783921827887_combined.npy"

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

basename = Path(path).stem
out_gif = os.path.join(output_dir, f"{basename}.gif")
out_png = os.path.join(output_dir, f"{basename}.png")

# 如果只想看标准 22 关节骨架，设为 True
# 如果想把 29 个点都画出来，设为 False
USE_22_ONLY = False


# =========================
# 读取 npy
# =========================
data = np.load(path, allow_pickle=True)

# 兼容 0 维 object 包
if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object:
    data = data.item()

# 如果是 dict，则优先取 motion
motion = data["motion"] if isinstance(data, dict) and "motion" in data else data
motion = np.asarray(motion)

print("raw motion shape:", motion.shape)


# =========================
# 统一转成 (T, J, 3)
# =========================
def normalize_motion_shape(motion_arr: np.ndarray) -> np.ndarray:
    """
    支持以下格式：
      (T, J, 3)
      (J, 3, T)
      (1, J, 3, T)
    其中 J 可以是 22 或 29。
    """
    if motion_arr.ndim == 4:
        # 常见旧格式: (1, J, 3, T)
        if motion_arr.shape[0] == 1 and motion_arr.shape[2] == 3:
            return motion_arr[0].transpose(2, 0, 1)

    if motion_arr.ndim == 3:
        # 新格式: (T, J, 3)
        if motion_arr.shape[2] == 3:
            return motion_arr

        # 旧格式: (J, 3, T)
        if motion_arr.shape[1] == 3:
            return motion_arr.transpose(2, 0, 1)

    raise ValueError(f"Unsupported motion shape: {motion_arr.shape}")


seq = normalize_motion_shape(motion)
N, J, _ = seq.shape

print("normalized seq shape:", seq.shape)

if J not in (22, 29):
    raise ValueError(f"Expected 22 or 29 joints, got J={J}, shape={seq.shape}")

if USE_22_ONLY and J >= 22:
    seq = seq[:, :22, :]
    J = 22
    print("using first 22 joints only:", seq.shape)


# =========================
# 坐标系旋转
# =========================
rot = R.from_quat([0.5, 0.5, 0.5, 0.5]).as_matrix()
seq = seq @ rot.T


# =========================
# 关节连接关系
# =========================
# 0 pelvis
# 1 left_hip
# 2 right_hip
# 3 spine1
# 4 left_knee
# 5 right_knee
# 6 spine2
# 7 left_ankle
# 8 right_ankle
# 9 spine3
# 10 left_foot
# 11 right_foot
# 12 neck
# 13 left_collar
# 14 right_collar
# 15 head
# 16 left_shoulder
# 17 right_shoulder
# 18 left_elbow
# 19 right_elbow
# 20 left_wrist
# 21 right_wrist
# 22 left_hand
# 23 right_hand
# 24 nose
# 25 right_eye
# 26 left_eye
# 27 right_ear
# 28 left_ear

edges_22 = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12),
    (12, 15),
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),
]

# 29 关节额外连接：
# wrist -> hand
# head -> nose / eyes / ears
edges_extra_29 = [
    (20, 22),  # left_wrist -> left_hand
    (21, 23),  # right_wrist -> right_hand
    (15, 24),  # head -> nose
    (24, 25),  # nose -> right_eye
    (24, 26),  # nose -> left_eye
    (25, 27),  # right_eye -> right_ear
    (26, 28),  # left_eye -> left_ear
]

edges = edges_22.copy()
if J >= 29:
    edges += edges_extra_29

# 防止越界
edges = [(a, b) for a, b in edges if a < J and b < J]


# =========================
# 坐标轴范围
# =========================
mins = seq.reshape(-1, 3).min(axis=0)
maxs = seq.reshape(-1, 3).max(axis=0)
center = (mins + maxs) / 2.0

span = (maxs - mins).max() * 0.6
if span <= 1e-8:
    span = 1.0

xlim = (center[0] - span, center[0] + span)
ylim = (center[1] - span, center[1] + span)
zlim = (center[2] - span, center[2] + span)


# =========================
# 动画 GIF
# =========================
fig = plt.figure(figsize=(6.5, 6.5))
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_zlim(*zlim)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 视角可按需要调整
ax.view_init(elev=0, azim=0)

pts = seq[0]
sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=18)

lines = []
for a, b in edges:
    ln, = ax.plot(
        [pts[a, 0], pts[b, 0]],
        [pts[a, 1], pts[b, 1]],
        [pts[a, 2], pts[b, 2]],
        linewidth=2,
    )
    lines.append(ln)


def update(frame):
    pts_frame = seq[frame]

    sc._offsets3d = (
        pts_frame[:, 0],
        pts_frame[:, 1],
        pts_frame[:, 2],
    )

    for ln, (a, b) in zip(lines, edges):
        ln.set_data(
            [pts_frame[a, 0], pts_frame[b, 0]],
            [pts_frame[a, 1], pts_frame[b, 1]],
        )
        ln.set_3d_properties(
            [pts_frame[a, 2], pts_frame[b, 2]]
        )

    ax.set_title(f"{basename}.npy - frame {frame + 1}/{N}, joints={J}")
    return [sc] + lines


anim = FuncAnimation(
    fig,
    update,
    frames=N,
    interval=50,
    blit=False,
)

anim.save(out_gif, writer=PillowWriter(fps=20))


# =========================
# 第一帧 PNG
# =========================
fig2 = plt.figure(figsize=(6.5, 6.5))
ax2 = fig2.add_subplot(111, projection="3d")

ax2.set_title(f"{basename}.npy - frame 0, joints={J}")
ax2.set_xlim(*xlim)
ax2.set_ylim(*ylim)
ax2.set_zlim(*zlim)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.view_init(elev=15, azim=-60)

pts0 = seq[0]
ax2.scatter(pts0[:, 0], pts0[:, 1], pts0[:, 2], s=18)

for a, b in edges:
    ax2.plot(
        [pts0[a, 0], pts0[b, 0]],
        [pts0[a, 1], pts0[b, 1]],
        [pts0[a, 2], pts0[b, 2]],
        linewidth=2,
    )

plt.tight_layout()
plt.savefig(out_png, dpi=160)
plt.close("all")


print("saved gif:", out_gif)
print("saved png:", out_png)
print("seq shape:", seq.shape)
print("span:", float(span))