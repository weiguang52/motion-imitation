"""Select the primary person from a multi-person tracking result.

The selector intentionally works on complete tracks instead of choosing a box
frame by frame.  This keeps the downstream pose sequence attached to one
person when several people cross the centre of the image.
"""

from collections import defaultdict

import numpy as np


DEFAULT_SELECTION_WEIGHTS = {
    "center": 0.45,
    "completeness": 0.25,
    "continuity": 0.20,
    "size": 0.10,
}


def collect_tracks(track_history):
    """Convert frame-wise detections into dictionaries indexed by track id."""
    id_to_frame_ids = defaultdict(list)
    id_to_bbx_xyxys = defaultdict(list)

    for frame_id, frame in enumerate(track_history):
        for detection in frame:
            track_id = detection["id"]
            box = np.asarray(detection["bbx_xyxy"], dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all():
                continue
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            id_to_frame_ids[track_id].append(frame_id)
            id_to_bbx_xyxys[track_id].append(box)

    id_to_bbx_xyxys = {
        track_id: np.asarray(boxes, dtype=np.float32)
        for track_id, boxes in id_to_bbx_xyxys.items()
    }
    return id_to_frame_ids, id_to_bbx_xyxys


def rank_primary_person_tracks(
    track_history,
    frame_width,
    frame_height,
    weights=None,
    edge_margin_ratio=0.02,
):
    """Rank person tracks by centre position, completeness and stability.

    ``completeness`` measures whether the detection stays clear of all four
    image boundaries.  It is a useful, inexpensive proxy for whether the head,
    hands or feet have been cut off.  Medians are used so that a brief
    occlusion or edge crossing does not dominate a whole video chunk.

    Returns:
        ``(frame_ids, boxes, ranked_ids, diagnostics)``.  ``ranked_ids`` is
        ordered best-first and diagnostics contains normalized component
        scores for logging and tests.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")
    if edge_margin_ratio <= 0:
        raise ValueError("edge_margin_ratio must be positive")

    selection_weights = dict(DEFAULT_SELECTION_WEIGHTS)
    if weights is not None:
        unknown = set(weights) - set(selection_weights)
        if unknown:
            raise ValueError(f"unknown person-selection weights: {sorted(unknown)}")
        selection_weights.update(weights)
    weight_sum = sum(selection_weights.values())
    if weight_sum <= 0 or any(value < 0 for value in selection_weights.values()):
        raise ValueError("person-selection weights must be non-negative with a positive sum")
    selection_weights = {
        name: value / weight_sum for name, value in selection_weights.items()
    }

    id_to_frame_ids, id_to_bbx_xyxys = collect_tracks(track_history)
    if not id_to_bbx_xyxys:
        return id_to_frame_ids, id_to_bbx_xyxys, [], {}

    max_track_length = max(len(frame_ids) for frame_ids in id_to_frame_ids.values())
    frame_size = np.asarray([frame_width, frame_height], dtype=np.float32)
    frame_center = frame_size / 2.0
    half_diagonal = np.sqrt(2.0)
    diagnostics = {}

    for track_id, boxes in id_to_bbx_xyxys.items():
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        normalized_offsets = (centers - frame_center) / (frame_size / 2.0)
        center_distances = np.linalg.norm(normalized_offsets, axis=1) / half_diagonal
        center_score = float(np.median(np.clip(1.0 - center_distances, 0.0, 1.0)))

        edge_clearances = np.column_stack(
            (
                boxes[:, 0] / frame_width,
                boxes[:, 1] / frame_height,
                (frame_width - boxes[:, 2]) / frame_width,
                (frame_height - boxes[:, 3]) / frame_height,
            )
        )
        per_box_completeness = np.clip(
            edge_clearances.min(axis=1) / edge_margin_ratio,
            0.0,
            1.0,
        )
        completeness_score = float(np.median(per_box_completeness))

        box_sizes = np.maximum(boxes[:, 2:] - boxes[:, :2], 0.0)
        area_ratios = box_sizes[:, 0] * box_sizes[:, 1] / (frame_width * frame_height)
        # Reaches 1.0 at 20% of the image. sqrt prevents size from dominating
        # centre and completeness when one person is much closer to the camera.
        size_score = float(np.clip(np.sqrt(np.median(area_ratios) / 0.20), 0.0, 1.0))
        continuity_score = len(id_to_frame_ids[track_id]) / max_track_length

        component_scores = {
            "center": center_score,
            "completeness": completeness_score,
            "continuity": continuity_score,
            "size": size_score,
        }
        total_score = sum(
            selection_weights[name] * component_scores[name]
            for name in selection_weights
        )
        diagnostics[track_id] = {**component_scores, "total": total_score}

    ranked_ids = sorted(
        diagnostics,
        key=lambda track_id: (
            diagnostics[track_id]["total"],
            diagnostics[track_id]["continuity"],
            -track_id if isinstance(track_id, (int, np.integer)) else 0,
        ),
        reverse=True,
    )
    return id_to_frame_ids, id_to_bbx_xyxys, ranked_ids, diagnostics
