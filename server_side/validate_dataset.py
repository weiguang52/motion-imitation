"""Validate one internet-motion sample before publishing it to object storage."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from dataset_export import CONTACT_PARTS, SCHEMA_VERSION


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sample(root, sample_id):
    root = Path(root)
    meta = json.loads((root / "metadata" / (sample_id + ".json")).read_text(encoding="utf-8"))
    assert meta["schema_version"] == SCHEMA_VERSION and meta["sample_id"] == sample_id
    assert meta["output_fps"] == 20, "tw_retargeting requires 20 FPS"
    arrays = meta["arrays"]
    for kind in ("new_joints", "new_joint_vecs", "annotations"):
        assert _hash(root / arrays[kind]) == meta["sha256"][kind], kind + " checksum mismatch"
    joints = np.load(root / arrays["new_joints"], allow_pickle=False)
    vec = np.load(root / arrays["new_joint_vecs"], allow_pickle=False)
    with np.load(root / arrays["annotations"], allow_pickle=False) as z:
        assert joints.ndim == 3 and joints.shape[1:] == (22, 3)
        n = len(joints)
        assert n >= 9 and vec.shape == (n, 263) and n == meta["frames"]
        assert joints.dtype == np.float32 and vec.dtype == np.float32
        assert joints.flags.c_contiguous and np.isfinite(joints).all() and np.isfinite(vec).all()
        assert z["ankle_wrist_rotation"].shape == (n, 4, 3, 3)
        assert z["hand_openness"].shape == (n, 2)
        assert z["contact_label"].shape == (n, len(CONTACT_PARTS))
        assert z["contact_candidate"].shape == (n, len(CONTACT_PARTS))
        assert z["support_candidate"].shape == (n,)
        assert z["active_side_candidate"].shape == (n,)
        assert z["transition_score"].shape == (n,)
        for key in ("support_state", "confidence_2d_keypoints", "confidence_3d", "active_side"):
            assert z[key].shape == (n,), key
        assert z["smpl_betas"].shape == (10,)
        rotation = z["ankle_wrist_rotation"]
        np.testing.assert_allclose(rotation @ rotation.transpose(0, 1, 3, 2),
                                   np.broadcast_to(np.eye(3), rotation.shape), atol=2e-3)
        np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=2e-3)
        assert np.isin(z["contact_label"], [-1, 0, 1]).all()
        assert np.isin(z["support_state"], list(meta["support_codes"].values())).all()
    for event in meta["caption_events"] + meta["transition_clips"]:
        assert 0 <= event["start_frame"] < event["end_frame"] <= n
    return {"sample_id": sample_id, "frames": n, "joints": list(joints.shape), "vectors": list(vec.shape)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("sample_id")
    args = p.parse_args()
    print(json.dumps(validate_sample(args.root, args.sample_id), ensure_ascii=False))
