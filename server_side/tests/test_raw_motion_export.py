import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raw_motion_export import (sample_times, resample_values, smpl_global_rotations,
                               resample_rotations, pack_extended, align_tw_joint_positions, align_tw_rotation_matrices)

class RawMotionExportTest(unittest.TestCase):
    def test_resampling_and_extended_contract(self):
        frames = 30
        joints = np.zeros((frames, 29, 3), dtype=np.float32)
        joints[:, 1, 0] = np.arange(frames, dtype=np.float32) / 30
        out = resample_values(joints, 30, 20)
        self.assertEqual(out.shape, (20, 29, 3))
        np.testing.assert_allclose(out[:, 1, 0], sample_times(30, 30, 20), atol=1e-6)
        matrices = np.broadcast_to(np.eye(3), (frames, 4, 3, 3)).copy()
        matrices = resample_rotations(matrices, 30, 20)
        hands = np.zeros((20, 2, 2), dtype=np.float32)
        hands[:, 0] = [0.8, 0.9]
        hands[:, 1] = [0.2, 0.7]
        packed = pack_extended(out, matrices, hands)
        self.assertEqual(packed.shape, (20, 43, 3))
        self.assertEqual(packed.dtype, np.float32)
        np.testing.assert_array_equal(packed[:, :29], out)
        np.testing.assert_allclose(packed[:, 29:41].reshape(20, 4, 3, 3), matrices)
        np.testing.assert_array_equal(packed[:, 41:43, :2], hands)
        self.assertTrue(np.isfinite(packed).all())

    def test_global_ankle_rotation_follows_parent_chain(self):
        parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                            9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
        root = np.zeros((2, 3))
        root[:, 2] = np.pi / 2
        body = np.zeros((2, 21, 3))
        body[:, 3, 0] = np.pi / 4  # left knee, parent of left ankle
        matrices = smpl_global_rotations(root, body, parents)
        expected = Rotation.from_euler('z', 90, degrees=True).as_matrix() @ Rotation.from_euler('x', 45, degrees=True).as_matrix()
        np.testing.assert_allclose(matrices[:, 0], np.broadcast_to(expected, (2, 3, 3)), atol=1e-6)
        np.testing.assert_allclose(matrices @ matrices.transpose(0, 1, 3, 2), np.broadcast_to(np.eye(3), (2, 4, 3, 3)), atol=1e-6)


class TwAlignmentTest(unittest.TestCase):
    def test_robot_forward_and_left_axes(self):
        joints = np.zeros((2, 29, 3), dtype=np.float32)
        joints[:, 3, 1] = 0.5
        joints[:, 13] = [-0.2, 0.8, 0.0]
        joints[:, 14] = [0.2, 0.8, 0.0]
        aligned = align_tw_joint_positions(joints)
        up_axis = Rotation.from_quat([0.5, 0.5, 0.5, 0.5]).as_matrix()
        xyz = aligned @ up_axis.T
        right = xyz[0, 13] - xyz[0, 14]
        up = (xyz[0, 13] + xyz[0, 14]) / 2 - xyz[0, 3]
        np.testing.assert_allclose(right / np.linalg.norm(right), [0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(np.cross(right, up) / np.linalg.norm(np.cross(right, up)), [1, 0, 0], atol=1e-6)
        matrices = np.broadcast_to(np.eye(3), (2, 4, 3, 3))
        rotated = align_tw_rotation_matrices(matrices)
        np.testing.assert_allclose(rotated[0, 0], np.diag([-1, 1, -1]))
        np.testing.assert_allclose(np.linalg.det(rotated), 1, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
