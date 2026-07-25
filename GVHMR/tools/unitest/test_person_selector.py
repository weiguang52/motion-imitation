"""Unit tests for multi-person primary-track selection."""

import importlib.util
import unittest
from pathlib import Path


# Import the pure selection module directly. Importing the preproc package also
# initializes YOLO/VitPose, which is intentionally unnecessary for these tests.
MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "hmr4d"
    / "utils"
    / "preproc"
    / "person_selector.py"
)
SPEC = importlib.util.spec_from_file_location("person_selector", MODULE_PATH)
person_selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(person_selector)
rank_primary_person_tracks = person_selector.rank_primary_person_tracks


def detection(track_id, box):
    return {"id": track_id, "bbx_xyxy": box}


class PrimaryPersonSelectorTest(unittest.TestCase):
    def test_prefers_central_complete_person_over_larger_edge_person(self):
        history = []
        for _ in range(20):
            history.append(
                [
                    detection(1, [400, 100, 600, 900]),
                    detection(2, [0, 50, 430, 980]),
                ]
            )

        _, _, ranked_ids, scores = rank_primary_person_tracks(history, 1000, 1000)

        self.assertEqual(ranked_ids[0], 1)
        self.assertGreater(scores[1]["center"], scores[2]["center"])
        self.assertGreater(scores[1]["completeness"], scores[2]["completeness"])

    def test_prefers_complete_person_when_center_distance_is_similar(self):
        history = []
        for _ in range(12):
            history.append(
                [
                    detection(7, [390, 0, 610, 760]),
                    detection(8, [390, 100, 610, 900]),
                ]
            )

        _, _, ranked_ids, _ = rank_primary_person_tracks(history, 1000, 1000)

        self.assertEqual(ranked_ids[0], 8)

    def test_track_level_scoring_does_not_switch_people_per_frame(self):
        history = []
        for frame_id in range(20):
            main_x = 420 if frame_id != 10 else 650
            visitor_x = 50 if frame_id != 10 else 420
            history.append(
                [
                    detection(11, [main_x, 100, main_x + 160, 900]),
                    detection(12, [visitor_x, 150, visitor_x + 160, 850]),
                ]
            )

        _, _, ranked_ids, _ = rank_primary_person_tracks(history, 1000, 1000)

        self.assertEqual(ranked_ids[0], 11)

    def test_continuity_breaks_tie_for_brief_central_detection(self):
        history = []
        for frame_id in range(20):
            frame = [detection(3, [390, 100, 610, 900])]
            if frame_id < 2:
                frame.append(detection(4, [400, 100, 600, 900]))
            history.append(frame)

        _, _, ranked_ids, scores = rank_primary_person_tracks(history, 1000, 1000)

        self.assertEqual(ranked_ids[0], 3)
        self.assertGreater(scores[3]["continuity"], scores[4]["continuity"])

    def test_empty_history_returns_no_track(self):
        _, _, ranked_ids, diagnostics = rank_primary_person_tracks(
            [[], []],
            1280,
            720,
        )

        self.assertEqual(ranked_ids, [])
        self.assertEqual(diagnostics, {})


if __name__ == "__main__":
    unittest.main()
