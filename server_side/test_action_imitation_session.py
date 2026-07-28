import unittest

try:
    from action_imitation_session import (
        ActionImitationSessionRegistry,
        register_session_request,
    )
except ModuleNotFoundError:
    from server_side.action_imitation_session import (
        ActionImitationSessionRegistry,
        register_session_request,
    )


class ActionImitationSessionRegistryTest(unittest.TestCase):
    def setUp(self):
        self.registry = ActionImitationSessionRegistry()

    def test_duplicate_registration_is_idempotent_and_preserves_frames(self):
        result, session = self.registry.register(robot_id="robot_001", task_id="task-A")
        self.assertEqual("created", result)
        self.assertEqual("armed", session.state)

        self.registry.note_frame("robot_001", now_unix_ms=100)
        result, duplicate = self.registry.register(
            robot_id="robot_001",
            task_id="task-A",
            source_turn_id="must-not-reset-session",
        )

        self.assertEqual("already_registered", result)
        self.assertEqual("recording", duplicate.state)
        self.assertEqual(1, duplicate.frame_count)
        self.assertEqual(100, duplicate.first_frame_unix_ms)
        self.assertEqual("", duplicate.source_turn_id)

    def test_different_task_conflicts_while_robot_is_recording(self):
        self.registry.register(robot_id="robot_001", task_id="task-A")
        self.registry.note_frame("robot_001", now_unix_ms=100)

        result, active = self.registry.register(
            robot_id="robot_001", task_id="task-B"
        )

        self.assertEqual("conflict", result)
        self.assertEqual("task-A", active.task_id)
        self.assertIsNone(self.registry.get("robot_001", "task-B"))

    def test_processing_snapshot_is_not_overwritten_by_next_task(self):
        self.registry.register(
            robot_id="robot_001",
            task_id="task-A",
            source_request_id="request-A",
        )
        self.registry.note_frame("robot_001", now_unix_ms=100)
        snapshot = self.registry.begin_processing("robot_001")

        result, next_session = self.registry.register(
            robot_id="robot_001",
            task_id="task-B",
            source_request_id="request-B",
        )

        self.assertIsNotNone(snapshot)
        self.assertEqual("task-A", snapshot.task_id)
        self.assertEqual("request-A", snapshot.source_request_id)
        self.assertEqual("processing", snapshot.state)
        self.assertEqual("created", result)
        self.assertEqual("task-B", next_session.task_id)

        self.registry.mark_state("robot_001", "task-A", "completed")
        self.assertEqual(
            "completed", self.registry.get("robot_001", "task-A").state
        )
        self.assertEqual(
            "armed", self.registry.get("robot_001", "task-B").state
        )

    def test_failed_callback_metadata_is_retained_for_retry(self):
        self.registry.register(robot_id="robot_001", task_id="task-A")
        self.registry.note_frame("robot_001", now_unix_ms=100)
        self.registry.begin_processing("robot_001")
        self.registry.record_result(
            "robot_001",
            "task-A",
            combined_path="/tmp/action.npy",
            sha256="a" * 64,
        )

        retained = self.registry.get("robot_001", "task-A")
        self.assertEqual("notify_pending", retained.state)
        self.assertEqual("/tmp/action.npy", retained.combined_path)
        self.assertEqual("a" * 64, retained.sha256)

    def test_frame_without_armed_session_is_ignored(self):
        self.assertIsNone(self.registry.note_frame("robot_001"))

    def test_http_contract_rejects_missing_identity(self):
        status, body, session = register_session_request(
            self.registry, robot_id=" ", task_id=None
        )
        self.assertEqual(400, status)
        self.assertEqual(
            {"ok": False, "reason": "missing_robot_id_or_task_id"}, body
        )
        self.assertIsNone(session)

    def test_http_contract_is_idempotent_and_reports_conflict(self):
        first_status, first_body, _ = register_session_request(
            self.registry, robot_id=" robot_001 ", task_id=" task-A "
        )
        duplicate_status, duplicate_body, _ = register_session_request(
            self.registry, robot_id="robot_001", task_id="task-A"
        )
        conflict_status, conflict_body, _ = register_session_request(
            self.registry, robot_id="robot_001", task_id="task-B"
        )

        self.assertEqual(200, first_status)
        self.assertEqual("armed", first_body["state"])
        self.assertEqual(200, duplicate_status)
        self.assertEqual("already_registered", duplicate_body["reason"])
        self.assertEqual("armed", duplicate_body["state"])
        self.assertEqual(409, conflict_status)
        self.assertEqual("active_session_conflict", conflict_body["reason"])
        self.assertEqual("task-A", conflict_body["active_task_id"])


if __name__ == "__main__":
    unittest.main()
