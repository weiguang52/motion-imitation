"""Action-imitation stream session state.

This module deliberately has no GVHMR/FastAPI dependencies so the identity
handoff rules can be unit-tested without loading model checkpoints.
"""

from dataclasses import dataclass, replace
from threading import RLock
import time


@dataclass
class ActionImitationStreamSession:
    robot_id: str
    task_id: str
    source_turn_id: str = ""
    source_request_id: str = ""
    function_call_id: str = ""
    state: str = "armed"
    created_unix_ms: int = 0
    first_frame_unix_ms: int = 0
    last_frame_unix_ms: int = 0
    frame_count: int = 0
    combined_path: str = ""
    sha256: str = ""


class ActionImitationSessionRegistry:
    """Thread-safe active and historical session registry.

    A robot has at most one armed/recording session.  Once recording is
    detached for processing, a new task may be armed while the old immutable
    snapshot continues through combine and callback.
    """

    def __init__(self) -> None:
        self.active_sessions: dict[str, ActionImitationStreamSession] = {}
        self._sessions: dict[tuple[str, str], ActionImitationStreamSession] = {}
        self._lock = RLock()

    def register(
        self,
        *,
        robot_id: str,
        task_id: str,
        source_turn_id: str = "",
        source_request_id: str = "",
        function_call_id: str = "",
    ) -> tuple[str, ActionImitationStreamSession]:
        """Register a task.

        Returns ``("created"|"already_registered"|"conflict", session)``.
        Re-registering an identity never resets its frame counters or state.
        """
        key = (robot_id, task_id)
        with self._lock:
            active = self.active_sessions.get(robot_id)
            if active is not None:
                if active.task_id == task_id:
                    return "already_registered", replace(active)
                return "conflict", replace(active)

            known = self._sessions.get(key)
            if known is not None:
                return "already_registered", replace(known)

            session = ActionImitationStreamSession(
                robot_id=robot_id,
                task_id=task_id,
                source_turn_id=source_turn_id,
                source_request_id=source_request_id,
                function_call_id=function_call_id,
                created_unix_ms=int(time.time() * 1000),
            )
            self.active_sessions[robot_id] = session
            self._sessions[key] = session
            return "created", replace(session)

    def note_frame(
        self, robot_id: str, now_unix_ms: int | None = None
    ) -> ActionImitationStreamSession | None:
        """Associate one received frame with the currently armed session."""
        with self._lock:
            session = self.active_sessions.get(robot_id)
            if session is None:
                return None
            now_ms = now_unix_ms or int(time.time() * 1000)
            if session.state == "armed":
                session.state = "recording"
                session.first_frame_unix_ms = now_ms
            session.last_frame_unix_ms = now_ms
            session.frame_count += 1
            return replace(session)

    def begin_processing(
        self, robot_id: str
    ) -> ActionImitationStreamSession | None:
        """Atomically detach and snapshot the active recording identity."""
        with self._lock:
            session = self.active_sessions.pop(robot_id, None)
            if session is None:
                return None
            session.state = "processing"
            snapshot = replace(session)
            self._sessions[(robot_id, session.task_id)] = snapshot
            return snapshot

    def mark_state(self, robot_id: str, task_id: str, state: str) -> None:
        with self._lock:
            key = (robot_id, task_id)
            session = self._sessions.get(key)
            if session is not None:
                session.state = state

    def record_result(
        self,
        robot_id: str,
        task_id: str,
        *,
        combined_path: str,
        sha256: str,
        state: str = "notify_pending",
    ) -> None:
        """Retain callback identity and artifact data across network failures."""
        with self._lock:
            session = self._sessions.get((robot_id, task_id))
            if session is not None:
                session.combined_path = combined_path
                session.sha256 = sha256
                session.state = state

    def get(self, robot_id: str, task_id: str) -> ActionImitationStreamSession | None:
        with self._lock:
            session = self._sessions.get((robot_id, task_id))
            return replace(session) if session is not None else None


def register_session_request(
    registry: ActionImitationSessionRegistry,
    *,
    robot_id: str | None,
    task_id: str | None,
    source_turn_id: str | None = "",
    source_request_id: str | None = "",
    function_call_id: str | None = "",
) -> tuple[int, dict, ActionImitationStreamSession | None]:
    """Apply the HTTP registration contract without depending on FastAPI."""
    normalized_robot_id = (robot_id or "").strip()
    normalized_task_id = (task_id or "").strip()
    if not normalized_robot_id or not normalized_task_id:
        return (
            400,
            {"ok": False, "reason": "missing_robot_id_or_task_id"},
            None,
        )

    result, session = registry.register(
        robot_id=normalized_robot_id,
        task_id=normalized_task_id,
        source_turn_id=(source_turn_id or "").strip(),
        source_request_id=(source_request_id or "").strip(),
        function_call_id=(function_call_id or "").strip(),
    )
    if result == "conflict":
        return (
            409,
            {
                "ok": False,
                "reason": "active_session_conflict",
                "active_task_id": session.task_id,
                "requested_task_id": normalized_task_id,
            },
            session,
        )

    response = {
        "ok": True,
        "robot_id": normalized_robot_id,
        "task_id": normalized_task_id,
        # Registration acknowledgement remains armed even if a late duplicate
        # arrives after internal processing has advanced.
        "state": "armed",
    }
    if result == "already_registered":
        response["reason"] = "already_registered"
    return 200, response, session
