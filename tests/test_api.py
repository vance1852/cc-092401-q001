from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path

from robot_trials.api import JsonApplication
from robot_trials.jsonio import load_json
from robot_trials.service import TrialService


ROOT = Path(__file__).resolve().parents[1]


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.service = TrialService(self.connection)
        self.app = JsonApplication(self.service)

    def tearDown(self) -> None:
        self.connection.close()

    def test_health(self) -> None:
        response = self.app.handle("GET", "/health")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.body["status"], "ok")

    def test_json_error_shape(self) -> None:
        response = self.app.handle("POST", "/users", body=b"not-json")
        self.assertEqual(response.status, 422)
        self.assertEqual(response.body["error"]["code"], "validation_failed")

    def test_user_route(self) -> None:
        payload = json.dumps({"user_id": "u1", "display_name": "操作员", "role": "operator"}).encode()
        response = self.app.handle("POST", "/users", body=payload)
        self.assertEqual(response.status, 201)
        self.assertEqual(response.body["role"], "operator")


class ExclusionReviewApiTests(unittest.TestCase):
    """从 HTTP 路由到持久化结果的复核输入契约回归。"""

    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.service = TrialService(self.connection)
        self.app = JsonApplication(self.service)
        for user_id, role in (
            ("operator", "operator"),
            ("stat", "statistician"),
            ("auditor", "auditor"),
        ):
            self.service.create_user(user_id, user_id, role)
        protocol = load_json(ROOT / "fixtures" / "demo_protocol.json")
        rows = [
            json.loads(line)
            for line in (ROOT / "fixtures" / "demo_observations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.service.register_robot("operator", "robot-a", "A 型", "厂商")
        self.service.register_build("operator", "build-a", "robot-a", "1.0", "b" * 64)
        self.service.publish_protocol("stat", protocol)
        self.service.create_batch("operator", "batch-a", "demo-delivery-v1", 1, "build-a")
        self.service.start_batch("operator", "batch-a", 1)
        self.service.import_observations("operator", "batch-a", "key-1", rows)
        # 额外克隆一行，保证七种畸形输入各自占用一条独立观测。
        extra_row = dict(rows[0], source_row="extra-6")
        self.service.import_observations("operator", "batch-a", "key-2", [extra_row])
        self.observation_ids = [
            row[0]
            for row in self.connection.execute(
                "SELECT observation_id FROM observations ORDER BY observation_id"
            ).fetchall()
        ]

    def tearDown(self) -> None:
        self.connection.close()

    def _open_exclusion(self, index: int) -> int:
        return self.service.request_exclusion(
            "operator", self.observation_ids[index], "现场记录失效"
        )["exclusion_id"]

    def _post_review(self, raw_body: bytes, actor: str = "stat", exclusion_id: int | None = None):
        target = f"/exclusions/{exclusion_id or self.exclusion_id}/review"
        return self.app.handle(
            "POST", target, headers={"X-Actor-Id": actor}, body=raw_body
        )

    def _snapshot(self, exclusion_id: int) -> tuple:
        return tuple(
            self.connection.execute(
                "SELECT status,reviewed_by,reviewed_at,review_note "
                "FROM exclusion_requests WHERE exclusion_id=?",
                (exclusion_id,),
            ).fetchone()
        )

    def _audit_count(self, exclusion_id: int) -> int:
        return self.connection.execute(
            "SELECT count(*) FROM audit_events WHERE entity_type='exclusion' AND entity_id=?",
            (str(exclusion_id),),
        ).fetchone()[0]

    PENDING = ("pending", None, None, None)

    def test_malformed_approve_is_rejected_and_leaves_no_trace(self) -> None:
        # 字符串、数字、null、数组、缺失字段都必须返回同一个稳定的校验错误。
        malformed_bodies = {
            "字符串 false": b'{"approve": "false"}',
            "字符串 true": b'{"approve": "true"}',
            "数字 0": b'{"approve": 0}',
            "数字 1": b'{"approve": 1}',
            "null": b'{"approve": null}',
            "数组": b'{"approve": [false]}',
            "缺失字段": b'{}',
        }
        for index, (label, body) in enumerate(malformed_bodies.items()):
            self.exclusion_id = self._open_exclusion(index)
            with self.subTest(label):
                response = self._post_review(body)
                self.assertEqual(response.status, 422)
                self.assertEqual(response.body["error"]["code"], "validation_failed")
                self.assertIn("approve", response.body["error"]["message"])
                self.assertEqual(self._snapshot(self.exclusion_id), self.PENDING)
                self.assertEqual(self._audit_count(self.exclusion_id), 0)

    def test_real_boolean_true_and_false_approve_and_reject(self) -> None:
        approved_id = self._open_exclusion(0)
        approved = self.app.handle(
            "POST", f"/exclusions/{approved_id}/review",
            headers={"X-Actor-Id": "stat"},
            body=json.dumps({"approve": True, "note": "证据充分"}).encode("utf-8"),
        )
        self.assertEqual(approved.status, 200)
        self.assertEqual(approved.body["status"], "approved")
        self.assertEqual(self._snapshot(approved_id)[0], "approved")
        self.assertEqual(self._snapshot(approved_id)[1], "stat")

        rejected_id = self._open_exclusion(1)
        rejected = self.app.handle(
            "POST", f"/exclusions/{rejected_id}/review",
            headers={"X-Actor-Id": "stat"},
            body=json.dumps({"approve": False, "note": "证据不足"}).encode("utf-8"),
        )
        self.assertEqual(rejected.status, 200)
        self.assertEqual(rejected.body["status"], "rejected")
        self.assertEqual(self._snapshot(rejected_id)[0], "rejected")

    def test_self_review_is_forbidden_and_leaves_no_trace(self) -> None:
        self.exclusion_id = self._open_exclusion(0)
        response = self._post_review(b'{"approve": true}', actor="operator")
        self.assertEqual(response.status, 403)
        self.assertEqual(response.body["error"]["code"], "forbidden")
        self.assertEqual(self._snapshot(self.exclusion_id), self.PENDING)
        self.assertEqual(self._audit_count(self.exclusion_id), 0)

    def test_report_confirms_malformed_request_left_no_business_trace(self) -> None:
        self.exclusion_id = self._open_exclusion(0)
        response = self._post_review(b'{"approve": "false"}')
        self.assertEqual(response.status, 422)

        report = self.app.handle(
            "GET", "/batches/batch-a/report", headers={"X-Actor-Id": "auditor"}
        )
        self.assertEqual(report.status, 200)
        entry = next(
            item for item in report.body["exclusions"]
            if item["exclusion_id"] == self.exclusion_id
        )
        self.assertEqual(entry["status"], "pending")
        self.assertIsNone(entry["reviewed_by"])


if __name__ == "__main__":
    unittest.main()
