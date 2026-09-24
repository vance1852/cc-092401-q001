"""排除复核接口 approve 字段的严格输入契约回归测试。

覆盖一次畸形请求（approve="false" 被错误当成批准的事故）：
从 HTTP 路由到数据库持久化与后续报告，确认非布尔输入返回稳定校验错误，
且不留下任何业务痕迹（状态、复核人、时间、审计事件均不变）。
"""

from __future__ import annotations

import http.client
import json
import sqlite3
import threading
import unittest
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

from robot_trials.api import JsonApplication, make_handler
from robot_trials.clock import FrozenClock
from robot_trials.errors import Forbidden, ValidationFailed
from robot_trials.jsonio import load_json
from robot_trials.service import TrialService


ROOT = Path(__file__).resolve().parents[1]

MALFORMED_APPROVE_VALUES = [
    ("字符串 false", "false"),
    ("字符串 true", "true"),
    ("字符串 0", "0"),
    ("数字 0", 0),
    ("数字 1", 1),
    ("null", None),
    ("数组", []),
    ("对象", {}),
]


class ExclusionReviewContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.service = TrialService(self.connection, self.clock)
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
        self.observation_id = self.connection.execute(
            "SELECT observation_id FROM observations ORDER BY observation_id LIMIT 1"
        ).fetchone()[0]

    def tearDown(self) -> None:
        self.connection.close()

    def _request_exclusion(self) -> int:
        # 每个观测只允许一个待处理/生效排除；先合法驳回上一子用例残留的待处理申请。
        leftover = self.connection.execute(
            "SELECT exclusion_id FROM exclusion_requests WHERE observation_id=? AND status='pending'",
            (self.observation_id,),
        ).fetchone()
        if leftover is not None:
            self.service.review_exclusion("stat", leftover[0], False, "子用例清理")
        return self.service.request_exclusion("operator", self.observation_id, "现场记录失效")["exclusion_id"]

    def _post_review(self, exclusion_id: int, raw_body: bytes, actor: str = "stat"):
        return self.app.handle(
            "POST",
            f"/exclusions/{exclusion_id}/review",
            {"X-Actor-Id": actor},
            raw_body,
        )

    def _row(self, exclusion_id: int) -> sqlite3.Row:
        return self.connection.execute(
            "SELECT * FROM exclusion_requests WHERE exclusion_id=?", (exclusion_id,)
        ).fetchone()

    def _review_audit_events(self, exclusion_id: int) -> list[tuple[str, str]]:
        return [
            (row[0], row[1])
            for row in self.connection.execute(
                "SELECT event_type, actor_id FROM audit_events "
                "WHERE entity_type='exclusion' AND entity_id=? ORDER BY event_id",
                (str(exclusion_id),),
            ).fetchall()
        ]

    def test_malformed_approve_returns_stable_validation_error(self) -> None:
        for label, value in MALFORMED_APPROVE_VALUES:
            with self.subTest(label):
                exclusion_id = self._request_exclusion()
                response = self._post_review(exclusion_id, json.dumps({"approve": value}).encode("utf-8"))
                self.assertEqual(response.status, 422, label)
                self.assertEqual(response.body["error"]["code"], "validation_failed", label)

    def test_missing_approve_field_returns_validation_error(self) -> None:
        exclusion_id = self._request_exclusion()
        response = self._post_review(exclusion_id, json.dumps({"note": "无结论"}).encode("utf-8"))
        self.assertEqual(response.status, 422)
        self.assertEqual(response.body["error"]["code"], "validation_failed")

    def test_malformed_request_leaves_no_business_trace(self) -> None:
        # 事故复现：调用方把 JSON 字符串 "false" 传成 approve。
        exclusion_id = self._request_exclusion()
        response = self._post_review(exclusion_id, json.dumps({"approve": "false"}).encode("utf-8"))
        self.assertEqual(response.status, 422)

        row = self._row(exclusion_id)
        self.assertEqual(row["status"], "pending")
        self.assertIsNone(row["reviewed_by"])
        self.assertIsNone(row["reviewed_at"])
        self.assertIsNone(row["review_note"])
        self.assertEqual(self._review_audit_events(exclusion_id), [])

        # 操作人员可以通过后续报告确认申请仍是待复核，且没有复核痕迹。
        report = self.service.report("auditor", "batch-a")
        entry = next(item for item in report["exclusions"] if item["exclusion_id"] == exclusion_id)
        self.assertEqual(entry["status"], "pending")
        self.assertIsNone(entry["reviewed_by"])
        self.assertFalse(
            any(event["event_type"].startswith("exclusion.") and event["event_type"] != "exclusion.requested"
                for event in self.connection.execute(
                    "SELECT event_type FROM audit_events WHERE entity_type='observation'",
                ).fetchall())
        )

    def test_all_malformed_values_leave_state_untouched(self) -> None:
        for label, value in MALFORMED_APPROVE_VALUES:
            with self.subTest(label):
                exclusion_id = self._request_exclusion()
                self._post_review(exclusion_id, json.dumps({"approve": value}).encode("utf-8"))
                row = self._row(exclusion_id)
                self.assertEqual(row["status"], "pending", label)
                self.assertIsNone(row["reviewed_by"], label)
                self.assertIsNone(row["reviewed_at"], label)
                self.assertEqual(self._review_audit_events(exclusion_id), [], label)

    def test_true_approves_and_false_rejects(self) -> None:
        approve_id = self._request_exclusion()
        approved = self._post_review(
            approve_id, json.dumps({"approve": True, "note": "证据充分"}).encode("utf-8")
        )
        self.assertEqual(approved.status, 200)
        self.assertEqual(approved.body["status"], "approved")
        row = self._row(approve_id)
        self.assertEqual(row["status"], "approved")
        self.assertEqual(row["reviewed_by"], "stat")
        self.assertEqual(row["review_note"], "证据充分")
        self.assertIsNotNone(row["reviewed_at"])
        self.assertEqual(self._review_audit_events(approve_id), [("exclusion.approved", "stat")])

        # 第二个观测的排除申请走驳回路径。
        second_observation = self.connection.execute(
            "SELECT observation_id FROM observations ORDER BY observation_id LIMIT 1 OFFSET 1"
        ).fetchone()[0]
        reject_id = self.service.request_exclusion("operator", second_observation, "理由存疑")["exclusion_id"]
        rejected = self._post_review(reject_id, json.dumps({"approve": False}).encode("utf-8"))
        self.assertEqual(rejected.status, 200)
        self.assertEqual(rejected.body["status"], "rejected")
        row = self._row(reject_id)
        self.assertEqual(row["status"], "rejected")
        self.assertEqual(row["reviewed_by"], "stat")
        self.assertEqual(self._review_audit_events(reject_id), [("exclusion.rejected", "stat")])

    def test_applicant_cannot_review_own_request(self) -> None:
        exclusion_id = self._request_exclusion()
        # 合法布尔值也要先过职责分离：申请人复核自己直接 403，且状态不变。
        response = self._post_review(
            exclusion_id, json.dumps({"approve": True}).encode("utf-8"), actor="operator"
        )
        self.assertEqual(response.status, 403)
        self.assertEqual(response.body["error"]["code"], "forbidden")
        self.assertEqual(self._row(exclusion_id)["status"], "pending")
        self.assertEqual(self._review_audit_events(exclusion_id), [])

    def test_applicant_malformed_request_also_leaves_no_trace(self) -> None:
        exclusion_id = self._request_exclusion()
        # 申请人 + 畸形输入：请求失败（参数契约先拦截为 422），同样不留痕迹。
        response = self._post_review(
            exclusion_id, json.dumps({"approve": "false"}).encode("utf-8"), actor="operator"
        )
        self.assertEqual(response.status, 422)
        self.assertEqual(self._row(exclusion_id)["status"], "pending")
        self.assertEqual(self._review_audit_events(exclusion_id), [])

    def test_domain_layer_rejects_non_bool_approve(self) -> None:
        exclusion_id = self._request_exclusion()
        with self.assertRaises(ValidationFailed):
            self.service.review_exclusion("stat", exclusion_id, "false", "")
        self.assertEqual(self._row(exclusion_id)["status"], "pending")
        with self.assertRaises(Forbidden):
            self.service.review_exclusion("operator", exclusion_id, True, "")


class ExclusionReviewHttpWireTests(unittest.TestCase):
    """真实 HTTP 服务器上的端到端回归：畸形 JSON over the wire 不得产生业务痕迹。"""

    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        service = TrialService(
            self.connection, FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        )
        for user_id, role in (("operator", "operator"), ("stat", "statistician")):
            service.create_user(user_id, user_id, role)
        protocol = load_json(ROOT / "fixtures" / "demo_protocol.json")
        rows = [
            json.loads(line)
            for line in (ROOT / "fixtures" / "demo_observations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        service.register_robot("operator", "robot-a", "A 型", "厂商")
        service.register_build("operator", "build-a", "robot-a", "1.0", "b" * 64)
        service.publish_protocol("stat", protocol)
        service.create_batch("operator", "batch-a", "demo-delivery-v1", 1, "build-a")
        service.start_batch("operator", "batch-a", 1)
        service.import_observations("operator", "batch-a", "key-1", rows)
        observation_id = self.connection.execute(
            "SELECT observation_id FROM observations ORDER BY observation_id LIMIT 1"
        ).fetchone()[0]
        self.exclusion_id = service.request_exclusion("operator", observation_id, "现场记录失效")["exclusion_id"]

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(JsonApplication(service)))
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.connection.close()

    def test_string_false_over_http_is_rejected_and_not_persisted(self) -> None:
        client = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = json.dumps({"approve": "false"}).encode("utf-8")
        client.request("POST", f"/exclusions/{self.exclusion_id}/review", body=body,
                       headers={"Content-Type": "application/json", "X-Actor-Id": "stat"})
        response = client.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        self.assertEqual(response.status, 422)
        self.assertEqual(payload["error"]["code"], "validation_failed")
        client.close()

        row = self.connection.execute(
            "SELECT status,reviewed_by,reviewed_at FROM exclusion_requests WHERE exclusion_id=?",
            (self.exclusion_id,),
        ).fetchone()
        self.assertEqual(row["status"], "pending")
        self.assertIsNone(row["reviewed_by"])
        self.assertIsNone(row["reviewed_at"])
        self.assertEqual(
            self.connection.execute(
                "SELECT count(*) FROM audit_events WHERE entity_type='exclusion'"
            ).fetchone()[0],
            0,
        )


if __name__ == "__main__":
    unittest.main()
