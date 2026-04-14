import unittest
from unittest.mock import patch

import wq_client
from wq_client import WQClient


class DummyResponse:
    def __init__(self, status_code, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


class WQClientSimulateRetryTests(unittest.TestCase):
    def _bare_client(self):
        client = object.__new__(WQClient)
        client._recent_submit_failures = []
        client._circuit_open_until = None
        client._record_submit_outcome = lambda *_: None
        return client

    def test_concurrent_limit_detector(self):
        resp = DummyResponse(400, {"detail": "CONCURRENT_SIMULATION_LIMIT_EXCEEDED"})
        self.assertTrue(WQClient._is_concurrent_sim_limit(resp))
        self.assertFalse(WQClient._is_concurrent_sim_limit(DummyResponse(400, {"detail": "other"})))

    def test_log_helpers_redact_private_values(self):
        expr = "group_neutralize(rank(ts_delta(close, 12)), subindustry)"

        self.assertEqual(wq_client._mask_email("alpha@example.com"), "a***@example.com")
        self.assertNotIn(expr, wq_client._safe_expr_ref(expr))
        self.assertEqual(len(wq_client._safe_expr_ref(expr)), 12)
        self.assertNotIn("BRAIN-123", wq_client._safe_alpha_ref("BRAIN-123"))
        self.assertEqual(len(wq_client._safe_alpha_ref("BRAIN-123")), 12)

    def test_simulate_retries_when_concurrent_limit_hit(self):
        client = self._bare_client()
        calls = {"post": 0}

        def fake_api_request(method, url, **kwargs):
            if method == "post" and url.endswith("/simulations"):
                calls["post"] += 1
                if calls["post"] == 1:
                    return DummyResponse(400, {"detail": "CONCURRENT_SIMULATION_LIMIT_EXCEEDED"})
                return DummyResponse(201, {}, headers={"Location": "https://example/progress"})
            if method == "get" and url == "https://example/progress":
                return DummyResponse(200, {"alpha": "ALPHA-1"})
            if method == "get" and url.endswith("/alphas/ALPHA-1"):
                return DummyResponse(
                    200,
                    {
                        "is": {
                            "sharpe": 1.7,
                            "fitness": 1.2,
                            "turnover": 0.21,
                            "returns": 0.04,
                            "drawdown": 0.09,
                            "checks": [],
                        }
                    },
                )
            raise AssertionError(f"Unexpected request: {method} {url}")

        client._api_request = fake_api_request

        with patch.object(wq_client, "SIM_SUBMIT_RETRIES", 1), patch.object(wq_client.time, "sleep", lambda *_: None):
            result = client.simulate("rank(close)")

        self.assertEqual(calls["post"], 2)
        self.assertEqual(result.alpha_id, "ALPHA-1")
        self.assertEqual(result.error, "")


if __name__ == "__main__":
    unittest.main()
