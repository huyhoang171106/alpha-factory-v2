import unittest

from wq_client import WQClient


class DummyResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class WQClientSubmissionTests(unittest.TestCase):
    def test_extract_submission_state_parser(self):
        self.assertEqual(WQClient._extract_submission_state({"submissionStatus": "ACCEPTED"}), "accepted")
        self.assertEqual(WQClient._extract_submission_state({"review": {"status": "rejected"}}), "rejected")
        self.assertEqual(WQClient._extract_submission_state({"status": "pending review"}), "submitted")
        self.assertEqual(WQClient._extract_submission_state({"foo": "bar"}), "unknown")

    def test_get_submission_decision_from_api_payload(self):
        client = object.__new__(WQClient)
        client._api_request = lambda method, url: DummyResponse(200, {"review": {"status": "accepted"}})
        decision, error_class, detail = client.get_submission_decision("ALPHA-X")
        self.assertEqual(decision, "accepted")
        self.assertEqual(error_class, "")
        self.assertEqual(detail, "")


if __name__ == "__main__":
    unittest.main()
