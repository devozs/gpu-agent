import unittest
from unittest.mock import Mock

import requests

from devozs_gpu_agent.agent import _ensure_enrolled


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(response=response)


class EnrollmentTest(unittest.TestCase):
    def config(self, enrollment_code="replacement-code"):
        config = Mock()
        config.enroll_code = enrollment_code
        config.mgmt_url = "http://management/api"
        config.load_token.return_value = "cached-token"
        config.stub = True
        config.name = "test-host"
        config.type = "HPU"
        return config

    def test_reuses_valid_cached_token(self):
        config = self.config()
        client = Mock()
        client.token = None
        client.heartbeat.return_value = {"ack": True}

        _ensure_enrolled(config, client)

        self.assertEqual(client.token, "cached-token")
        client.heartbeat.assert_called_once_with("IDLE")
        client.enroll.assert_not_called()
        config.clear_token.assert_not_called()

    def test_replaces_cached_token_rejected_with_401(self):
        config = self.config()
        client = Mock()
        client.token = None
        client.heartbeat.side_effect = http_error(401)
        client.enroll.return_value = {
            "token": "new-token",
            "name": "gaudi-host",
            "resourceId": "resource-id",
        }

        _ensure_enrolled(config, client)

        config.clear_token.assert_called_once_with()
        client.enroll.assert_called_once()
        self.assertEqual(client.enroll.call_args.args[0], "replacement-code")
        config.save_token.assert_called_once_with("new-token")
        self.assertEqual(client.token, "new-token")

    def test_keeps_rejected_token_when_no_replacement_code_exists(self):
        config = self.config(enrollment_code=None)
        client = Mock()
        client.token = None
        client.heartbeat.side_effect = http_error(401)

        with self.assertRaisesRegex(SystemExit, "Re-issue"):
            _ensure_enrolled(config, client)

        config.clear_token.assert_not_called()
        client.enroll.assert_not_called()

    def test_keeps_token_on_transient_network_failure(self):
        config = self.config()
        client = Mock()
        client.token = None
        client.heartbeat.side_effect = requests.ConnectionError("offline")

        with self.assertRaises(requests.ConnectionError):
            _ensure_enrolled(config, client)

        config.clear_token.assert_not_called()
        client.enroll.assert_not_called()

    def test_keeps_token_on_server_failure(self):
        config = self.config()
        client = Mock()
        client.token = None
        client.heartbeat.side_effect = http_error(503)

        with self.assertRaises(requests.HTTPError):
            _ensure_enrolled(config, client)

        config.clear_token.assert_not_called()
        client.enroll.assert_not_called()


if __name__ == "__main__":
    unittest.main()
