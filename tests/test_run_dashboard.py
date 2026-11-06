import unittest
from unittest.mock import Mock, patch

import requests

import run_dashboard


class DashboardLauncherTests(unittest.TestCase):

    @patch("run_dashboard.time.sleep")
    @patch("run_dashboard.requests.get")
    def test_wait_for_api_retries_connection_errors(
        self,
        get,
        _sleep,
    ):
        api_process = Mock()
        api_process.poll.return_value = None
        response = Mock(status_code=200)
        response.json.return_value = {"products_loaded": 800}
        get.side_effect = [
            requests.exceptions.ConnectTimeout("connection timed out"),
            response,
        ]

        self.assertTrue(run_dashboard.wait_for_api(api_process, timeout=10))
        self.assertEqual(get.call_count, 2)

    @patch("run_dashboard.requests.get")
    def test_wait_for_api_reports_process_exit(self, get):
        api_process = Mock()
        api_process.poll.return_value = 1

        with patch("builtins.print") as print_output:
            self.assertFalse(
                run_dashboard.wait_for_api(api_process, timeout=10)
            )

        get.assert_not_called()
        self.assertIn(
            "exit code 1",
            " ".join(
                str(arg)
                for call in print_output.call_args_list
                for arg in call.args
            ),
        )

    @patch("run_dashboard.time.sleep")
    @patch("run_dashboard.time.monotonic", side_effect=[0, 0, 0, 2])
    @patch("run_dashboard.requests.get")
    def test_wait_for_api_reports_last_timeout(
        self,
        get,
        _monotonic,
        _sleep,
    ):
        api_process = Mock()
        api_process.poll.return_value = None
        get.side_effect = requests.exceptions.ConnectTimeout(
            "connection timed out"
        )

        with patch("builtins.print") as print_output:
            self.assertFalse(run_dashboard.wait_for_api(api_process, timeout=1))

        self.assertIn(
            "connection timed out",
            " ".join(
                str(arg)
                for call in print_output.call_args_list
                for arg in call.args
            ),
        )


if __name__ == "__main__":
    unittest.main()
