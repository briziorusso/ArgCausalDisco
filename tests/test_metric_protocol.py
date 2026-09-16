import json
from pathlib import Path
import tempfile
import unittest

from utils.metric_protocol import GRAPH_METRIC_PROTOCOL, ensure_metric_protocol


class MetricProtocolTests(unittest.TestCase):
    def test_fresh_and_current_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'metric_protocol.json'
            results = Path(directory) / 'summary.json'
            ensure_metric_protocol(marker, existing_paths=[results])
            results.write_text('{}')
            ensure_metric_protocol(marker, existing_paths=[results])
            self.assertEqual(json.loads(marker.read_text())['graph_metric_protocol'], GRAPH_METRIC_PROTOCOL)

    def test_old_results_are_not_relabelled_or_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'metric_protocol.json'
            results = Path(directory) / 'summary.json'
            results.write_text('{"old": true}')
            with self.assertRaisesRegex(ValueError, 'new results version'):
                ensure_metric_protocol(marker, existing_paths=[results])
            self.assertFalse(marker.exists())
            self.assertEqual(results.read_text(), '{"old": true}')

    def test_different_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'metric_protocol.json'
            marker.write_text('{"graph_metric_protocol": "legacy"}')
            with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
                ensure_metric_protocol(marker)
