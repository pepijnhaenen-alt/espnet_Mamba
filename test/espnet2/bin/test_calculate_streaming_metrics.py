import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[3] / "egs2" / "mamba" / "asr1" / "pyscripts" / "utils" / "calculate_streaming_metrics.py"
    spec = importlib.util.spec_from_file_location("calculate_streaming_metrics", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_log(path: Path, payloads):
    lines = []
    for payload in payloads:
        if isinstance(payload, dict):
            lines.append(f'2026-01-01 00:00:00,000 INFO: stream_event: {json.dumps(payload, sort_keys=True)}\n')
        else:
            lines.append(payload)
    path.write_text("".join(lines), encoding="utf-8")


class TestCalculateStreamingMetrics(unittest.TestCase):
    def test_aggregate_metrics_basic(self):
        mod = _load_module()
        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td) / "logs"
            log_dir.mkdir()

            _write_log(
                log_dir / "asr_inference.1.log",
                [
                    {"event": "update", "utt_id": "utt1", "token_count": 0, "lag_ms": 8.0, "emit_elapsed_ms": 100.0},
                    {"event": "update", "utt_id": "utt1", "token_count": 2, "lag_ms": 12.0, "emit_elapsed_ms": 250.0},
                    {"event": "first_token", "utt_id": "utt1", "first_token_latency_ms": 250.0},
                    {"event": "final", "utt_id": "utt1", "endpoint_delay_ms": 40.0},
                ],
            )
            _write_log(
                log_dir / "asr_inference.2.log",
                [
                    {"event": "update", "utt_id": "utt2", "token_count": 1, "lag_ms": 20.0, "emit_elapsed_ms": 300.0},
                    {"event": "first_token", "utt_id": "utt2", "first_token_latency_ms": 300.0},
                    {"event": "final", "utt_id": "utt2", "endpoint_delay_ms": 60.0},
                ],
            )

            summary = mod.aggregate_metrics(str(log_dir), "asr_inference")
            self.assertEqual(summary["num_log_files"], 2)
            self.assertEqual(summary["num_update_events"], 3)
            self.assertEqual(summary["num_utterances_with_ftl"], 2)
            self.assertEqual(summary["num_utterances_with_epd"], 2)

            # Only token-bearing updates are counted for lag by default.
            self.assertEqual(summary["lag_ms"]["count"], 2)
            self.assertEqual(summary["lag_ms"]["min"], 12.0)
            self.assertEqual(summary["lag_ms"]["max"], 20.0)
            self.assertEqual(summary["ftl_ms"]["median"], 275.0)
            self.assertEqual(summary["epd_ms"]["median"], 50.0)


    def test_ftl_fallback_from_update_when_first_token_missing(self):
        mod = _load_module()
        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td) / "logs"
            log_dir.mkdir()

            _write_log(
                log_dir / "asr_inference.1.log",
                [
                    {"event": "update", "utt_id": "utt1", "token_count": 0, "lag_ms": 2.0, "emit_elapsed_ms": 20.0},
                    {"event": "update", "utt_id": "utt1", "token_count": 3, "lag_ms": 4.0, "emit_elapsed_ms": 90.0},
                    {"event": "final", "utt_id": "utt1", "endpoint_delay_ms": 10.0},
                ],
            )

            summary = mod.aggregate_metrics(str(log_dir), "asr_inference")
            self.assertEqual(summary["num_first_token_events"], 0)
            self.assertEqual(summary["num_utterances_with_ftl"], 1)
            self.assertEqual(summary["ftl_ms"]["min"], 90.0)


    def test_include_empty_updates_and_malformed_lines(self):
        mod = _load_module()
        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td) / "logs"
            log_dir.mkdir()

            _write_log(
                log_dir / "asr_inference.1.log",
                [
                    "this line has no event\n",
                    "2026-01-01 00:00:00,000 INFO: stream_event: not-json\n",
                    {"event": "update", "utt_id": "utt1", "token_count": 0, "lag_ms": 7.0, "emit_elapsed_ms": 12.0},
                    {"event": "update", "utt_id": "utt1", "token_count": 1, "lag_ms": 5.0, "emit_elapsed_ms": 20.0},
                    {"event": "final", "utt_id": "utt1", "endpoint_delay_ms": 8.0},
                ],
            )

            summary_default = mod.aggregate_metrics(str(log_dir), "asr_inference")
            self.assertEqual(summary_default["lag_ms"]["count"], 1)
            self.assertEqual(summary_default["lag_ms"]["min"], 5.0)

            summary_all = mod.aggregate_metrics(
                str(log_dir),
                "asr_inference",
                include_empty_updates=True,
            )
            self.assertEqual(summary_all["lag_ms"]["count"], 2)
            self.assertEqual(summary_all["lag_ms"]["max"], 7.0)


if __name__ == "__main__":
    unittest.main()
