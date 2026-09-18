import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from stream_infer.streaming_pipeline import YoloBBoxStream


class StreamingBatchTests(unittest.TestCase):
    def make_detector(self, batch_size):
        detector = YoloBBoxStream.__new__(YoloBBoxStream)
        detector.engine = None
        detector._onnx_batch_size = batch_size
        detector.device = "cpu"
        detector.use_half = False
        detector.conf = 0.35
        detector.backend = "test"
        calls = []

        def predict(source, **kwargs):
            values = [int(frame[0, 0, 0]) for frame in source]
            calls.append(values)
            return values

        detector.model = SimpleNamespace(predict=predict)
        detector._records_from_ultralytics = lambda results, frames, ids: list(zip(ids, results))
        return detector, calls

    def test_static_batches_preserve_order_without_emitting_padding(self):
        for count in (0, 1, 2, 3, 8, 16):
            with self.subTest(count=count), patch("stream_infer.streaming_pipeline.sync_cuda"):
                detector, calls = self.make_detector(2)
                frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(count)]
                ids = list(range(100, 100 + count))
                records, timing = detector.detect_batch(frames, ids)
                self.assertEqual(records, list(zip(ids, range(count))))
                self.assertTrue(all(len(batch) == 2 for batch in calls))
                if count % 2:
                    self.assertEqual(calls[-1], [count - 1, count - 1])
                self.assertEqual(timing.extra["batch"], count)

    def test_dynamic_and_pytorch_inputs_are_not_split_or_padded(self):
        detector, calls = self.make_detector(None)
        frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(3)]
        with patch("stream_infer.streaming_pipeline.sync_cuda"):
            records, _ = detector.detect_batch(frames, [0, 1, 2])
        self.assertEqual(calls, [[0, 1, 2]])
        self.assertEqual(records, [(0, 0), (1, 1), (2, 2)])

    def test_cpu_onnx_reads_static_and_dynamic_batch_metadata(self):
        for size, expected in ((2, 2), ("batch", None)):
            detector, _ = self.make_detector(None)
            detector.yolo_ckpt = "test.onnx"
            session = SimpleNamespace(
                get_inputs=lambda: [SimpleNamespace(shape=[size, 3, 640, 640])],
                get_providers=lambda: ["CPUExecutionProvider"],
            )
            with patch("onnxruntime.InferenceSession", return_value=session):
                detector._select_onnx_device()
            self.assertEqual(detector._onnx_batch_size, expected)
            self.assertFalse(detector.use_half)


if __name__ == "__main__":
    unittest.main()
