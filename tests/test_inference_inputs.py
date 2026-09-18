import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lib.util.inference_inputs import load_bbox, load_visual_features
from scripts.validate_inputs import discover_sequences, validate_sequence


class InferenceInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.features = self.root / "vit_features.pt"

    def test_features_work_without_boxes_or_annotations(self):
        torch.save(torch.zeros(3, 1024), self.features)
        self.assertEqual(validate_sequence(self.root), [])
        self.assertEqual(discover_sequences(self.root), [self.root])

    def test_invalid_feature_payloads_are_rejected(self):
        for value in ({"features": torch.zeros(2, 1024)}, torch.zeros(0, 1024), torch.zeros(2, 512), torch.full((2, 1024), float("nan")), torch.zeros(2, 1024, dtype=torch.int64)):
            with self.subTest(kind=str(type(value))):
                torch.save(value, self.features)
                with self.assertRaises(ValueError):
                    load_visual_features(self.features)

    def test_invalid_boxes_and_frame_ids_are_rejected(self):
        path = self.root / "bbox.npy"
        boxes = np.array([[0, 0, 0, 8, 8, 1, 0, 0], [1, 0, 0, 8, 8, 1, 0, 0]], dtype=np.float32)
        np.save(path, boxes)
        self.assertEqual(load_bbox(path).shape, (2, 8))
        boxes[1, 0] = 0
        np.save(path, boxes)
        with self.assertRaisesRegex(ValueError, "frame IDs"):
            load_bbox(path)
        boxes[1, 0] = 1
        boxes[1, 3] = 0
        np.save(path, boxes)
        with self.assertRaisesRegex(ValueError, "positive"):
            load_bbox(path)

    def test_missing_features_in_raw_or_empty_sequences_are_reported(self):
        raw = self.root / "subject" / "raw"
        empty = self.root / "subject" / "empty"
        raw.mkdir(parents=True)
        empty.mkdir()
        (raw / "1.mp4").touch()
        self.assertEqual(discover_sequences(self.root), [empty, raw])
        self.assertTrue(any("vit_features.pt" in error for error in validate_sequence(empty)))

    def test_frame_count_mismatch_is_reported(self):
        torch.save(torch.zeros(3, 1024), self.features)
        np.save(self.root / "bbox.npy", np.array([[0, 0, 0, 8, 8, 1, 0, 0]]))
        self.assertTrue(any("frame-count mismatch" in error for error in validate_sequence(self.root)))


if __name__ == "__main__":
    unittest.main()
