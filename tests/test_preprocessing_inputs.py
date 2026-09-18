import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from lib.util.gen_bbox_feature import load_rgb_frames, validate_outputs


class PreprocessingInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "color").mkdir()
        for name in ("002.png", "001.jpg"):
            cv2.imwrite(str(self.root / "color" / name), np.zeros((8, 8, 3), dtype=np.uint8))
        self.bbox = self.root / "bbox.npy"
        self.features = self.root / "vit_features.pt"
        boxes = np.array([[0, 0, 0, 8, 8, 1, 0, 0], [1, 0, 0, 8, 8, 1, 0, 0]], dtype=np.float32)
        np.save(self.bbox, boxes)
        torch.save(torch.zeros(2, 1024), self.features)

    def test_images_are_validated_without_opening_a_video(self):
        with patch("lib.util.gen_bbox_feature.count_frames_from_video", side_effect=AssertionError("directory opened as video")):
            result = validate_outputs(self.root, self.bbox, self.features)
        self.assertEqual(result["source_frames"], 2)
        paths, frames = load_rgb_frames(self.root / "color")
        self.assertEqual([Path(path).name for path in paths], ["001.jpg", "002.png"])
        self.assertEqual(frames.shape, (2, 8, 8, 3))

    def test_image_frame_mismatch_is_rejected(self):
        torch.save(torch.zeros(1, 1024), self.features)
        with self.assertRaisesRegex(ValueError, "feature length mismatch"):
            validate_outputs(self.root, self.bbox, self.features)

    def test_video_input_keeps_video_frame_validation(self):
        video = self.root / "video.mp4"
        with patch("lib.util.gen_bbox_feature.count_frames_from_video", return_value=2) as count:
            result = validate_outputs(video, self.bbox, self.features)
        count.assert_called_once_with(video)
        self.assertEqual(result["source_frames"], 2)

    def test_empty_and_unreadable_images_report_source(self):
        for image in (self.root / "color").iterdir():
            image.unlink()
        with self.assertRaises(FileNotFoundError):
            validate_outputs(self.root, self.bbox, self.features)
        (self.root / "color" / "broken.png").write_bytes(b"invalid")
        with self.assertRaisesRegex(FileNotFoundError, "broken.png"):
            load_rgb_frames(self.root / "color")


if __name__ == "__main__":
    unittest.main()
