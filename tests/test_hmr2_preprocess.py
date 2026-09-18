import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lib.util.hmr2_preprocess import get_bbx_xys_from_xyxy, prepare_crops
from lib.util.person_track import complete_track, select_person_track
from lib.util.video_io import read_rgb_video, video_info, write_rgb_video
from lib.util.visual_feature_config import resolve_visual_feature_cfg


class EncoderPreprocessingTests(unittest.TestCase):
    def test_box_center_and_aspect_ratio(self):
        result = get_bbx_xys_from_xyxy(torch.tensor([[0., 0., 300., 200.], [20., 40., 80., 240.]]))
        torch.testing.assert_close(result, torch.tensor([[150., 100., 480.], [50., 140., 240.]]))

    def test_crop_pixel_grid_and_normalization(self):
        image = np.arange(256, dtype=np.uint8)[None, :, None].repeat(256, 0).repeat(3, 2)
        crop = prepare_crops(image[None], torch.tensor([[127.5, 127.5, 255.]]))
        expected = (torch.from_numpy(image).float() / 255 - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        torch.testing.assert_close(crop[0], expected.permute(2, 0, 1), rtol=0, atol=0)

    def test_invalid_crops_fail_explicitly(self):
        images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        for boxes in [torch.zeros(1, 3), torch.zeros(2, 3), torch.full((2, 3), float("nan"))]:
            with self.assertRaises(ValueError):
                prepare_crops(images, boxes)

    def test_single_detection_is_held_across_video(self):
        box = torch.tensor([[10., 20., 100., 200.]])
        result = complete_track([3], box, 8)
        torch.testing.assert_close(result, box.expand(8, -1))

    def test_gaps_are_interpolated_and_smoothed(self):
        corners = torch.tensor([[0., 10., 20., 30.], [10., 20., 30., 40.]])
        result = complete_track([0, 10], corners, 11)
        # A linear trajectory is unchanged away from the four-frame edge zone.
        torch.testing.assert_close(result[4:7], corners[:1] + torch.arange(4., 7.)[:, None])
        self.assertTrue(torch.all(result[1:] >= result[:-1]))
        with self.assertRaises(ValueError):
            complete_track([4, 2], corners, 11)

    def test_person_selection_uses_total_visible_area(self):
        def frame(ids, boxes):
            return SimpleNamespace(boxes=SimpleNamespace(id=torch.tensor(ids), xyxy=torch.tensor(boxes, dtype=torch.float32)))
        result = select_person_track([
            frame([1, 2], [[0, 0, 20, 20], [0, 0, 10, 10]]),
            frame([2], [[0, 0, 10, 10]]),
        ], 2, 100, 100)
        torch.testing.assert_close(result, torch.tensor([[0., 0., 20., 20.]]).expand(2, -1))
        with self.assertRaises(RuntimeError):
            select_person_track([], 2, 100, 100)

    def test_video_writer_reader_and_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mp4"
            frames = np.zeros((3, 32, 48, 3), dtype=np.uint8)
            frames[:, :, :, 0] = 200
            write_rgb_video(frames, path)
            self.assertEqual(video_info(path), (3, 48, 32))
            decoded = read_rgb_video(path)
            self.assertEqual(decoded.shape, frames.shape)
            self.assertGreater(decoded[:, :, :, 0].mean(), 190)
            self.assertEqual(read_rgb_video(path, scale=.5).shape, (3, 16, 24, 3))

    def test_qualified_legacy_feature_names_normalize(self):
        self.assertEqual(resolve_visual_feature_cfg({"type": "legacy_hmr2_1024"})["type"], "hmr2_1024")
        with self.assertRaises(ValueError):
            resolve_visual_feature_cfg({"type": "unknown"})


if __name__ == "__main__":
    unittest.main()
