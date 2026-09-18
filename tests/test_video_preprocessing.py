import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import preprocess_videos


class VideoPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sequence = self.root / "sample"
        self.sequence.mkdir()
        (self.sequence / "1.mp4").touch()
        self.args = SimpleNamespace(input_root=self.root, output_root=None, video_glob="1.mp4", overwrite=False, fps=30, yolo_ckpt="yolo.pt", hmr2_ckpt="hmr2.ckpt")
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.real_parse_args = preprocess_videos.parse_args
        stack.enter_context(patch.object(preprocess_videos, "parse_args", return_value=self.args))
        self.bbox = stack.enter_context(patch.object(preprocess_videos, "generate_bbox", return_value=("bbox.npy", None)))
        self.feature = stack.enter_context(patch.object(preprocess_videos, "generate_hmr2_feature", return_value="vit_features.pt"))
        stack.enter_context(patch.object(preprocess_videos, "validate_outputs", return_value={"source_frames": 2}))
        self.output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def test_video_alone_is_sufficient(self):
        preprocess_videos.main()
        self.bbox.assert_called_once()
        self.feature.assert_called_once()
        self.assertEqual(sorted(p.name for p in self.sequence.iterdir()), ["1.mp4"])
        self.assertIn("prepared=1 failed=0", self.output.getvalue())

    def test_default_selects_front_view(self):
        with patch("sys.argv", ["preprocess_videos.py", "--input_root", str(self.root)]):
            self.assertEqual(self.real_parse_args().video_glob, "1.mp4")

    def test_multiple_views_cannot_overwrite_same_features(self):
        (self.sequence / "0.mp4").touch()
        self.args.video_glob = "*.mp4"
        with self.assertRaisesRegex(ValueError, "Multiple videos"):
            preprocess_videos.main()
        self.bbox.assert_not_called()

    def test_output_root_separates_videos_and_preserves_relative_paths(self):
        (self.sequence / "0.mp4").touch()
        self.args.video_glob = "*.mp4"
        self.args.output_root = self.root / "prepared"
        preprocess_videos.main()
        destinations = [call.kwargs["output_dir"] for call in self.bbox.call_args_list]
        self.assertEqual(destinations, [str(self.args.output_root / "sample" / stem) for stem in ("0", "1")])

    def test_failed_video_returns_nonzero(self):
        self.bbox.side_effect = ValueError("no person")
        with self.assertRaises(SystemExit) as caught:
            preprocess_videos.main()
        self.assertEqual(caught.exception.code, 1)
        self.feature.assert_not_called()


if __name__ == "__main__":
    unittest.main()
