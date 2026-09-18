import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stream_infer.run_streaming_video import main, parse_args, prepare_output_dir


class StreamingOutputTests(unittest.TestCase):
    def test_each_output_is_protected_even_after_partial_run(self):
        for name in ("bbox_stream.npy", "vit_features.pt", "stream_benchmark.json",
                     "g1_raw_pred.npz", "g1_export_meta.json", "stream_meta.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / name
                output.write_bytes(b"existing result")
                with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                    prepare_output_dir(root, overwrite=False)
                self.assertEqual(output.read_bytes(), b"existing result")
                prepare_output_dir(root, overwrite=True)
                # Permission to overwrite must not delete results before inference succeeds.
                self.assertEqual(output.read_bytes(), b"existing result")

    def test_new_directory_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "new"
            prepare_output_dir(root, overwrite=False)
            unrelated = root / "notes.txt"
            unrelated.write_text("keep")
            prepare_output_dir(root, overwrite=False)
            prepare_output_dir(root, overwrite=True)
            self.assertEqual(unrelated.read_text(), "keep")

    def test_cli_checks_collision_before_loading_models_or_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "g1_raw_pred.npz").touch()
            argv = ["run_streaming_video", "--video_path", "missing.mp4", "--output_dir", tmp]
            with patch("sys.argv", argv), self.assertRaisesRegex(FileExistsError, "--overwrite"):
                main()
            with patch("sys.argv", argv + ["--overwrite"]):
                self.assertTrue(parse_args().overwrite)


if __name__ == "__main__":
    unittest.main()
