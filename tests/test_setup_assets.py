import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.download_rgb2robo_assets import (
    ASSETS,
    EXPECTED_ROBOT_DIRS,
    Asset,
    _parse_content_range,
    _safe_zip_members,
    download_drive_file,
    install_robot_bundle,
    parse_drive_folder_html,
    resolve_assets,
    validate_asset_file,
)


class SetupAssetTests(unittest.TestCase):
    def test_eight_robot_bundle_installs_without_romeo(self):
        self.assertEqual(len(EXPECTED_ROBOT_DIRS), 8)
        self.assertNotIn("romeo", EXPECTED_ROBOT_DIRS)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "assets").mkdir()
            with zipfile.ZipFile(root / "assets/robot.zip", "w") as archive:
                for name in EXPECTED_ROBOT_DIRS:
                    archive.writestr(f"robot/{name}/LICENSE", "test license")
            install_robot_bundle(root)
            for name in EXPECTED_ROBOT_DIRS:
                self.assertTrue((root / "assets/robot" / name / "LICENSE").is_file())
            with zipfile.ZipFile(root / "assets/robot.zip", "a") as archive:
                archive.writestr("robot/romeo/romeo.urdf", "excluded")
            with self.assertRaisesRegex(RuntimeError, "excluded robot directories: romeo"):
                install_robot_bundle(root)

    def test_drive_folder_parser_extracts_name_id_and_size(self):
        row = [None] * 14
        row[0] = "file-id"
        row[2] = r"epoch\u003d10-step\u003d25000.ckpt"
        row[13] = 123
        payload = json.dumps([[row], None, None, None, None, None], separators=(",", ":"))
        encoded = "".join(f"\\x{ord(char):02x}" if char in '[]"' else char for char in payload)
        html = f"<script>window['_DRIVE_ivd'] = '{encoded}';if (window['_DRIVE_ivdc']) {{}}</script>"

        entries = parse_drive_folder_html(html)

        self.assertEqual(entries["epoch=10-step=25000.ckpt"], {"id": "file-id", "size": 123})

    def test_asset_validator_rejects_symbolic_link(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "target.bin"
            target.write_bytes(b"data")
            link = root / "asset.bin"
            link.symlink_to(target)
            asset = Asset("asset.bin", "asset.bin", size=4)

            self.assertIn("symbolic link", validate_asset_file(link, asset))

    def test_robot_zip_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "robot.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("robot/../outside.txt", "unsafe")
            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaisesRegex(RuntimeError, "unsafe path"):
                    list(_safe_zip_members(archive))

    def test_resolver_rejects_wrong_remote_size_before_download(self):
        entries = {"rgb2robo_multirobot.pth": {"id": "bad-file", "size": 1}}
        expected_size = next(asset.size for asset in ASSETS if asset.remote_name == "rgb2robo_multirobot.pth")
        with patch("scripts.download_rgb2robo_assets.fetch_drive_folder", return_value=entries):
            with self.assertRaisesRegex(RuntimeError, f"expected {expected_size}"):
                list(resolve_assets("https://example.invalid/folder", None))

    def test_content_range_parser_requires_exact_range_metadata(self):
        self.assertEqual(_parse_content_range("bytes 5-9/10"), (5, 9, 10))
        with self.assertRaisesRegex(RuntimeError, "invalid Google Drive Content-Range"):
            _parse_content_range("bytes */10")

    def test_range_downloader_resumes_only_matching_partial_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "asset.part"
            output.write_bytes(b"abcd")
            metadata = {"file_id": "file-id", "size": 10, "fingerprint": "sha"}
            output.with_name(output.name + ".meta.json").write_text(
                json.dumps(metadata, sort_keys=True),
                encoding="utf-8",
            )
            with patch(
                "scripts.download_rgb2robo_assets._request_drive_range",
                return_value=b"efghij",
            ) as request_range:
                download_drive_file("file-id", output, 10, "sha")

            request_range.assert_called_once_with("file-id", 4, 9, 10)
            self.assertEqual(output.read_bytes(), b"abcdefghij")
            self.assertFalse(output.with_name(output.name + ".meta.json").exists())


if __name__ == "__main__":
    unittest.main()
