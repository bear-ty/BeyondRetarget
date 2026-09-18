import os
import tempfile
import unittest
from pathlib import Path

from scripts.run_multirobot_pipeline import (
    infer_stage_fresh,
    postprocess_stage_fresh,
    run_worker_payloads,
)


class PipelineCacheTests(unittest.TestCase):
    def test_empty_payloads_do_not_create_zero_process_pool(self):
        self.assertEqual(run_worker_payloads(None, lambda payload: payload, []), [])

    def test_feature_update_invalidates_inference_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = {"raw": root / "raw", "contact": root / "contact"}
            feature = root / "input" / "seq" / "vit_features.pt"
            raw = roots["raw"] / "seq" / "g1" / "g1_raw_pred.npz"
            contact = roots["contact"] / "seq" / "contact_label.npy"
            for path in (feature, raw, contact):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            os.utime(feature, ns=(1, 1))
            os.utime(raw, ns=(2, 2))
            os.utime(contact, ns=(2, 2))
            self.assertTrue(infer_stage_fresh("seq", roots, ["g1"], feature))
            os.utime(feature, ns=(3, 3))
            self.assertFalse(infer_stage_fresh("seq", roots, ["g1"], feature))

    def test_raw_or_contact_update_invalidates_postprocess_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = {
                "raw": root / "raw",
                "contact": root / "contact",
                "postprocess": root / "postprocess",
            }
            raw = roots["raw"] / "seq" / "g1" / "g1_raw_pred.npz"
            contact = roots["contact"] / "seq" / "contact_label.npy"
            postprocess = roots["postprocess"] / "seq" / "g1" / "g1_raw_pred.npz"
            for path in (raw, contact, postprocess):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            os.utime(raw, ns=(1, 1))
            os.utime(contact, ns=(1, 1))
            os.utime(postprocess, ns=(2, 2))
            self.assertTrue(postprocess_stage_fresh("seq", roots, ["g1"]))
            os.utime(contact, ns=(3, 3))
            self.assertFalse(postprocess_stage_fresh("seq", roots, ["g1"]))


if __name__ == "__main__":
    unittest.main()
