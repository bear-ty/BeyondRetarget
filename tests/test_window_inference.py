import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch

from lib.util.window_inference import build_context_kv
from stream_infer.streaming_pipeline import StreamMotionRunner


class ContextLayoutTests(unittest.TestCase):
    def setUp(self):
        self.features = torch.arange(1, 7, dtype=torch.float32).unsqueeze(1)
        self.starts = list(range(6))

    def assert_context(self, result, values, valid):
        context, mask = result
        torch.testing.assert_close(context["img_kv"][0, :, 0], torch.tensor(values, dtype=torch.float32))
        self.assertEqual(mask[0].tolist(), valid)

    def test_first_window_has_only_future_context(self):
        self.assert_context(
            build_context_kv(self.features, self.starts, 0, 3, True, 1),
            [0, 0, 0, 2, 3, 4], [False, False, False, True, True, True],
        )

    def test_partial_past_keeps_nearest_offset(self):
        self.assert_context(
            build_context_kv(self.features, self.starts, 1, 3, True, 1),
            [0, 0, 1, 3, 4, 5], [False, False, True, True, True, True],
        )

    def test_full_past_and_partial_future(self):
        self.assert_context(
            build_context_kv(self.features, self.starts, 4, 3, True, 1),
            [2, 3, 4, 6, 0, 0], [True, True, True, True, False, False],
        )

    def test_causal_context_and_empty_cases(self):
        self.assert_context(
            build_context_kv(self.features, self.starts, 1, 3, False, 1),
            [0, 0, 1], [False, False, True],
        )
        for index, span, bidirectional, starts in (
            (0, 3, False, self.starts), (0, 0, True, self.starts), (0, 3, True, [0]),
        ):
            self.assertEqual(build_context_kv(self.features, starts, index, span, bidirectional, 1), (None, None))

    def test_streaming_causal_never_uses_future(self):
        for bidirectional in (True, False):
            runner = SimpleNamespace(context_span=3, bidirectional=bidirectional, window_length=1)
            tail = [0, 0, 0] if bidirectional else []
            self.assert_context(
                StreamMotionRunner._build_stream_context(runner, self.features, self.starts, 1, "causal"),
                [0, 0, 1] + tail, [False, False, True] + [False] * len(tail),
            )
            self.assertEqual(
                StreamMotionRunner._build_stream_context(runner, self.features, self.starts, 0, "causal"),
                (None, None),
            )

    def test_live_camera_cached_history(self):
        # The camera runner is local to main; extract its method without opening devices.
        source = Path(__file__).resolve().parents[1] / "stream_infer/run_live_camera.py"
        tree = ast.parse(source.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_build_context")
        namespace = dict(torch=torch, Optional=Optional, Dict=Dict, Tuple=Tuple)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        runner = SimpleNamespace(context_span=3, window_means=[torch.tensor([1.])],
                                 feature_dim=1, model_dtype=torch.float32, device="cpu")
        self.assert_context(namespace["_build_context"](runner), [0, 0, 1], [False, False, True])
        runner.window_means = []
        self.assertEqual(namespace["_build_context"](runner), (None, None))


if __name__ == "__main__":
    unittest.main()
