import unittest

from lib.model.cross_window_attention import CrossWindowAttentionBlock
from lib.model.rgb2robo_temporal_encoder import RGB2RoboTemporalEncoder


class CrossWindowCompatibilityTests(unittest.TestCase):
    def test_explicit_position_length_matches_released_checkpoint(self):
        block = CrossWindowAttentionBlock(
            d_model=16,
            n_heads=4,
            context_span=8,
            max_position_length=66,
        )

        self.assertEqual(tuple(block.pos_encoding.pe.shape), (1, 66, 16))

    def test_encoder_derives_position_length_from_window_and_context(self):
        encoder = RGB2RoboTemporalEncoder(
            enable_cross_window=True,
            cross_window_config={
                "window_length": 50,
                "context_span": 8,
                "attention": {
                    "num_heads": 4,
                    "dropout": 0.0,
                    "bidirectional": False,
                },
            },
            g1_dof=2,
            input_feature_dim=16,
            model_feature_dim=16,
            frontend_config={"visual_adapter": {"enabled": False}},
            g1_head_config={"hidden_dim": 16, "num_layers": 1},
        )

        self.assertEqual(tuple(encoder.img_cross_window.pos_encoding.pe.shape), (1, 66, 16))
        self.assertEqual(encoder.img_cross_window.n_heads, 4)
        self.assertFalse(encoder.img_cross_window.bidirectional)
        self.assertEqual(encoder.cross_window_config["max_position_length"], 66)


if __name__ == "__main__":
    unittest.main()
