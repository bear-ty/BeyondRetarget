import unittest

import numpy as np

from lib.util.contact_labels import normalize_contact_lr


class ContactLabelTests(unittest.TestCase):
    def test_two_channel_labels_are_preserved(self):
        contact = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        result = normalize_contact_lr(contact, expected_frames=2)
        np.testing.assert_array_equal(result, contact.astype(np.float32))
        self.assertEqual(result.dtype, np.float32)

    def test_legacy_labels_use_columns_six_and_seven(self):
        contact = np.zeros((3, 10), dtype=np.float32)
        contact[:, 6] = [0.0, 1.0, 0.5]
        contact[:, 7] = [1.0, 0.0, 0.25]
        np.testing.assert_array_equal(normalize_contact_lr(contact), contact[:, [6, 7]])

    def test_motionpro_labels_use_forefeet_not_ankles(self):
        contact = np.array([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.float64)
        result = normalize_contact_lr(contact, expected_frames=2)
        np.testing.assert_array_equal(result, [[0, 1], [1, 0]])
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)

    def test_invalid_shapes_values_and_lengths_are_rejected(self):
        invalid = [
            np.zeros(3),
            np.zeros((3, 1)),
            np.zeros((3, 3)),
            np.zeros((3, 7)),
            np.zeros((1, 2, 2)),
            np.array([[np.nan, 0.0]]),
            np.array([[1.1, 0.0]]),
        ]
        for contact in invalid:
            with self.subTest(shape=contact.shape):
                with self.assertRaises(ValueError):
                    normalize_contact_lr(contact)
        with self.assertRaises(ValueError):
            normalize_contact_lr(np.zeros((3, 2)), expected_frames=4)


if __name__ == "__main__":
    unittest.main()
