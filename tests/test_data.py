import unittest

import numpy as np

from torchlike.data import ArrayDataset, DataLoader


class TestArrayDataset(unittest.TestCase):
    def test_reshape_and_len(self):
        ds = ArrayDataset(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0]))
        self.assertEqual(ds.x.shape, (3, 1))
        self.assertEqual(ds.y.shape, (3, 1))
        self.assertEqual(len(ds), 3)

    def test_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ArrayDataset(np.array([1.0, 2.0]), np.array([1.0]))

    def test_batch(self):
        ds = ArrayDataset(np.arange(6).reshape(3, 2), np.array([0, 1, 0]))
        x, y = ds.batch(np.array([0, 2]))
        np.testing.assert_array_equal(x, np.array([[0.0, 1.0], [4.0, 5.0]]))
        np.testing.assert_array_equal(y, np.array([[0.0], [0.0]]))


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.ds = ArrayDataset(np.arange(10).reshape(5, 2), np.array([0, 1, 0, 1, 1]))

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            DataLoader(self.ds, batch_size=0)

    def test_iteration_no_shuffle(self):
        loader = DataLoader(self.ds, batch_size=2, shuffle=False)
        batches = list(loader)
        self.assertEqual(len(batches), 3)
        np.testing.assert_array_equal(
            batches[0][0].data, np.array([[0.0, 1.0], [2.0, 3.0]])
        )

    def test_iteration_shuffle_reproducible(self):
        l1 = DataLoader(self.ds, batch_size=2, shuffle=True, seed=123)
        l2 = DataLoader(self.ds, batch_size=2, shuffle=True, seed=123)
        b1 = [xb.data.copy() for xb, _ in l1]
        b2 = [xb.data.copy() for xb, _ in l2]
        self.assertEqual(len(b1), len(b2))
        for a, b in zip(b1, b2):
            np.testing.assert_array_equal(a, b)

    def test_len_drop_last(self):
        self.assertEqual(len(DataLoader(self.ds, batch_size=2, drop_last=False)), 3)
        self.assertEqual(len(DataLoader(self.ds, batch_size=2, drop_last=True)), 2)


if __name__ == "__main__":
    unittest.main()
