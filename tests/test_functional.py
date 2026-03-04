import unittest
from typing import Any, cast

import numpy as np

import torchlike.functional as F
from torchlike.tensor import Tensor
from torchlike.functional.losses import Reduction
from torchlike.functional.losses.binary_cross_entropy import binary_cross_entropy
from torchlike.functional.losses.cross_entropy import cross_entropy
from torchlike.functional.losses.cross_entropy_from_probs import (
    cross_entropy_from_probs,
)
from torchlike.functional.losses.l1_loss import l1_loss
from torchlike.functional.losses.mse_loss import mse_loss
from torchlike.functional.losses.reduce import reduce_losses


class TestFunctionalLayersAndActivations(unittest.TestCase):
    def test_linear_numpy_and_tensor(self):
        x = np.array([[1.0, 2.0]])
        w = np.array([[3.0], [4.0]])
        b = np.array([[1.0]])
        out_np = F.linear(x, w, b)
        np.testing.assert_array_equal(out_np, np.array([[12.0]]))

        x_t = Tensor(x, requires_grad=True)
        w_t = Tensor(w, requires_grad=True)
        b_t = Tensor(b, requires_grad=True)
        out_t = F.linear(x_t, w_t, b_t)
        self.assertIsInstance(out_t, Tensor)
        out_t.backward(np.array([[1.0]]))
        np.testing.assert_array_equal(x_t.grad, np.array([[3.0, 4.0]]))

    def test_activation_numpy(self):
        x = np.array([[-1.0, 0.0, 1.0]])
        np.testing.assert_array_equal(F.relu(x), np.array([[0.0, 0.0, 1.0]]))
        np.testing.assert_allclose(F.sigmoid(x), 1.0 / (1.0 + np.exp(-x)))
        np.testing.assert_allclose(F.tanh(x), np.tanh(x))
        sm = F.softmax(x, axis=1)
        np.testing.assert_allclose(np.sum(sm, axis=1), np.array([1.0]))

        gelu = F.gelu(x, alpha=2.0)
        np.testing.assert_allclose(gelu, x * (1.0 / (1.0 + np.exp(-(2.0 * x)))))

    def test_activation_tensor(self):
        x = Tensor(np.array([[-1.0, 0.0, 1.0]]), requires_grad=True)
        self.assertIsInstance(F.relu(x), Tensor)
        self.assertIsInstance(F.sigmoid(x), Tensor)
        self.assertIsInstance(F.tanh(x), Tensor)
        self.assertIsInstance(F.softmax(x), Tensor)
        self.assertIsInstance(F.gelu(x), Tensor)
        self.assertIsInstance(F.selu(x), Tensor)


class TestFunctionalLosses(unittest.TestCase):
    def test_reduce_losses(self):
        losses = np.array([1.0, 2.0, 3.0])
        self.assertEqual(reduce_losses(losses, "mean"), 2.0)
        self.assertEqual(reduce_losses(losses, "sum"), 6.0)
        np.testing.assert_array_equal(reduce_losses(losses, "none"), losses)
        with self.assertRaises(ValueError):
            reduce_losses(losses, cast(Any, "bad"))

    def test_mse_and_l1_loss(self):
        pred = np.array([1.0, 3.0])
        target = np.array([2.0, 1.0])
        self.assertEqual(mse_loss(pred, target), 2.5)
        self.assertEqual(l1_loss(pred, target), 1.5)

    def test_binary_cross_entropy(self):
        pred = np.array([0.8, 0.2])
        target = np.array([1.0, 0.0])
        expected = float(
            np.mean(-(target * np.log(pred) + (1 - target) * np.log(1 - pred)))
        )
        self.assertAlmostEqual(binary_cross_entropy(pred, target), expected)

    def test_cross_entropy_indices_and_onehot(self):
        logits = np.array([[2.0, 0.0], [0.0, 2.0]])
        target_idx = np.array([0, 1])
        target_oh = np.array([[1.0, 0.0], [0.0, 1.0]])
        ce_idx = cross_entropy(logits, target_idx)
        ce_oh = cross_entropy(logits, target_oh)
        self.assertAlmostEqual(ce_idx, ce_oh)

    def test_cross_entropy_errors(self):
        with self.assertRaises(ValueError):
            cross_entropy(np.array([1.0, 2.0]), np.array([0]))
        with self.assertRaises(ValueError):
            cross_entropy(np.array([[1.0, 2.0]]), np.array([[1.0, 0.0, 0.0]]))

    def test_cross_entropy_from_probs(self):
        probs = np.array([[0.8, 0.2], [0.1, 0.9]])
        target_idx = np.array([0, 1])
        target_oh = np.array([[1.0, 0.0], [0.0, 1.0]])
        ce_idx = cross_entropy_from_probs(probs, target_idx)
        ce_oh = cross_entropy_from_probs(probs, target_oh)
        self.assertAlmostEqual(ce_idx, ce_oh)

        with self.assertRaises(ValueError):
            cross_entropy_from_probs(np.array([0.5, 0.5]), target_idx)
        with self.assertRaises(ValueError):
            cross_entropy_from_probs(probs, np.array([[1.0, 0.0, 0.0]]))

    def test_functional_exports(self):
        expected = [
            "linear",
            "relu",
            "sigmoid",
            "tanh",
            "gelu",
            "selu",
            "softmax",
            "mse_loss",
            "l1_loss",
            "binary_cross_entropy",
            "cross_entropy",
            "cross_entropy_from_probs",
        ]
        for name in expected:
            self.assertTrue(hasattr(F, name))

    def test_reduction_type_alias_values(self):
        reduction: Reduction = "mean"
        self.assertEqual(reduction, "mean")


if __name__ == "__main__":
    unittest.main()
