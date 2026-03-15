import unittest

import numpy as np

from torchlike.tensor import Tensor, _to_array, _unbroadcast


class TestTensorHelpers(unittest.TestCase):
    def test_to_array_converts_to_float64(self):
        arr = _to_array([1, 2, 3])
        self.assertEqual(arr.dtype, np.float64)
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

    def test_unbroadcast_reduces_to_target_shape(self):
        grad = np.ones((2, 3))
        reduced = _unbroadcast(grad, (1, 3))
        np.testing.assert_array_equal(reduced, np.array([[2.0, 2.0, 2.0]]))


class TestTensorCore(unittest.TestCase):
    def test_properties_and_repr(self):
        t = Tensor([[1, 2], [3, 4]], requires_grad=True)
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.ndim, 2)
        self.assertEqual(t.size, 4)
        self.assertIn("requires_grad=True", repr(t))

    def test_item_and_item_error(self):
        self.assertEqual(Tensor([5.0]).item(), 5.0)
        with self.assertRaises(ValueError):
            Tensor([1.0, 2.0]).item()

    def test_numpy_returns_copy(self):
        t = Tensor([1.0, 2.0])
        arr = t.numpy()
        arr[0] = 99.0
        self.assertEqual(t.data[0], 1.0)

    def test_zero_grad_and_detach(self):
        t = Tensor([1.0, 2.0], requires_grad=True)
        y = (t * 3.0).sum()
        y.backward()
        self.assertIsNotNone(t.grad)
        t.zero_grad()
        np.testing.assert_array_equal(t.grad, np.zeros_like(t.data))

        d = t.detach()
        self.assertFalse(d.requires_grad)
        d.data[0] = 42.0
        self.assertNotEqual(t.data[0], d.data[0])

    def test_add_sub_mul_div_backward(self):
        x = Tensor([2.0, 4.0], requires_grad=True)
        y = ((x + 1.0) - 2.0) * 3.0 / 2.0
        loss = y.sum()
        loss.backward()
        np.testing.assert_allclose(x.grad, np.array([1.5, 1.5]))

    def test_broadcast_backward(self):
        x = Tensor(np.ones((2, 3)), requires_grad=True)
        b = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
        out = (x + b).sum()
        out.backward()
        np.testing.assert_array_equal(x.grad, np.ones((2, 3)))
        np.testing.assert_array_equal(b.grad, np.array([[2.0, 2.0, 2.0]]))

    def test_neg_radd_rsub_rmul_rdiv(self):
        x = Tensor([2.0], requires_grad=True)
        y = -x + 3 + (10 - x) + (2 * x) + (8 / x)
        y.backward()
        expected = -1.0 - 1.0 + 2.0 - 8.0 / (2.0**2)
        self.assertAlmostEqual(x.grad.item(), expected)

    def test_pow_and_matmul_backward(self):
        x = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        w = Tensor(np.array([[3.0], [4.0]]), requires_grad=True)
        out = (x @ w) ** 2
        out.backward()
        np.testing.assert_allclose(x.grad, np.array([[22.0 * 3.0, 22.0 * 4.0]]))
        np.testing.assert_allclose(w.grad, np.array([[22.0 * 1.0], [22.0 * 2.0]]))

    def test_sum_mean_backward(self):
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        out = x.sum(axis=0).mean()
        out.backward()
        np.testing.assert_allclose(x.grad, np.full((2, 2), 0.5))

    def test_sum_backward_with_tuple_axes(self):
        x = Tensor(np.arange(24.0).reshape(2, 3, 4), requires_grad=True)
        out = x.sum(axis=(1, 2)).mean()
        out.backward()
        np.testing.assert_allclose(x.grad, np.full((2, 3, 4), 0.5))

    def test_mean_backward_with_negative_axis_and_keepdims(self):
        x = Tensor(np.arange(24.0).reshape(2, 3, 4), requires_grad=True)
        out = x.mean(axis=-1, keepdims=True).sum()
        out.backward()
        np.testing.assert_allclose(x.grad, np.full((2, 3, 4), 0.25))

    def test_log_exp_tanh_relu_sigmoid_backward(self):
        x1 = Tensor(np.array([2.0]), requires_grad=True)
        x1.log().backward()
        self.assertAlmostEqual(x1.grad.item(), 0.5)

        x2 = Tensor(np.array([1.0]), requires_grad=True)
        x2.exp().backward()
        self.assertAlmostEqual(x2.grad.item(), np.e)

        x3 = Tensor(np.array([0.0]), requires_grad=True)
        x3.tanh().backward()
        self.assertAlmostEqual(x3.grad.item(), 1.0)

        x4 = Tensor(np.array([-1.0, 2.0]), requires_grad=True)
        x4.relu().sum().backward()
        np.testing.assert_array_equal(x4.grad, np.array([0.0, 1.0]))

        x5 = Tensor(np.array([0.0]), requires_grad=True)
        x5.sigmoid().backward()
        self.assertAlmostEqual(x5.grad.item(), 0.25)

    def test_softmax_backward_sum_is_zero(self):
        x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
        out = x.softmax(axis=1).sum()
        out.backward()
        np.testing.assert_allclose(x.grad, np.zeros_like(x.data), atol=1e-12)

    def test_clamp_backward(self):
        x = Tensor(np.array([-1.0, 0.0, 2.0]), requires_grad=True)
        out = x.clamp(0.0, 1.0).sum()
        out.backward()
        np.testing.assert_array_equal(x.grad, np.array([0.0, 1.0, 0.0]))

    def test_backward_non_scalar_requires_grad_argument(self):
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        with self.assertRaises(RuntimeError):
            x.backward()

    def test_backward_non_requires_grad_noop(self):
        x = Tensor(np.array([1.0, 2.0]), requires_grad=False)
        x.backward(np.array([1.0, 1.0]))
        self.assertIsNone(x.grad)


if __name__ == "__main__":
    unittest.main()
