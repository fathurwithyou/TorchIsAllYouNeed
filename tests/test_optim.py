import unittest
from typing import Any, cast

import numpy as np

from torchlike.optim import (
    Adam,
    Optimizer,
    SGD,
)
from torchlike.tensor import Tensor


class DummyOptimizer(Optimizer):
    def __init__(self, params):
        super().__init__(params, {"lr": 0.1})

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.param_groups[0]["lr"] * p.grad


class TestBaseOptimizer(unittest.TestCase):
    def test_init_and_repr_and_params(self):
        p = Tensor(np.array([1.0]), requires_grad=True)
        opt = DummyOptimizer([p])
        self.assertEqual(len(opt.params), 1)
        self.assertIn("Parameter Group 0", repr(opt))

    def test_init_validation(self):
        with self.assertRaises(TypeError):
            DummyOptimizer(cast(Any, Tensor([1.0])))
        with self.assertRaises(ValueError):
            DummyOptimizer([])
        with self.assertRaises(TypeError):
            DummyOptimizer([cast(Any, object())])

    def test_add_param_group_validation(self):
        p1 = Tensor(np.array([1.0]), requires_grad=True)
        p2 = Tensor(np.array([2.0]), requires_grad=True)
        opt = DummyOptimizer([p1])
        opt.add_param_group({"params": [p2], "lr": 0.2})
        self.assertEqual(len(opt.param_groups), 2)

        with self.assertRaises(ValueError):
            opt.add_param_group({"params": [p1]})
        with self.assertRaises(TypeError):
            opt.add_param_group({"params": [cast(Any, object())]})
        with self.assertRaises(ValueError):
            opt.add_param_group({"params": []})
        with self.assertRaises(TypeError):
            opt.add_param_group(cast(Any, []))
        with self.assertRaises(ValueError):
            opt.add_param_group({"lr": 0.1})

    def test_state_dict_and_load_state_dict(self):
        p = Tensor(np.array([1.0]), requires_grad=True)
        opt = DummyOptimizer([p])
        opt.state[p] = {"moment": np.array([0.5])}
        state = opt.state_dict()

        p2 = Tensor(np.array([3.0]), requires_grad=True)
        opt2 = DummyOptimizer([p2])
        opt2.load_state_dict(state)
        self.assertIn("moment", opt2.state[p2])
        np.testing.assert_array_equal(opt2.state[p2]["moment"], np.array([0.5]))

        with self.assertRaises(ValueError):
            opt2.load_state_dict({"state": {}, "param_groups": []})

    def test_step_runs(self):
        p = Tensor(np.array([1.0]), requires_grad=True)
        p.grad = np.array([1.0])
        opt = DummyOptimizer([p])
        opt.step()
        self.assertLess(p.data.item(), 1.0)


class TestSGD(unittest.TestCase):
    def test_step_without_regularization(self):
        p = Tensor(np.array([1.0, -1.0]), requires_grad=True)
        p.grad = np.array([0.1, -0.2])
        opt = SGD([p], lr=0.5)
        opt.step()
        np.testing.assert_allclose(p.data, np.array([0.95, -0.9]))

    def test_step_with_l1_l2(self):
        p = Tensor(np.array([1.0, -2.0]), requires_grad=True)
        p.grad = np.array([0.0, 0.0])
        opt = SGD([p], lr=0.1, l1_lambda=0.5, l2_lambda=0.2)
        opt.step()
        np.testing.assert_allclose(p.data, np.array([0.93, -1.91]))

    def test_zero_grad(self):
        p = Tensor(np.array([1.0]), requires_grad=True)
        p.grad = np.array([3.0])
        opt = SGD([p], lr=0.1)
        opt.zero_grad()
        np.testing.assert_array_equal(p.grad, np.array([0.0]))


class TestAdam(unittest.TestCase):
    def test_invalid_hyperparameters(self):
        p = Tensor(np.array([1.0]), requires_grad=True)
        with self.assertRaises(ValueError):
            Adam([p], beta1=1.0)
        with self.assertRaises(ValueError):
            Adam([p], beta2=1.0)
        with self.assertRaises(ValueError):
            Adam([p], lr=0.0)
        with self.assertRaises(ValueError):
            Adam([p], eps=0.0)

    def test_first_step_direction(self):
        p = Tensor(np.array([1.0, -1.0]), requires_grad=True)
        p.grad = np.array([0.1, -0.2])
        opt = Adam([p], lr=0.01, eps=1e-8)
        opt.step()
        np.testing.assert_allclose(p.data, np.array([0.99, -0.99]), atol=1e-6)

    def test_step_with_regularization(self):
        p = Tensor(np.array([1.0]), requires_grad=True)
        p.grad = np.array([0.0])
        opt = Adam([p], lr=0.01, l2_lambda=0.1)
        old = p.data.copy()
        opt.step()
        self.assertLess(p.data.item(), old.item())

    def test_zero_grad(self):
        p = Tensor(np.array([1.0]), requires_grad=True)
        p.grad = np.array([2.0])
        opt = Adam([p], lr=0.01)
        opt.zero_grad()
        np.testing.assert_array_equal(p.grad, np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
