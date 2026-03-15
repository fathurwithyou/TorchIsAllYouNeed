import os
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import matplotlib
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)

from torchlike import Tensor, nn
from torchlike.nn.ffnn import _clone_activation
from torchlike.nn.linear import _initialize
from torchlike.nn.module import Module


class DummyModule(Module):
    def __init__(self):
        self.param = Tensor(np.array([1.0]), requires_grad=True)
        self.non_param = Tensor(np.array([2.0]), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.param


class ContainerModule(Module):
    def __init__(self):
        self.sub = DummyModule()
        shared = self.sub.param
        self.list_params = [shared]
        self.dict_params = {"k": shared}

    def forward(self, x: Tensor) -> Tensor:
        return self.sub(x)


class TestModuleBase(unittest.TestCase):
    def test_forward_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Module().forward()

    def test_parameters_deduplicated_and_zero_grad(self):
        mod = ContainerModule()
        params = mod.parameters()
        self.assertEqual(len(params), 1)

        x = Tensor(np.array([2.0]))
        y = mod(x).sum()
        y.backward()
        self.assertIsNotNone(params[0].grad)
        mod.zero_grad()
        np.testing.assert_array_equal(params[0].grad, np.zeros_like(params[0].data))

    def test_save_and_load(self):
        model = nn.Sequential([nn.Linear(2, 1, seed=1)])
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.pkl"
            model.save(p)
            loaded = nn.Sequential.load(p)
            self.assertIsInstance(loaded, nn.Sequential)

            p2 = Path(tmpdir) / "tensor.pkl"
            with p2.open("wb") as f:
                pickle.dump(Tensor([1.0]), f)
            with self.assertRaises(TypeError):
                nn.Sequential.load(p2)


class TestLinearAndRMSNorm(unittest.TestCase):
    def test_initialize_methods_and_errors(self):
        rng = np.random.default_rng(0)
        self.assertEqual(
            _initialize((2, 3), method="zero", params={}, rng=rng).shape, (2, 3)
        )
        self.assertEqual(
            _initialize(
                (2, 3),
                method="uniform",
                params={"lower_bound": -1, "upper_bound": 1},
                rng=rng,
            ).shape,
            (2, 3),
        )
        self.assertEqual(
            _initialize(
                (2, 3), method="normal", params={"mean": 0, "variance": 1}, rng=rng
            ).shape,
            (2, 3),
        )
        self.assertEqual(
            _initialize((2, 3), method="xavier", params={}, rng=rng).shape, (2, 3)
        )
        self.assertEqual(
            _initialize((2, 3), method="xavier_normal", params={}, rng=rng).shape,
            (2, 3),
        )
        self.assertEqual(
            _initialize((2, 3), method="he", params={}, rng=rng).shape, (2, 3)
        )
        self.assertEqual(
            _initialize((2, 3), method="he_uniform", params={}, rng=rng).shape, (2, 3)
        )

        with self.assertRaises(ValueError):
            _initialize((2, 3), method="normal", params={"variance": -1}, rng=rng)
        with self.assertRaises(ValueError):
            _initialize((2, 3), method="unknown", params={}, rng=rng)

    def test_linear_forward_and_bias_flag(self):
        layer = nn.Linear(2, 1, bias=False, init="zero")
        self.assertIsNone(layer.bias)
        out = layer(Tensor(np.array([[1.0, 2.0]])))
        np.testing.assert_array_equal(out.data, np.array([[0.0]]))

        layer2 = nn.Linear(2, 1, seed=42)
        out2 = layer2(Tensor(np.array([[1.0, 2.0]])))
        self.assertEqual(out2.shape, (1, 1))

    def test_rmsnorm_forward_and_errors(self):
        with self.assertRaises(ValueError):
            nn.RMSNorm(0)

        norm = nn.RMSNorm(3)
        x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
        out = norm(x)
        self.assertEqual(out.shape, (1, 3))

        norm_no_affine = nn.RMSNorm(3, affine=False)
        self.assertIsNone(norm_no_affine.weight)

        with self.assertRaises(ValueError):
            norm(Tensor(np.array([[1.0, 2.0]])))


class TestActivationsAndLosses(unittest.TestCase):
    def test_activation_modules(self):
        x = Tensor(np.array([[-1.0, 0.0, 1.0]]))
        for layer in [
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
            nn.Softmax(),
            nn.GELU(),
            nn.SELU(),
        ]:
            out = layer(x)
            self.assertEqual(out.shape, x.shape)

    def test_losses(self):
        pred = Tensor(np.array([[0.9], [0.2]]), requires_grad=True)
        target = Tensor(np.array([[1.0], [0.0]]))

        self.assertGreater(nn.MSELoss()(pred, target).item(), 0.0)
        self.assertGreater(nn.BCELoss()(pred, target).item(), 0.0)

        logits = Tensor(np.array([[2.0, 0.0], [0.0, 2.0]]), requires_grad=True)
        idx_target = np.array([0, 1])
        oh_target = np.array([[1.0, 0.0], [0.0, 1.0]])
        ce = nn.CrossEntropyLoss()
        self.assertAlmostEqual(
            ce(logits, idx_target).item(), ce(logits, oh_target).item()
        )

        with self.assertRaises(ValueError):
            ce(logits, np.array([[[1.0]]]))

        bce_logits = nn.BCEWithLogitsLoss()
        self.assertAlmostEqual(
            bce_logits(Tensor(np.array([[0.0]])), Tensor(np.array([[1.0]]))).item(),
            float(np.log(2.0)),
        )

    def test_loss_base_reduction_validation(self):
        with self.assertRaises(ValueError):
            nn.MSELoss(reduction=cast(Any, "bad"))


class TestSequential(unittest.TestCase):
    def test_forward_len_getitem(self):
        model = nn.Sequential([nn.Linear(2, 2, init="zero"), nn.ReLU()])
        self.assertEqual(len(model), 2)
        self.assertIsInstance(model[0], nn.Linear)
        out = model(Tensor(np.array([[1.0, -1.0]])))
        np.testing.assert_array_equal(out.data, np.array([[0.0, 0.0]]))

    def test_plot_distribution(self):
        model = nn.Sequential(
            [nn.Linear(2, 3, seed=1), nn.ReLU(), nn.Linear(3, 1, seed=2)]
        )
        x = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        y = model(x).sum()
        y.backward()

        with self.assertRaises(ValueError):
            model.plot_weight_distribution([])
        with self.assertRaises(IndexError):
            model.plot_weight_distribution([99])

        with patch("torchlike.nn.sequential.plt.show"):
            model.plot_weight_distribution([0, 1])
            model.plot_gradient_distribution([0, 1])


class TestFFNN(unittest.TestCase):
    def test_clone_activation_variants(self):
        relu = nn.ReLU()
        self.assertIs(_clone_activation(relu), relu)
        self.assertIsInstance(_clone_activation(nn.Sigmoid), nn.Sigmoid)
        with self.assertRaises(TypeError):
            _clone_activation(cast(Any, 123))

    def test_builds_from_layer_sizes_and_activation_names(self):
        model = nn.FFNN(
            [3, 4, 2],
            activations=["relu", "softmax"],
            init="uniform",
            seed=7,
        )
        self.assertEqual(model.layer_sizes, [3, 4, 2])
        self.assertEqual(len(model.layers), 4)
        self.assertIsInstance(model.layers[0], nn.Linear)
        self.assertIsInstance(model.layers[1], nn.ReLU)
        self.assertIsInstance(model.layers[2], nn.Linear)
        self.assertIsInstance(model.layers[3], nn.Softmax)

    def test_linear_activation_and_autodiff(self):
        model = nn.FFNN([2, 3, 1], activations=["linear", None], init="normal", seed=3)
        x = Tensor(np.array([[0.5, -1.5], [1.0, 2.0]]), requires_grad=True)
        target = Tensor(np.array([[0.0], [1.0]]))

        out = model(x)
        self.assertEqual(out.shape, (2, 1))

        loss = nn.MSELoss()(out, target)
        loss.backward()

        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertEqual(param.grad.shape, param.data.shape)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            nn.FFNN([4])
        with self.assertRaises(ValueError):
            nn.FFNN([2, 0, 1])
        with self.assertRaises(ValueError):
            nn.FFNN([2, 3, 1], activations=["relu"])
        with self.assertRaises(ValueError):
            nn.FFNN([2, 3, 1], activations=["unknown", None])


if __name__ == "__main__":
    unittest.main()
