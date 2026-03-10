import csv
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import matplotlib
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)

import main
from torchlike import Tensor, nn


def _write_small_dataset(csv_path: Path, n_rows: int = 20) -> None:
    headers = [
        "cgpa",
        "backlogs",
        "internship_count",
        "aptitude_score",
        "communication_score",
        "internship_quality_score",
        "college_tier",
        "country",
        "university_ranking_band",
        "specialization",
        "industry",
        "placement_status",
    ]

    tiers = ["Tier 1", "Tier 2", "Tier 3"]
    countries = ["USA", "India", "UK"]
    bands = ["Top 100", "100-300", "300+"]
    specs = ["AI/ML", "Core CS", "Cloud"]
    industries = ["Tech", "Finance", "Healthcare"]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for i in range(n_rows):
            writer.writerow(
                {
                    "cgpa": f"{7.0 + (i % 5) * 0.3:.2f}",
                    "backlogs": str(i % 3),
                    "internship_count": str(i % 4),
                    "aptitude_score": str(60 + (i % 20)),
                    "communication_score": str(55 + (i % 25)),
                    "internship_quality_score": str(50 + (i % 30)),
                    "college_tier": tiers[i % len(tiers)],
                    "country": countries[i % len(countries)],
                    "university_ranking_band": bands[i % len(bands)],
                    "specialization": specs[i % len(specs)],
                    "industry": industries[i % len(industries)],
                    "placement_status": "Placed" if i % 2 == 0 else "Not Placed",
                }
            )


def _fake_result(name: str) -> main.ExperimentResult:
    model = nn.Sequential([nn.Linear(2, 1, seed=1)])
    return main.ExperimentResult(
        name=name,
        model=model,
        train_losses=[0.6, 0.5],
        val_losses=[0.7, 0.6],
        val_accs=[0.7, 0.75],
        final_val_loss=0.6,
        final_val_acc=0.75,
        val_probs=np.array([[0.8], [0.2]]),
        val_preds=np.array([[1.0], [0.0]]),
        best_epoch=2,
    )


class TestMainUtilities(unittest.TestCase):
    def test_load_dataset_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "tiny.csv"
            _write_small_dataset(p, n_rows=20)
            x_train, x_val, y_train, y_val = main.load_dataset_features(
                p, split_seed=123
            )

            self.assertEqual(x_train.shape[0], 16)
            self.assertEqual(x_val.shape[0], 4)
            self.assertEqual(y_train.shape, (16, 1))
            self.assertEqual(y_val.shape, (4, 1))
            self.assertTrue(set(np.unique(y_train)).issubset({0.0, 1.0}))

    def test_load_dataset_features_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "empty.csv"
            with p.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "cgpa",
                        "backlogs",
                        "internship_count",
                        "aptitude_score",
                        "communication_score",
                        "internship_quality_score",
                        "college_tier",
                        "country",
                        "university_ranking_band",
                        "specialization",
                        "industry",
                        "placement_status",
                    ],
                )
                writer.writeheader()

            with self.assertRaises(ValueError):
                main.load_dataset_features(p)

    def test_build_binary_model(self):
        model_no = main.build_binary_model(10, use_rmsnorm=False)
        self.assertFalse(
            any(isinstance(layer, nn.RMSNorm) for layer in model_no.layers)
        )

        model_yes = main.build_binary_model(10, use_rmsnorm=True)
        self.assertTrue(
            any(isinstance(layer, nn.RMSNorm) for layer in model_yes.layers)
        )

    def test_evaluate_binary_and_attach_grad(self):
        model = main.build_binary_model(4, use_rmsnorm=False)
        criterion = nn.BCEWithLogitsLoss()
        x = np.random.default_rng(0).normal(size=(8, 4))
        y = (np.random.default_rng(1).random((8, 1)) > 0.5).astype(np.float64)

        loss, acc, probs, preds = main.evaluate_binary(model, x, y, criterion)
        self.assertIsInstance(loss, float)
        self.assertTrue(0.0 <= acc <= 1.0)
        self.assertEqual(probs.shape, y.shape)
        self.assertEqual(preds.shape, y.shape)

        main.attach_full_batch_gradients(model, x, y, criterion)
        for p in model.parameters():
            self.assertIsNotNone(p.grad)

    def test_create_optimizer(self):
        model = main.build_binary_model(3, use_rmsnorm=False)
        self.assertEqual(main.create_optimizer("sgd", model).__class__.__name__, "SGD")
        self.assertEqual(
            main.create_optimizer("adam", model).__class__.__name__, "Adam"
        )
        with self.assertRaises(ValueError):
            main.create_optimizer(cast(Any, "bad"), model)

    def test_train_experiment(self):
        rng = np.random.default_rng(42)
        x_train = rng.normal(size=(32, 6))
        y_train = (rng.random((32, 1)) > 0.5).astype(np.float64)
        x_val = rng.normal(size=(12, 6))
        y_val = (rng.random((12, 1)) > 0.5).astype(np.float64)

        result = main.train_experiment(
            name="unit",
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            use_rmsnorm=False,
            optimizer_name="sgd",
            epochs=6,
            batch_size=8,
            early_stopping_patience=2,
            min_delta=0.0,
        )

        self.assertGreaterEqual(len(result.train_losses), 1)
        self.assertGreaterEqual(result.best_epoch, 1)
        self.assertEqual(result.val_preds.shape, y_val.shape)
        self.assertIsInstance(result.final_val_loss, float)

    def test_linear_layers_and_collect_distribution(self):
        model = nn.Sequential(
            [nn.Linear(3, 4, seed=0), nn.ReLU(), nn.Linear(4, 1, seed=1)]
        )
        x = Tensor(np.random.default_rng(0).normal(size=(5, 3)), requires_grad=True)
        y = model(x).sum()
        y.backward()

        layers = main.linear_layers(model)
        self.assertEqual(len(layers), 2)

        weights = main.collect_layer_distribution(model, use_grad=False)
        grads = main.collect_layer_distribution(model, use_grad=True)
        self.assertEqual(len(weights), 2)
        self.assertEqual(len(grads), 2)

    def test_save_and_reload_demo_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "tiny_persistence_demo.pkl"
            loaded = main.save_and_reload_demo_model(p, input_dim=3, hidden_dim=2)

            self.assertTrue(p.exists())
            self.assertIsInstance(loaded, nn.Sequential)
            self.assertEqual(len(main.linear_layers(loaded)), 2)

    def test_plot_functions(self):
        left = _fake_result("left")
        right = _fake_result("right")

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "loss.png"
            main.plot_loss_comparison([left, right], title="t", output_path=p1)
            self.assertTrue(p1.exists())

            p2 = Path(tmpdir) / "dist.png"
            main.plot_distribution_comparison(
                left, right, use_grad=False, output_path=p2, title="t"
            )
            self.assertTrue(p2.exists())

    def test_first_epoch_below(self):
        self.assertEqual(main.first_epoch_below([0.9, 0.6, 0.3], 0.5), 3)
        self.assertIsNone(main.first_epoch_below([0.9, 0.6], 0.5))

    def test_summaries_print(self):
        a = _fake_result("A")
        b = _fake_result("B")
        b.val_preds = np.array([[1.0], [1.0]])

        buf = io.StringIO()
        with redirect_stdout(buf):
            main.summarize_rmsnorm_effect(a, b)
            main.summarize_optimizer_speed(a, b)
        out = buf.getvalue()
        self.assertIn("RMSNorm Analysis", out)
        self.assertIn("SGD vs Adam Convergence", out)

    def test_train_on_dataset_with_mocked_train(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "tiny.csv"
            _write_small_dataset(dataset, n_rows=20)

            with patch(
                "main.train_experiment",
                side_effect=[
                    _fake_result("sgd1"),
                    _fake_result("sgd2"),
                    _fake_result("adam"),
                ],
            ):
                cwd = Path.cwd()
                os.chdir(tmpdir)
                try:
                    main.train_on_dataset(dataset)
                    self.assertTrue(
                        (Path("reports") / "rmsnorm_loss_comparison.png").exists()
                    )
                    self.assertTrue(
                        (Path("reports") / "optimizer_loss_comparison.png").exists()
                    )
                    self.assertTrue(
                        (Path("reports") / "rmsnorm_weight_distribution.png").exists()
                    )
                    self.assertTrue(
                        (Path("reports") / "rmsnorm_gradient_distribution.png").exists()
                    )
                    self.assertTrue(
                        (Path("reports") / "tiny_persistence_demo.pkl").exists()
                    )
                finally:
                    os.chdir(cwd)

    def test_main_entrypoint_behaviors(self):
        buf = io.StringIO()
        with patch("main.Path.exists", return_value=False):
            with redirect_stdout(buf):
                main.main()
        self.assertIn("Dataset not found", buf.getvalue())

        with (
            patch("main.Path.exists", return_value=True),
            patch("main.train_on_dataset") as mocked,
        ):
            main.main()
            mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
