from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import torchlike.functional as F
from torchlike import ArrayDataset, DataLoader, Tensor, nn, optim

OptimizerName = Literal["sgd", "adam"]
TARGET_COL = "placement_status"
NUMERIC_COLS = (
    "cgpa",
    "backlogs",
    "internship_count",
    "aptitude_score",
    "communication_score",
    "internship_quality_score",
)
ONEHOT_COLS = (
    "college_tier",
    "country",
    "university_ranking_band",
    "specialization",
    "industry",
)


@dataclass
class ExperimentResult:
    name: str
    model: nn.Sequential
    train_losses: list[float]
    val_losses: list[float]
    val_accs: list[float]
    final_val_loss: float
    final_val_acc: float
    val_probs: np.ndarray
    val_preds: np.ndarray
    best_epoch: int


def load_dataset_features(
    csv_path: Path,
    *,
    split_seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Dataset is empty")

    y_all = np.array(
        [1.0 if row[TARGET_COL] == "Placed" else 0.0 for row in rows], dtype=np.float64
    ).reshape(-1, 1)
    train_rows, val_rows, y_train, y_val = train_test_split(
        rows,
        y_all,
        test_size=0.2,
        random_state=split_seed,
        shuffle=True,
    )

    x_train_raw = [
        [
            *([float(row[col]) for col in NUMERIC_COLS]),
            *([row[col] for col in ONEHOT_COLS]),
        ]
        for row in train_rows
    ]
    x_val_raw = [
        [
            *([float(row[col]) for col in NUMERIC_COLS]),
            *([row[col] for col in ONEHOT_COLS]),
        ]
        for row in val_rows
    ]

    num_idx = list(range(len(NUMERIC_COLS)))
    cat_idx = list(range(len(NUMERIC_COLS), len(NUMERIC_COLS) + len(ONEHOT_COLS)))
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_idx),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_idx,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    x_train = preprocessor.fit_transform(x_train_raw).astype(np.float64)
    x_val = preprocessor.transform(x_val_raw).astype(np.float64)

    return x_train, x_val, y_train, y_val


def build_binary_model(
    input_dim: int,
    *,
    use_rmsnorm: bool,
    init: str = "he",
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, 128, init=init, seed=7)]
    if use_rmsnorm:
        layers.append(nn.RMSNorm(128))
    layers.extend([nn.GELU(), nn.Linear(128, 96, init=init, seed=11)])
    if use_rmsnorm:
        layers.append(nn.RMSNorm(96))
    layers.extend([nn.SELU(), nn.Linear(96, 64, init=init, seed=13)])
    if use_rmsnorm:
        layers.append(nn.RMSNorm(64))
    layers.extend(
        [
            nn.GELU(),
            nn.Linear(64, 32, init=init, seed=17),
            nn.SELU(),
            nn.Linear(32, 1, init=init, seed=19),
        ]
    )
    return nn.Sequential(layers)


def evaluate_binary(
    model: nn.Sequential,
    x: np.ndarray,
    y: np.ndarray,
    criterion: nn.BCEWithLogitsLoss,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    logits = model(Tensor(x))
    loss = criterion(logits, Tensor(y)).item()
    probs = F.sigmoid(logits).data
    preds = (probs >= 0.5).astype(np.float64)
    y_true_flat = y.reshape(-1).astype(np.int64)
    y_pred_flat = preds.reshape(-1).astype(np.int64)
    acc = float(accuracy_score(y_true_flat, y_pred_flat))
    return loss, acc, probs, preds


def attach_full_batch_gradients(
    model: nn.Sequential,
    x: np.ndarray,
    y: np.ndarray,
    criterion: nn.BCEWithLogitsLoss,
) -> None:
    for p in model.parameters():
        p.zero_grad()
    logits = model(Tensor(x))
    loss = criterion(logits, Tensor(y))
    loss.backward()


def create_optimizer(name: OptimizerName, model: nn.Sequential):
    params = model.parameters()
    if name == "sgd":
        return optim.SGD(params, lr=2e-5, l2_lambda=1e-4)
    if name == "adam":
        return optim.Adam(params, lr=2e-4, l2_lambda=1e-4)
    raise ValueError(f"Unsupported optimizer: {name}")


def train_experiment(
    *,
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    use_rmsnorm: bool,
    optimizer_name: OptimizerName,
    epochs: int = 500,
    batch_size: int = 256,
    dataloader_seed: int = 123,
    early_stopping_patience: int = 200,
    min_delta: float = 1e-4,
) -> ExperimentResult:
    train_dataset = ArrayDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=dataloader_seed,
    )

    model = build_binary_model(x_train.shape[1], use_rmsnorm=use_rmsnorm, init="he")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = create_optimizer(optimizer_name, model)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_params: list[np.ndarray] | None = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, _, _, _ = evaluate_binary(model, x_train, y_train, criterion)
        val_loss, val_acc, _, _ = evaluate_binary(model, x_val, y_val, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_params = [p.data.copy() for p in model.parameters()]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 50 == 0:
            print(
                f"[{name}] epoch {epoch + 1:03d}/{epochs} "
                f"- train_loss: {train_loss:.6f} "
                f"- val_loss: {val_loss:.6f} "
                f"- val_acc: {val_acc:.4f}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"[{name}] early stopping at epoch {epoch + 1:03d}; "
                f"best val_loss {best_val_loss:.6f} at epoch {best_epoch:03d}"
            )
            break

    if best_params is not None:
        for param, best_value in zip(model.parameters(), best_params):
            param.data[...] = best_value

    attach_full_batch_gradients(model, x_train, y_train, criterion)
    final_val_loss, final_val_acc, val_probs, val_preds = evaluate_binary(
        model, x_val, y_val, criterion
    )

    return ExperimentResult(
        name=name,
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        val_accs=val_accs,
        final_val_loss=final_val_loss,
        final_val_acc=final_val_acc,
        val_probs=val_probs,
        val_preds=val_preds,
        best_epoch=best_epoch,
    )


def linear_layers(model: nn.Sequential) -> list[nn.Module]:
    return [
        layer
        for layer in model.layers
        if hasattr(layer, "in_features")
        and hasattr(layer, "out_features")
        and hasattr(layer, "weight")
    ]


def collect_layer_distribution(
    model: nn.Sequential, *, use_grad: bool
) -> list[np.ndarray]:
    values_per_layer: list[np.ndarray] = []
    for layer in linear_layers(model):
        weight_values = layer.weight.grad if use_grad else layer.weight.data
        flat_values = [weight_values.reshape(-1)]
        if getattr(layer, "bias", None) is not None:
            bias_values = layer.bias.grad if use_grad else layer.bias.data
            flat_values.append(bias_values.reshape(-1))
        values_per_layer.append(np.concatenate(flat_values))
    return values_per_layer


def plot_loss_comparison(
    results: list[ExperimentResult],
    *,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    for result in results:
        epochs = np.arange(1, len(result.train_losses) + 1)
        plt.plot(
            epochs, result.train_losses, linestyle="--", label=f"{result.name} train"
        )
        plt.plot(epochs, result.val_losses, label=f"{result.name} val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_distribution_comparison(
    left: ExperimentResult,
    right: ExperimentResult,
    *,
    use_grad: bool,
    output_path: Path,
    title: str,
) -> None:
    left_values = collect_layer_distribution(left.model, use_grad=use_grad)
    right_values = collect_layer_distribution(right.model, use_grad=use_grad)
    layer_count = min(len(left_values), len(right_values))
    if layer_count == 0:
        return

    fig, axes = plt.subplots(1, layer_count, figsize=(6 * layer_count, 4))
    if layer_count == 1:
        axes = [axes]

    for idx in range(layer_count):
        ax = axes[idx]
        ax.hist(left_values[idx], bins=30, alpha=0.65, label=left.name)
        ax.hist(right_values[idx], bins=30, alpha=0.65, label=right.name)
        ax.set_title(f"Linear Layer {idx}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def first_epoch_below(history: list[float], target: float) -> int | None:
    for idx, value in enumerate(history, start=1):
        if value <= target:
            return idx
    return None


def summarize_rmsnorm_effect(
    no_norm: ExperimentResult,
    with_norm: ExperimentResult,
) -> None:
    final_loss_no_norm = no_norm.final_val_loss
    final_loss_with_norm = with_norm.final_val_loss
    final_acc_no_norm = no_norm.final_val_acc
    final_acc_with_norm = with_norm.final_val_acc

    changed_fraction = float(np.mean(no_norm.val_preds != with_norm.val_preds))
    positive_rate_no_norm = float(np.mean(no_norm.val_preds))
    positive_rate_with_norm = float(np.mean(with_norm.val_preds))

    print("\n[RMSNorm Analysis]")
    print(
        f"NoNorm   -> val_loss: {final_loss_no_norm:.6f}, val_acc: {final_acc_no_norm:.4f}, "
        f"positive_rate: {positive_rate_no_norm:.4f}"
    )
    print(
        f"RMSNorm  -> val_loss: {final_loss_with_norm:.6f}, val_acc: {final_acc_with_norm:.4f}, "
        f"positive_rate: {positive_rate_with_norm:.4f}"
    )
    print(f"Prediction difference on validation set: {changed_fraction:.4f}")


def summarize_optimizer_speed(
    sgd_result: ExperimentResult, adam_result: ExperimentResult
) -> None:
    sgd_best_loss = min(sgd_result.val_losses)
    adam_best_loss = min(adam_result.val_losses)
    target_val_loss = max(sgd_best_loss, adam_best_loss)
    sgd_epoch = first_epoch_below(sgd_result.val_losses, target_val_loss)
    adam_epoch = first_epoch_below(adam_result.val_losses, target_val_loss)
    sgd_best_epoch = int(np.argmin(sgd_result.val_losses) + 1)
    adam_best_epoch = int(np.argmin(adam_result.val_losses) + 1)

    print("\n[SGD vs Adam Convergence]")
    print(
        f"SGD  -> final val_loss: {sgd_result.final_val_loss:.6f}, "
        f"final val_acc: {sgd_result.final_val_acc:.4f}, "
        f"best val_loss: {sgd_best_loss:.6f} @ epoch {sgd_best_epoch}"
    )
    print(
        f"Adam -> final val_loss: {adam_result.final_val_loss:.6f}, "
        f"final val_acc: {adam_result.final_val_acc:.4f}, "
        f"best val_loss: {adam_best_loss:.6f} @ epoch {adam_best_epoch}"
    )
    print(f"Common target val_loss (both can reach): {target_val_loss:.6f}")
    print(f"Epoch to reach target - SGD: {sgd_epoch}, Adam: {adam_epoch}")


def train_on_dataset(csv_path: Path) -> None:
    x_train, x_val, y_train, y_val = load_dataset_features(csv_path)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    sgd_no_norm = train_experiment(
        name="SGD-NoNorm",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        use_rmsnorm=False,
        optimizer_name="sgd",
    )
    sgd_rmsnorm = train_experiment(
        name="SGD-RMSNorm",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        use_rmsnorm=True,
        optimizer_name="sgd",
    )
    adam_no_norm = train_experiment(
        name="Adam-NoNorm",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        use_rmsnorm=False,
        optimizer_name="adam",
    )

    report_paths = [
        reports_dir / "rmsnorm_loss_comparison.png",
        reports_dir / "optimizer_loss_comparison.png",
        reports_dir / "rmsnorm_weight_distribution.png",
        reports_dir / "rmsnorm_gradient_distribution.png",
    ]
    plot_loss_comparison(
        [sgd_no_norm, sgd_rmsnorm],
        title="RMSNorm vs NoNorm (SGD): Train/Validation Loss",
        output_path=report_paths[0],
    )
    plot_loss_comparison(
        [sgd_no_norm, adam_no_norm],
        title="SGD vs Adam (NoNorm): Train/Validation Loss",
        output_path=report_paths[1],
    )
    plot_distribution_comparison(
        sgd_no_norm,
        sgd_rmsnorm,
        use_grad=False,
        output_path=report_paths[2],
        title="Weight Distribution: NoNorm vs RMSNorm",
    )
    plot_distribution_comparison(
        sgd_no_norm,
        sgd_rmsnorm,
        use_grad=True,
        output_path=report_paths[3],
        title="Gradient Distribution: NoNorm vs RMSNorm",
    )

    summarize_rmsnorm_effect(sgd_no_norm, sgd_rmsnorm)
    summarize_optimizer_speed(sgd_no_norm, adam_no_norm)

    print("\nSaved artifacts:")
    for report_path in report_paths:
        print(f"- {report_path}")


def main() -> None:
    dataset_path = Path("data/datasetml_2026.csv")
    if not dataset_path.exists():
        print("Dataset not found at data/datasetml_2026.csv")
        return

    print("Training on datasetml_2026.csv")
    train_on_dataset(dataset_path)


if __name__ == "__main__":
    main()
