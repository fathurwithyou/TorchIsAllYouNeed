# TorchIsAllYouNeed

This repository contains a from-scratch NumPy implementation of a `feedforward neural network` with an API inspired by PyTorch. It includes automatic differentiation, layers, activation functions, losses, optimizers, and several experiments for job placement prediction using the `data/datasetml_2026.csv` dataset.

## Setup

Use Python `>= 3.14`.

If you use `uv`:

```bash
uv sync
```

If you prefer a standard `venv` setup:

```bash
python -m venv .venv
./.venv/bin/pip install numpy matplotlib scikit-learn ruff
```

## Project Structure

- `src/`: main source code, including `main.py` and the `torchlike` package
- `data/`: dataset used in the experiments
- `reports/`: generated figures and experiment artifacts
- `tests/`: unit tests
- `doc/`: report materials, appendices, and LaTeX files
- `src/experiments.ipynb`: notebook for exploration and interactive experiments

## Run

The code can be executed either through the main script or the notebook. Run commands from the repository root so that the `data/` and `reports/` paths are resolved correctly.

Primary method:

```bash
uv run env PYTHONPATH=src python src/main.py
```

The command above will:

- load the `data/datasetml_2026.csv` dataset
- train the experimental model
- save the results to the `reports/` directory

Alternative:

- open the `src/experiments.ipynb` notebook
- run the experiment cells interactively

Note: although the main file is located in `src`, it is best not to run `python main.py` directly from that directory, as the script uses paths relative to the repository root.

## Task Distribution

| Name | Student ID | Responsibility |
| --- | --- | --- |
| Shanice Feodora Tjahjono | 13523097 | Linear layer, activation functions, and experimental design |
| Muhammad Fathur Rizky | 13523105 | SGD, Adam, automatic differentiation, and RMSNorm |
| Ahmad Wicaksono | 13523121 | Loss functions and experimental design |
