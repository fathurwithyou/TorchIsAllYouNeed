# TorchIsAllYouNeed

Repositori ini berisi implementasi `feedforward neural network` dari awal berbasis NumPy dengan API yang terinspirasi dari PyTorch. Di dalamnya ada automatic differentiation, layer, activation function, loss, optimizer, dan beberapa eksperimen untuk prediksi penempatan kerja menggunakan dataset `data/datasetml_2026.csv`.

## Setup

Gunakan Python `>= 3.14`.

Kalau pakai `uv`:

```bash
uv sync
```

Kalau pakai `venv` biasa:

```bash
python -m venv .venv
./.venv/bin/pip install numpy matplotlib scikit-learn ruff
```

## Struktur Folder

- `src/`: source code utama, termasuk `main.py` dan package `torchlike`
- `data/`: dataset yang dipakai
- `reports/`: output berupa gambar dan artefak hasil eksperimen
- `tests/`: unit test
- `doc/`: laporan, lampiran, dan file LaTeX
- `experiments_global_student_placement_and_salary.ipynb`: notebook untuk eksplorasi dan eksperimen interaktif

## Run

Code bisa dijalankan lewat script utama maupun notebook. Jalankan dari root repositori supaya path `data/` dan `reports/` terbaca dengan benar.

Cara utama:

```bash
uv run env PYTHONPATH=src python src/main.py
```

Perintah di atas akan:

- membaca dataset `data/datasetml_2026.csv`
- melatih model eksperimen
- menyimpan hasil ke folder `reports/`

Alternatif:

- buka notebook `experiments_global_student_placement_and_salary.ipynb`
- jalankan sel eksperimen secara interaktif

Catatan: meskipun file utamanya ada di folder `src`, sebaiknya jangan menjalankan `python main.py` langsung dari dalam folder tersebut karena script memakai path relatif terhadap root repositori.

## Task Distribution

| Nama | NIM | Tugas |
| --- | --- | --- |
| Shanice Feodora Tjahjono | 13523097 | Linear, activation functions, dan experimental design |
| Muhammad Fathur Rizky | 13523105 | SGD, Adam, automatic differentiation, dan RMSNorm |
| Ahmad Wicaksono | 13523121 | Loss functions dan experimental design |
