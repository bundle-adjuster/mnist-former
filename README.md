# MNIST-Former

A small Vision Transformer (default **N=2** encoder blocks) trained on **Fashion-MNIST**, with training and inference separated from visualization. Profiling hooks are optional; analysis lives in Jupyter notebooks that read checkpoints and metrics from disk.

## Install

```bash
cd mnist-former
pip install -e .
```

Training and inference only need the base dependencies (`torch`, `torchvision`, `tqdm`).

For notebooks (plots):

```bash
pip install -e ".[viz]"
```

For tests:

```bash
pip install -e ".[dev]"
```

## Train

```bash
python scripts/train.py --output-dir runs/exp1
```

Optional: profile a short segment and write a Chrome trace under the run directory:

```bash
python scripts/train.py --output-dir runs/exp1 --profile
```

### NVIDIA Nsight Systems

Use [Nsight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`) for a full GPU/CPU timeline (CUDA kernels, memory, OS runtime). Install the CUDA toolkit or the standalone Nsight Systems package so `nsys` is on your `PATH`.

**Short capture** (keeps reports small; use GPU, few epochs):

```bash
nsys profile -o runs/nsys_train --trace=cuda,nvtx,osrt --sample=cpu \
  python scripts/train.py --output-dir runs/nsys_exp --device cuda --epochs 1 --nvtx
```

- `--trace=cuda,nvtx,osrt` — CUDA API, NVTX markers, OS runtime (add `cudnn`, `cublas`, etc. if your `nsys --help` lists them and you want more detail).
- `--nvtx` — enables PyTorch `emit_nvtx()` so autograd regions show up in the NVTX row (CUDA only).

**Limit duration** (optional) if you only want the first tens of seconds:

```bash
nsys profile -o runs/nsys_clip --trace=cuda,nvtx,osrt --duration=30 \
  python scripts/train.py --output-dir runs/nsys_exp --device cuda --epochs 1 --nvtx
```

Open the `.nsys-rep` file in the Nsight Systems GUI, or summarize from the CLI:

```bash
nsys stats runs/nsys_train.nsys-rep
```

Or use the console entry point:

```bash
mnist-former-train --output-dir runs/exp1
```

Artifacts:

- `runs/exp1/checkpoints/best.pt` — model weights, configs, epoch
- `runs/exp1/metrics.jsonl` — one JSON object per epoch (loss, accuracy, etc.)

## Inference

```bash
python scripts/infer.py --checkpoint runs/exp1/checkpoints/best.pt --split test
```

```bash
mnist-former-infer --checkpoint runs/exp1/checkpoints/best.pt --split test
```

Optional: save predictions to CSV with `--predictions-csv path/to/preds.csv`.

## Notebooks

After installing `[viz]`, open `notebooks/` and set `OUTPUT_DIR` (or equivalent) to your run directory, for example `runs/exp1`.

- `01_training_curves.ipynb` — loss and accuracy from `metrics.jsonl`
- `02_attention_and_samples.ipynb` — attention maps, sample images, confusion matrix
- `03_profiler_trace.ipynb` — notes on Chrome traces from `--profile`

## Tests

```bash
pytest
```

If your environment auto-loads unrelated pytest plugins (for example ROS) and collection fails, run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```
