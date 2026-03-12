# Running LingBot-World with Gradio

Step-by-step guide to run the LingBot-World interactive world model via a
browser-based Gradio UI. Written for someone who just cloned the repo and has
access to a machine with 2 GPUs (~48 GB VRAM each).

---

## Prerequisites

- Linux with NVIDIA GPUs (tested on 2x RTX A6000, 48 GB each)
- CUDA 12.x driver installed
- conda (Miniconda or Anaconda)
- ~160 GB free disk space for model weights
- ~100 GB system RAM (models are partially resident on CPU)

---

## 1. Environment setup

```bash
conda create -n lingbot python=3.10 -y
conda activate lingbot
```

## 2. Install Python dependencies

```bash
cd lingbot-world
pip install -r requirements.txt
pip install gradio
```

### flash-attn (the tricky one)

`flash-attn` requires CUDA compilation and building from source takes 30+ min.
Use a prebuilt wheel instead. Match your **exact** Python, PyTorch, and CUDA
versions. Find wheels at:
<https://github.com/mjun0812/flash-attention-prebuild-wheels/releases>

Check your versions first:

```bash
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')"
python --version
```

Then install the matching wheel. Example for torch 2.10 + CUDA 12.8 + Python 3.10:

```bash
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp310-cp310-linux_x86_64.whl
```

Verify it loads:

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

### Common install issues

**torchvision ABI mismatch** -- If you see `operator torchvision::nms does not
exist`, the system-level torchvision conflicts with the pip-installed torch.
Fix:

```bash
pip install --force-reinstall torchvision
```

**numpy version** -- The codebase requires numpy < 2:

```bash
pip install "numpy>=1.23.5,<2"
```

## 3. Download model weights

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download robbyant/lingbot-world-base-cam \
    --local-dir ./lingbot-world-base-cam
```

This downloads ~150 GB into `./lingbot-world-base-cam/`:

| Component | Size | Description |
|-----------|------|-------------|
| `high_noise_model/` | ~70 GB | DiT for high-noise denoising steps (8 safetensor shards) |
| `low_noise_model/` | ~70 GB | DiT for low-noise denoising steps (8 safetensor shards) |
| `models_t5_umt5-xxl-enc-bf16.pth` | ~10 GB | T5-XXL text encoder |
| `Wan2.1_VAE.pth` | ~500 MB | Video VAE |
| `google/umt5-xxl/` | tokenizer | T5 tokenizer files |

## 4. Launch

The Gradio app (`gradio_app.py`) is designed for multi-GPU inference. It uses
PyTorch FSDP to shard the two DiT models across GPUs and Ulysses sequence
parallelism to split the attention computation.

```bash
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 gradio_app.py --port 7860
```

- `--nproc_per_node=2` -- one process per GPU.
- `PYTHONUNBUFFERED=1` -- so you see log output in real time.

Startup takes ~2 min (loading 150 GB from disk, FSDP sharding, NCCL init).
When ready:

```
* Running on local URL:  http://0.0.0.0:7860
```

### Accessing from your local machine

If you are on a remote cluster, forward the port via SSH:

```bash
ssh -L 7860:localhost:7860 user@cluster-host
```

Then open <http://localhost:7860>.

Or use `--share` to get a public Gradio URL (no tunnel needed):

```bash
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 gradio_app.py --port 7860 --share
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--ckpt_dir` | `./lingbot-world-base-cam` | Path to model weights |
| `--port` | `7860` | Server port |
| `--share` | off | Public Gradio link |

## 5. Using the UI

1. **Load an example** -- Pick `00`, `01`, or `02` from the dropdown. This
   auto-fills the input image, text prompt, and camera control signals from the
   `examples/` directory.
2. **Or bring your own** -- Upload any image and write a prompt. Set the example
   dropdown to `none` (no camera control will be applied).
3. **Frame count** -- Slider from 17 to 161 (must satisfy `4n+1`). At 16 fps,
   81 frames is ~5 seconds of video.
4. **Seed** -- Fixed by default (42). Set to -1 for random.
5. Press **Generate**.

---

## How it works under the hood

### Why not just `python generate.py`?

The upstream `generate.py` uses `torchrun` with 8 GPUs and writes a video file
to disk. We needed something interactive, running on 2 GPUs, accessible through
a browser on a cluster.

### What `gradio_app.py` does differently

1. **Multi-GPU coordination with Gradio** -- `torchrun` launches 2 processes
   (rank 0 and rank 1). Only rank 0 runs the Gradio web server. When the user
   clicks Generate, rank 0 broadcasts the generation parameters (prompt, image
   bytes, seed, etc.) to rank 1 via `dist.broadcast_object_list`. Both ranks
   then call `pipeline.generate()` in lockstep. Only rank 0 collects the output
   video and sends it back to the browser.

2. **Memory-aware configuration** -- The model has two 14B-param DiT
   transformers (~28 GB each in bf16) plus a T5-XXL text encoder (~10 GB) and a
   VAE (~500 MB). On 2x 48 GB GPUs:
   - **DiTs**: sharded via FSDP (`dit_fsdp=True`), so each GPU holds ~14 GB per
     DiT, totalling ~28 GB/GPU for both.
   - **T5**: kept on CPU (`t5_cpu=True, t5_fsdp=False`). Text encoding runs on
     CPU and only the output embeddings are copied to GPU. This saves ~10 GB of
     GPU memory that would otherwise be needed for T5 parameters.
   - **VAE**: lives on GPU 0 (~500 MB). It encodes the input image and decodes
     the final latents into video frames.
   - Result: ~39 GB used per GPU at rest, leaving ~10 GB for activations.

3. **Resolution choice** -- 480P (`480*832`) is hardcoded because 720P does not
   fit in memory with this GPU configuration.

4. **Worker loop** -- Rank 1 runs a simple `while True` loop that waits for
   broadcast, runs generation, and repeats. A sentinel value shuts it down when
   Gradio exits.

### Architecture diagram

```
                    ┌──────────────────────────────────────┐
                    │           User Browser                │
                    │         localhost:7860                │
                    └──────────────┬───────────────────────┘
                                   │ HTTP
                    ┌──────────────▼───────────────────────┐
                    │     Rank 0  (GPU 0)                  │
                    │  ┌─────────────────────────────┐     │
                    │  │  Gradio Server               │     │
                    │  └──────────┬──────────────────┘     │
                    │             │                         │
                    │  broadcast_object_list (NCCL)         │
                    │             │                         │
                    │  ┌──────────▼──────────────────┐     │
                    │  │  T5 (CPU) → embeddings       │     │
                    │  │  VAE encode (GPU)            │     │
                    │  │  DiT denoise (FSDP, half)    │     │
                    │  │  VAE decode (GPU)            │     │
                    │  └─────────────────────────────┘     │
                    └──────────────┬───────────────────────┘
                                   │ NCCL all-gather / all-to-all
                    ┌──────────────▼───────────────────────┐
                    │     Rank 1  (GPU 1)                  │
                    │  ┌─────────────────────────────┐     │
                    │  │  Worker loop (wait/generate) │     │
                    │  │  T5 (CPU) → embeddings       │     │
                    │  │  DiT denoise (FSDP, half)    │     │
                    │  └─────────────────────────────┘     │
                    └──────────────────────────────────────┘
```

### Adapting to other hardware

| Setup | What to change |
|-------|---------------|
| **1 GPU, >= 80 GB** (A100/H100) | `python gradio_app.py` (no torchrun). Set `t5_cpu=False` in `load_pipeline`. Single-process, no FSDP. |
| **1 GPU, 48 GB** (A6000) | Likely OOMs at 480P/81 frames. Use the [nf4 quantized model](https://huggingface.co/cahlen/lingbot-world-base-cam-nf4) instead. |
| **4+ GPUs** | `torchrun --nproc_per_node=4 gradio_app.py`. The code auto-detects world_size; `num_heads` (40) must be divisible by `nproc_per_node`. |
| **Different model** (Act variant) | Download `robbyant/lingbot-world-base-act`, pass `--ckpt_dir ./lingbot-world-base-act`. The code auto-detects `cam` vs `act` from the directory name. |
