# Requirements

This file lists the recommended environment and dependencies for running the TouchSafeBench data generation and evaluation pipeline.

The code has two dependency levels:

```text
Core Habitat dependencies
  Required for episode generation and video dataset generation.

VLM evaluation dependencies
  Required only when running safety benchmark evaluation.
```

## 1. System Requirements

Recommended hardware:

```text
OS: Linux
GPU: NVIDIA GPU
CUDA: CUDA-compatible driver
RAM: 32 GB or more recommended
Storage: large HDD/SSD recommended for rendered frames and videos
```

Recommended software:

```text
Conda or Miniconda
Python 3.9
Habitat-Sim 0.3.3
```

The data generation stage is GPU-heavy and writes many image files. Make sure the output disk has enough free space before running the full dataset generation.

## 2. Create the Conda Environment

Recommended:

```bash
conda env create -f environment.yml
conda activate habitat
```

Manual setup:

```bash
conda create -n habitat python=3.9 cmake=3.14.0 -y
conda activate habitat
```

Install Habitat-Sim:

```bash
conda install habitat-sim=0.3.3 withbullet headless \
  -c conda-forge -c aihabitat -y
```

If your machine has display / EGL issues, see `TROUBLESHOOTING.md`.

## 3. Install Habitat-Lab and Habitat-Baselines

Run from the repository root:

```bash
pip install -e habitat-lab
pip install -e habitat-baselines
```

If you used the manual setup instead of `environment.yml`, also install the pinned Python packages:

```bash
pip install -r requirements.txt
```

The root `requirements.txt` is a compact list of packages and versions taken from the working `habitat` conda environment used for this project. `environment.yml` uses this file automatically.

The Habitat-Lab requirements include packages such as:

```text
gym
numpy
numpy-quaternion
attrs
opencv-python
hydra-core
omegaconf
numba
imageio
imageio-ffmpeg
scipy
tqdm
```

These are installed through the editable Habitat packages above.

## 4. Download Habitat Data Assets

Download the required assets:

```bash
python -m habitat_sim.utils.datasets_download \
  --uids hab3-episodes hab3_bench_assets habitat_humanoids hssd-hab ycb \
  --data-path data/
```

Expected paths:

```text
data/scene_datasets/
data/humanoids/
data/hab3_bench_assets/
data/versioned_data/hab3-episodes/
data/versioned_data/ycb/
```

The video generator expects:

```text
data/versioned_data/hab3-episodes/val/social_rearrange_diverse.json.gz
data/versioned_data/hab3-episodes/checkpoint/social_nav_latest.pth
```

Only the `social_navigation` task needs a policy checkpoint (`social_nav_latest.pth`, included in the download). The `social_rearrangement` task runs with oracle skills and needs no checkpoint.

If `social_rearrange_diverse.json.gz` is regenerated locally, run `fix_dataset.py` before video generation.

## 5. Extra Packages for Dataset Generation

The dataset generation scripts use OpenCV, PIL/image tools, and ffmpeg-backed video writing. Most packages are installed with Habitat-Lab, but if imports fail, install:

```bash
pip install opencv-python pillow imageio imageio-ffmpeg matplotlib
```

Check that `ffmpeg` is available:

```bash
ffmpeg -version
```

If not available, install it with conda:

```bash
conda install -c conda-forge ffmpeg -y
```

## 6. Extra Packages for VLM Evaluation

Install common evaluation dependencies:

```bash
pip install pillow matplotlib scikit-learn openai google-genai
```

For local open-source VLMs, install PyTorch and Transformers. Choose the PyTorch build that matches your CUDA driver:

```bash
pip install transformers accelerate qwen-vl-utils
```

Example PyTorch install for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If your CUDA version is different, use the command from the official PyTorch install page.

## 7. Optional Dependencies for Predicted Depth Modes

The evaluation modes below require Depth Anything 3:

```text
pred_depth
pred_depth_color
rgbd_pred
rgbd_pred_color
```

Depth Anything 3 is already pinned in `requirements.txt` (`depth-anything-3==0.1.1`, which requires Python 3.9–3.13), so it is installed by default. If you installed manually and skipped it, add it with:

```bash
pip install depth-anything-3==0.1.1
```

The evaluation code imports:

```python
from depth_anything_3.api import DepthAnything3
```

If this package is not installed, avoid the predicted-depth modalities and use `rgb`, `depth`, `depth_color`, `rgbd`, or `rgbd_color`.

## 8. API Keys

For Gemini:

```bash
export GEMINI_API_KEYS="key1,key2"
```

For OpenAI:

```bash
export OPENAI_API_KEY="your-openai-key"
```

Local open-source models do not require API keys, but they require enough GPU memory for the selected model.

## 9. Hugging Face Cache

Local VLM evaluation may download large model weights. Set the Hugging Face cache to a disk with enough space:

```bash
export HF_HOME=/path/to/large_disk/huggingface
```

`evaluate_safety_benchmark_v2_open.py` also respects `HF_HOME`.

## 10. Quick Verification

Check the Habitat installation:

```bash
python -c "import habitat; import habitat_sim; print('Habitat OK')"
```

Check core Python packages:

```bash
python -c "import cv2, numpy, PIL, torch; print('Core packages OK')"
```

Check VLM evaluation packages:

```bash
python -c "import transformers, sklearn, openai; from google import genai; print('Eval packages OK')"
```

Run a small dataset generation smoke test:

```bash
python fix_dataset.py
python generate_video_dataset.py --gpu 0 --episodes 2 --resume
```

Run an evaluation dry run:

```bash
python evaluate_safety_benchmark_v2.py \
  --task all \
  --dry-run \
  --max-samples 5 \
  --views arm head \
  --modality rgb
```

## 11. Large Files

The following are usually large and should not be committed to GitHub:

```text
habitat_video_dataset/
eval_safety_benchmark/
topdown_ortho/
data/scene_datasets/
data/versioned_data/
data/humanoids/
data/hab3_bench_assets/
*.pth
*.mp4
*.jpg
*.png
```
