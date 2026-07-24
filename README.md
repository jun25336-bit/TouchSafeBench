# TouchSafeBench

TouchSafeBench is a Habitat-based pipeline for generating multi-view human-robot interaction episodes and evaluating VLMs on robot safety understanding.

The pipeline has four main stages:

```text
Episode dataset -> fixed episode dataset -> simulated video dataset -> VLM safety evaluation
```

This repository is based on Habitat-Lab / Habitat-Baselines and adds scripts for dataset generation, frame extraction, trajectory recording, and VLM evaluation.

## Pre-Generated Dataset

If you prefer not to generate the data yourself, download the ready-made
TouchSafeBench dataset (public) from Hugging Face:

https://huggingface.co/datasets/face12345/TouchSafeBench

With the downloaded data you can skip the generation stages (sections 1-4) and run
the safety benchmark evaluation directly.

## Requirements

- Linux
- NVIDIA GPU with CUDA support
- Conda
- Python 3.9
- Habitat-Sim 0.3.3

See `REQUIREMENTS.md` for the detailed setup, optional VLM dependencies, API keys, and verification commands.

Quick setup with `environment.yml`:

```bash
cd /path/to/habitat-lab
conda env create -f environment.yml
conda activate habitat

pip install -e habitat-lab
pip install -e habitat-baselines
```

Manual setup:

```bash
conda create -n habitat python=3.9 cmake=3.14.0 -y
conda activate habitat

conda install habitat-sim=0.3.3 withbullet headless \
  -c conda-forge -c aihabitat -y

cd /path/to/habitat-lab
pip install -r requirements.txt
pip install -e habitat-lab
pip install -e habitat-baselines
```

Download the required Habitat assets:

```bash
python -m habitat_sim.utils.datasets_download \
  --uids hab3-episodes hab3_bench_assets habitat_humanoids hssd-hab ycb \
  --data-path data/
```

Expected data paths:

```text
data/scene_datasets/
data/humanoids/
data/hab3_bench_assets/
data/versioned_data/hab3-episodes/
data/versioned_data/ycb/
```

## Reproduce Paper Results

Use the following order to reproduce the full pipeline:

```bash
# 1. Create environment and install this repository
conda env create -f environment.yml
conda activate habitat
pip install -e habitat-lab
pip install -e habitat-baselines

# 2. Download Habitat assets
python -m habitat_sim.utils.datasets_download \
  --uids hab3-episodes hab3_bench_assets habitat_humanoids hssd-hab ycb \
  --data-path data/

# 3. Fix episode metadata
python fix_dataset.py

# 4. Generate the video dataset
python generate_video_dataset.py \
  --gpu 0 \
  --tasks social_navigation social_rearrangement \
  --human-types female_2 male_2 neutral_0 \
  --topdown-height auto \
  --episodes -1 \
  --resume

# 5. Render orthographic maps for top-down evaluation
python render_topdown_ortho.py --gpu 0

# 6. Run open-source VLM evaluations
bash run_open_eval.sh qwen
bash run_open_eval.sh internvl
```

Before running the full generation, verify that these files exist:

```text
data/versioned_data/hab3-episodes/val/social_rearrange_diverse.json.gz
data/versioned_data/hab3-episodes/checkpoint/social_nav_latest.pth
data/scene_datasets/hssd-hab/
data/humanoids/
data/hab3_bench_assets/
data/versioned_data/ycb/
```

Notes:

- `social_rearrange_diverse.json.gz` is the episode file used in the paper. It is not
  committed to this GitHub repo; get it from the Hugging Face dataset (see
  "Pre-Generated Dataset" above) or regenerate it (see "1. Generate the Episode Dataset").
- The `social_navigation` task loads the `social_nav_latest.pth` policy, which is
  included in the `hab3-episodes` download.
- The `social_rearrangement` task runs with oracle skills (`should_load_ckpt: False`
  in `pop_play_all.yaml`), so it needs no policy checkpoint.

For a quick smoke test:

```bash
python fix_dataset.py
python generate_video_dataset.py --gpu 0 --episodes 2 --resume

python evaluate_safety_benchmark_v2.py \
  --task all \
  --dry-run \
  --max-samples 5 \
  --views arm head \
  --modality rgb
```

## 1. Generate the Episode Dataset

The episode file used by the video generator is:

```text
data/versioned_data/hab3-episodes/val/social_rearrange_diverse.json.gz
```

If you need to regenerate it, run:

```bash
python habitat-lab/habitat/datasets/rearrange/run_episode_generator.py \
  --run \
  --config habitat-lab/habitat/datasets/rearrange/configs/hssd_diverse.yaml \
  --episodes-per-scene 10 \
  --out data/versioned_data/hab3-episodes/val/social_rearrange_diverse.json.gz
```

This file stores episode definitions such as scenes, objects, target objects, and receptacles. It does not contain rendered frames.

## 2. Fix the Episode Dataset

Before generating videos, fix invalid ghost-object references in the episode dataset:

```bash
python fix_dataset.py
```

The script edits:

```text
data/versioned_data/hab3-episodes/val/social_rearrange_diverse.json.gz
```

and creates a backup:

```text
data/versioned_data/hab3-episodes/val/social_rearrange_diverse-old.json.gz
```

Do not skip this step. Without it, `generate_video_dataset.py` may fail during Habitat task initialization.

## 3. Generate the Video Dataset

Run from the repository root:

```bash
python generate_video_dataset.py \
  --gpu 0 \
  --tasks social_navigation social_rearrangement \
  --human-types female_2 male_2 neutral_0 \
  --topdown-height auto \
  --episodes -1 \
  --resume
```

Output:

```text
habitat_video_dataset/
```

The command runs both simulation tasks for all three humanoid types. It uses merged mode by default, so RGB, depth, top-down video, trajectories, and camera parameters are collected in one simulator pass per task.

Useful variants:

```bash
# Small test run
python generate_video_dataset.py --gpu 0 --episodes 10 --resume

# One task and one humanoid
python generate_video_dataset.py \
  --gpu 0 \
  --tasks social_navigation \
  --human-types female_2 \
  --episodes -1 \
  --resume
```

Main output structure:

```text
habitat_video_dataset/
  female_2/
    social_navigation/
      scene_102344022/
        episode_0005/
          episode_meta.json
          trajectory.json
          camera_params.json
          rgb_meta.json
          depth_meta.json
          topdown.mp4
          rgb_full/
            arm/
            head/
            third_robot/
            third_human/
          depth/
            arm/
            head/
            third_robot/
            third_human/
```

## 4. Render Top-Down Orthographic Maps

Top-down VLM evaluation uses orthographic scene maps. Generate them with:

```bash
python render_topdown_ortho.py --gpu 0
```

Output:

```text
topdown_ortho/
```

## 5. Run Safety Benchmark Evaluation

The evaluation scripts read from:

```text
habitat_video_dataset/
```

and write results to:

```text
eval_safety_benchmark/
```

The benchmark supports two tasks:

```text
Task 1: Safety event classification
  A = no event
  B = scene collision
  C = human collision risk

Task 2: Collision warning
  A = no risk
  B = operational hazard
  C = catastrophic risk
```

### Closed-Source / API Models

Set API keys first:

```bash
export GEMINI_API_KEYS="key1,key2"
export OPENAI_API_KEY="your-openai-key"
```

Gemini example:

```bash
python evaluate_safety_benchmark_v2.py \
  --task all \
  --provider gemini \
  --views third_robot third_human arm head \
  --modality rgb \
  --max-samples 450 \
  --resume
```

OpenAI example:

```bash
python evaluate_safety_benchmark_v2.py \
  --task all \
  --provider openai \
  --openai-model gpt-5.5 \
  --views third_robot third_human arm head \
  --modality rgb \
  --max-samples 450 \
  --resume
```

### Open-Source Local Models

Single run:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n habitat python evaluate_safety_benchmark_v2_open.py \
  --task all \
  --provider local \
  --local-model qwen2.5-vl-7b \
  --views arm head \
  --modality rgb \
  --max-samples 450 \
  --request-interval 0 \
  --resume
```

Batch runs:

```bash
bash run_open_eval.sh qwen
bash run_open_eval.sh internvl
```

`run_open_eval.sh` evaluates several models, view settings, and input modalities. Results are grouped by model name under `eval_safety_benchmark/`. The GPU IDs are hard-coded inside the script (`GPU=3` for `qwen`, `GPU=0,2` for `internvl`); edit them to match your machine.

## 6. Evaluation Outputs

Example:

```text
eval_safety_benchmark/
  qwen2.5-vl-7b/
    all_qwen2.5-vl-7b_4v_rgb_n900.log
    all_qwen2.5-vl-7b_4v_rgb_n900_progress.jsonl
    all_qwen2.5-vl-7b_4v_rgb_n900_report.json
    all_qwen2.5-vl-7b_4v_rgb_n900_failures.jsonl
```

Files:

```text
*.log
  Run log.

*_progress.jsonl
  Per-sample predictions. Used by --resume.

*_report.json
  Accuracy, macro-F1, per-class F1, confusion matrix, structured metrics,
  and token usage.

*_failures.jsonl
  Incorrect predictions with episode path, sampled frames, ground truth,
  prediction, metadata, and raw model response.
```

The run name encodes the setting:

```text
all_qwen2.5-vl-7b_4v_rgb_n900_report.json
│   │             │  │   │    └── total samples
│   │             │  │   └────── input modality
│   │             │  └────────── views: 4v means third_robot + third_human + arm + head
│   │             └───────────── model name
│   └────────────────────────── task setting: all = Task 1 + Task 2
```

Note: `--max-samples` caps the number of samples per task, so with `--task all`
the total (`n`) is roughly double that value (e.g. `--max-samples 450` -> `n900`).

For paper tables, read the corresponding `*_report.json` files. The main fields are:

```text
task1_metrics.accuracy
task1_metrics.f1_macro
task1_metrics.f1_per_class
task1_metrics.confusion_matrix

task2_metrics.accuracy
task2_metrics.f1_macro
task2_metrics.far_c
task2_metrics.far_any
task2_metrics.confusion_matrix

task1_structured_metrics
task2_structured_metrics
```

Common result files (each is prefixed by `<task>_<model>_`, e.g. `all_qwen2.5-vl-7b_`):

```text
<task>_<model>_arm+head_rgb_n900_report.json
  Ego-view RGB setting.

<task>_<model>_arm+head_depth_n900_report.json
  Ego-view depth setting.

<task>_<model>_arm+head_rgbd_n900_report.json
  Ego-view RGB-D setting.

<task>_<model>_4v_rgb_n900_report.json
  Four-view RGB setting: third_robot, third_human, arm, head.

<task>_<model>_topdown_rgb_n900_report.json
  Top-down trajectory map setting.
```

## Recommended GitHub Ignore List

The following paths are usually large and should not be committed:

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

## Citation

```bibtex
@article{wang2025touchsafebench,
  title={Probing Collision Grounding in Vision-Language Models for Safe Human--Robot Collaboration},
  author={Wang, Jun and others},
  journal={arXiv preprint arXiv:2605.31196},
  year={2025}
}
```

## License

This project builds on Habitat-Lab, which is released under the MIT License. See `HABITAT_ORIGINAL_README.md` for the original Habitat-Lab README.
