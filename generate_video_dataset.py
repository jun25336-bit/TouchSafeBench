#!/usr/bin/env python3
"""
Habitat Social Video Dataset Generator
=======================================
Generates structured video datasets for Social Navigation / Social Rearrangement.

Per episode output (each sensor independent):
  RGB preview videos (frame_skip subsampled, per-sensor MP4):
    - rgb/arm.mp4                Spot arm (10fps)
    - rgb/head.mp4               Human head
    - rgb/third_robot.mp4        Third-person Spot
    - rgb/third_human.mp4        Third-person Human
  RGB full frames (every step, JPEG, for VLM input):
    - rgb_full/arm/000000.jpg .. Spot arm full
    - rgb_full/head/             Human head full
    - rgb_full/third_robot/      Third-person Spot full
    - rgb_full/third_human/      Third-person Human full
  Depth maps (every step, 8-bit PNG, 640x480):
    - depth/arm/                 Spot arm depth
    - depth/head/                Human head depth
    - depth/third_robot/         Third-person Spot depth
    - depth/third_human/         Third-person Human depth

Output directory structure:
  habitat_video_dataset/
  ├── female_0/
  │   ├── social_navigation/
  │   │   ├── scene_XXXXXX/
  │   │   │   ├── episode_0000/
  │   │   │   │   ├── episode_meta.json
  │   │   │   │   ├── camera_params.json
  │   │   │   │   ├── trajectory.json
  │   │   │   │   ├── depth_meta.json
  │   │   │   │   ├── rgb_meta.json
  │   │   │   │   ├── rgb/
  │   │   │   │   │   ├── arm.mp4
  │   │   │   │   │   ├── head.mp4
  │   │   │   │   │   ├── third_robot.mp4
  │   │   │   │   │   └── third_human.mp4
  │   │   │   │   ├── rgb_full/
  │   │   │   │   │   ├── arm/000000.jpg ...
  │   │   │   │   │   ├── head/000000.jpg ...
  │   │   │   │   │   ├── third_robot/000000.jpg ...
  │   │   │   │   │   └── third_human/000000.jpg ...
  │   │   │   │   └── depth/
  │   │   │   │       ├── arm/000000.png ...
  │   │   │   │       ├── head/000000.png ...
  │   │   │   │       ├── third_robot/000000.png ...
  │   │   │   │       └── third_human/000000.png ...
  │   │   │   └── ...
  │   │   └── ...
  │   └── social_rearrangement/
  │       └── ...
  ├── male_0/
  │   └── ...
  └── neutral_0/
      └── ...

Usage:
  python generate_video_dataset.py                # all (3 humanoids x 2 tasks)
  python generate_video_dataset.py --tasks social_navigation
  python generate_video_dataset.py --human-types female_0 male_0
  python generate_video_dataset.py --views rgb_third depth_first
"""

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# Configuration
# ============================================================

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "output_dir": os.path.join(_REPO_ROOT, "habitat_video_dataset"),
    "work_dir": _REPO_ROOT,

    # ---- Episode data ----
    "episode_data": {
        "social_navigation": "data/versioned_data/hab3-episodes/val/social_rearrange_diverse.json.gz",
        "social_rearrangement": "data/versioned_data/hab3-episodes/val/social_rearrange_diverse.json.gz",
    },

    # ---- Checkpoint ----
    "checkpoints": {
        "social_navigation": os.path.join(_REPO_ROOT, "data/versioned_data/hab3-episodes/checkpoint/social_nav_latest.pth"),
        "social_rearrangement": os.path.join(_REPO_ROOT, "data/versioned_data/hab3-episodes/checkpoint/social_rearrange_latest.pth"),
    },

    # ---- Video parameters ----
    "video_height": 1080,
    "video_width": 1920,
    "video_fps": 10,
    "frame_skip": 3,  # sample 1 frame every N steps (1=every step, 3=1/3 of frames)

    # ---- Depth map resolution (independent of video) ----
    "depth_height": 480,
    "depth_width": 640,

    # ---- Episode control ----
    "num_episodes": -1,  # -1 = all
    "num_environments": 1,
    "video_option": '["disk"]',
    "resume": True,  # resume from checkpoint: skip completed episodes in _raw

    # ---- Humanoid types ----
    "human_types": ["female_2", "male_2", "neutral_0"],

    # ---- Additional object paths ----
    "additional_object_paths": [
        os.path.join(_REPO_ROOT, "data/hab3_bench_assets/"),
        os.path.join(_REPO_ROOT, "data/versioned_data/ycb/configs/"),
    ],
}

# ============================================================
# View definitions:  (config_yaml, sensor_filter_env, extra_cli_args)
# ============================================================

# view_key mapping:
#   config yaml file                       mounts which extra_sim_sensor
#   HABITAT_VIDEO_SENSOR_FILTER env var    controls which sensors evaluator renders
#   extra_cli_args                         CLI overrides for resolution etc.

VIEW_DEFS = {
    "social_navigation": {
        "topdown": {
            "config": "social_nav/social_nav.yaml",
            "filter": "third_rgb",
            "cli": [
                "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.height={h}",
                "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.width={w}",
            ],
        },
        "rgb_third": {
            "config": "social_nav/social_nav.yaml",
            "filter": "third_rgb",
            "cli": [
                "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.height={h}",
                "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.width={w}",
            ],
        },
        "rgb_first": {
            "config": "social_nav/social_nav_first.yaml",
            "filter": "arm_rgb,head_rgb",
            "cli": [
                "habitat_baselines.eval.extra_sim_sensors.arm_rgb_sensor.height={h}",
                "habitat_baselines.eval.extra_sim_sensors.arm_rgb_sensor.width={w}",
                "habitat.simulator.agents.agent_1.sim_sensors.head_rgb_sensor.height={h}",
                "habitat.simulator.agents.agent_1.sim_sensors.head_rgb_sensor.width={w}",
            ],
        },
        "depth_third": {
            "config": "social_nav/social_nav_depth.yaml",
            "filter": "third_depth",
            "cli": [
                "habitat_baselines.eval.extra_sim_sensors.third_depth_sensor.height={h}",
                "habitat_baselines.eval.extra_sim_sensors.third_depth_sensor.width={w}",
            ],
        },
        "depth_first": {
            "config": "social_nav/social_nav_first_depth.yaml",
            "filter": "depth_render",
            "cli": [
                "habitat_baselines.eval.extra_sim_sensors.render_arm_depth_sensor.height={h}",
                "habitat_baselines.eval.extra_sim_sensors.render_arm_depth_sensor.width={w}",
                "habitat_baselines.eval.extra_sim_sensors.render_head_depth_sensor.height={h}",
                "habitat_baselines.eval.extra_sim_sensors.render_head_depth_sensor.width={w}",
            ],
        },
    },
    "social_rearrangement": {
        "topdown": {
            "config": "social_rearrange/pop_play.yaml",
            "filter": "third_rgb",
            "cli": [
                "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.height={h}",
                "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.width={w}",
            ],
        },
        "rgb_third": {
            "config": "social_rearrange/pop_play.yaml",
            "filter": "third_rgb",
            "cli": [
                "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.height={h}",
                "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.width={w}",
            ],
        },
        "rgb_first": {
            "config": "social_rearrange/pop_play_first.yaml",
            "filter": "arm_rgb,head_rgb",
            "cli": [
                "habitat_baselines.eval.extra_sim_sensors.arm_rgb_sensor.height={h}",
                "habitat_baselines.eval.extra_sim_sensors.arm_rgb_sensor.width={w}",
                "habitat.simulator.agents.agent_1.sim_sensors.head_rgb_sensor.height={h}",
                "habitat.simulator.agents.agent_1.sim_sensors.head_rgb_sensor.width={w}",
            ],
        },
        "depth_third": {
            "config": "social_rearrange/pop_play_depth.yaml",
            "filter": "third_depth",
            "cli": [
                "habitat_baselines.eval.extra_sim_sensors.third_depth_sensor.height={h}",
                "habitat_baselines.eval.extra_sim_sensors.third_depth_sensor.width={w}",
            ],
        },
        "depth_first": {
            "config": "social_rearrange/pop_play_first_depth.yaml",
            "filter": "depth_render",
            "cli": [
                "habitat_baselines.eval.extra_sim_sensors.render_arm_depth_sensor.height={h}",
                "habitat_baselines.eval.extra_sim_sensors.render_arm_depth_sensor.width={w}",
                "habitat_baselines.eval.extra_sim_sensors.render_head_depth_sensor.height={h}",
                "habitat_baselines.eval.extra_sim_sensors.render_head_depth_sensor.width={w}",
            ],
        },
    },
}

# ============================================================
# Merged configuration: collect all sensors in a single simulation
# ============================================================

MERGED_DEFS = {
    "social_navigation": {
        "config": "social_nav/social_nav_all.yaml",
        "cli": [
            # RGB sensors (video resolution)
            "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.height={vh}",
            "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.width={vw}",
            "habitat_baselines.eval.extra_sim_sensors.arm_rgb_sensor.height={vh}",
            "habitat_baselines.eval.extra_sim_sensors.arm_rgb_sensor.width={vw}",
            "habitat.simulator.agents.agent_1.sim_sensors.head_rgb_sensor.height={vh}",
            "habitat.simulator.agents.agent_1.sim_sensors.head_rgb_sensor.width={vw}",
            # Depth sensors (depth resolution)
            "habitat_baselines.eval.extra_sim_sensors.third_depth_sensor.height={dh}",
            "habitat_baselines.eval.extra_sim_sensors.third_depth_sensor.width={dw}",
            "habitat_baselines.eval.extra_sim_sensors.render_arm_depth_sensor.height={dh}",
            "habitat_baselines.eval.extra_sim_sensors.render_arm_depth_sensor.width={dw}",
            "habitat.simulator.agents.agent_0.sim_sensors.render_arm_depth_sensor.height={dh}",
            "habitat.simulator.agents.agent_0.sim_sensors.render_arm_depth_sensor.width={dw}",
            "habitat_baselines.eval.extra_sim_sensors.render_head_depth_sensor.height={dh}",
            "habitat_baselines.eval.extra_sim_sensors.render_head_depth_sensor.width={dw}",
            "habitat.simulator.agents.agent_1.sim_sensors.render_head_depth_sensor.height={dh}",
            "habitat.simulator.agents.agent_1.sim_sensors.render_head_depth_sensor.width={dw}",
        ],
    },
    "social_rearrangement": {
        "config": "social_rearrange/pop_play_all.yaml",
        "cli": [
            "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.height={vh}",
            "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.width={vw}",
            "habitat_baselines.eval.extra_sim_sensors.arm_rgb_sensor.height={vh}",
            "habitat_baselines.eval.extra_sim_sensors.arm_rgb_sensor.width={vw}",
            "habitat.simulator.agents.agent_1.sim_sensors.head_rgb_sensor.height={vh}",
            "habitat.simulator.agents.agent_1.sim_sensors.head_rgb_sensor.width={vw}",
            "habitat_baselines.eval.extra_sim_sensors.third_depth_sensor.height={dh}",
            "habitat_baselines.eval.extra_sim_sensors.third_depth_sensor.width={dw}",
            "habitat_baselines.eval.extra_sim_sensors.render_arm_depth_sensor.height={dh}",
            "habitat_baselines.eval.extra_sim_sensors.render_arm_depth_sensor.width={dw}",
            "habitat.simulator.agents.agent_0.sim_sensors.render_arm_depth_sensor.height={dh}",
            "habitat.simulator.agents.agent_0.sim_sensors.render_arm_depth_sensor.width={dw}",
            "habitat_baselines.eval.extra_sim_sensors.render_head_depth_sensor.height={dh}",
            "habitat_baselines.eval.extra_sim_sensors.render_head_depth_sensor.width={dw}",
            "habitat.simulator.agents.agent_1.sim_sensors.render_head_depth_sensor.height={dh}",
            "habitat.simulator.agents.agent_1.sim_sensors.render_head_depth_sensor.width={dw}",
        ],
    },
}


# ============================================================
# Social Nav shared parameters
# ============================================================

def _common_nav_args() -> List[str]:
    return [
        "benchmark/multi_agent=hssd_spot_human_social_nav",
        "habitat.task.actions.agent_0_base_velocity.longitudinal_lin_speed=10.0",
        "habitat.task.actions.agent_0_base_velocity.ang_speed=10.0",
        "habitat.task.actions.agent_0_base_velocity.allow_dyn_slide=True",
        "habitat.task.actions.agent_0_base_velocity.enable_rotation_check_for_dyn_slide=False",
        "habitat.task.actions.agent_1_oracle_nav_randcoord_action.human_stop_and_walk_to_robot_distance_threshold=-1.0",
        "habitat.task.actions.agent_1_oracle_nav_randcoord_action.lin_speed=10.0",
        "habitat.task.actions.agent_1_oracle_nav_randcoord_action.ang_speed=10.0",
        "habitat.task.actions.agent_1_oracle_nav_action.lin_speed=10.0",
        "habitat.task.actions.agent_1_oracle_nav_action.ang_speed=10.0",
        "habitat.task.measurements.social_nav_reward.facing_human_reward=3.0",
        "habitat.task.measurements.social_nav_reward.count_coll_pen=0.01",
        "habitat.task.measurements.social_nav_reward.max_count_colls=-1",
        "habitat.task.measurements.social_nav_reward.count_coll_end_pen=5",
        "habitat.task.measurements.social_nav_reward.use_geo_distance=True",
        "habitat.task.measurements.social_nav_reward.facing_human_dis=3.0",
        "habitat.task.measurements.social_nav_seek_success.following_step_succ_threshold=400",
        "habitat.task.measurements.social_nav_seek_success.need_to_face_human=True",
        "habitat.task.measurements.social_nav_seek_success.use_geo_distance=True",
        "habitat.task.measurements.social_nav_seek_success.facing_threshold=0.5",
        "habitat.task.lab_sensors.humanoid_detector_sensor.return_image=True",
        "habitat.task.lab_sensors.humanoid_detector_sensor.is_return_image_bbox=True",
        "habitat.task.success_reward=10.0",
        "habitat.task.end_on_success=False",
        "habitat.task.slack_reward=-0.1",
        "habitat.environment.max_episode_steps=1500",
        "habitat.simulator.kinematic_mode=True",
        "habitat.simulator.ac_freq_ratio=4",
        "habitat.simulator.ctrl_freq=120",
        "habitat.simulator.agents.agent_0.joint_start_noise=0.0",
        "habitat_baselines.load_resume_state_config=False",
    ]


def _common_rearrange_args() -> List[str]:
    return [
        "+habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.select_random_goal=False",
        "+habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx=1",
    ]


# ============================================================
# Resume: detect completed episodes and generate filtered dataset
# ============================================================

def detect_completed_episodes(
    raw_dir: str, output_dir: str = None, task: str = None,
) -> Set[str]:
    """Return the set of completed episode IDs.

    Scans two locations:
    1. _raw directory for trajectory_ep{id}.json (produced mid-run).
    2. Final output directory for episode directories.

    An episode in the output directory is considered complete only when all
    of these exist: trajectory.json, camera_params.json, topdown.mp4,
    rgb_full/, depth/. Incomplete episodes are cleaned up for re-generation.
    """
    completed: Set[str] = set()
    cleaned_info: List[str] = []

    # 1. Scan _raw directory (produced mid-run, not yet organized)
    if os.path.exists(raw_dir):
        for f in Path(raw_dir).glob("trajectory_ep*.json"):
            m = re.match(r"trajectory_ep(\d+)\.json", f.name)
            if m:
                completed.add(str(int(m.group(1))))

    # 2. Scan final output directory
    if output_dir and task:
        task_dir = os.path.join(output_dir, task)
        if os.path.exists(task_dir):
            for scene_dir in sorted(Path(task_dir).iterdir()):
                if not scene_dir.is_dir() or scene_dir.name.startswith("_"):
                    continue
                for ep_dir in sorted(scene_dir.iterdir()):
                    if not ep_dir.is_dir():
                        continue
                    m = re.match(r"episode_(\d+)", ep_dir.name)
                    if not m:
                        continue
                    ep_id = str(int(m.group(1)))
                    has_all = (
                        (ep_dir / "trajectory.json").exists()
                        and (ep_dir / "camera_params.json").exists()
                        and (ep_dir / "topdown.mp4").exists()
                        and (ep_dir / "rgb_full").is_dir()
                        and (ep_dir / "depth").is_dir()
                    )
                    if has_all:
                        completed.add(ep_id)
                    else:
                        missing = []
                        for name in ("trajectory.json", "camera_params.json", "topdown.mp4"):
                            if not (ep_dir / name).exists():
                                missing.append(name)
                        for name in ("rgb_full", "depth"):
                            if not (ep_dir / name).is_dir():
                                missing.append(name + "/")
                        cleaned_info.append(
                            f"    episode {ep_id} ({scene_dir.name}): "
                            f"missing {', '.join(missing)}"
                        )
                        shutil.rmtree(str(ep_dir))

    if cleaned_info:
        print(f"  [resume] Cleaned {len(cleaned_info)} incomplete episodes (will regenerate):")
        for line in cleaned_info[:10]:
            print(line)
        if len(cleaned_info) > 10:
            print(f"    ... and {len(cleaned_info) - 10} more")

    return completed


def cleanup_partial_raw_data(raw_dir: str):
    """Clean up partial episode data in the raw directory.

    Completed episodes have trajectory files; interrupted episodes only
    have partial topdown/rgb/depth data. Remove the latter to avoid
    conflicts with the next evaluator run.
    """
    if not os.path.exists(raw_dir):
        return

    valid = _get_valid_episode_ids(raw_dir) or set()

    patterns = [
        ("topdown_ep*", r"topdown_ep(\d+)"),
        ("rgb_all_ep*", r"rgb_all_ep(\d+)"),
        ("rgb_split_ep*", r"rgb_split_ep(\d+)"),
        ("depth_png_ep*", r"depth_png_ep(\d+)"),
    ]

    cleaned = 0
    for glob_pat, re_pat in patterns:
        for item in Path(raw_dir).glob(glob_pat):
            if not item.is_dir():
                continue
            m = re.match(re_pat, item.name)
            if m and m.group(1) not in valid:
                shutil.rmtree(str(item))
                cleaned += 1

    if cleaned:
        print(f"  [resume] Cleaned {cleaned} partial episode artifacts from raw directory")


def create_filtered_dataset(
    original_data_path: str,
    work_dir: str,
    completed_ids: Set[str],
    output_path: str,
) -> int:
    """Read the original dataset, remove completed episodes, write a new file.

    Returns the number of remaining episodes.
    """
    full_path = os.path.join(work_dir, original_data_path)
    with gzip.open(full_path, "rt") as f:
        data = json.load(f)

    original_count = len(data.get("episodes", []))
    data["episodes"] = [
        ep for ep in data.get("episodes", [])
        if str(ep["episode_id"]) not in completed_ids
    ]
    remaining = len(data["episodes"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wt") as f:
        json.dump(data, f)

    print(f"  [resume] Original {original_count} episodes, "
          f"completed {len(completed_ids)}, remaining {remaining}")
    return remaining


# ============================================================
# Load episode data -> episode_id -> scene_id mapping
# ============================================================

def count_dataset_episodes(data_path: str, work_dir: str) -> int:
    """Return total number of episodes in the dataset."""
    full_path = os.path.join(work_dir, data_path)
    with gzip.open(full_path, "rt") as f:
        data = json.load(f)
    return len(data.get("episodes", []))


def count_dataset_scenes(data_path: str, work_dir: str) -> int:
    """Return number of scenes in the dataset."""
    full_path = os.path.join(work_dir, data_path)
    with gzip.open(full_path, "rt") as f:
        data = json.load(f)
    scenes = set(ep["scene_id"] for ep in data.get("episodes", []))
    return len(scenes)


def load_episode_metadata(data_path: str, work_dir: str) -> Dict[str, dict]:
    """Read per-episode metadata from the dataset JSON.gz.

    Returns episode_id -> {scene_name, rigid_objs, target_objs, ...} mapping.
    """
    full_path = os.path.join(work_dir, data_path)
    print(f"  Loading episode data: {full_path}")

    with gzip.open(full_path, "rt") as f:
        data = json.load(f)

    mapping = {}
    for ep in data.get("episodes", []):
        ep_id = str(ep["episode_id"])
        scene_id = ep["scene_id"]
        # scene_id format: "data/.../107734479_176000442.scene_instance.json"
        #              or: "data/.../102344022.scene_instance.json"
        scene_name = Path(scene_id).stem  # e.g. "107734479_176000442.scene_instance"
        if ".scene_instance" in scene_name:
            scene_name = scene_name.split(".scene_instance")[0]
        else:
            scene_name = scene_name.replace(".", "_")

        # Extract object names in the scene (strip .object_config.json suffix)
        rigid_objs = []
        for obj_entry in ep.get("rigid_objs", []):
            obj_handle = obj_entry[0] if isinstance(obj_entry, list) else obj_entry
            obj_name = obj_handle.replace(".object_config.json", "")
            rigid_objs.append(obj_name)

        # Extract target objects (objects to be rearranged)
        target_objs = list(ep.get("targets", {}).keys())

        # Extract object_labels from info (object role labels)
        object_labels = ep.get("info", {}).get("object_labels", {})

        mapping[ep_id] = {
            "scene_name": scene_name,
            "rigid_objs": rigid_objs,
            "target_objs": target_objs,
            "object_labels": object_labels,
        }

    print(f"  Loaded metadata for {len(mapping)} episodes")
    return mapping


# ============================================================
# Build evaluation commands
# ============================================================

def _humanoid_cli_overrides(human_type: str) -> List[str]:
    """Generate CLI overrides for humanoid URDF and motion data."""
    urdf = f"data/humanoids/humanoid_data/{human_type}/{human_type}.urdf"
    motion = f"data/humanoids/humanoid_data/{human_type}/{human_type}_motion_data_smplx.pkl"
    return [
        f"habitat.simulator.agents.agent_1.articulated_agent_urdf={urdf}",
        f"habitat.simulator.agents.agent_1.motion_data_path={motion}",
    ]


def build_command(
    task: str,
    view_key: str,
    raw_video_dir: str,
    cfg: dict,
    human_type: str = "female_0",
) -> List[str]:
    """Build the complete evaluation command."""
    view_def = VIEW_DEFS[task][view_key]
    ckpt = cfg["checkpoints"][task]
    ep_data = cfg["episode_data"][task]

    # Depth views use independent lower resolution; RGB views use video resolution
    if view_key.startswith("depth_"):
        h = cfg["depth_height"]
        w = cfg["depth_width"]
    else:
        h = cfg["video_height"]
        w = cfg["video_width"]

    cmd = [
        sys.executable, "-u", "-m", "habitat_baselines.run",
        f"--config-name={view_def['config']}",
    ]

    # Task-specific parameters
    if task == "social_navigation":
        cmd += _common_nav_args()
    elif task == "social_rearrangement":
        cmd += _common_rearrange_args()

    # Humanoid type overrides
    cmd += _humanoid_cli_overrides(human_type)

    # View-specific sensor resolution
    cmd += [arg.format(h=h, w=w) for arg in view_def["cli"]]

    num_ep = cfg["num_episodes"]
    if num_ep == -1:
        num_ep = count_dataset_episodes(ep_data, cfg.get("work_dir", "."))

    n_envs = cfg.get("num_environments", 1)
    n_scenes = count_dataset_scenes(ep_data, cfg.get("work_dir", "."))
    if num_ep > 0:
        n_envs = min(n_envs, num_ep // 2 or 1)
    if n_scenes > 0:
        n_envs = min(n_envs, n_scenes)

    cmd += [
        "habitat_baselines.evaluate=True",
        f"habitat_baselines.num_environments={n_envs}",
        'habitat_baselines.eval.video_option=["disk"]',
        f"habitat_baselines.video_fps={cfg['video_fps']}",
        f"habitat_baselines.video_dir={raw_video_dir}",
        f"habitat_baselines.eval_ckpt_path_dir={ckpt}",
        f"habitat.dataset.data_path={ep_data}",
        "habitat.dataset.scenes_dir=data/scene_datasets/",
    ]

    if num_ep > 0:
        cmd.append(f"habitat_baselines.test_episode_count={num_ep}")

    # Additional object paths
    if cfg["additional_object_paths"]:
        paths_str = ",".join(f'"{p}"' for p in cfg["additional_object_paths"])
        cmd.append(f'+habitat.simulator.additional_object_paths=[{paths_str}]')

    return cmd


def build_merged_command(
    task: str,
    raw_dir: str,
    cfg: dict,
    human_type: str = "female_0",
) -> List[str]:
    """Build the merged-mode evaluation command (all sensors in one simulation)."""
    merged_def = MERGED_DEFS[task]
    ckpt = cfg["checkpoints"][task]
    ep_data = cfg["episode_data"][task]

    vh = cfg["video_height"]
    vw = cfg["video_width"]
    dh = cfg["depth_height"]
    dw = cfg["depth_width"]

    cmd = [
        sys.executable, "-u", "-m", "habitat_baselines.run",
        f"--config-name={merged_def['config']}",
    ]

    if task == "social_navigation":
        cmd += _common_nav_args()
    elif task == "social_rearrangement":
        cmd += _common_rearrange_args()

    cmd += _humanoid_cli_overrides(human_type)

    cmd += [arg.format(vh=vh, vw=vw, dh=dh, dw=dw) for arg in merged_def["cli"]]

    num_ep = cfg["num_episodes"]
    if num_ep == -1:
        num_ep = count_dataset_episodes(ep_data, cfg.get("work_dir", "."))

    n_envs = cfg.get("num_environments", 1)
    n_scenes = count_dataset_scenes(ep_data, cfg.get("work_dir", "."))
    if num_ep > 0:
        n_envs = min(n_envs, num_ep // 2 or 1)
    if n_scenes > 0:
        n_envs = min(n_envs, n_scenes)

    video_option = cfg.get("video_option", '["disk"]')
    cmd += [
        "habitat_baselines.evaluate=True",
        f"habitat_baselines.num_environments={n_envs}",
        f"habitat_baselines.eval.video_option={video_option}",
        f"habitat_baselines.video_fps={cfg['video_fps']}",
        f"habitat_baselines.video_dir={raw_dir}",
        f"habitat_baselines.eval_ckpt_path_dir={ckpt}",
        f"habitat.dataset.data_path={ep_data}",
        "habitat.dataset.scenes_dir=data/scene_datasets/",
    ]

    if num_ep > 0:
        cmd.append(f"habitat_baselines.test_episode_count={num_ep}")

    if cfg["additional_object_paths"]:
        paths_str = ",".join(f'"{p}"' for p in cfg["additional_object_paths"])
        cmd.append(f'+habitat.simulator.additional_object_paths=[{paths_str}]')

    return cmd


# ============================================================
# Organize videos into target directory structure
# ============================================================

_RENUM_MAP: Optional[Dict[str, Tuple[str, str]]] = None


def build_renumber_map(filtered_dataset_path: str) -> Dict[str, Tuple[str, str]]:
    """Build renumbered_index -> (original_episode_id, scene_name) mapping
    from the filtered dataset.

    Habitat's RearrangeDataset.from_json renumbers episode_id to index str(i).
    This function reads the filtered dataset and builds the mapping in load order.
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    if not os.path.exists(filtered_dataset_path):
        return mapping
    try:
        with gzip.open(filtered_dataset_path, "rt") as f:
            data = json.load(f)
        for i, ep in enumerate(data.get("episodes", [])):
            orig_id = str(ep["episode_id"])
            scene_name = Path(ep["scene_id"]).stem.replace(".scene_instance", "")
            mapping[str(i)] = (orig_id, scene_name)
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    return mapping


def _resolve_episode_info(raw_dir: str, ep_id: str, ep_meta_map: dict) -> Tuple[str, str]:
    """Resolve the true scene_name and episode_id for a given evaluator ep_id.

    scene_name is preferentially read from the trajectory file (the scene
    actually used by the evaluator), since multi-env parallelism may assign
    scenes differently than the original dataset.

    episode_id is preferentially resolved via _RENUM_MAP (filtered dataset
    index -> original ID mapping), since Habitat renumbers episode_ids
    when loading a filtered dataset.

    Returns (scene_name, real_episode_id).
    """
    real_ep_id = ep_id
    if _RENUM_MAP and ep_id in _RENUM_MAP:
        real_ep_id = _RENUM_MAP[ep_id][0]

    traj_path = os.path.join(raw_dir, f"trajectory_ep{ep_id}.json")
    if os.path.exists(traj_path):
        try:
            with open(traj_path) as f:
                traj = json.load(f)
            scene_id = traj.get("scene_id", "")
            scene_name = Path(scene_id).stem.replace(".scene_instance", "")
            if scene_name:
                return scene_name, real_ep_id
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    if _RENUM_MAP and ep_id in _RENUM_MAP:
        return _RENUM_MAP[ep_id][1], real_ep_id

    ep_info = ep_meta_map.get(str(int(ep_id)), {})
    scene_name = ep_info.get("scene_name", "unknown_scene")
    return scene_name, real_ep_id


AVAILABLE_HUMAN_TYPES = [
    "female_0", "female_1", "female_2", "female_3",
    "male_0", "male_1", "male_2", "male_3",
    "neutral_0", "neutral_1", "neutral_2", "neutral_3",
]



# ============================================================
# Organize trajectory files
# ============================================================

def organize_trajectories(
    task: str,
    raw_dir: str,
    output_dir: str,
    ep_meta_map: Dict[str, dict],
) -> int:
    """Move trajectory JSON files from the raw directory to the dataset directory."""
    if not os.path.exists(raw_dir):
        return 0

    traj_files = sorted(Path(raw_dir).glob("trajectory_ep*.json"))
    count = 0
    for tf in traj_files:
        # Extract episode_id from filename: trajectory_ep1140.json -> "1140"
        match = re.match(r"trajectory_ep(\d+)\.json", tf.name)
        if not match:
            continue
        ep_id = match.group(1)

        scene_name, real_ep_id = _resolve_episode_info(raw_dir, ep_id, ep_meta_map)

        ep_dir = os.path.join(
            output_dir, task,
            f"scene_{scene_name}",
            f"episode_{real_ep_id.zfill(4)}",
        )
        os.makedirs(ep_dir, exist_ok=True)
        target_path = os.path.join(ep_dir, "trajectory.json")

        if real_ep_id != ep_id:
            try:
                with open(str(tf), "r") as f:
                    traj = json.load(f)
                traj["episode_id"] = int(real_ep_id)
                with open(target_path, "w") as f:
                    json.dump(traj, f, indent=2)
                tf.unlink()
            except (json.JSONDecodeError, OSError):
                os.replace(str(tf), target_path)
        else:
            os.replace(str(tf), target_path)
        count += 1

    return count


# ============================================================
# Organize camera parameter files
# ============================================================

_CAMERA_PARAM_PATTERNS = ("arm_depth_render", "head_depth_render", "third_depth")


def _filter_camera_params(data: dict) -> dict:
    """Keep only intrinsics and extrinsics for the 4 cameras matching depth directories."""
    if "intrinsics" in data:
        data["intrinsics"] = {
            k: v for k, v in data["intrinsics"].items()
            if any(p in k for p in _CAMERA_PARAM_PATTERNS)
        }
    for frame_key in ("frames", "extrinsics"):
        if frame_key in data and isinstance(data[frame_key], list):
            data[frame_key] = [
                {k: v for k, v in frame.items()
                 if any(p in k for p in _CAMERA_PARAM_PATTERNS)}
                for frame in data[frame_key]
            ]
    return data


def organize_camera_params(
    task: str,
    raw_dir: str,
    output_dir: str,
    ep_meta_map: Dict[str, dict],
) -> int:
    """Move camera parameter JSON files from raw to dataset directory.
    Keeps only the 4 depth-aligned cameras (arm, head, third_robot, third_human).
    """
    if not os.path.exists(raw_dir):
        return 0

    cam_files = sorted(Path(raw_dir).glob("camera_params_ep*.json"))
    count = 0
    for cf in cam_files:
        match = re.match(r"camera_params_ep(\d+)\.json", cf.name)
        if not match:
            continue
        ep_id = match.group(1)

        scene_name, real_ep_id = _resolve_episode_info(raw_dir, ep_id, ep_meta_map)

        ep_dir = os.path.join(
            output_dir, task,
            f"scene_{scene_name}",
            f"episode_{real_ep_id.zfill(4)}",
        )
        os.makedirs(ep_dir, exist_ok=True)
        target_path = os.path.join(ep_dir, "camera_params.json")

        with open(str(cf), "r") as f:
            cam_data = json.load(f)
        cam_data = _filter_camera_params(cam_data)
        if real_ep_id != ep_id and "episode_id" in cam_data:
            cam_data["episode_id"] = int(real_ep_id)
        with open(target_path, "w") as f:
            json.dump(cam_data, f, indent=2)
        cf.unlink()
        count += 1

    return count


# ============================================================
# Write episode_meta.json
# ============================================================

def write_episode_metas(
    task: str,
    output_dir: str,
    ep_meta_map: Dict[str, dict],
    human_type: str,
) -> int:
    """Scan generated episode directories and write episode_meta.json where missing.

    episode_meta.json contains episode-level info (scene, task, objects, etc.),
    independent of the specific output format (video/PNG/JPEG).
    """
    task_dir = os.path.join(output_dir, task)
    if not os.path.exists(task_dir):
        return 0

    count = 0
    for scene_dir in sorted(Path(task_dir).iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith("_"):
            continue
        for ep_dir in sorted(scene_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            meta_path = ep_dir / "episode_meta.json"
            if meta_path.exists():
                continue
            match = re.match(r"episode_(\d+)", ep_dir.name)
            if not match:
                continue
            ep_id = match.group(1)
            traj_file = ep_dir / "trajectory.json"
            real_ep_id = ep_id
            real_scene = scene_dir.name.replace("scene_", "")
            if traj_file.exists():
                try:
                    with open(traj_file) as tf:
                        tdata = json.load(tf)
                    real_ep_id = str(tdata.get("episode_id", ep_id))
                    sid = tdata.get("scene_id", "")
                    if sid:
                        real_scene = Path(sid).stem.replace(".scene_instance", "")
                except (json.JSONDecodeError, KeyError):
                    pass
            ep_info = ep_meta_map.get(real_ep_id, ep_meta_map.get(str(int(ep_id)), {}))

            meta = {
                "episode_id": real_ep_id,
                "scene_id": ep_info.get("scene_name", real_scene),
                "task": task,
                "human_type": human_type,
                "rigid_objs": ep_info.get("rigid_objs", []),
                "target_objs": ep_info.get("target_objs", []),
                "object_labels": ep_info.get("object_labels", {}),
            }
            os.makedirs(str(ep_dir), exist_ok=True)
            with open(str(meta_path), "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            count += 1

    return count


# ============================================================
# Organize per-sensor RGB videos
# ============================================================

RGB_SENSOR_SHORT_NAMES = {
    "articulated_agent_arm_rgb": "arm",
    "head_rgb": "head",
}


def _rgb_short_name(sensor_uuid: str) -> str:
    """Map RGB sensor UUID to a short directory name."""
    for pattern, short in RGB_SENSOR_SHORT_NAMES.items():
        if pattern in sensor_uuid:
            return short
    if "third_rgb" in sensor_uuid:
        if "agent_0" in sensor_uuid:
            return "third_robot"
        if "agent_1" in sensor_uuid:
            return "third_human"
        return "third"
    return sensor_uuid


def organize_rgb_split(
    task: str,
    raw_dir: str,
    output_dir: str,
    ep_meta_map: Dict[str, dict],
) -> int:
    """Move per-sensor RGB videos from raw directory to dataset directory.

    Raw structure:   rgb_split_ep{id}/{sensor_uuid}.mp4
    Target structure: episode_{id}/rgb/{short_name}.mp4
    """
    if not os.path.exists(raw_dir):
        return 0

    split_dirs = sorted(Path(raw_dir).glob("rgb_split_ep*"))
    count = 0
    for sd in split_dirs:
        if not sd.is_dir():
            continue
        match = re.match(r"rgb_split_ep(\d+)", sd.name)
        if not match:
            continue

        # Skip residual directories without videos (first frame of next episode at evaluator exit)
        mp4_files = sorted(sd.glob("*.mp4"))
        if not mp4_files:
            shutil.rmtree(str(sd))
            continue

        ep_id = match.group(1)
        scene_name, real_ep_id = _resolve_episode_info(raw_dir, ep_id, ep_meta_map)

        ep_dir = os.path.join(
            output_dir, task,
            f"scene_{scene_name}",
            f"episode_{real_ep_id.zfill(4)}",
        )

        for mp4_file in mp4_files:
            sensor_uuid = mp4_file.stem
            short = _rgb_short_name(sensor_uuid)
            target_dir = os.path.join(ep_dir, "rgb")
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, f"{short}.mp4")
            os.replace(str(mp4_file), target_path)

        try:
            sd.rmdir()
        except OSError:
            pass

        count += 1

    return count


# ============================================================
# Organize RGB full-frame directories
# ============================================================

def organize_rgb_frames(
    task: str,
    raw_dir: str,
    output_dir: str,
    ep_meta_map: Dict[str, dict],
) -> int:
    """Move per-sensor RGB full frames (JPEG) from raw to dataset directory.

    Raw structure:    rgb_all_ep{id}/{sensor_uuid}/000000.jpg
    Target structure: episode_{id}/rgb_full/{short_name}/000000.jpg
    """
    if not os.path.exists(raw_dir):
        return 0

    frame_dirs = sorted(Path(raw_dir).glob("rgb_all_ep*"))
    count = 0
    for fd in frame_dirs:
        if not fd.is_dir():
            continue
        match = re.match(r"rgb_all_ep(\d+)", fd.name)
        if not match:
            continue

        meta_src = fd / "rgb_meta.json"
        if not meta_src.exists():
            shutil.rmtree(str(fd))
            continue

        ep_id = match.group(1)
        scene_name, real_ep_id = _resolve_episode_info(raw_dir, ep_id, ep_meta_map)

        ep_dir = os.path.join(
            output_dir, task,
            f"scene_{scene_name}",
            f"episode_{real_ep_id.zfill(4)}",
        )

        for sensor_dir in sorted(fd.iterdir()):
            if not sensor_dir.is_dir():
                continue
            short = _rgb_short_name(sensor_dir.name)
            target_dir = os.path.join(ep_dir, "rgb_full", short)
            os.makedirs(os.path.dirname(target_dir), exist_ok=True)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.move(str(sensor_dir), target_dir)

        target_meta = os.path.join(ep_dir, "rgb_meta.json")
        os.makedirs(ep_dir, exist_ok=True)
        if real_ep_id != ep_id:
            try:
                with open(str(meta_src), "r") as f:
                    rgb_meta = json.load(f)
                if "episode_id" in rgb_meta:
                    rgb_meta["episode_id"] = int(real_ep_id)
                with open(target_meta, "w") as f:
                    json.dump(rgb_meta, f, indent=2)
                meta_src.unlink()
            except (json.JSONDecodeError, OSError):
                os.replace(str(meta_src), target_meta)
        else:
            os.replace(str(meta_src), target_meta)

        try:
            fd.rmdir()
        except OSError:
            pass

        count += 1

    return count


# ============================================================
# Organize depth PNG directories
# ============================================================

DEPTH_SENSOR_SHORT_NAMES = {
    "articulated_agent_arm_depth_render": "arm",
    "head_depth_render": "head",
}


def _depth_short_name(sensor_uuid: str) -> str:
    """Map full sensor UUID to a short directory name.

    Third-person sensors are mounted on two agents and need disambiguation:
      agent_0_third_depth -> third_robot
      agent_1_third_depth -> third_human
    """
    for pattern, short in DEPTH_SENSOR_SHORT_NAMES.items():
        if pattern in sensor_uuid:
            return short
    if "third_depth" in sensor_uuid:
        if "agent_0" in sensor_uuid:
            return "third_robot"
        if "agent_1" in sensor_uuid:
            return "third_human"
        return "third"
    return sensor_uuid


def organize_depth_pngs(
    task: str,
    raw_dir: str,
    output_dir: str,
    ep_meta_map: Dict[str, dict],
) -> int:
    """Move depth PNG directories and depth_meta.json from raw to dataset directory."""
    if not os.path.exists(raw_dir):
        return 0

    depth_dirs = sorted(Path(raw_dir).glob("depth_png_ep*"))
    count = 0
    for dd in depth_dirs:
        if not dd.is_dir():
            continue
        match = re.match(r"depth_png_ep(\d+)", dd.name)
        if not match:
            continue

        # Skip incomplete episodes (residual first frame of next episode at evaluator exit)
        meta_src = dd / "depth_meta.json"
        if not meta_src.exists():
            shutil.rmtree(str(dd))
            continue

        ep_id = match.group(1)
        scene_name, real_ep_id = _resolve_episode_info(raw_dir, ep_id, ep_meta_map)

        ep_dir = os.path.join(
            output_dir, task,
            f"scene_{scene_name}",
            f"episode_{real_ep_id.zfill(4)}",
        )

        for sensor_dir in sorted(dd.iterdir()):
            if not sensor_dir.is_dir():
                continue
            short = _depth_short_name(sensor_dir.name)
            target_dir = os.path.join(ep_dir, "depth", short)
            os.makedirs(os.path.dirname(target_dir), exist_ok=True)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.move(str(sensor_dir), target_dir)

        target_meta = os.path.join(ep_dir, "depth_meta.json")
        os.makedirs(ep_dir, exist_ok=True)
        if real_ep_id != ep_id:
            try:
                with open(str(meta_src), "r") as f:
                    depth_meta = json.load(f)
                if "episode_id" in depth_meta:
                    depth_meta["episode_id"] = int(real_ep_id)
                with open(target_meta, "w") as f:
                    json.dump(depth_meta, f, indent=2)
                meta_src.unlink()
            except (json.JSONDecodeError, OSError):
                os.replace(str(meta_src), target_meta)
        else:
            os.replace(str(meta_src), target_meta)

        try:
            dd.rmdir()
        except OSError:
            pass

        count += 1

    return count


# ============================================================
# Main flow
# ============================================================

def run_command(cmd: List[str], work_dir: str, env: dict) -> bool:
    """Execute a command and return whether it succeeded."""
    full_env = os.environ.copy()
    full_env["HYDRA_FULL_ERROR"] = "1"
    full_env.update(env)

    try:
        subprocess.run(cmd, cwd=work_dir, check=True, env=full_env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"    Command failed (exit code {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"    User interrupted")
        raise


def _get_valid_episode_ids(raw_dir: str) -> Optional[Set[str]]:
    """Determine valid (evaluator-counted) episode IDs from trajectory files in raw dir.

    In parallel-env mode, extra environments may process uncounted episodes;
    trajectory files are only generated for counted episodes and serve as
    the reliable indicator. Returns None if no trajectory files exist (no filtering).
    """
    traj_files = list(Path(raw_dir).glob("trajectory_ep*.json"))
    if not traj_files:
        return None
    ids = set()
    for tf in traj_files:
        m = re.match(r"trajectory_ep(\d+)\.json", tf.name)
        if m:
            ids.add(m.group(1))
    return ids


def organize_topdown(task, raw_dir, output_dir, ep_meta_map, human_type: str = "unknown",
                     valid_episode_ids: Optional[Set[str]] = None):
    """Organize top-down videos into the dataset directory, writing episode_meta.json in sync."""
    topdown_dirs = sorted(Path(raw_dir).glob("topdown_ep*"))
    count = 0
    for td in topdown_dirs:
        if not td.is_dir():
            continue
        mp4 = td / "topdown.mp4"
        if not mp4.exists():
            shutil.rmtree(str(td))
            continue
        match = re.match(r"topdown_ep(\d+)", td.name)
        if not match:
            continue
        ep_id = match.group(1)
        if valid_episode_ids is not None and ep_id not in valid_episode_ids:
            shutil.rmtree(str(td))
            continue
        scene_name, real_ep_id = _resolve_episode_info(raw_dir, ep_id, ep_meta_map)
        ep_info = ep_meta_map.get(real_ep_id, ep_meta_map.get(str(int(ep_id)), {}))
        ep_dir = os.path.join(
            output_dir, task,
            f"scene_{scene_name}",
            f"episode_{real_ep_id.zfill(4)}",
        )
        os.makedirs(ep_dir, exist_ok=True)
        os.replace(str(mp4), os.path.join(ep_dir, "topdown.mp4"))

        # Write episode_meta.json alongside topdown.mp4
        meta_path = os.path.join(ep_dir, "episode_meta.json")
        if not os.path.exists(meta_path):
            meta = {
                "episode_id": real_ep_id,
                "scene_id": ep_info.get("scene_name", scene_name),
                "task": task,
                "human_type": human_type,
                "rigid_objs": ep_info.get("rigid_objs", []),
                "target_objs": ep_info.get("target_objs", []),
                "object_labels": ep_info.get("object_labels", {}),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        try:
            td.rmdir()
        except OSError:
            pass
        count += 1
    return count


# ============================================================
# Merged mode main flow
# ============================================================

def organize_all(task, raw_dir, output_dir, ep_meta_map, human_type: str = "unknown"):
    """Organize all data from the _raw directory into the final directory structure."""
    valid_ids = _get_valid_episode_ids(raw_dir)

    print("  Organizing top-down videos...")
    n = organize_topdown(task, raw_dir, output_dir, ep_meta_map, human_type, valid_ids)
    print(f"  Done: {n} episodes top-down videos")

    print("  Organizing RGB full frames...")
    n = organize_rgb_frames(task, raw_dir, output_dir, ep_meta_map)
    print(f"  Done: {n} episodes RGB full frames")

    print("  Organizing depth PNGs...")
    n = organize_depth_pngs(task, raw_dir, output_dir, ep_meta_map)
    print(f"  Done: {n} episodes depth PNGs")

    print("  Organizing trajectory files...")
    n = organize_trajectories(task, raw_dir, output_dir, ep_meta_map)
    print(f"  Done: {n} trajectory files")

    print("  Organizing camera parameter files...")
    n = organize_camera_params(task, raw_dir, output_dir, ep_meta_map)
    print(f"  Done: {n} camera parameter files")


def run_merged(task, raw_dir, cfg, ep_meta_map, topdown_height, human_type="female_0"):
    """Merged mode: run the evaluator to collect data. Returns success status."""
    cmd = build_merged_command(task, raw_dir, cfg, human_type)

    env = {
        "HABITAT_VIDEO_SENSOR_FILTER": "third_rgb,arm_rgb,head_rgb,third_depth,depth_render",
        "HABITAT_FRAME_SKIP": str(cfg.get("frame_skip", 1)),
        "HABITAT_TOPDOWN_VIEW": "1",
        "HABITAT_TOPDOWN_HEIGHT": topdown_height,
        "HABITAT_SAVE_RGB_FRAMES": "1",
        "HABITAT_DEPTH_FORMAT": "png8",
        "HABITAT_SAVE_TRAJECTORY": "1",
        "HABITAT_SAVE_CAMERA_PARAMS": "1",
    }

    return run_command(cmd, cfg["work_dir"], env)


# ============================================================
# Per-view mode main flow (legacy)
# ============================================================

def run_per_view(task, views, cfg, ep_meta_map, topdown_height,
                 human_type: str = "unknown", output_dir: str = None):
    """Per-view mode: run a full simulation separately for each view."""
    base_output_dir = cfg["output_dir"]
    if output_dir is None:
        output_dir = base_output_dir

    for view_key in views:
        view_def = VIEW_DEFS[task][view_key]
        if "_" in view_key:
            modality, perspective = view_key.split("_")
        else:
            modality, perspective = view_key, ""

        raw_dir = os.path.join(base_output_dir, "_raw", human_type, task, view_key)
        os.makedirs(raw_dir, exist_ok=True)

        print()
        print("-" * 50)
        print(f"  {task} — {modality.upper()} {perspective}")
        print(f"  config: {view_def['config']}")
        print(f"  filter: {view_def['filter']}")
        print("-" * 50)

        cmd = build_command(task, view_key, raw_dir, cfg, human_type)

        env = {
            "HABITAT_VIDEO_SENSOR_FILTER": view_def["filter"],
            "HABITAT_FRAME_SKIP": str(cfg.get("frame_skip", 1)),
        }

        is_depth_view = view_key.startswith("depth_")
        is_rgb_view = view_key.startswith("rgb_")
        is_topdown = view_key == "topdown"
        if is_depth_view:
            env["HABITAT_DEPTH_FORMAT"] = "png8"
        if is_rgb_view:
            env["HABITAT_VIDEO_SPLIT_SENSORS"] = "1"
            env["HABITAT_SAVE_RGB_FRAMES"] = "1"
        if is_topdown:
            env["HABITAT_TOPDOWN_VIEW"] = "1"
            env["HABITAT_TOPDOWN_HEIGHT"] = topdown_height

        if view_key == views[0]:
            env["HABITAT_SAVE_TRAJECTORY"] = "1"
            env["HABITAT_SAVE_CAMERA_PARAMS"] = "1"
        else:
            env["HABITAT_SAVE_TRAJECTORY"] = "0"
            env["HABITAT_SAVE_CAMERA_PARAMS"] = "0"

        ok = run_command(cmd, cfg["work_dir"], env)
        if not ok:
            print("  Skipping organization")
            continue

        if is_topdown:
            print("  Organizing top-down videos...")
            n = organize_topdown(task, raw_dir, output_dir, ep_meta_map, human_type)
            print(f"  Done: {n} episodes top-down videos")
        elif is_depth_view:
            print("  Organizing depth PNGs...")
            n = organize_depth_pngs(task, raw_dir, output_dir, ep_meta_map)
            print(f"  Done: {n} episodes depth PNGs")
        elif is_rgb_view:
            print("  Organizing per-sensor RGB videos...")
            n = organize_rgb_split(task, raw_dir, output_dir, ep_meta_map)
            print(f"  Done: {n} episodes RGB videos")
            print("  Organizing RGB full frames...")
            n = organize_rgb_frames(task, raw_dir, output_dir, ep_meta_map)
            print(f"  Done: {n} episodes RGB full frames")

        if view_key == views[0]:
            print("  Organizing trajectory files...")
            n = organize_trajectories(task, raw_dir, output_dir, ep_meta_map)
            print(f"  Done: {n} trajectory files")
            print("  Organizing camera parameter files...")
            n = organize_camera_params(task, raw_dir, output_dir, ep_meta_map)
            print(f"  Done: {n} camera parameter files")


def main():
    global _RENUM_MAP
    parser = argparse.ArgumentParser(description="Habitat Video Dataset Generator")
    parser.add_argument(
        "--tasks", nargs="+",
        default=["social_navigation", "social_rearrangement"],
        choices=["social_navigation", "social_rearrangement"],
        help="Tasks to generate",
    )
    parser.add_argument(
        "--views", nargs="+",
        default=["topdown", "rgb_third", "rgb_first", "depth_third", "depth_first"],
        choices=["topdown", "rgb_third", "rgb_first", "depth_third", "depth_first"],
        help="Views to generate (used in legacy mode)",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count")
    parser.add_argument(
        "--human-types", nargs="+",
        default=None,
        choices=AVAILABLE_HUMAN_TYPES,
        help="Humanoid types to generate (default: female_0 male_0 neutral_0)",
    )
    parser.add_argument(
        "--topdown-height", type=str, default="auto",
        help="Top-down camera height above floor (meters), 'auto' for optimal (default)",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="Use legacy mode (separate simulation per view), default is merged mode",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume: skip completed episodes in _raw, run remaining only",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU device ID (e.g. --gpu 0), default is system auto-assign",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = CONFIG.copy()
    if args.episodes is not None:
        cfg["num_episodes"] = args.episodes

    resume = args.resume or cfg.get("resume", False)

    tasks = args.tasks
    human_types = args.human_types or cfg.get("human_types", ["female_0", "male_0", "neutral_0"])
    merged = not args.legacy

    print("=" * 60)
    print("Habitat Social Video Dataset Generator")
    print("=" * 60)
    print(f"  Output dir:     {cfg['output_dir']}")
    print(f"  Tasks:          {tasks}")
    print(f"  Humanoid:       {human_types}")
    print(f"  Mode:           {'merged (1 pass/task)' if merged else f'per-view ({len(args.views)} pass/task)'}")
    print(f"  Episodes:       {cfg['num_episodes']}")
    print(f"  Top-down height:{args.topdown_height}")
    print(f"  Video res:      {cfg['video_width']}x{cfg['video_height']}")
    print(f"  Depth res:      {cfg['depth_width']}x{cfg['depth_height']}")
    if args.gpu is not None:
        print(f"  GPU:         {args.gpu}")
    if resume:
        print(f"  Resume:         enabled")
    print()

    base_output_dir = cfg["output_dir"]
    os.makedirs(base_output_dir, exist_ok=True)

    ep_meta_maps: Dict[str, Dict[str, dict]] = {}
    for task in tasks:
        ep_meta_maps[task] = load_episode_metadata(
            cfg["episode_data"][task], cfg["work_dir"]
        )

    for human_type in human_types:
        ht_output_dir = os.path.join(base_output_dir, human_type)

        print()
        print("#" * 60)
        print(f"Humanoid: {human_type}")
        print("#" * 60)

        for task in tasks:
            print()
            print("=" * 60)
            print(f"Task: {task}  |  Humanoid: {human_type}")
            print("=" * 60)

            if merged:
                raw_dir = os.path.join(base_output_dir, "_raw", human_type, task, "merged")
                os.makedirs(raw_dir, exist_ok=True)

                if resume:
                    completed = detect_completed_episodes(raw_dir, ht_output_dir, task)
                    cleanup_partial_raw_data(raw_dir)

                    if completed:
                        print(f"  [resume] {len(completed)} episodes already completed")
                        filtered_path = os.path.join(
                            base_output_dir, f"_resume_{human_type}_{task}_episodes.json.gz"
                        )
                        remaining = create_filtered_dataset(
                            cfg["episode_data"][task],
                            cfg["work_dir"],
                            completed,
                            filtered_path,
                        )
                        requested = cfg["num_episodes"]
                        if requested > 0:
                            still_needed = max(0, requested - len(completed))
                            to_run = min(still_needed, remaining)
                        else:
                            to_run = remaining

                        _RENUM_MAP = build_renumber_map(filtered_path)

                        if to_run == 0:
                            print("  [resume] All episodes completed, skipping evaluation")
                        else:
                            print(f"  [resume] Target {requested if requested > 0 else 'all'}, "
                                  f"completed {len(completed)}, running {to_run} episodes")
                            task_cfg = cfg.copy()
                            task_cfg["episode_data"] = cfg["episode_data"].copy()
                            task_cfg["episode_data"][task] = os.path.relpath(
                                filtered_path, cfg["work_dir"]
                            )
                            task_cfg["num_episodes"] = to_run
                            ok = run_merged(task, raw_dir, task_cfg, ep_meta_maps[task],
                                            args.topdown_height, human_type)
                            if not ok:
                                print("  Evaluation failed, organizing existing data anyway")
                        organize_all(task, raw_dir, ht_output_dir, ep_meta_maps[task], human_type)
                        _RENUM_MAP = None
                    else:
                        print("  [resume] No completed episodes found, starting from scratch")
                        ok = run_merged(task, raw_dir, cfg, ep_meta_maps[task],
                                        args.topdown_height, human_type)
                        if ok:
                            organize_all(task, raw_dir, ht_output_dir, ep_meta_maps[task], human_type)
                else:
                    ok = run_merged(task, raw_dir, cfg, ep_meta_maps[task],
                                    args.topdown_height, human_type)
                    if ok:
                        organize_all(task, raw_dir, ht_output_dir, ep_meta_maps[task], human_type)
            else:
                run_per_view(task, args.views, cfg, ep_meta_maps[task],
                             args.topdown_height, human_type, output_dir=ht_output_dir)

    info = {
        "dataset_name": "Habitat Social Video Dataset",
        "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": tasks,
        "human_types": human_types,
        "mode": "merged" if merged else "per_view",
        "video_resolution": [cfg["video_width"], cfg["video_height"]],
        "depth_resolution": [cfg["depth_width"], cfg["depth_height"]],
        "num_episodes_requested": cfg["num_episodes"],
    }
    info_path = os.path.join(base_output_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    raw_root = os.path.join(base_output_dir, "_raw")
    if os.path.exists(raw_root):
        for dirpath, dirnames, filenames in os.walk(raw_root, topdown=False):
            if not filenames and not dirnames:
                os.rmdir(dirpath)
        if os.path.exists(raw_root) and not os.listdir(raw_root):
            os.rmdir(raw_root)

    print()
    print("=" * 60)
    print("Dataset generation complete!")
    print(f"   Output: {base_output_dir}")
    print(f"   Humanoid types: {', '.join(human_types)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
