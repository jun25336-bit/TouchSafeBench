#!/usr/bin/env python3
"""
Check social rearrangement success rate by running the evaluator.

This actually runs the habitat evaluator with NO video output and
minimal sensor resolution to be as fast as possible. It collects
pddl_success from the episode metrics printed by the evaluator.

The trajectory files saved will contain an 'episode_metrics' field
with pddl_success and other scalar metrics.

Usage:
  python check_rearrange_success.py --gpu 0
  python check_rearrange_success.py --gpu 0 --episodes 50
  python check_rearrange_success.py --gpu 0 --human-types female_2 male_2
"""

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "work_dir": _REPO_ROOT,
    "episode_data": "data/versioned_data/hab3-episodes/val/social_rearrange_diverse.json.gz",
    "checkpoint": os.path.join(_REPO_ROOT, "data/versioned_data/hab3-episodes/checkpoint/social_rearrange_latest.pth"),
    "additional_object_paths": [
        os.path.join(_REPO_ROOT, "data/hab3_bench_assets/"),
        os.path.join(_REPO_ROOT, "data/versioned_data/ycb/configs/"),
    ],
}

AVAILABLE_HUMAN_TYPES = [
    "female_0", "female_1", "female_2", "female_3",
    "male_0", "male_1", "male_2", "male_3",
    "neutral_0", "neutral_1", "neutral_2", "neutral_3",
]


def count_dataset_episodes(data_path: str, work_dir: str) -> int:
    full_path = os.path.join(work_dir, data_path)
    with gzip.open(full_path, "rt") as f:
        data = json.load(f)
    return len(data.get("episodes", []))


def count_dataset_scenes(data_path: str, work_dir: str) -> int:
    full_path = os.path.join(work_dir, data_path)
    with gzip.open(full_path, "rt") as f:
        data = json.load(f)
    return len(set(ep["scene_id"] for ep in data.get("episodes", [])))


def _humanoid_cli_overrides(human_type: str) -> List[str]:
    urdf = f"data/humanoids/humanoid_data/{human_type}/{human_type}.urdf"
    motion = f"data/humanoids/humanoid_data/{human_type}/{human_type}_motion_data_smplx.pkl"
    return [
        f"habitat.simulator.agents.agent_1.articulated_agent_urdf={urdf}",
        f"habitat.simulator.agents.agent_1.motion_data_path={motion}",
    ]


def build_command(cfg: dict, human_type: str, num_ep: int, output_dir: str) -> List[str]:
    ep_data = cfg["episode_data"]

    cmd = [
        sys.executable, "-u", "-m", "habitat_baselines.run",
        "--config-name=social_rearrange/pop_play.yaml",
    ]

    cmd += [
        "+habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.select_random_goal=False",
        "+habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx=1",
    ]
    cmd += _humanoid_cli_overrides(human_type)

    # Minimal sensor resolution to speed up
    cmd += [
        "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.height=120",
        "habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.width=160",
    ]

    n_envs = 1
    n_scenes = count_dataset_scenes(ep_data, cfg["work_dir"])
    n_envs = min(n_envs, n_scenes) if n_scenes > 0 else n_envs
    n_envs = max(n_envs, 1)

    cmd += [
        "habitat_baselines.evaluate=True",
        f"habitat_baselines.num_environments={n_envs}",
        # No video output
        'habitat_baselines.eval.video_option=[]',
        f"habitat_baselines.video_dir={output_dir}",
        f"habitat_baselines.eval_ckpt_path_dir={cfg['checkpoint']}",
        f"habitat.dataset.data_path={ep_data}",
        "habitat.dataset.scenes_dir=data/scene_datasets/",
        "habitat_baselines.load_resume_state_config=False",
    ]

    if num_ep > 0:
        cmd.append(f"habitat_baselines.test_episode_count={num_ep}")

    if cfg["additional_object_paths"]:
        paths_str = ",".join(f'"{p}"' for p in cfg["additional_object_paths"])
        cmd.append(f'+habitat.simulator.additional_object_paths=[{paths_str}]')

    return cmd


def run_evaluator(cfg: dict, human_type: str, num_ep: int, output_dir: str) -> bool:
    os.makedirs(output_dir, exist_ok=True)
    cmd = build_command(cfg, human_type, num_ep, output_dir)

    full_env = os.environ.copy()
    full_env["HYDRA_FULL_ERROR"] = "1"
    full_env["HABITAT_DROP_ENABLED"] = "0"
    # Save trajectory with episode_metrics (including pddl_success)
    full_env["HABITAT_SAVE_TRAJECTORY"] = "1"

    print(f"  Running evaluator ({num_ep} episodes, no video)...")
    print(f"  Output: {output_dir}")
    print()

    try:
        subprocess.run(cmd, cwd=cfg["work_dir"], check=True, env=full_env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Command failed (exit code {e.returncode})")
        return False
    except KeyboardInterrupt:
        print("  Interrupted by user")
        raise


def parse_results(output_dir: str, human_type: str, save_dir: str = None):
    """Parse trajectory files and extract pddl_success from episode_metrics."""
    traj_files = sorted(Path(output_dir).glob("trajectory_ep*.json"))

    if not traj_files:
        print(f"  No trajectory files found in {output_dir}")
        return

    total = 0
    success = 0
    failed = 0
    no_metric = 0
    results = []

    for tf in traj_files:
        m = re.match(r"trajectory_ep(\d+)\.json", tf.name)
        if not m:
            continue

        with open(tf) as f:
            data = json.load(f)

        ep_id = data.get("episode_id", m.group(1))
        num_steps = data.get("num_steps", 0)
        metrics = data.get("episode_metrics", {})
        pddl_success = metrics.get("pddl_success", None)

        total += 1
        if pddl_success is None:
            no_metric += 1
            status = "NO_METRIC"
        elif pddl_success > 0.5:
            success += 1
            status = "SUCCESS"
        else:
            failed += 1
            status = "FAILED"

        results.append({
            "episode_id": str(ep_id),
            "num_steps": num_steps,
            "pddl_success": pddl_success,
            "status": status,
        })

    # Print summary
    print()
    print("=" * 60)
    print(f"  Results: {human_type}")
    print("=" * 60)
    print(f"  Total episodes:     {total}")
    if no_metric > 0:
        print(f"  No pddl_success:    {no_metric}  (trajectory missing episode_metrics)")
    print(f"  SUCCESS:            {success}  ({success/total*100:.1f}%)" if total > 0 else "")
    print(f"  FAILED:             {failed}  ({failed/total*100:.1f}%)" if total > 0 else "")
    print()

    # Show successful episodes
    successes = [r for r in results if r["status"] == "SUCCESS"]
    failures = [r for r in results if r["status"] == "FAILED"]

    if successes:
        print(f"  --- Successful Episodes ({len(successes)}) ---")
        for r in sorted(successes, key=lambda x: int(x["episode_id"]))[:30]:
            print(f"    ep {r['episode_id']:>6}  steps={r['num_steps']:>5}  pddl_success={r['pddl_success']}")
        if len(successes) > 30:
            print(f"    ... and {len(successes) - 30} more")
        print()

    # Save results
    if save_dir is None:
        save_dir = output_dir

    success_ids = [r["episode_id"] for r in successes]
    result_file = os.path.join(save_dir, f"success_episodes_{human_type}.json")
    with open(result_file, "w") as f:
        json.dump({
            "human_type": human_type,
            "total_episodes": total,
            "successful_count": len(success_ids),
            "failed_count": failed,
            "success_rate": success / total if total > 0 else 0,
            "successful_episode_ids": success_ids,
            "all_results": results,
        }, f, indent=2)
    print(f"  Results saved to: {result_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Check social rearrangement success rate via actual evaluator run"
    )
    parser.add_argument(
        "--episodes", type=int, default=-1,
        help="Number of episodes (-1 = all)",
    )
    parser.add_argument(
        "--human-types", nargs="+", default=["female_2"],
        choices=AVAILABLE_HUMAN_TYPES,
        help="Humanoid types (default: female_2)",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU device index",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(_REPO_ROOT, "habitat_video_dataset/_success_check"),
        help="Directory to save trajectory files with metrics",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = CONFIG.copy()
    total_ep = count_dataset_episodes(cfg["episode_data"], cfg["work_dir"])
    total_scenes = count_dataset_scenes(cfg["episode_data"], cfg["work_dir"])
    num_ep = args.episodes if args.episodes > 0 else total_ep

    print("=" * 60)
    print("Social Rearrangement Success Checker (actual evaluator)")
    print("=" * 60)
    print(f"  Dataset:       {cfg['episode_data']}")
    print(f"  Total eps:     {total_ep}")
    print(f"  Total scenes:  {total_scenes}")
    print(f"  Evaluating:    {num_ep} episodes")
    print(f"  Humanoids:     {args.human_types}")
    print(f"  Video output:  NONE (speed mode)")
    if args.gpu is not None:
        print(f"  GPU:           {args.gpu}")
    print()

    for human_type in args.human_types:
        print(f"{'#' * 60}")
        print(f"  Humanoid: {human_type}")
        print(f"{'#' * 60}")

        ht_output = os.path.join(args.output_dir, human_type)
        ok = run_evaluator(cfg, human_type, num_ep, ht_output)

        if ok:
            parse_results(ht_output, human_type, save_dir=args.output_dir)
        else:
            print("  Evaluator failed")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
