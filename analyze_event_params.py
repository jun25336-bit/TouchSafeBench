#!/usr/bin/env python3
"""
Event Parameter Analysis Tool
==============================
Sweep different parameter values and report how many events each combination
produces from the dataset. Helps choose good defaults for:

  - gap_threshold       : merging collision / near-miss frames into events
  - safe_event_window   : partitioning safe segments into events
  - near_miss_dist      : distance threshold for near-miss frames
  - near_miss_gap       : gap for clustering near-miss events
  - near_miss_half_span : half-span around near-miss event center

Usage:
    python analyze_event_params.py --dataset-dir ./habitat_video_dataset
    python analyze_event_params.py --human-types neutral_0 female_2 male_2 \
        --sim-task-types social_navigation social_rearrangement \
        --gap-thresholds 5 10 15 20 30 \
        --safe-windows 30 60 90 120 180 \
        --near-miss-dists 1.0 1.5 2.0 2.5 \
        --near-miss-gaps 5 10 15 20 30
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from evaluate_safety_benchmark_v2 import (
    DATASET_DIR,
    NEAR_MISS_DIST,
    EpisodeData,
    cluster_collision_events,
    cluster_near_miss_frames,
    find_safe_segments,
    get_near_miss_frames,
    partition_safe_events,
    assign_event_frame_roles,
    scan_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("param_analysis")


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_scene_collision_events(
    episodes: List[EpisodeData],
    gap_thresholds: List[int],
) -> Dict[int, dict]:
    """For each gap_threshold, count scene collision events and their sizes."""
    results = {}
    for gap in gap_thresholds:
        total_events = 0
        total_frames_in_events = 0
        event_sizes = []
        episodes_with_events = 0
        for ep in episodes:
            if not ep.scene_collision_steps:
                continue
            events = cluster_collision_events(ep.scene_collision_steps, gap)
            if events:
                episodes_with_events += 1
            total_events += len(events)
            for ev in events:
                event_sizes.append(len(ev))
                total_frames_in_events += len(ev)

        results[gap] = {
            "total_events": total_events,
            "episodes_with_events": episodes_with_events,
            "total_coll_frames": total_frames_in_events,
            "avg_event_size": round(np.mean(event_sizes), 1) if event_sizes else 0,
            "median_event_size": int(np.median(event_sizes)) if event_sizes else 0,
            "min_event_size": int(np.min(event_sizes)) if event_sizes else 0,
            "max_event_size": int(np.max(event_sizes)) if event_sizes else 0,
            "size_percentiles": {
                "p25": int(np.percentile(event_sizes, 25)) if event_sizes else 0,
                "p75": int(np.percentile(event_sizes, 75)) if event_sizes else 0,
                "p90": int(np.percentile(event_sizes, 90)) if event_sizes else 0,
            },
        }
    return results


def analyze_human_collision_episodes(
    episodes: List[EpisodeData],
) -> dict:
    """
    Analyze human collision episodes for C-class sampling.
    C sampling uses human_collision_steps[-1] per episode (no gap clustering).
    """
    hc_episodes = [e for e in episodes if e.did_collide_human]
    hc_frame_positions = []
    n_hc_frames_list = []
    for ep in hc_episodes:
        if not ep.human_collision_steps:
            continue
        hc_frame = ep.human_collision_steps[-1]
        hc_frame_positions.append(hc_frame)
        n_hc_frames_list.append(len(ep.human_collision_steps))

    return {
        "hc_episodes": len(hc_episodes),
        "hc_episodes_with_steps": len(hc_frame_positions),
        "last_hc_frame_position": {
            "avg": round(np.mean(hc_frame_positions), 1) if hc_frame_positions else 0,
            "median": int(np.median(hc_frame_positions)) if hc_frame_positions else 0,
            "min": int(np.min(hc_frame_positions)) if hc_frame_positions else 0,
            "max": int(np.max(hc_frame_positions)) if hc_frame_positions else 0,
        },
        "hc_frames_per_episode": {
            "avg": round(np.mean(n_hc_frames_list), 1) if n_hc_frames_list else 0,
            "median": int(np.median(n_hc_frames_list)) if n_hc_frames_list else 0,
            "min": int(np.min(n_hc_frames_list)) if n_hc_frames_list else 0,
            "max": int(np.max(n_hc_frames_list)) if n_hc_frames_list else 0,
        },
    }


def analyze_task2_c_lead_times(
    episodes: List[EpisodeData],
    lead_times: List[int],
    clip_size: int = 3,
) -> dict:
    """
    For each lead_time K, count how many valid C samples can be produced
    from human-collision episodes. Overlapping human/scene collision frames
    are treated as human collision (C takes priority).
    """
    hc_episodes = [e for e in episodes if e.did_collide_human]
    results = {
        "hc_episodes_total": len(hc_episodes),
    }

    per_k = {}
    for k in lead_times:
        valid_samples = 0
        skipped_too_early = 0
        skipped_overlap = 0
        for ep in hc_episodes:
            if not ep.human_collision_steps:
                continue
            hc_frame = ep.human_collision_steps[-1]
            end_frame = hc_frame - k
            pre_start = end_frame - k + 1
            if pre_start < 0 or end_frame < 0:
                skipped_too_early += 1
                continue
            if (end_frame - pre_start + 1) < k:
                skipped_too_early += 1
                continue
            length = end_frame - pre_start + 1
            step = (length - 1) / (clip_size - 1) if clip_size > 1 else 0
            frames = [pre_start + int(round(i * step)) for i in range(clip_size)]
            if any(f >= hc_frame for f in frames):
                skipped_overlap += 1
                continue
            valid_samples += 1

        per_k[k] = {
            "valid_samples": valid_samples,
            "skipped_too_early": skipped_too_early,
            "skipped_overlap": skipped_overlap,
        }
    results["per_lead_time"] = per_k
    return results


def analyze_safe_segments(
    episodes: List[EpisodeData],
    safe_windows: List[int],
    margin: int = 5,
) -> Dict[int, dict]:
    """For each safe_event_window, count safe events."""
    all_segments = []
    for ep in episodes:
        all_coll = set(ep.scene_collision_steps) | set(ep.human_collision_steps)
        segs = find_safe_segments(ep.num_steps, all_coll, margin)
        all_segments.append((ep, segs))

    seg_lengths = []
    for _, segs in all_segments:
        for s, e in segs:
            seg_lengths.append(e - s + 1)

    results = {}
    for window in safe_windows:
        total_events = 0
        episodes_with_events = 0
        events_per_ep = []
        for ep, segs in all_segments:
            events = partition_safe_events(segs, window)
            if events:
                episodes_with_events += 1
            events_per_ep.append(len(events))
            total_events += len(events)

        results[window] = {
            "total_safe_events": total_events,
            "episodes_with_safe_events": episodes_with_events,
            "avg_events_per_episode": round(np.mean(events_per_ep), 2),
            "max_events_per_episode": int(np.max(events_per_ep)) if events_per_ep else 0,
        }

    results["_segment_stats"] = {
        "total_segments": len(seg_lengths),
        "avg_segment_length": round(np.mean(seg_lengths), 1) if seg_lengths else 0,
        "median_segment_length": int(np.median(seg_lengths)) if seg_lengths else 0,
        "min_segment_length": int(np.min(seg_lengths)) if seg_lengths else 0,
        "max_segment_length": int(np.max(seg_lengths)) if seg_lengths else 0,
        "percentiles": {
            "p10": int(np.percentile(seg_lengths, 10)) if seg_lengths else 0,
            "p25": int(np.percentile(seg_lengths, 25)) if seg_lengths else 0,
            "p50": int(np.percentile(seg_lengths, 50)) if seg_lengths else 0,
            "p75": int(np.percentile(seg_lengths, 75)) if seg_lengths else 0,
            "p90": int(np.percentile(seg_lengths, 90)) if seg_lengths else 0,
        },
    }
    return results


def analyze_near_miss_events(
    episodes: List[EpisodeData],
    near_miss_dists: List[float],
    near_miss_gaps: List[int],
    near_miss_half_spans: List[int],
) -> Dict[str, dict]:
    """
    For each (dist, gap) combo, count near-miss events.
    For each half_span, report how many events have enough room to expand.
    """
    log.info(f"Loading agent distances for {len(episodes)} episodes (this may take a while)...")
    for i, ep in enumerate(episodes):
        _ = ep.agent_distances
        if (i + 1) % 20 == 0:
            log.info(f"  ...distances loaded for {i+1}/{len(episodes)}")

    results = {}

    for dist in near_miss_dists:
        nm_frames_per_ep = {}
        for ep in episodes:
            dists = ep.agent_distances
            if dists is None:
                continue
            frames = [i for i, d in enumerate(dists) if not np.isnan(d) and d < dist]
            if frames:
                nm_frames_per_ep[ep.episode_id] = (ep, frames)

        eps_with_nm = len(nm_frames_per_ep)
        total_nm_frames = sum(len(f) for _, f in nm_frames_per_ep.values())

        for gap in near_miss_gaps:
            total_events = 0
            event_sizes = []
            for ep, frames in nm_frames_per_ep.values():
                events = cluster_collision_events(frames, gap)
                total_events += len(events)
                for ev in events:
                    event_sizes.append(len(ev))

            span_stats = {}
            for half_span in near_miss_half_spans:
                valid = 0
                for ep, frames in nm_frames_per_ep.values():
                    events = cluster_collision_events(frames, gap)
                    for ev in events:
                        center = ev[len(ev) // 2]
                        s = max(0, center - half_span)
                        e = min(ep.num_steps - 1, center + half_span)
                        if (e - s + 1) >= 2 * half_span + 1:
                            valid += 1
                span_stats[half_span] = valid

            key = f"dist={dist}_gap={gap}"
            results[key] = {
                "near_miss_dist": dist,
                "near_miss_gap": gap,
                "episodes_with_near_miss": eps_with_nm,
                "total_near_miss_frames": total_nm_frames,
                "total_events": total_events,
                "avg_event_size": round(np.mean(event_sizes), 1) if event_sizes else 0,
                "median_event_size": int(np.median(event_sizes)) if event_sizes else 0,
                "max_event_size": int(np.max(event_sizes)) if event_sizes else 0,
                "valid_events_per_half_span": span_stats,
            }

    return results


def analyze_task2_pre_post_zones(
    episodes: List[EpisodeData],
    gap_thresholds: List[int],
    pre_windows: List[int],
    post_windows: List[int],
    clip_size: int = 3,
) -> Dict[int, dict]:
    """
    Analyze pre/post zone availability for Task 2 B and A-post sampling.

    The actual sampling takes a fixed-size window (pre_window / post_window)
    before/after the collision event, clamped to the available safe interval.
    This function reports:
      - Full safe interval length distribution (how much room there is)
      - For each pre_window / post_window value, how many events have enough
        frames for sampling (actual clamped length >= clip_size)
    """
    results = {}
    for gap in gap_thresholds:
        pre_full_sizes = []
        post_full_sizes = []
        n_events = 0
        all_roles = []
        for ep in episodes:
            if not ep.scene_collision_steps:
                continue
            events = cluster_collision_events(ep.scene_collision_steps, gap)
            roles_list = assign_event_frame_roles(events, ep.num_steps)
            for roles in roles_list:
                n_events += 1
                pre_len = max(0, roles.pre_upper - roles.pre_lower + 1)
                post_len = max(0, roles.post_upper - roles.post_lower + 1)
                pre_full_sizes.append(pre_len)
                post_full_sizes.append(post_len)
                all_roles.append(roles)

        pre_window_stats = {}
        for pw in pre_windows:
            valid = 0
            actual_sizes = []
            for roles in all_roles:
                desired_start = roles.start - pw
                actual_start = max(roles.pre_lower, desired_start)
                actual_len = max(0, roles.pre_upper - actual_start + 1)
                if actual_len >= pw:
                    valid += 1
                    actual_sizes.append(actual_len)
            pre_window_stats[pw] = {
                "valid_events": valid,
                "avg_actual_len": round(np.mean(actual_sizes), 1) if actual_sizes else 0,
            }

        post_window_stats = {}
        for ptw in post_windows:
            valid = 0
            actual_sizes = []
            for roles in all_roles:
                desired_end = roles.end + ptw
                actual_end = min(roles.post_upper, desired_end)
                actual_len = max(0, actual_end - roles.post_lower + 1)
                if actual_len >= ptw:
                    valid += 1
                    actual_sizes.append(actual_len)
            post_window_stats[ptw] = {
                "valid_events": valid,
                "avg_actual_len": round(np.mean(actual_sizes), 1) if actual_sizes else 0,
            }

        def _dist_stats(sizes):
            if not sizes:
                return {"avg": 0, "med": 0, "min": 0, "max": 0, "p25": 0, "p75": 0}
            return {
                "avg": round(np.mean(sizes), 1),
                "med": int(np.median(sizes)),
                "min": int(np.min(sizes)),
                "max": int(np.max(sizes)),
                "p25": int(np.percentile(sizes, 25)),
                "p75": int(np.percentile(sizes, 75)),
            }

        results[gap] = {
            "total_scene_events": n_events,
            "pre_full_interval": _dist_stats(pre_full_sizes),
            "post_full_interval": _dist_stats(post_full_sizes),
            "per_pre_window": pre_window_stats,
            "per_post_window": post_window_stats,
        }
    return results


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_table(headers: List[str], rows: List[list], col_widths: List[int] = None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 2
                      for i, h in enumerate(headers)]
    header_line = "".join(str(h).rjust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))
    for row in rows:
        print("".join(str(v).rjust(w) for v, w in zip(row, col_widths)))


def analyze_per_subdataset_episodes(
    dataset_dir: Path,
    human_types: List[str],
    sim_task_types: List[str],
    gap_threshold: int = 30,
    clip_size: int = 3,
    safe_event_window: int = 90,
    near_miss_dist: float = 1.5,
    near_miss_gap: int = 15,
    near_miss_half_span: int = 30,
    pre_window: int = 30,
    post_window: int = 30,
) -> Dict[str, dict]:
    """
    For each sub-dataset (human_type x sim_task_type), count how many
    episodes qualify for each sample type under the given parameters.

    Returns a dict keyed by "{human_type}/{sim_task_type}".
    """
    results = {}

    for human_type in human_types:
        for sim_task_type in sim_task_types:
            key = f"{human_type}/{sim_task_type}"
            episodes = scan_dataset(
                dataset_dir, [human_type], [sim_task_type],
            )
            if not episodes:
                results[key] = {"total_episodes": 0}
                continue

            hc_episodes = [e for e in episodes if e.did_collide_human]

            # Task 1 B: episodes with at least one scene collision event >= clip_size
            t1_b_eps = 0
            t1_b_events = 0
            for ep in episodes:
                events = cluster_collision_events(ep.scene_collision_steps, gap_threshold)
                valid = [ev for ev in events if len(ev) >= clip_size]
                if valid:
                    t1_b_eps += 1
                    t1_b_events += len(valid)

            # Task 1 C: hc episodes with last hc_frame >= clip_size - 1
            t1_c_eps = 0
            for ep in hc_episodes:
                if ep.human_collision_steps and ep.human_collision_steps[-1] >= clip_size - 1:
                    t1_c_eps += 1

            # Task 1 A-easy: episodes with at least one safe event of safe_event_window
            # Uses same exclusion logic as evaluate_safety_benchmark_v2._build_exclusion_zones
            t1_a_easy_eps = 0
            t1_a_easy_events = 0
            for ep in episodes:
                exclusion = set()
                scene_events = cluster_collision_events(ep.scene_collision_steps, gap_threshold)
                for ev_frames in scene_events:
                    exclusion.update(range(ev_frames[0], ev_frames[-1] + 1))
                for f in ep.human_collision_steps:
                    for d in range(-5, 6):
                        if 0 <= f + d < ep.num_steps:
                            exclusion.add(f + d)
                segs = find_safe_segments(ep.num_steps, exclusion, margin=0)
                safe_evts = partition_safe_events(segs, safe_event_window)
                if safe_evts:
                    t1_a_easy_eps += 1
                    t1_a_easy_events += len(safe_evts)

            # Task 1 A-hard: episodes with at least one valid near-miss event
            # Discard event if its span overlaps any exclusion zone
            log.info(f"  Loading distances for {key} ({len(episodes)} eps)...")
            t1_a_hard_eps = 0
            t1_a_hard_events = 0
            for ep in episodes:
                nm_frames = get_near_miss_frames(ep, near_miss_dist)
                if not nm_frames:
                    continue
                exclusion = set()
                scene_events = cluster_collision_events(ep.scene_collision_steps, gap_threshold)
                for ev_frames in scene_events:
                    exclusion.update(range(ev_frames[0], ev_frames[-1] + 1))
                for f in ep.human_collision_steps:
                    for d in range(-5, 6):
                        if 0 <= f + d < ep.num_steps:
                            exclusion.add(f + d)

                nm_events = cluster_near_miss_frames(nm_frames, near_miss_gap)
                has_valid = False
                for nm_event in nm_events:
                    center = nm_event[len(nm_event) // 2]
                    s = max(0, center - near_miss_half_span)
                    e = min(ep.num_steps - 1, center + near_miss_half_span)
                    if (e - s + 1) < 2 * near_miss_half_span + 1:
                        continue
                    span_frames = set(range(s, e + 1))
                    if span_frames & exclusion:
                        continue
                    has_valid = True
                    t1_a_hard_events += 1
                if has_valid:
                    t1_a_hard_eps += 1

            # Task 2 C: hc episodes where >= clip_size lead_time points are valid
            # Each episode produces at most 1 sample using clip_size frames
            # chosen from the 4 lead_time time points (hc_frame - K).
            lead_times = [5, 10, 15, 20]
            t2_c_eps = 0
            t2_c_samples = 0
            for ep in hc_episodes:
                if not ep.human_collision_steps:
                    continue
                hc_frame = ep.human_collision_steps[-1]
                candidate_frames = [hc_frame - k for k in lead_times if hc_frame - k >= 0]
                if len(candidate_frames) >= clip_size:
                    t2_c_eps += 1
                    t2_c_samples += 1

            # Task 2 B: episodes with valid pre-window scene collision events
            # Must also check that enough frames survive hc_exclusion
            t2_b_eps = 0
            t2_b_events = 0
            for ep in episodes:
                events = cluster_collision_events(ep.scene_collision_steps, gap_threshold)
                if not events:
                    continue
                hc_excl = set()
                for f in ep.human_collision_steps:
                    for d in range(-5, 6):
                        if 0 <= f + d < ep.num_steps:
                            hc_excl.add(f + d)
                roles_list = assign_event_frame_roles(events, ep.num_steps)
                has_valid = False
                for roles in roles_list:
                    desired_start = roles.start - pre_window
                    actual_start = max(roles.pre_lower, desired_start)
                    actual_end = roles.pre_upper
                    if actual_end < actual_start or (actual_end - actual_start + 1) < pre_window:
                        continue
                    valid_frames = [f for f in range(actual_start, actual_end + 1) if f not in hc_excl]
                    if len(valid_frames) >= clip_size:
                        has_valid = True
                        t2_b_events += 1
                if has_valid:
                    t2_b_eps += 1

            # Task 2 A-post: episodes with valid post-collision windows
            # Must also check that enough frames survive hc_exclusion
            t2_a_post_eps = 0
            t2_a_post_events = 0
            for ep in episodes:
                events = cluster_collision_events(ep.scene_collision_steps, gap_threshold)
                if not events:
                    continue
                hc_excl = set()
                for f in ep.human_collision_steps:
                    for d in range(-5, 6):
                        if 0 <= f + d < ep.num_steps:
                            hc_excl.add(f + d)
                roles_list = assign_event_frame_roles(events, ep.num_steps)
                has_valid = False
                for roles in roles_list:
                    actual_start = roles.post_lower
                    desired_end = roles.end + post_window
                    actual_end = min(roles.post_upper, desired_end)
                    if actual_end < actual_start or (actual_end - actual_start + 1) < post_window:
                        continue
                    valid_frames = [f for f in range(actual_start, actual_end + 1) if f not in hc_excl]
                    if len(valid_frames) >= clip_size:
                        has_valid = True
                        t2_a_post_events += 1
                if has_valid:
                    t2_a_post_eps += 1

            results[key] = {
                "total_episodes": len(episodes),
                "hc_episodes": len(hc_episodes),
                "with_scene_coll": sum(1 for e in episodes if e.scene_collision_steps),
                "task1_B_episodes": t1_b_eps,
                "task1_B_events": t1_b_events,
                "task1_C_episodes": t1_c_eps,
                "task1_A_easy_episodes": t1_a_easy_eps,
                "task1_A_easy_events": t1_a_easy_events,
                "task1_A_hard_episodes": t1_a_hard_eps,
                "task1_A_hard_events": t1_a_hard_events,
                "task2_C_episodes": t2_c_eps,
                "task2_C_samples": t2_c_samples,
                "task2_B_episodes": t2_b_eps,
                "task2_B_events": t2_b_events,
                "task2_A_post_episodes": t2_a_post_eps,
                "task2_A_post_events": t2_a_post_events,
            }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze event parameter choices")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--gap-thresholds", nargs="+", type=int,
                        default=[5, 10, 15, 20, 30, 50])
    parser.add_argument("--safe-windows", nargs="+", type=int,
                        default=[30, 45, 60, 90, 120, 150, 180])
    parser.add_argument("--near-miss-dists", nargs="+", type=float,
                        default=[0.8, 1.0, 1.2, 1.5, 2.0, 2.5])
    parser.add_argument("--near-miss-gaps", nargs="+", type=int,
                        default=[5, 10, 15, 20, 30])
    parser.add_argument("--near-miss-half-spans", nargs="+", type=int,
                        default=[15, 20, 30, 45, 60])
    parser.add_argument("--lead-times", nargs="+", type=int,
                        default=[3, 5, 10, 15, 20, 25, 30],
                        help="Lead times K for Task 2 C (human collision imminent)")
    parser.add_argument("--pre-windows", nargs="+", type=int,
                        default=[10, 15, 20, 30, 45, 60],
                        help="Pre-collision window sizes for Task 2 B sampling")
    parser.add_argument("--post-windows", nargs="+", type=int,
                        default=[10, 15, 20, 30, 45, 60],
                        help="Post-collision window sizes for Task 2 A-post sampling")
    parser.add_argument("--clip-size", type=int, default=3)
    parser.add_argument("--human-types", nargs="+",
                        default=["neutral_0", "female_2", "male_2"],
                        help="Human types to include (default: all three)")
    parser.add_argument("--sim-task-types", nargs="+",
                        default=["social_navigation", "social_rearrangement"],
                        help="Simulation task types to include (default: both)")
    parser.add_argument("--output-json", default="", help="Save full results to JSON")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    episodes = scan_dataset(dataset_dir, args.human_types, args.sim_task_types)
    if not episodes:
        log.error("No episodes found.")
        sys.exit(1)

    n_hc = sum(1 for e in episodes if e.did_collide_human)
    n_sc = sum(1 for e in episodes if e.scene_collision_steps)
    print_section("Dataset Overview")
    print(f"  Total episodes     : {len(episodes)}")
    print(f"  With human coll    : {n_hc}")
    print(f"  With scene coll    : {n_sc}")
    print(f"  Total steps        : {sum(e.num_steps for e in episodes)}")
    print(f"  Avg steps/episode  : {np.mean([e.num_steps for e in episodes]):.0f}")

    all_results = {}

    # ---- 1. Scene collision events ----
    print_section("Scene Collision Events (B-class) vs gap_threshold")
    sc_results = analyze_scene_collision_events(episodes, args.gap_thresholds)
    all_results["scene_collision"] = sc_results

    headers = ["gap", "events", "ep_w_ev", "avg_sz", "med_sz", "min", "max", "p25", "p75", "p90"]
    rows = []
    for gap in args.gap_thresholds:
        r = sc_results[gap]
        rows.append([
            gap, r["total_events"], r["episodes_with_events"],
            r["avg_event_size"], r["median_event_size"],
            r["min_event_size"], r["max_event_size"],
            r["size_percentiles"]["p25"], r["size_percentiles"]["p75"],
            r["size_percentiles"]["p90"],
        ])
    print_table(headers, rows)

    # ---- 2. Human collision episodes (C-class) ----
    print_section("Human Collision Episodes (C-class)")
    hc_results = analyze_human_collision_episodes(episodes)
    all_results["human_collision"] = hc_results

    print(f"  HC episodes          : {hc_results['hc_episodes']}")
    print(f"  With valid hc_steps  : {hc_results['hc_episodes_with_steps']}")
    pos = hc_results["last_hc_frame_position"]
    print(f"  Last HC frame pos    : avg={pos['avg']}, median={pos['median']}, "
          f"min={pos['min']}, max={pos['max']}")
    fpep = hc_results["hc_frames_per_episode"]
    print(f"  HC frames/episode    : avg={fpep['avg']}, median={fpep['median']}, "
          f"min={fpep['min']}, max={fpep['max']}")

    # ---- 3. Safe segments ----
    print_section("Safe Events (A-easy) vs safe_event_window")
    safe_results = analyze_safe_segments(episodes, args.safe_windows)
    all_results["safe_segments"] = safe_results

    seg_stats = safe_results.pop("_segment_stats")
    print(f"\n  Raw safe segment stats (before windowing):")
    print(f"    Total segments : {seg_stats['total_segments']}")
    print(f"    Length stats   : avg={seg_stats['avg_segment_length']}, "
          f"median={seg_stats['median_segment_length']}, "
          f"min={seg_stats['min_segment_length']}, max={seg_stats['max_segment_length']}")
    pct = seg_stats["percentiles"]
    print(f"    Percentiles    : p10={pct['p10']}, p25={pct['p25']}, "
          f"p50={pct['p50']}, p75={pct['p75']}, p90={pct['p90']}")
    print()

    headers = ["window", "events", "ep_w_ev", "avg/ep", "max/ep"]
    rows = []
    for window in args.safe_windows:
        r = safe_results[window]
        rows.append([
            window, r["total_safe_events"], r["episodes_with_safe_events"],
            r["avg_events_per_episode"], r["max_events_per_episode"],
        ])
    print_table(headers, rows)
    safe_results["_segment_stats"] = seg_stats

    # ---- 4. Near-miss events ----
    print_section("Near-Miss Events (A-hard) vs (dist, gap)")
    nm_results = analyze_near_miss_events(
        episodes, args.near_miss_dists, args.near_miss_gaps,
        args.near_miss_half_spans,
    )
    all_results["near_miss"] = nm_results

    headers = ["dist", "gap", "eps_nm", "nm_frames", "events", "avg_sz", "med_sz", "max_sz"]
    for hs in args.near_miss_half_spans:
        headers.append(f"hs={hs}")
    rows = []
    for key in sorted(nm_results.keys()):
        r = nm_results[key]
        row = [
            r["near_miss_dist"], r["near_miss_gap"],
            r["episodes_with_near_miss"], r["total_near_miss_frames"],
            r["total_events"], r["avg_event_size"],
            r["median_event_size"], r["max_event_size"],
        ]
        for hs in args.near_miss_half_spans:
            row.append(r["valid_events_per_half_span"].get(hs, 0))
        rows.append(row)
    print_table(headers, rows)

    # ---- 5. Task 2 C lead times ----
    print_section("Task 2 C (Human Collision Imminent) vs lead_time K")
    lt_results = analyze_task2_c_lead_times(episodes, args.lead_times, args.clip_size)
    all_results["task2_c_lead_times"] = lt_results

    print(f"  HC episodes total         : {lt_results['hc_episodes_total']}")
    print()

    headers = ["K", "valid", "skip_early", "skip_overlap"]
    rows = []
    for k in args.lead_times:
        r = lt_results["per_lead_time"].get(k)
        if r:
            rows.append([k, r["valid_samples"], r["skipped_too_early"], r["skipped_overlap"]])
    print_table(headers, rows)

    # ---- 6. Task 2 pre/post zones ----
    print_section("Task 2 Pre/Post Zones vs gap_threshold & window size")
    pp_results = analyze_task2_pre_post_zones(
        episodes, args.gap_thresholds,
        args.pre_windows, args.post_windows, args.clip_size,
    )
    all_results["pre_post_zones"] = pp_results

    for gap in args.gap_thresholds:
        r = pp_results[gap]
        pre_iv = r["pre_full_interval"]
        post_iv = r["post_full_interval"]
        print(f"\n  gap_threshold={gap}  ({r['total_scene_events']} scene events)")
        print(f"    Full pre  interval: avg={pre_iv['avg']}, med={pre_iv['med']}, "
              f"min={pre_iv['min']}, max={pre_iv['max']}, p25={pre_iv['p25']}, p75={pre_iv['p75']}")
        print(f"    Full post interval: avg={post_iv['avg']}, med={post_iv['med']}, "
              f"min={post_iv['min']}, max={post_iv['max']}, p25={post_iv['p25']}, p75={post_iv['p75']}")

        print(f"    Pre-window (B sampling):")
        headers = ["pre_win", "valid_ev", "avg_actual"]
        rows = []
        for pw in args.pre_windows:
            ps = r["per_pre_window"][pw]
            rows.append([pw, ps["valid_events"], ps["avg_actual_len"]])
        print_table(headers, rows)

        print(f"    Post-window (A-post sampling):")
        headers = ["post_win", "valid_ev", "avg_actual"]
        rows = []
        for ptw in args.post_windows:
            ps = r["per_post_window"][ptw]
            rows.append([ptw, ps["valid_events"], ps["avg_actual_len"]])
        print_table(headers, rows)

    # ---- 7. Per sub-dataset episode counts (with chosen params) ----
    print_section("Per Sub-Dataset Qualifying Episodes (chosen params)")
    print(f"  Parameters: gap={30}, clip={args.clip_size}, safe_window={90}, "
          f"nm_dist={1.5}, nm_gap={15}, nm_half_span={30}, "
          f"pre_win={30}, post_win={30}")
    print()

    subdataset_results = analyze_per_subdataset_episodes(
        dataset_dir,
        human_types=args.human_types,
        sim_task_types=args.sim_task_types,
        gap_threshold=30,
        clip_size=args.clip_size,
        safe_event_window=90,
        near_miss_dist=1.5,
        near_miss_gap=15,
        near_miss_half_span=30,
        pre_window=30,
        post_window=30,
    )
    all_results["per_subdataset"] = subdataset_results

    headers = [
        "sub-dataset", "total", "hc", "sc",
        "T1-B ep", "T1-B ev", "T1-C", "T1-Ae ep", "T1-Ae ev",
        "T1-Ah ep", "T1-Ah ev",
        "T2-C ep", "T2-C samp", "T2-B ep", "T2-B ev",
        "T2-Ap ep", "T2-Ap ev",
    ]
    rows = []
    totals = {h: 0 for h in headers[1:]}
    for key in sorted(subdataset_results.keys()):
        r = subdataset_results[key]
        if r["total_episodes"] == 0:
            continue
        row = [
            key, r["total_episodes"], r["hc_episodes"], r["with_scene_coll"],
            r["task1_B_episodes"], r["task1_B_events"],
            r["task1_C_episodes"],
            r["task1_A_easy_episodes"], r["task1_A_easy_events"],
            r["task1_A_hard_episodes"], r["task1_A_hard_events"],
            r["task2_C_episodes"], r["task2_C_samples"],
            r["task2_B_episodes"], r["task2_B_events"],
            r["task2_A_post_episodes"], r["task2_A_post_events"],
        ]
        for i, h in enumerate(headers[1:]):
            totals[h] += row[i + 1]
        rows.append(row)

    total_row = ["TOTAL"] + [totals[h] for h in headers[1:]]
    rows.append(total_row)
    print_table(headers, rows)

    # ---- Save JSON ----
    if args.output_json:
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2, default=_convert)
        log.info(f"Full results saved to {args.output_json}")

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
