#!/usr/bin/env python3
"""
Trajectory characteristics analysis: correct vs incorrect predictions
across all 27 model-modality conditions in the safety benchmark.
"""

import json
import glob
import numpy as np
from collections import defaultdict

# ── Load all progress data ──────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(f"{DATA_DIR}/all_*_n900_progress.jsonl"))
print(f"Found {len(files)} progress files\n")

records = []
for fpath in files:
    with open(fpath) as f:
        for line in f:
            d = json.loads(line)
            meta = d["metadata"]
            # Compute correct
            correct = d["gt_label"] == d["pred_label"]
            # Average distance across 3 frames
            dists = meta["gt_distances"]
            avg_dist = np.mean(dists)
            # Human visible in ego views (arm OR head) in at least 1 frame
            hv = meta["gt_human_visible"]
            arm_any = any(hv["arm"])
            head_any = any(hv["head"])
            ego_visible = arm_any or head_any
            # Scene collision count (how many of 3 frames have scene collision)
            scene_col_count = sum(meta["gt_collision_flags_scene"])
            # Human collision count
            human_col_count = sum(meta["gt_collision_flags_human"])

            records.append({
                "task": d["task"],
                "gt_label": d["gt_label"],
                "sample_type": d["sample_type"],
                "correct": correct,
                "avg_dist": avg_dist,
                "gt_distances": dists,
                "distance_trend": meta["gt_distance_trend"],
                "ego_visible": ego_visible,
                "arm_visible": arm_any,
                "head_visible": head_any,
                "scene_col_count": scene_col_count,
                "human_col_count": human_col_count,
                "gt_collision_flags_scene": meta["gt_collision_flags_scene"],
            })

print(f"Total records loaded: {len(records)}")

# ── Helper ──────────────────────────────────────────────────────────────────
def acc(subset):
    if len(subset) == 0:
        return float("nan"), 0
    c = sum(1 for r in subset if r["correct"])
    return c / len(subset), len(subset)

def fmt_acc(val, n):
    if n == 0:
        return "  --  (n=0)"
    return f"{val*100:5.1f}% (n={n})"

# Build groups by (task, gt_label)
groups = defaultdict(list)
for r in records:
    groups[(r["task"], r["gt_label"])].append(r)

task_labels = sorted(groups.keys())
print(f"Unique (task, class) combos: {len(task_labels)}")
for k in task_labels:
    a, n = acc(groups[k])
    print(f"  {k[0]}-{k[1]}: n={n}, accuracy={a*100:.1f}%")
print()

# ═══════════════════════════════════════════════════════════════════════════
# 1. DISTANCE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("1. DISTANCE DISTRIBUTION: Correct vs Incorrect")
print("=" * 90)

for tk, label in task_labels:
    subset = groups[(tk, label)]
    corr = [r for r in subset if r["correct"]]
    incorr = [r for r in subset if not r["correct"]]

    print(f"\n--- {tk} / Class {label} ---")
    for name, grp in [("Correct", corr), ("Incorrect", incorr)]:
        if len(grp) == 0:
            print(f"  {name:10s}: no samples")
            continue
        dists = [r["avg_dist"] for r in grp]
        print(f"  {name:10s}: n={len(grp):5d}  mean={np.mean(dists):.3f}m  "
              f"median={np.median(dists):.3f}m  "
              f"std={np.std(dists):.3f}m  "
              f"range=[{np.min(dists):.2f}, {np.max(dists):.2f}]")

print(f"\n{'':3s}{'':20s}", end="")
bins = [("<1m", 0, 1), ("1-2m", 1, 2), ("2-4m", 2, 4), ("4-8m", 4, 8), (">8m", 8, 999)]
for bname, _, _ in bins:
    print(f"{bname:>16s}", end="")
print()

for tk, label in task_labels:
    subset = groups[(tk, label)]
    print(f"   {tk}-{label:1s}               ", end="")
    for bname, lo, hi in bins:
        bsub = [r for r in subset if lo <= r["avg_dist"] < hi]
        a, n = acc(bsub)
        print(f"{fmt_acc(a, n):>16s}", end="")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# 2. HUMAN VISIBILITY EFFECT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("2. HUMAN VISIBILITY IN EGO VIEWS (arm OR head, at least 1 frame)")
print("=" * 90)

print(f"\n{'Task-Class':12s} {'Vis Acc':>18s} {'Not-Vis Acc':>18s} "
      f"{'Vis frac(correct)':>18s} {'Vis frac(incorr)':>18s}")
print("-" * 78)

for tk, label in task_labels:
    subset = groups[(tk, label)]
    vis = [r for r in subset if r["ego_visible"]]
    nvis = [r for r in subset if not r["ego_visible"]]
    a_vis, n_vis = acc(vis)
    a_nvis, n_nvis = acc(nvis)

    corr = [r for r in subset if r["correct"]]
    incorr = [r for r in subset if not r["correct"]]
    vis_frac_corr = sum(1 for r in corr if r["ego_visible"]) / max(len(corr), 1)
    vis_frac_incorr = sum(1 for r in incorr if r["ego_visible"]) / max(len(incorr), 1)

    print(f"{tk}-{label:1s}        {fmt_acc(a_vis, n_vis):>18s} {fmt_acc(a_nvis, n_nvis):>18s} "
          f"{vis_frac_corr:>17.1%} {vis_frac_incorr:>17.1%}")

# Breakdown by arm vs head
print(f"\n  Breakdown by view type:")
print(f"  {'Task-Class':12s} {'Arm-only Acc':>18s} {'Head-only Acc':>18s} "
      f"{'Both Acc':>18s} {'Neither Acc':>18s}")
print("  " + "-" * 78)

for tk, label in task_labels:
    subset = groups[(tk, label)]
    arm_only = [r for r in subset if r["arm_visible"] and not r["head_visible"]]
    head_only = [r for r in subset if r["head_visible"] and not r["arm_visible"]]
    both = [r for r in subset if r["arm_visible"] and r["head_visible"]]
    neither = [r for r in subset if not r["arm_visible"] and not r["head_visible"]]

    print(f"  {tk}-{label:1s}        "
          f"{fmt_acc(*acc(arm_only)):>18s} "
          f"{fmt_acc(*acc(head_only)):>18s} "
          f"{fmt_acc(*acc(both)):>18s} "
          f"{fmt_acc(*acc(neither)):>18s}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. DISTANCE TREND EFFECT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("3. DISTANCE TREND EFFECT (closing / stable / separating)")
print("=" * 90)

trends = ["closing", "stable", "separating"]
print(f"\n{'Task-Class':12s}", end="")
for t in trends:
    print(f"  {t:>22s}", end="")
print()
print("-" * 80)

for tk, label in task_labels:
    subset = groups[(tk, label)]
    print(f"{tk}-{label:1s}        ", end="")
    for trend in trends:
        tsub = [r for r in subset if r["distance_trend"] == trend]
        a, n = acc(tsub)
        print(f"  {fmt_acc(a, n):>22s}", end="")
    print()

# Also show the sample-type composition for each trend
print(f"\n  Sample-type composition per trend:")
for tk, label in task_labels:
    subset = groups[(tk, label)]
    print(f"\n  {tk}-{label}:")
    for trend in trends:
        tsub = [r for r in subset if r["distance_trend"] == trend]
        if len(tsub) == 0:
            continue
        st_counts = defaultdict(int)
        for r in tsub:
            st_counts[r["sample_type"]] += 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(st_counts.items()))
        print(f"    {trend:12s}: n={len(tsub):5d} | {parts}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. SCENE COLLISION DENSITY (Task1-B and Task2-B)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("4. SCENE COLLISION DENSITY (frames with gt_collision_flags_scene=True)")
print("   Restricted to Task1-B and Task2-B (scene collision class)")
print("=" * 90)

print("  NOTE: Task2-B is 'scene_collision_imminent' — sampled frames are PRE-collision,")
print("        so gt_collision_flags_scene are all False. Analysis is most meaningful for Task1-B.")

for tk in ["task1", "task2"]:
    label = "B"
    subset = groups.get((tk, label), [])
    if not subset:
        continue
    print(f"\n--- {tk}-{label} (sample_type: {subset[0]['sample_type']}) ---")
    print(f"  {'Scene-col frames':20s} {'Accuracy':>18s}")
    print(f"  " + "-" * 40)
    for count in [0, 1, 2, 3]:
        csub = [r for r in subset if r["scene_col_count"] == count]
        a, n = acc(csub)
        print(f"  {count} of 3 frames       {fmt_acc(a, n):>18s}")

    # Also show mean distance for each scene collision count
    print(f"\n  Mean distance by scene-col count:")
    for count in [0, 1, 2, 3]:
        csub = [r for r in subset if r["scene_col_count"] == count]
        if csub:
            d = np.mean([r["avg_dist"] for r in csub])
            print(f"    {count} frames: mean_dist={d:.3f}m (n={len(csub)})")

# Also show scene collision density for ALL (task, class) combos
print(f"\n  Scene collision density across ALL classes:")
print(f"  {'Task-Class':12s}  {'0 frames':>16s}  {'1 frame':>16s}  {'2 frames':>16s}  {'3 frames':>16s}")
print(f"  " + "-" * 80)
for tk, label in task_labels:
    subset = groups[(tk, label)]
    print(f"  {tk}-{label:1s}        ", end="")
    for count in [0, 1, 2, 3]:
        csub = [r for r in subset if r["scene_col_count"] == count]
        a, n = acc(csub)
        print(f"  {fmt_acc(a, n):>16s}", end="")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# 5. CROSS-TABULATION: Task1-B by (human_visible_ego, distance_bin)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("5. CROSS-TABULATION: Task1-B — Accuracy by (ego_visible x distance_bin)")
print("=" * 90)

subset_t1b = groups.get(("task1", "B"), [])
if subset_t1b:
    # Header
    print(f"\n  {'':18s}", end="")
    for bname, _, _ in bins:
        print(f"  {bname:>16s}", end="")
    print(f"  {'TOTAL':>16s}")
    print(f"  " + "-" * (18 + 17 * (len(bins) + 1)))

    for vis_label, vis_val in [("Visible", True), ("Not visible", False)]:
        vsub = [r for r in subset_t1b if r["ego_visible"] == vis_val]
        print(f"  {vis_label:18s}", end="")
        for bname, lo, hi in bins:
            bsub = [r for r in vsub if lo <= r["avg_dist"] < hi]
            a, n = acc(bsub)
            print(f"  {fmt_acc(a, n):>16s}", end="")
        a_tot, n_tot = acc(vsub)
        print(f"  {fmt_acc(a_tot, n_tot):>16s}")

    # Total row
    print(f"  {'TOTAL':18s}", end="")
    for bname, lo, hi in bins:
        bsub = [r for r in subset_t1b if lo <= r["avg_dist"] < hi]
        a, n = acc(bsub)
        print(f"  {fmt_acc(a, n):>16s}", end="")
    a_tot, n_tot = acc(subset_t1b)
    print(f"  {fmt_acc(a_tot, n_tot):>16s}")

# Also do the same for all Task1 classes for completeness
print(f"\n  Extended: All Task1 classes — Accuracy by (ego_visible x distance_bin)")
for label in ["A", "B", "C"]:
    subset = groups.get(("task1", label), [])
    if not subset:
        continue
    print(f"\n  Task1-{label}:")
    print(f"  {'':18s}", end="")
    for bname, _, _ in bins:
        print(f"  {bname:>16s}", end="")
    print(f"  {'TOTAL':>16s}")
    print(f"  " + "-" * (18 + 17 * (len(bins) + 1)))

    for vis_label, vis_val in [("Visible", True), ("Not visible", False)]:
        vsub = [r for r in subset if r["ego_visible"] == vis_val]
        print(f"  {vis_label:18s}", end="")
        for bname, lo, hi in bins:
            bsub = [r for r in vsub if lo <= r["avg_dist"] < hi]
            a, n = acc(bsub)
            print(f"  {fmt_acc(a, n):>16s}", end="")
        a_tot, n_tot = acc(vsub)
        print(f"  {fmt_acc(a_tot, n_tot):>16s}")

    print(f"  {'TOTAL':18s}", end="")
    for bname, lo, hi in bins:
        bsub = [r for r in subset if lo <= r["avg_dist"] < hi]
        a, n = acc(bsub)
        print(f"  {fmt_acc(a, n):>16s}", end="")
    a_tot, n_tot = acc(subset)
    print(f"  {fmt_acc(a_tot, n_tot):>16s}")

print("\n\nDone.")
