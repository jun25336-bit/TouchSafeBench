#!/usr/bin/env python3
"""
Analyze failure JSONL files for distance estimation errors, trend misclassification,
collision flag errors, and human visibility effects.
"""

import json
import glob
import re
import os
import numpy as np
from collections import defaultdict, Counter


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_raw_response(raw):
    """Parse raw_response string, handling markdown-wrapped JSON."""
    rr = raw.strip()
    if rr.startswith("```"):
        # Remove markdown code fence
        rr = rr.strip("`").strip()
        if rr.startswith("json"):
            rr = rr[4:].strip()
        # May have trailing ```
        if rr.endswith("```"):
            rr = rr[:-3].strip()
    try:
        return json.loads(rr)
    except json.JSONDecodeError:
        return None


def is_visible_in_ego_at_frame(gt_human_visible, frame_idx):
    """Check if human is visible in any ego view (arm or head) at a given frame index."""
    arm_vis = gt_human_visible.get("arm", [])
    head_vis = gt_human_visible.get("head", [])
    arm_ok = arm_vis[frame_idx] if frame_idx < len(arm_vis) else False
    head_ok = head_vis[frame_idx] if frame_idx < len(head_vis) else False
    return arm_ok or head_ok


def any_visible_in_ego(gt_human_visible):
    """Check if human is visible in any ego view in any frame."""
    arm_vis = gt_human_visible.get("arm", [])
    head_vis = gt_human_visible.get("head", [])
    for i in range(max(len(arm_vis), len(head_vis))):
        a = arm_vis[i] if i < len(arm_vis) else False
        h = head_vis[i] if i < len(head_vis) else False
        if a or h:
            return True
    return False


def extract_model(filename):
    """Extract model name from filename."""
    basename = os.path.basename(filename)
    for model in ["gpt-5.5", "gemini-3.1-pro-preview", "gemini-robotics-er-1.6-preview"]:
        if f"_{model}_" in basename:
            return model
    return "unknown"


def extract_modality(filename):
    """Extract modality config from filename."""
    basename = os.path.basename(filename)
    model = extract_model(filename)
    prefix = f"all_{model}_arm+head_"
    suffix = "_n900_failures.jsonl"
    if basename.startswith(prefix) and basename.endswith(suffix):
        return basename[len(prefix):-len(suffix)]
    return "unknown"


def load_all_failures():
    """Load all failure files and return records grouped by model and by (model, modality)."""
    pattern = os.path.join(BASE_DIR, "*_failures.jsonl")
    files = sorted(glob.glob(pattern))

    records_by_model = defaultdict(list)
    records_by_model_modality = defaultdict(list)

    for fpath in files:
        model = extract_model(fpath)
        modality = extract_modality(fpath)

        with open(fpath) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                d["_model"] = model
                d["_modality"] = modality
                records_by_model[model].append(d)
                records_by_model_modality[(model, modality)].append(d)

    return records_by_model, records_by_model_modality


# =============================================================================
# SECTION 1: Distance estimation errors in failures
# =============================================================================
def analyze_distance_errors(records_by_model):
    print("=" * 100)
    print("SECTION 1: DISTANCE ESTIMATION ERRORS IN FAILURES")
    print("=" * 100)

    for model in sorted(records_by_model.keys()):
        print(f"\n{'─' * 90}")
        print(f"MODEL: {model}")
        print(f"{'─' * 90}")

        records = records_by_model[model]

        # Group by (task, gt->pred)
        by_pattern = defaultdict(list)
        for d in records:
            by_pattern[(d["task"], d["gt"], d["pred"])].append(d)

        for pattern_key in sorted(by_pattern.keys()):
            task, gt, pred = pattern_key
            recs = by_pattern[pattern_key]

            all_maes = []
            visible_maes = []      # sample-level MAE when human visible in ego
            not_visible_maes = []  # sample-level MAE when human NOT visible in ego
            per_frame_visible_errors = []
            per_frame_not_visible_errors = []
            parse_failures = 0

            for d in recs:
                parsed = parse_raw_response(d["raw_response"])
                if parsed is None:
                    parse_failures += 1
                    continue

                pred_distances = parsed.get("distances", [])
                gt_distances = d["metadata"].get("gt_distances", [])
                gt_vis = d["metadata"].get("gt_human_visible", {})

                if not pred_distances or not gt_distances:
                    continue

                n = min(len(pred_distances), len(gt_distances))
                if n == 0:
                    continue

                # Handle non-numeric predictions
                try:
                    pred_d = [float(x) if x is not None else np.nan for x in pred_distances[:n]]
                    gt_d = [float(x) for x in gt_distances[:n]]
                except (ValueError, TypeError):
                    parse_failures += 1
                    continue

                errors = [abs(p - g) for p, g in zip(pred_d, gt_d) if not np.isnan(p)]
                if errors:
                    mae = np.mean(errors)
                    all_maes.append(mae)

                # Per-frame visibility analysis
                for i in range(n):
                    if np.isnan(pred_d[i]):
                        continue
                    err = abs(pred_d[i] - gt_d[i])
                    if is_visible_in_ego_at_frame(gt_vis, i):
                        per_frame_visible_errors.append(err)
                    else:
                        per_frame_not_visible_errors.append(err)

                # Sample-level: any frame visible in ego?
                sample_visible = any_visible_in_ego(gt_vis)
                if errors:
                    if sample_visible:
                        visible_maes.append(np.mean(errors))
                    else:
                        not_visible_maes.append(np.mean(errors))

            if not all_maes:
                continue

            print(f"\n  {task} | {gt} -> {pred} (n={len(recs)}, parsed={len(all_maes)}, parse_fail={parse_failures})")
            print(f"    Overall MAE:           {np.mean(all_maes):.4f} m  (median {np.median(all_maes):.4f}, std {np.std(all_maes):.4f})")

            if visible_maes:
                print(f"    MAE (human visible in ego):     {np.mean(visible_maes):.4f} m  (n={len(visible_maes)})")
            if not_visible_maes:
                print(f"    MAE (human NOT visible in ego): {np.mean(not_visible_maes):.4f} m  (n={len(not_visible_maes)})")

            if per_frame_visible_errors:
                print(f"    Per-frame abs error (visible in ego):     {np.mean(per_frame_visible_errors):.4f} m  (n={len(per_frame_visible_errors)})")
            if per_frame_not_visible_errors:
                print(f"    Per-frame abs error (NOT visible in ego): {np.mean(per_frame_not_visible_errors):.4f} m  (n={len(per_frame_not_visible_errors)})")


# =============================================================================
# SECTION 2: Distance trend misclassification
# =============================================================================
def analyze_trend_misclassification(records_by_model):
    print("\n\n" + "=" * 100)
    print("SECTION 2: DISTANCE TREND MISCLASSIFICATION")
    print("=" * 100)

    for model in sorted(records_by_model.keys()):
        print(f"\n{'─' * 90}")
        print(f"MODEL: {model}")
        print(f"{'─' * 90}")

        records = records_by_model[model]

        by_pattern = defaultdict(list)
        for d in records:
            by_pattern[(d["task"], d["gt"], d["pred"])].append(d)

        for pattern_key in sorted(by_pattern.keys()):
            task, gt, pred = pattern_key
            recs = by_pattern[pattern_key]

            total = 0
            wrong_trend = 0
            trend_confusions = Counter()  # (gt_trend, pred_trend) -> count
            parse_failures = 0

            for d in recs:
                parsed = parse_raw_response(d["raw_response"])
                if parsed is None:
                    parse_failures += 1
                    continue

                gt_trend = d["metadata"].get("gt_distance_trend")
                pred_trend = parsed.get("distance_trend")

                if gt_trend is None or pred_trend is None:
                    continue

                gt_trend_n = gt_trend.strip().lower()
                pred_trend_n = pred_trend.strip().lower()

                total += 1
                if gt_trend_n != pred_trend_n:
                    wrong_trend += 1
                    trend_confusions[(gt_trend_n, pred_trend_n)] += 1

            if total == 0:
                continue

            wrong_frac = wrong_trend / total
            print(f"\n  {task} | {gt} -> {pred} (n={total}, parse_fail={parse_failures})")
            print(f"    Wrong trend fraction: {wrong_trend}/{total} = {wrong_frac:.3f}")

            if trend_confusions:
                print(f"    Most common (gt_trend -> pred_trend) errors:")
                for (gt_t, pred_t), cnt in trend_confusions.most_common(5):
                    print(f"      {gt_t:12s} -> {pred_t:12s}: {cnt:4d} ({cnt/total:.3f})")


# =============================================================================
# SECTION 3: Collision flag errors
# =============================================================================
def analyze_collision_flag_errors(records_by_model):
    print("\n\n" + "=" * 100)
    print("SECTION 3: COLLISION FLAG ERRORS")
    print("=" * 100)
    print("For failures where ground truth has collisions, does the model also detect them?")

    for model in sorted(records_by_model.keys()):
        print(f"\n{'─' * 90}")
        print(f"MODEL: {model}")
        print(f"{'─' * 90}")

        records = records_by_model[model]

        # ── Scene collision ──
        scene_by_pattern = defaultdict(lambda: {"gt_true": 0, "pred_true": 0, "pred_false": 0, "parse_fail": 0})

        # ── Human collision ──
        human_by_pattern = defaultdict(lambda: {"gt_true": 0, "pred_true": 0, "pred_false": 0, "parse_fail": 0})

        # Also per-frame analysis
        scene_frame_tp = 0  # gt=True, pred=True
        scene_frame_fn = 0  # gt=True, pred=False
        human_frame_tp = 0
        human_frame_fn = 0

        for d in records:
            pattern = (d["task"], d["gt"], d["pred"])
            parsed = parse_raw_response(d["raw_response"])

            gt_scene_flags = d["metadata"].get("gt_collision_flags_scene", [])
            gt_human_flags = d["metadata"].get("gt_collision_flags_human", [])

            has_scene_collision = any(gt_scene_flags)
            has_human_collision = any(gt_human_flags)

            if has_scene_collision:
                scene_by_pattern[pattern]["gt_true"] += 1
                if parsed:
                    pred_scene_flags = parsed.get("collision_flags_scene", [])
                    if any(pred_scene_flags):
                        scene_by_pattern[pattern]["pred_true"] += 1
                    else:
                        scene_by_pattern[pattern]["pred_false"] += 1
                    # Per-frame
                    for i in range(min(len(gt_scene_flags), len(pred_scene_flags))):
                        if gt_scene_flags[i]:
                            if pred_scene_flags[i]:
                                scene_frame_tp += 1
                            else:
                                scene_frame_fn += 1
                else:
                    scene_by_pattern[pattern]["parse_fail"] += 1

            if has_human_collision:
                human_by_pattern[pattern]["gt_true"] += 1
                if parsed:
                    pred_human_flags = parsed.get("collision_flags_human", [])
                    if any(pred_human_flags):
                        human_by_pattern[pattern]["pred_true"] += 1
                    else:
                        human_by_pattern[pattern]["pred_false"] += 1
                    # Per-frame
                    for i in range(min(len(gt_human_flags), len(pred_human_flags))):
                        if gt_human_flags[i]:
                            if pred_human_flags[i]:
                                human_frame_tp += 1
                            else:
                                human_frame_fn += 1
                else:
                    human_by_pattern[pattern]["parse_fail"] += 1

        # Print scene collision results
        total_scene_gt = sum(v["gt_true"] for v in scene_by_pattern.values())
        total_scene_pred = sum(v["pred_true"] for v in scene_by_pattern.values())
        total_scene_pf = sum(v["parse_fail"] for v in scene_by_pattern.values())

        print(f"\n  SCENE COLLISION (gt has scene collision somewhere):")
        if total_scene_gt > 0:
            print(f"    Model also detected (any frame): {total_scene_pred}/{total_scene_gt} = {total_scene_pred/total_scene_gt:.3f}")
            print(f"    (parse failures excluded: {total_scene_pf})")
            if scene_frame_tp + scene_frame_fn > 0:
                print(f"    Per-frame recall (gt=True frames): {scene_frame_tp}/{scene_frame_tp+scene_frame_fn} = {scene_frame_tp/(scene_frame_tp+scene_frame_fn):.3f}")
            print(f"    By (task, gt->pred):")
            for pk in sorted(scene_by_pattern.keys()):
                task, g, p = pk
                v = scene_by_pattern[pk]
                if v["gt_true"] > 0:
                    frac = v["pred_true"] / v["gt_true"]
                    print(f"      {task} {g}->{p}: model detected {v['pred_true']}/{v['gt_true']} = {frac:.3f}")
        else:
            print(f"    No failures with gt scene collision.")

        # Print human collision results
        total_human_gt = sum(v["gt_true"] for v in human_by_pattern.values())
        total_human_pred = sum(v["pred_true"] for v in human_by_pattern.values())
        total_human_pf = sum(v["parse_fail"] for v in human_by_pattern.values())

        print(f"\n  HUMAN COLLISION (gt has human collision somewhere):")
        if total_human_gt > 0:
            print(f"    Model also detected (any frame): {total_human_pred}/{total_human_gt} = {total_human_pred/total_human_gt:.3f}")
            print(f"    (parse failures excluded: {total_human_pf})")
            if human_frame_tp + human_frame_fn > 0:
                print(f"    Per-frame recall (gt=True frames): {human_frame_tp}/{human_frame_tp+human_frame_fn} = {human_frame_tp/(human_frame_tp+human_frame_fn):.3f}")
            print(f"    By (task, gt->pred):")
            for pk in sorted(human_by_pattern.keys()):
                task, g, p = pk
                v = human_by_pattern[pk]
                if v["gt_true"] > 0:
                    frac = v["pred_true"] / v["gt_true"]
                    print(f"      {task} {g}->{p}: model detected {v['pred_true']}/{v['gt_true']} = {frac:.3f}")
        else:
            print(f"    No failures with gt human collision.")


# =============================================================================
# SECTION 4: Human visibility effect on Task1 B->A and B->C errors
# =============================================================================
def analyze_visibility_task1(records_by_model):
    print("\n\n" + "=" * 100)
    print("SECTION 4: HUMAN VISIBILITY EFFECT ON TASK1 ERRORS")
    print("=" * 100)

    for model in sorted(records_by_model.keys()):
        print(f"\n{'─' * 90}")
        print(f"MODEL: {model}")
        print(f"{'─' * 90}")

        records = records_by_model[model]
        task1_records = [d for d in records if d["task"] == "task1"]

        if not task1_records:
            print("  No task1 failures.")
            continue

        # ── 4a: When human NOT visible in ego views, what fraction are B->A? ──
        not_visible_records = [d for d in task1_records if not any_visible_in_ego(d["metadata"].get("gt_human_visible", {}))]
        not_visible_total = len(not_visible_records)
        not_visible_ba = sum(1 for d in not_visible_records if d["gt"] == "B" and d["pred"] == "A")
        not_visible_by_gt_pred = Counter()
        for d in not_visible_records:
            not_visible_by_gt_pred[(d["gt"], d["pred"])] += 1

        print(f"\n  4a. When human NOT visible in ego views (arm/head):")
        print(f"    Total task1 failures (human not visible): {not_visible_total}")
        if not_visible_total > 0:
            print(f"    Fraction that are B->A: {not_visible_ba}/{not_visible_total} = {not_visible_ba/not_visible_total:.3f}")
            print(f"    Full breakdown:")
            for (g, p), cnt in not_visible_by_gt_pred.most_common():
                print(f"      {g}->{p}: {cnt:4d} ({cnt/not_visible_total:.3f})")

        # ── 4b: When human IS visible and close (<1.5m), what fraction are B->C? ──
        visible_close_records = []
        for d in task1_records:
            gt_vis = d["metadata"].get("gt_human_visible", {})
            gt_dists = d["metadata"].get("gt_distances", [])
            if any_visible_in_ego(gt_vis) and gt_dists and min(gt_dists) < 1.5:
                visible_close_records.append(d)

        visible_close_total = len(visible_close_records)
        visible_close_bc = sum(1 for d in visible_close_records if d["gt"] == "B" and d["pred"] == "C")
        visible_close_by_gt_pred = Counter()
        for d in visible_close_records:
            visible_close_by_gt_pred[(d["gt"], d["pred"])] += 1

        print(f"\n  4b. When human IS visible in ego AND close (<1.5m):")
        print(f"    Total task1 failures (visible + close): {visible_close_total}")
        if visible_close_total > 0:
            print(f"    Fraction that are B->C: {visible_close_bc}/{visible_close_total} = {visible_close_bc/visible_close_total:.3f}")
            print(f"    Full breakdown:")
            for (g, p), cnt in visible_close_by_gt_pred.most_common():
                print(f"      {g}->{p}: {cnt:4d} ({cnt/visible_close_total:.3f})")

        # ── Additional context: overall task1 B->A and B->C rates for comparison ──
        all_task1 = len(task1_records)
        all_ba = sum(1 for d in task1_records if d["gt"] == "B" and d["pred"] == "A")
        all_bc = sum(1 for d in task1_records if d["gt"] == "B" and d["pred"] == "C")
        print(f"\n  Context: overall task1 failure distribution:")
        print(f"    Total task1 failures: {all_task1}")
        if all_task1 > 0:
            print(f"    B->A overall rate: {all_ba}/{all_task1} = {all_ba/all_task1:.3f}")
            print(f"    B->C overall rate: {all_bc}/{all_task1} = {all_bc/all_task1:.3f}")


# =============================================================================
# SECTION 5 (BONUS): Distance MAE by model x modality
# =============================================================================
def analyze_distance_by_modality(records_by_model_modality):
    print("\n\n" + "=" * 100)
    print("SECTION 5 (BONUS): DISTANCE MAE BY MODEL x MODALITY")
    print("=" * 100)

    results = []
    for (model, modality), records in sorted(records_by_model_modality.items()):
        all_maes = []
        for d in records:
            parsed = parse_raw_response(d["raw_response"])
            if parsed is None:
                continue
            pred_distances = parsed.get("distances", [])
            gt_distances = d["metadata"].get("gt_distances", [])
            if not pred_distances or not gt_distances:
                continue
            n = min(len(pred_distances), len(gt_distances))
            try:
                pred_d = [float(x) if x is not None else np.nan for x in pred_distances[:n]]
                gt_d = [float(x) for x in gt_distances[:n]]
            except (ValueError, TypeError):
                continue
            errors = [abs(p - g) for p, g in zip(pred_d, gt_d) if not np.isnan(p)]
            if errors:
                all_maes.append(np.mean(errors))

        if all_maes:
            results.append((model, modality, np.mean(all_maes), np.median(all_maes), len(all_maes)))

    current_model = None
    for model, modality, mean_mae, median_mae, n in sorted(results):
        if model != current_model:
            print(f"\n  {model}:")
            current_model = model
        print(f"    {modality:25s}  MAE={mean_mae:.4f}  median={median_mae:.4f}  (n={n})")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Loading all failure JSONL files...\n")
    records_by_model, records_by_model_modality = load_all_failures()

    total = sum(len(v) for v in records_by_model.values())
    print(f"Loaded {total} failure records across {len(records_by_model)} models "
          f"and {len(records_by_model_modality)} (model, modality) combos.\n")

    for model, recs in sorted(records_by_model.items()):
        task_counts = Counter(d["task"] for d in recs)
        print(f"  {model}: {len(recs)} failures  (task1={task_counts.get('task1',0)}, task2={task_counts.get('task2',0)})")

    analyze_distance_errors(records_by_model)
    analyze_trend_misclassification(records_by_model)
    analyze_collision_flag_errors(records_by_model)
    analyze_visibility_task1(records_by_model)
    analyze_distance_by_modality(records_by_model_modality)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
