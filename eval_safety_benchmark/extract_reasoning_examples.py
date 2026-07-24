#!/usr/bin/env python3
"""
Extract representative reasoning examples from failure JSONL files
for different confusion patterns across models.
"""

import json
import glob
import os
import random
import re
import sys
from collections import defaultdict

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models of interest
MODELS = ["gpt-5.5", "gemini-3.1-pro-preview", "gemini-robotics-er-1.6-preview"]

# Priority confusion patterns with custom sample counts
# Format: (task, gt, pred, count, description)
PRIORITY_PATTERNS = [
    ("task1", "B", "A", 5, "Task1 B->A: scene collision missed as safe"),
    ("task1", "B", "C", 5, "Task1 B->C: scene collision -> human collision"),
    ("task2", "B", "A", 5, "Task2 B->A: scene collision imminent missed as safe"),
    ("task2", "B", "C", 5, "Task2 B->C: scene collision imminent -> human collision"),
    ("task1", "C", "A", 3, "Task1 C->A: human collision missed as safe"),
    ("task2", "A", "C", 3, "Task2 A->C: false alarm human collision"),
]

# All other confusion directions (sample up to 3 each)
ALL_DIRECTIONS = []
for task in ["task1", "task2"]:
    for gt in ["A", "B", "C"]:
        for pred in ["A", "B", "C"]:
            if gt != pred:
                # Check if already in priority
                is_priority = any(
                    p[0] == task and p[1] == gt and p[2] == pred
                    for p in PRIORITY_PATTERNS
                )
                if not is_priority:
                    ALL_DIRECTIONS.append(
                        (task, gt, pred, 3, f"{task} {gt}->{pred}")
                    )

# Combine: priority first, then the rest
ALL_PATTERNS = PRIORITY_PATTERNS + ALL_DIRECTIONS


def parse_raw_response(raw_response):
    """Parse raw_response JSON, handling optional ```json``` markdown wrapping."""
    if raw_response is None:
        return {}
    text = raw_response.strip()
    # Remove markdown code fences
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r'^```[a-zA-Z]*\s*\n?', '', text)
        # Remove closing fence
        text = re.sub(r'\n?```\s*$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": True, "_raw_text": text[:500]}


def extract_modality_from_filename(filename):
    """Extract modality description from filename."""
    basename = os.path.basename(filename)
    # Pattern: all_{model}_{modality}_n900_failures.jsonl
    # Remove prefix and suffix
    basename = basename.replace("all_", "").replace("_failures.jsonl", "")
    # Remove the model name part and n900
    for model in MODELS:
        basename = basename.replace(model + "_", "")
    basename = basename.replace("_n900", "").replace("_n6", "")
    return basename


def print_separator(char="=", width=120):
    print(char * width)


def print_entry(entry, modality, model, idx):
    """Print a single failure entry with full reasoning."""
    meta = entry.get("metadata", {})
    parsed = parse_raw_response(entry.get("raw_response", ""))

    print(f"\n  --- Sample {idx} ---")
    print(f"  Model:         {model}")
    print(f"  Modality:      {modality}")
    print(f"  Sample ID:     {entry.get('sample_id')}")
    print(f"  Task:          {entry.get('task')}")
    print(f"  GT -> Pred:    {entry.get('gt')} -> {entry.get('pred')}")
    print(f"  Sample Type:   {entry.get('sample_type')}")
    print()
    print(f"  GT distances:           {meta.get('gt_distances')}")
    print(f"  GT distance trend:      {meta.get('gt_distance_trend')}")
    print(f"  GT collision human:     {meta.get('gt_collision_flags_human')}")
    print(f"  GT collision scene:     {meta.get('gt_collision_flags_scene')}")

    hv = meta.get("gt_human_visible", {})
    print(f"  GT human visible (arm): {hv.get('arm')}")
    print(f"  GT human visible (head):{hv.get('head')}")

    # Print predicted values from raw_response
    print()
    print(f"  Pred distances:         {parsed.get('distances')}")
    print(f"  Pred distance trend:    {parsed.get('distance_trend')}")
    print(f"  Pred collision human:   {parsed.get('collision_flags_human')}")
    print(f"  Pred collision scene:   {parsed.get('collision_flags_scene')}")
    print(f"  Pred answer:            {parsed.get('answer')}")

    # Print reasoning fields
    print()
    cr = parsed.get("collision_reasoning", "")
    dtr = parsed.get("distance_trend_reasoning", "")
    dr = parsed.get("distances_reasoning", "")

    if cr:
        print(f"  COLLISION REASONING:")
        # Wrap long text
        for line in _wrap_text(cr, width=110, indent="    "):
            print(line)

    if dtr:
        print(f"\n  DISTANCE TREND REASONING:")
        for line in _wrap_text(dtr, width=110, indent="    "):
            print(line)

    if dr:
        print(f"\n  DISTANCES REASONING:")
        for line in _wrap_text(dr, width=110, indent="    "):
            print(line)

    if parsed.get("_parse_error"):
        print(f"\n  [RAW RESPONSE PARSE ERROR] First 500 chars:")
        print(f"    {parsed.get('_raw_text', 'N/A')}")

    print()


def _wrap_text(text, width=110, indent="    "):
    """Simple text wrapping."""
    lines = []
    words = text.split()
    current = indent
    for w in words:
        if len(current) + len(w) + 1 > width:
            lines.append(current)
            current = indent + w
        else:
            if current == indent:
                current += w
            else:
                current += " " + w
    if current.strip():
        lines.append(current)
    return lines


def main():
    # Collect all failure files grouped by model
    failure_files = sorted(glob.glob(os.path.join(BASE_DIR, "*_failures.jsonl")))

    model_files = defaultdict(list)
    for f in failure_files:
        for model in MODELS:
            if model in f:
                model_files[model].append(f)
                break

    print("=" * 120)
    print("FAILURE REASONING EXTRACTION REPORT")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Total failure files found: {len(failure_files)}")
    for m in MODELS:
        print(f"  {m}: {len(model_files[m])} files")
    print("=" * 120)

    # For each pattern, for each model, collect matching entries across all modalities
    total_samples_printed = 0

    for pattern_idx, (task, gt, pred, max_samples, desc) in enumerate(ALL_PATTERNS):
        pattern_has_any = False

        for model in MODELS:
            # Collect all matching entries across modalities for this model
            matching_entries = []  # (entry, modality)

            for fpath in model_files[model]:
                modality = extract_modality_from_filename(fpath)
                with open(fpath) as fh:
                    for line in fh:
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if (
                            entry.get("task") == task
                            and entry.get("gt") == gt
                            and entry.get("pred") == pred
                        ):
                            matching_entries.append((entry, modality))

            if not matching_entries:
                continue

            if not pattern_has_any:
                # Print pattern header on first model that has entries
                print_separator("=")
                print(f"\nPATTERN: {desc}")
                print(f"  (gt={gt}, pred={pred}, task={task}, max_samples={max_samples}/model)")
                pattern_has_any = True

            # Sample
            sampled = random.sample(
                matching_entries, min(max_samples, len(matching_entries))
            )

            print(f"\n  [{model}] -- {len(matching_entries)} total matches across modalities, showing {len(sampled)}")

            for i, (entry, modality) in enumerate(sampled, 1):
                print_entry(entry, modality, model, i)
                total_samples_printed += 1

    print_separator("=")
    print(f"\nTotal reasoning samples printed: {total_samples_printed}")
    print("Done.")


if __name__ == "__main__":
    main()
