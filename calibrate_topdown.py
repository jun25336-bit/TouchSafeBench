"""
Batch-calibrate topdown projection parameters for all scenes.

Loads each scene's stage GLB mesh (via trimesh) to get bounding box,
then computes the topdown camera projection using the same formula as
the dataset generation pipeline (HFOV=90, 1920x1080, margin=1.10).

Outputs topdown_calibration.json mapping scene_id -> {sx, sz, ox, oz, ...}.

Usage:
    conda run -n habitat python calibrate_topdown.py
"""

import trimesh
import math
import json
import os
from pathlib import Path

SCENE_DIR = Path(__file__).parent / "data/scene_datasets/hssd-hab/scenes-uncluttered"
STAGE_DIR = Path(__file__).parent / "data/scene_datasets/hssd-hab/stages"
OUTPUT_PATH = Path(__file__).parent / "topdown_calibration.json"

IMG_W, IMG_H = 1920, 1080
HFOV_DEG = 90.0
MARGIN = 1.10


def compute_calibration_from_glb(stage_path: str) -> dict:
    """Compute topdown projection parameters from a stage GLB file."""
    mesh = trimesh.load(stage_path, force='scene')
    bounds = mesh.bounds  # (2, 3): [min_xyz, max_xyz]

    bmin = bounds[0]
    bmax = bounds[1]

    cx = (bmin[0] + bmax[0]) / 2.0
    cz = (bmin[2] + bmax[2]) / 2.0
    floor_y = float(bmin[1])
    ext_x = bmax[0] - bmin[0]
    ext_z = bmax[2] - bmin[2]

    hfov_rad = math.radians(HFOV_DEG)
    vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * (IMG_H / IMG_W))

    h_x = (ext_x / 2.0) / math.tan(hfov_rad / 2.0)
    h_z = (ext_z / 2.0) / math.tan(vfov_rad / 2.0)
    height = max(h_x, h_z) * MARGIN

    # Focal lengths in pixels
    fx = IMG_W / (2.0 * math.tan(hfov_rad / 2.0))
    fy = IMG_H / (2.0 * math.tan(vfov_rad / 2.0))

    # world_to_pixel: u = sx * world_x + ox, v = sz * world_z + oz
    sx = fx / height
    sz = fy / height
    ox = IMG_W / 2.0 - sx * cx
    oz = IMG_H / 2.0 - sz * cz

    return {
        "sx": round(float(sx), 4),
        "sz": round(float(sz), 4),
        "ox": round(float(ox), 4),
        "oz": round(float(oz), 4),
        "height": round(float(height), 4),
        "cx": round(float(cx), 4),
        "cz": round(float(cz), 4),
        "floor_y": round(float(floor_y), 4),
        "ext_x": round(float(ext_x), 4),
        "ext_z": round(float(ext_z), 4),
    }


def extract_scene_id(filename: str) -> str:
    """Extract numeric scene ID from filename."""
    return filename.replace(".scene_instance.json", "")


def main():
    scene_files = sorted(SCENE_DIR.glob("*.scene_instance.json"))
    print(f"Found {len(scene_files)} scene files")

    calibration = {}
    for i, scene_file in enumerate(scene_files):
        scene_id = extract_scene_id(scene_file.name)
        # Scene ID may be compound (e.g. "103997424_171030444"), stage uses first part
        stage_candidates = [
            STAGE_DIR / f"{scene_id}.glb",
            STAGE_DIR / f"{scene_id.split('_')[0]}.glb",
        ]
        stage_path = None
        for candidate in stage_candidates:
            if candidate.exists():
                stage_path = candidate
                break

        if stage_path is None:
            print(f"[{i+1}/{len(scene_files)}] {scene_id}: SKIP (no stage GLB)")
            continue

        print(f"[{i+1}/{len(scene_files)}] {scene_id}...", end=" ", flush=True)
        try:
            result = compute_calibration_from_glb(str(stage_path))
            calibration[scene_id] = result
            print(f"OK (sx={result['sx']:.2f}, h={result['height']:.1f}m)")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    with open(OUTPUT_PATH, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nSaved calibration for {len(calibration)} scenes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
