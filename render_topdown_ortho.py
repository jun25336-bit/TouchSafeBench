#!/usr/bin/env python3
"""
Render orthographic top-down views for all 49 scenes in the dataset.

Uses the full Habitat-lab pipeline (RearrangeSim) to load scenes,
avoiding the ceiling occlusion issue when loading with bare habitat-sim.
Automatically hides the Spot robot and Humanoid character before rendering.

Output directory structure:
  topdown_ortho/
  ├── {scene_name}.png             Orthographic top-down RGB image
  ├── {scene_name}.json            Coordinate mapping metadata
  └── scenes_index.json            Index of all scenes

Coordinate mapping (world → pixel):
  px = (world_x - center_x) / pixel_scale + width / 2
  py = (world_z - center_z) / pixel_scale + height / 2

  where pixel_scale is the number of meters per pixel (meters/pixel).
  Output images are cropped to the scene extent with no excess black borders.

Usage:
  python render_topdown_ortho.py
  python render_topdown_ortho.py --resolution 2048
  python render_topdown_ortho.py --gpu 0
  python render_topdown_ortho.py --scenes 102344022 107734479
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# ============================================================
# Default configuration
# ============================================================

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = "data/versioned_data/hab3-episodes/val/social_rearrange_diverse.json.gz"
OUTPUT_DIR = os.path.join(WORK_DIR, "topdown_ortho")

DEFAULT_RESOLUTION = 2560
MARGIN = 1.10

ADDITIONAL_OBJECT_PATHS = [
    os.path.join(WORK_DIR, "data/hab3_bench_assets/"),
    os.path.join(WORK_DIR, "data/versioned_data/ycb/configs/"),
]


# ============================================================
# Dataset loading
# ============================================================

def load_unique_scenes(dataset_path: str, work_dir: str) -> Dict[str, str]:
    """Extract all unique scenes from the dataset. Returns {scene_id: scene_name}."""
    full_path = os.path.join(work_dir, dataset_path)
    with gzip.open(full_path, "rt") as f:
        data = json.load(f)

    scenes: Dict[str, str] = {}
    for ep in data.get("episodes", []):
        sid = ep["scene_id"]
        if sid not in scenes:
            name = Path(sid).stem.replace(".scene_instance", "")
            scenes[sid] = name
    return dict(sorted(scenes.items(), key=lambda x: x[1]))


def count_episodes(dataset_path: str, work_dir: str) -> int:
    full_path = os.path.join(work_dir, dataset_path)
    with gzip.open(full_path, "rt") as f:
        data = json.load(f)
    return len(data.get("episodes", []))


def create_filtered_dataset(
    dataset_path: str,
    work_dir: str,
    target_scene_names: Optional[set] = None,
) -> str:
    """Create a filtered dataset: keep only the first episode per scene, with target scenes first.

    Returns the path of the temporary file (relative to work_dir).
    """
    full_path = os.path.join(work_dir, dataset_path)
    with gzip.open(full_path, "rt") as f:
        data = json.load(f)

    seen_scenes: Dict[str, dict] = {}
    for ep in data.get("episodes", []):
        sid = ep["scene_id"]
        if sid not in seen_scenes:
            seen_scenes[sid] = ep

    episodes = list(seen_scenes.values())

    if target_scene_names:
        def sort_key(ep):
            name = Path(ep["scene_id"]).stem.replace(".scene_instance", "")
            return (0 if name in target_scene_names else 1, name)
        episodes.sort(key=sort_key)

    data["episodes"] = episodes

    out_dir = os.path.join(work_dir, "topdown_ortho")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "_filtered_dataset.json.gz")
    with gzip.open(out_path, "wt") as f:
        json.dump(data, f)

    print(f"  Filtered dataset: {len(episodes)} episodes (1 per scene)")
    return os.path.relpath(out_path, work_dir)


# ============================================================
# Habitat-lab environment configuration
# ============================================================

def build_habitat_config(
    resolution: int = DEFAULT_RESOLUTION,
    dataset_path: str = DATASET_PATH,
):
    """
    Build the full Habitat-lab configuration using habitat.config.default.get_config.
    Based on benchmark/multi_agent/hssd_spot_human_social_nav.yaml,
    including RearrangeSim, dual agents (Spot + Human), scene datasets, and full config.
    """
    from habitat.config.default import get_config
    from habitat.config.default_structured_configs import ThirdRGBSensorConfig
    from habitat.config.read_write import read_write

    cfg = get_config(
        "benchmark/multi_agent/hssd_spot_human_social_nav.yaml",
        overrides=[
            f"habitat.dataset.data_path={dataset_path}",
            "habitat.dataset.scenes_dir=data/scene_datasets/",
            "habitat.simulator.concur_render=False",
            "habitat.simulator.auto_sleep=False",
            "habitat.simulator.kinematic_mode=True",
            "habitat.simulator.ac_freq_ratio=4",
            "habitat.simulator.ctrl_freq=120",
            "habitat.simulator.agents.agent_0.joint_start_noise=0.0",
            "habitat.environment.max_episode_steps=10",
        ],
    )

    with read_write(cfg):
        cfg.habitat.simulator.agents.agent_0.sim_sensors[
            "third_rgb_sensor"
        ] = ThirdRGBSensorConfig(height=resolution, width=resolution)
        cfg.habitat.simulator.additional_object_paths = ADDITIONAL_OBJECT_PATHS

    return cfg


# ============================================================
# Agent hiding
# ============================================================

def hide_agents(sim) -> None:
    """Move all agents (Spot + Humanoid) to y=-1000 so they don't appear in the render."""
    import magnum as mn

    for agent_data in sim.agents_mgr._all_agent_data:
        agent_data.articulated_agent.base_pos = mn.Vector3(0.0, -1000.0, 0.0)


# ============================================================
# Orthographic projection rendering
# ============================================================

def _get_scene_bounds(sim):
    """Get scene bounds from the pathfinder or scene graph."""
    pf = sim.pathfinder
    if pf.is_loaded:
        b = pf.get_bounds()
        return {
            "cx": (b[0][0] + b[1][0]) / 2.0,
            "cz": (b[0][2] + b[1][2]) / 2.0,
            "floor_y": float(b[0][1]),
            "ceiling_y": float(b[1][1]),
            "extent_x": float(b[1][0] - b[0][0]),
            "extent_z": float(b[1][2] - b[0][2]),
        }
    bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    return {
        "cx": (bb.min[0] + bb.max[0]) / 2.0,
        "cz": (bb.min[2] + bb.max[2]) / 2.0,
        "floor_y": float(bb.min[1]),
        "ceiling_y": float(bb.max[1]),
        "extent_x": float(bb.max[0] - bb.min[0]),
        "extent_z": float(bb.max[2] - bb.min[2]),
    }


def _do_ortho_render(sim, sensor_key, resolution, cx, cz, floor_y,
                     camera_y, height, ortho_scale):
    """Perform a single orthographic render, returning an RGB image (H, W, 3)."""
    import magnum as mn

    sensor_obj = sim._sensors[sensor_key]._sensor_object
    old_transform = mn.Matrix4(sensor_obj.node.transformation)

    eye = mn.Vector3(cx, camera_y, cz)
    target = mn.Vector3(cx, floor_y, cz)
    cam_world = mn.Matrix4.look_at(eye, target, mn.Vector3(0, 0, -1))

    agent_T = sim._default_agent.scene_node.transformation
    sensor_obj.node.transformation = agent_T.inverted() @ cam_world

    render_camera = sensor_obj.render_camera
    spec = sensor_obj.specification()

    render_camera.set_orthographic_projection_matrix(
        resolution, resolution, 0.01, height + 5.0, ortho_scale,
    )

    sim_obs = sim.get_sensor_observations()

    render_camera.set_projection_matrix(
        resolution, resolution, spec.near, spec.far,
        getattr(spec, "hfov", mn.Deg(90.0)),
    )
    sensor_obj.node.transformation = old_transform

    img = sim_obs.get(sensor_key)
    if img is None:
        return None
    return img[:, :, :3]


def _do_perspective_probe(sim, sensor_key, cx, cz, floor_y,
                          extent_x, extent_z):
    """Render a single probe frame using the perspective camera.

    Camera height is automatically computed from the pathfinder bounds + FOV to cover the entire scene.
    Returns (img, img_w, img_h, hfov_rad, actual_cam_height_above_floor).
    """
    import magnum as mn
    import math

    sensor_obj = sim._sensors[sensor_key]._sensor_object
    old_transform = mn.Matrix4(sensor_obj.node.transformation)

    spec = sensor_obj.specification()
    img_h, img_w = spec.resolution[0], spec.resolution[1]
    hfov_rad = float(math.radians(float(getattr(spec, "hfov", 90.0))))
    vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * img_h / img_w)

    h_for_x = (extent_x * 1.15) / (2.0 * math.tan(hfov_rad / 2.0))
    h_for_z = (extent_z * 1.15) / (2.0 * math.tan(vfov_rad / 2.0))
    probe_height = max(h_for_x, h_for_z)
    camera_y = floor_y + probe_height

    eye = mn.Vector3(cx, camera_y, cz)
    target = mn.Vector3(cx, floor_y, cz)
    cam_world = mn.Matrix4.look_at(eye, target, mn.Vector3(0, 0, -1))

    agent_T = sim._default_agent.scene_node.transformation
    sensor_obj.node.transformation = agent_T.inverted() @ cam_world

    sim_obs = sim.get_sensor_observations()
    sensor_obj.node.transformation = old_transform

    img = sim_obs.get(sensor_key)
    if img is None:
        return None

    return img[:, :, :3], img_w, img_h, hfov_rad, probe_height


def _detect_building_from_perspective(img, cx, cz, cam_height,
                                      img_w, img_h, hfov_rad,
                                      debug_path=None):
    """Detect the building region from a perspective probe image and back-project to world coordinates.

    Uses image gradient (edge density) instead of brightness to avoid interference from uniform ground/sky.
    """
    import math

    if debug_path:
        from PIL import Image
        Image.fromarray(img).save(debug_path)

    gray = np.mean(img.astype(np.float32), axis=2)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(np.diff(gray, axis=1))
    gy[1:, :] = np.abs(np.diff(gray, axis=0))
    grad = np.maximum(gx, gy)

    EDGE_THRESH = 8
    MIN_EDGE_FRAC = 0.015

    strong = grad > EDGE_THRESH
    row_frac = strong.mean(axis=1)
    col_frac = strong.mean(axis=0)

    content_rows = np.where(row_frac > MIN_EDGE_FRAC)[0]
    content_cols = np.where(col_frac > MIN_EDGE_FRAC)[0]

    if len(content_rows) == 0 or len(content_cols) == 0:
        return None

    r0, r1 = content_rows[0], content_rows[-1]
    c0, c1 = content_cols[0], content_cols[-1]

    focal_x = (img_w / 2.0) / math.tan(hfov_rad / 2.0)
    vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * img_h / img_w)
    focal_z = (img_h / 2.0) / math.tan(vfov_rad / 2.0)

    x_min = cx + (c0 - img_w / 2.0) * cam_height / focal_x
    x_max = cx + (c1 - img_w / 2.0) * cam_height / focal_x
    z_min = cz + (r0 - img_h / 2.0) * cam_height / focal_z
    z_max = cz + (r1 - img_h / 2.0) * cam_height / focal_z

    return x_min, x_max, z_min, z_max


def render_ortho_topdown(
    sim,
    sensor_key: str,
    resolution: int,
    margin: float = MARGIN,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Render an orthographic top-down view using an existing RGB sensor in the sim.

    Two-pass rendering:
      1. Perspective camera probes the actual building extent (more accurate when the building fills the frame)
      2. Orthographic camera renders at full resolution based on the detected building bounds
    """
    import math

    sb = _get_scene_bounds(sim)
    cx, cz = sb["cx"], sb["cz"]
    floor_y, ceiling_y = sb["floor_y"], sb["ceiling_y"]

    height = (ceiling_y - floor_y) + 5.0
    camera_y = floor_y + height

    # === Pass 1: Perspective camera probes the building region ===
    probe = _do_perspective_probe(
        sim, sensor_key, cx, cz, floor_y,
        sb["extent_x"], sb["extent_z"],
    )

    bld_cx, bld_cz = cx, cz
    bld_w = sb["extent_x"] * margin
    bld_h = sb["extent_z"] * margin

    if probe is not None:
        probe_img, img_w, img_h, hfov_rad, probe_h = probe
        import os
        debug_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "topdown_ortho"
        )
        debug_path = os.path.join(debug_dir, "_probe_debug.png")
        building = _detect_building_from_perspective(
            probe_img, cx, cz, probe_h,
            img_w, img_h, hfov_rad,
            debug_path=debug_path,
        )
        if building is not None:
            bx_min, bx_max, bz_min, bz_max = building
            bld_cx = (bx_min + bx_max) / 2.0
            bld_cz = (bz_min + bz_max) / 2.0
            bld_w = (bx_max - bx_min) * margin
            bld_h = (bz_max - bz_min) * margin

    # === Pass 2a: Orthographic render (loose bounds) to detect actual building pixel extent ===
    ortho_scale_1 = 1.0 / max(bld_w, bld_h)
    pixel_scale_1 = max(bld_w, bld_h) / resolution

    img1 = _do_ortho_render(
        sim, sensor_key, resolution,
        bld_cx, bld_cz, floor_y, camera_y, height, ortho_scale_1,
    )

    if img1 is None:
        return None, None

    gray = np.mean(img1.astype(np.float32), axis=2)
    non_black = gray > 2
    rows_mask = np.any(non_black, axis=1)
    cols_mask = np.any(non_black, axis=0)

    if not (np.any(rows_mask) and np.any(cols_mask)):
        return None, None

    r_idx = np.where(rows_mask)[0]
    c_idx = np.where(cols_mask)[0]
    r0, r1 = int(r_idx[0]), int(r_idx[-1])
    c0, c1 = int(c_idx[0]), int(c_idx[-1])

    actual_cx = bld_cx + ((c0 + c1) / 2.0 - resolution / 2.0) * pixel_scale_1
    actual_cz = bld_cz + ((r0 + r1) / 2.0 - resolution / 2.0) * pixel_scale_1
    actual_w = (c1 - c0 + 1) * pixel_scale_1 * 1.02
    actual_h = (r1 - r0 + 1) * pixel_scale_1 * 1.02

    # === Pass 2b: Orthographic render (tight bounds), building fills full resolution ===
    ortho_scale_2 = 1.0 / max(actual_w, actual_h)
    pixel_scale = max(actual_w, actual_h) / resolution

    img = _do_ortho_render(
        sim, sensor_key, resolution,
        actual_cx, actual_cz, floor_y, camera_y, height, ortho_scale_2,
    )

    if img is None:
        return None, None

    gray = np.mean(img.astype(np.float32), axis=2)
    non_black = gray > 2
    rows_mask = np.any(non_black, axis=1)
    cols_mask = np.any(non_black, axis=0)
    if np.any(rows_mask) and np.any(cols_mask):
        r_idx = np.where(rows_mask)[0]
        c_idx = np.where(cols_mask)[0]
        r0, r1 = int(r_idx[0]), int(r_idx[-1])
        c0, c1 = int(c_idx[0]), int(c_idx[-1])
        pad = max(int(max(r1 - r0, c1 - c0) * 0.01), 2)
        r0 = max(0, r0 - pad)
        r1 = min(resolution - 1, r1 + pad)
        c0 = max(0, c0 - pad)
        c1 = min(resolution - 1, c1 + pad)
        actual_cx += ((c0 + c1) / 2.0 - resolution / 2.0) * pixel_scale
        actual_cz += ((r0 + r1) / 2.0 - resolution / 2.0) * pixel_scale
        img = img[r0:r1 + 1, c0:c1 + 1]
    out_h, out_w = img.shape[:2]

    meta = {
        "width": out_w,
        "height": out_h,
        "pixel_scale": pixel_scale,
        "center_x": actual_cx,
        "center_z": actual_cz,
        "floor_y": floor_y,
        "ceiling_y": ceiling_y,
        "extent_x": actual_w / 1.02,
        "extent_z": actual_h / 1.02,
        "camera_height": camera_y,
        "margin": margin,
        "world_to_pixel": (
            "px = (world_x - center_x) / pixel_scale + width / 2; "
            "py = (world_z - center_z) / pixel_scale + height / 2"
        ),
        "pixel_to_world": (
            "world_x = (px - width / 2) * pixel_scale + center_x; "
            "world_z = (py - height / 2) * pixel_scale + center_z"
        ),
    }

    return img, meta


# ============================================================
# Coordinate conversion (fully compatible with bev_ortho.py)
# ============================================================

def world_to_pixel(
    world_x: float, world_z: float, meta: Dict[str, Any],
) -> Tuple[float, float]:
    """Habitat world coordinates (x, z) -> image pixel coordinates (px, py)."""
    cx, cz = meta["center_x"], meta["center_z"]
    ps = meta["pixel_scale"]
    w, h = meta["width"], meta["height"]
    px = (world_x - cx) / ps + w / 2.0
    py = (world_z - cz) / ps + h / 2.0
    return px, py


def pixel_to_world(
    px: float, py: float, meta: Dict[str, Any],
) -> Tuple[float, float]:
    """Image pixel coordinates (px, py) -> Habitat world coordinates (x, z)."""
    cx, cz = meta["center_x"], meta["center_z"]
    ps = meta["pixel_scale"]
    w, h = meta["width"], meta["height"]
    world_x = (px - w / 2.0) * ps + cx
    world_z = (py - h / 2.0) * ps + cz
    return world_x, world_z


def meters_to_pixels(meters: float, meta: Dict[str, Any]) -> float:
    """Convert a real-world distance (meters) to pixels."""
    return meters / meta["pixel_scale"]


# ============================================================
# Find the third_rgb sensor
# ============================================================

def find_rgb_sensor_key(sim) -> Optional[str]:
    """Find an available third_rgb sensor in sim._sensors."""
    for k in sim._sensors:
        if "third_rgb" in k:
            return k
    for k in sim._sensors:
        if "rgb" in k:
            return k
    return None


# ============================================================
# Main workflow
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Render orthographic top-down views for all scenes in the dataset (Habitat-lab pipeline)",
    )
    parser.add_argument(
        "--resolution", type=int, default=DEFAULT_RESOLUTION,
        help=f"Square image resolution (default: {DEFAULT_RESOLUTION})",
    )
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--scenes", type=str, nargs="*", default=None,
        help="Only render specified scenes (prefix match on scene_name)",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    os.chdir(WORK_DIR)
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Orthographic Top-Down Renderer (Habitat-lab pipeline)")
    print("=" * 60)

    all_scenes = load_unique_scenes(DATASET_PATH, WORK_DIR)
    target_scenes = all_scenes
    if args.scenes:
        target_scenes = {
            k: v for k, v in all_scenes.items()
            if any(v.startswith(prefix) for prefix in args.scenes)
        }

    already_done = set()
    for scene_name in target_scenes.values():
        img_path = os.path.join(output_dir, f"{scene_name}.png")
        meta_path = os.path.join(output_dir, f"{scene_name}.json")
        if os.path.exists(img_path) and os.path.exists(meta_path):
            already_done.add(scene_name)

    to_render = {k: v for k, v in target_scenes.items() if v not in already_done}

    print(f"  Total scenes:   {len(all_scenes)}")
    print(f"  Target scenes:  {len(target_scenes)}")
    print(f"  Already cached: {len(already_done)}")
    print(f"  To render:      {len(to_render)}")
    print(f"  Resolution:     {args.resolution} x {args.resolution}")
    print(f"  Output:         {output_dir}")
    print()

    if not to_render:
        print("All scenes already rendered, nothing to do.")
        _write_index(output_dir, target_scenes, already_done)
        return

    import habitat

    target_names_set = set(to_render.values())
    filtered_path = create_filtered_dataset(
        DATASET_PATH, WORK_DIR, target_names_set,
    )
    total_episodes = count_episodes(filtered_path, WORK_DIR)
    print(f"  Filtered episodes: {total_episodes}")
    print("  Building config...")

    cfg = build_habitat_config(
        resolution=args.resolution,
        dataset_path=filtered_path,
    )

    print("  Creating environment (loading dataset)...")
    env = habitat.Env(config=cfg.habitat)
    print("  Environment created")
    print()

    rendered_scenes: Dict[str, Dict[str, Any]] = {}
    target_names = set(to_render.values())
    failed: List[str] = []

    for reset_i in range(total_episodes):
        if not target_names:
            break

        try:
            env.reset()
        except Exception as e:
            print(f"  reset #{reset_i} failed: {e}")
            continue

        episode = env.current_episode
        scene_id = episode.scene_id
        scene_name = Path(scene_id).stem.replace(".scene_instance", "")

        if scene_name not in target_names:
            continue

        tag = f"[{len(rendered_scenes) + len(already_done) + 1}/{len(target_scenes)}]"
        print(f"{tag} {scene_name} ...", end="", flush=True)
        t0 = time.time()

        try:
            sim = env._sim

            hide_agents(sim)

            sensor_key = find_rgb_sensor_key(sim)
            if sensor_key is None:
                print(f"  RGB sensor not found")
                failed.append(scene_name)
                target_names.discard(scene_name)
                continue

            img, meta = render_ortho_topdown(
                sim, sensor_key, args.resolution,
            )

            if img is None:
                print(f"  Render returned empty")
                failed.append(scene_name)
                target_names.discard(scene_name)
                continue

            meta["scene_id"] = scene_id
            meta["scene_name"] = scene_name

            img_path = os.path.join(output_dir, f"{scene_name}.png")
            meta_path = os.path.join(output_dir, f"{scene_name}.json")

            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            elapsed = time.time() - t0
            print(
                f"  {meta['width']}x{meta['height']}px  "
                f"({meta['pixel_scale']:.4f} m/px)  "
                f"scene {meta['extent_x']:.1f}x{meta['extent_z']:.1f}m  "
                f"[{elapsed:.1f}s]"
            )

            rendered_scenes[scene_name] = meta
            target_names.discard(scene_name)

        except Exception as e:
            print(f"  Error: {e}")
            failed.append(scene_name)
            target_names.discard(scene_name)

    env.close()

    all_done = already_done | set(rendered_scenes.keys())
    _write_index(output_dir, target_scenes, all_done)

    print()
    print("=" * 60)
    print(f"Done: {len(all_done)}/{len(target_scenes)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    if target_names:
        print(f"Not covered (no matching episode in dataset): {', '.join(sorted(target_names))}")
    print(f"Output: {output_dir}")
    print("=" * 60)


def _write_index(
    output_dir: str,
    target_scenes: Dict[str, str],
    done_names: set,
):
    """Write scenes_index.json."""
    index = []
    for scene_id, scene_name in target_scenes.items():
        if scene_name not in done_names:
            continue
        meta_path = os.path.join(output_dir, f"{scene_name}.json")
        entry = {
            "scene_name": scene_name,
            "scene_id": scene_id,
            "image": f"{scene_name}.png",
            "meta": f"{scene_name}.json",
        }
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            entry["ortho_scale"] = meta.get("ortho_scale")
        index.append(entry)

    with open(os.path.join(output_dir, "scenes_index.json"), "w") as f:
        json.dump({"total": len(index), "scenes": index}, f, indent=2)


if __name__ == "__main__":
    main()
