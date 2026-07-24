# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info


class HabitatEvaluator(Evaluator):
    """
    Evaluator for Habitat environments.
    """

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        _actual_num_envs = envs.num_envs
        test_recurrent_hidden_states = torch.zeros(
            (
                _actual_num_envs,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            _actual_num_envs,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            _actual_num_envs,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        # HABITAT_VIDEO_SENSOR_FILTER controls which sensors appear in video.
        # Comma-separated patterns, e.g.:
        #   "rgb"              -> keep all sensors containing "rgb" (default)
        #   "depth"            -> keep all sensors containing "depth"
        #   "arm_rgb,head_rgb" -> keep only arm_rgb and head_rgb
        #   "third_depth"      -> keep only third_depth
        #   "all"              -> keep all 2D+ visual sensors
        _sensor_filter = os.environ.get("HABITAT_VIDEO_SENSOR_FILTER", "rgb")
        _frame_skip = max(1, int(os.environ.get("HABITAT_FRAME_SKIP", "1")))
        _step_counters: List[int] = [
            0 for _ in range(_actual_num_envs)
        ]
        if _frame_skip > 1:
            logger.info(f"Frame sampling interval: 1 frame per {_frame_skip} steps")

        def _write_json_file(path, data):
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        def _should_capture(env_i):
            """Whether the current step should capture a frame."""
            return _step_counters[env_i] % _frame_skip == 0

        _filter_patterns = (
            None if _sensor_filter == "all"
            else [p.strip() for p in _sensor_filter.split(",")]
        )

        def _filter_obs(obs_dict):
            if _filter_patterns is None:
                return obs_dict
            return {
                k: v for k, v in obs_dict.items()
                if any(p in k for p in _filter_patterns)
            }

        # ── Trajectory recording ──
        # Enabled via HABITAT_SAVE_TRAJECTORY=1.
        # Records all agent positions + collision info per step.
        _save_trajectory = os.environ.get("HABITAT_SAVE_TRAJECTORY", "0") == "1"
        if _save_trajectory:
            trajectory_buffers: List[List[dict]] = [
                [] for _ in range(_actual_num_envs)
            ]
            # For computing per-step deltas of robot_collisions (raw metric is cumulative)
            _prev_robot_colls: List[dict] = [
                {"robot_obj_colls": 0, "robot_scene_colls": 0}
                for _ in range(_actual_num_envs)
            ]
            logger.info("Trajectory recording enabled (HABITAT_SAVE_TRAJECTORY=1)")

        # ── Camera parameter recording ──
        # Enabled via HABITAT_SAVE_CAMERA_PARAMS=1.
        # Only records the 4 cameras matching the depth directory:
        #   arm_depth_render  → arm
        #   head_depth_render → head
        #   third_depth       → third_robot / third_human
        _save_camera_params = os.environ.get("HABITAT_SAVE_CAMERA_PARAMS", "0") == "1"
        _CAMERA_PARAM_PATTERNS = ("arm_depth_render", "head_depth_render", "third_depth")
        if _save_camera_params:
            camera_params_buffers: List[List[dict]] = [
                [] for _ in range(_actual_num_envs)
            ]
            camera_intrinsics_cache: Dict[str, dict] = {}
            logger.info("Camera parameter recording enabled (HABITAT_SAVE_CAMERA_PARAMS=1)")

        # ── Per-sensor RGB video mode ──
        # When HABITAT_VIDEO_SPLIT_SENSORS=1, each sensor produces a separate MP4
        # instead of the default tiled video, for per-view VLM input.
        _split_rgb = os.environ.get("HABITAT_VIDEO_SPLIT_SENSORS", "0") == "1"
        if _split_rgb:
            from habitat.utils.visualizations.utils import images_to_video
            _rgb_split: List[Dict[str, List[np.ndarray]]] = [
                {} for _ in range(_actual_num_envs)
            ]
            _rgb_video_dir = config.habitat_baselines.video_dir
            os.makedirs(_rgb_video_dir, exist_ok=True)
            logger.info("Per-sensor RGB video mode enabled (HABITAT_VIDEO_SPLIT_SENSORS=1)")

        # ── RGB full-frame saving ──
        # When HABITAT_SAVE_RGB_FRAMES=1, saves every sensor's RGB frame as JPEG
        # every step (unaffected by frame_skip), for VLM input.
        _save_rgb_frames = os.environ.get("HABITAT_SAVE_RGB_FRAMES", "0") == "1"
        if _save_rgb_frames:
            import cv2 as _cv2_rgb
            _rgb_jpeg_quality = int(os.environ.get("HABITAT_RGB_JPEG_QUALITY", "85"))
            _rgb_frame_counters: List[Dict[str, int]] = [
                {} for _ in range(_actual_num_envs)
            ]
            _rgb_frames_dir = config.habitat_baselines.video_dir
            os.makedirs(_rgb_frames_dir, exist_ok=True)
            logger.info(
                f"RGB full-frame saving enabled (JPEG quality={_rgb_jpeg_quality})"
            )

            _rgb_patterns = (
                None if _sensor_filter == "all"
                else [p.strip() for p in _sensor_filter.split(",")]
            )

            def _save_rgb_obs(obs_dict, env_i, episode_id):
                """Save sensor_filter-matched RGB observations as JPEG files."""
                for obs_key, obs_val in obs_dict.items():
                    if _rgb_patterns is not None and not any(p in obs_key for p in _rgb_patterns):
                        continue
                    if "depth" in obs_key:
                        continue
                    obs_np = np.asarray(obs_val)
                    if obs_np.ndim < 3:
                        continue
                    if obs_np.dtype != np.uint8:
                        obs_np = (obs_np * 255.0).clip(0, 255).astype(np.uint8)
                    if obs_np.shape[2] == 1:
                        obs_np = np.concatenate([obs_np] * 3, axis=2)
                    bgr = obs_np[:, :, ::-1].copy()
                    sensor_dir = os.path.join(
                        _rgb_frames_dir,
                        f"rgb_all_ep{episode_id}",
                        obs_key,
                    )
                    os.makedirs(sensor_dir, exist_ok=True)
                    frame_idx = _rgb_frame_counters[env_i].get(obs_key, 0)
                    fpath = os.path.join(sensor_dir, f"{frame_idx:06d}.jpg")
                    params = [int(_cv2_rgb.IMWRITE_JPEG_QUALITY), _rgb_jpeg_quality]
                    _cv2_rgb.imwrite(fpath, bgr, params)
                    _rgb_frame_counters[env_i][obs_key] = frame_idx + 1

        # ── Top-down view ──
        # When HABITAT_TOPDOWN_VIEW=1, renders a fixed-position overhead RGB frame each step.
        _topdown_view = os.environ.get("HABITAT_TOPDOWN_VIEW", "0") == "1"
        if _topdown_view:
            _topdown_height_env = os.environ.get("HABITAT_TOPDOWN_HEIGHT", "auto")
            if _topdown_height_env.lower() == "auto":
                _topdown_height = 0.0  # <= 0 triggers auto-compute in render_topdown
            else:
                _topdown_height = float(_topdown_height_env)
            _topdown_procs: List[Optional[subprocess.Popen]] = [
                None for _ in range(_actual_num_envs)
            ]
            _topdown_dir = config.habitat_baselines.video_dir
            _topdown_fps = config.habitat_baselines.video_fps
            os.makedirs(_topdown_dir, exist_ok=True)
            _height_desc = "auto" if _topdown_height <= 0 else f"{_topdown_height}m"

            def _topdown_feed(env_i, frame, episode_id):
                """Pipe one frame to ffmpeg; auto-starts the process on the first frame. Zero memory accumulation."""
                if _topdown_procs[env_i] is None:
                    h, w = frame.shape[:2]
                    out_dir = os.path.join(_topdown_dir, f"topdown_ep{episode_id}")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, "topdown.mp4")
                    _topdown_procs[env_i] = subprocess.Popen(
                        [
                            "ffmpeg", "-y", "-loglevel", "error",
                            "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{w}x{h}", "-r", str(_topdown_fps),
                            "-i", "-",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            out_path,
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                _topdown_procs[env_i].stdin.write(frame.tobytes())

            def _topdown_finalize(proc):
                """Close the pipe and wait for ffmpeg to finish encoding."""
                proc.stdin.close()
                proc.wait()

            logger.info(
                f"Top-down view enabled (height={_height_desc}, streaming encoding)"
            )

        # ── Depth PNG mode ──
        # HABITAT_DEPTH_FORMAT=png8: save depth as 8-bit PNG (normalized 0-255).
        # HABITAT_DEPTH_FORMAT=png16: save depth as 16-bit PNG (millimeters).
        _depth_format = os.environ.get("HABITAT_DEPTH_FORMAT", "mp4")
        _depth_png_mode = _depth_format in ("png8", "png16")
        if _depth_png_mode:
            import cv2 as _cv2
            _depth_max = float(os.environ.get("HABITAT_DEPTH_MAX", "10.0"))
            _depth_8bit = _depth_format == "png8"
            _depth_scale = 255 if _depth_8bit else 1000
            _depth_frame_counters: List[Dict[str, int]] = [
                {} for _ in range(_actual_num_envs)
            ]
            _depth_video_dir = config.habitat_baselines.video_dir
            os.makedirs(_depth_video_dir, exist_ok=True)
            logger.info(
                f"Depth PNG mode enabled (format={'8-bit' if _depth_8bit else '16-bit'}, "
                f"scale={_depth_scale}, max_depth={_depth_max}m)"
            )

            _depth_patterns = (
                None if _sensor_filter == "all"
                else [p.strip() for p in _sensor_filter.split(",")]
            )

            def _save_depth_obs(obs_dict, env_i, episode_id):
                """Save sensor_filter-matched depth data from obs_dict as PNG files."""
                for obs_key, obs_val in obs_dict.items():
                    if _depth_patterns is not None and not any(p in obs_key for p in _depth_patterns):
                        continue
                    if "depth" not in obs_key:
                        continue
                    depth_np = np.asarray(obs_val).squeeze()
                    if _depth_8bit:
                        depth_img = (
                            (depth_np * _depth_scale)
                            .clip(0, 255)
                            .astype(np.uint8)
                        )
                    else:
                        depth_img = (
                            (depth_np * _depth_max * _depth_scale)
                            .clip(0, 65535)
                            .astype(np.uint16)
                        )
                    sensor_dir = os.path.join(
                        _depth_video_dir,
                        f"depth_png_ep{episode_id}",
                        obs_key,
                    )
                    os.makedirs(sensor_dir, exist_ok=True)
                    frame_idx = _depth_frame_counters[env_i].get(obs_key, 0)
                    fpath = os.path.join(sensor_dir, f"{frame_idx:06d}.png")
                    _cv2.imwrite(fpath, depth_img)
                    _depth_frame_counters[env_i][obs_key] = frame_idx + 1

        _skip_main_video = _depth_png_mode or _save_rgb_frames or _split_rgb
        if len(config.habitat_baselines.eval.video_option) > 0 and not _skip_main_video:
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        _filter_obs({k: v[env_idx] for k, v in batch.items()}), {}
                    )
                ]
                for env_idx in range(_actual_num_envs)
            ]
        else:
            rgb_frames = [[] for _ in range(_actual_num_envs)]

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        # ── Save first frame: depth PNG + RGB full frames ──
        if _depth_png_mode or _save_rgb_frames:
            _init_ep_info = envs.current_episodes()
            for env_idx in range(envs.num_envs):
                if _depth_png_mode:
                    _save_depth_obs(
                        observations[env_idx],
                        env_idx,
                        _init_ep_info[env_idx].episode_id,
                    )
                if _save_rgb_frames:
                    _save_rgb_obs(
                        observations[env_idx],
                        env_idx,
                        _init_ep_info[env_idx].episode_id,
                    )

        # ── Per-sensor RGB: save first frame (RGB image sensors only) ──
        if _split_rgb:
            for env_idx in range(_actual_num_envs):
                filtered = _filter_obs(
                    {k: v[env_idx] for k, v in batch.items()}
                )
                for sk, sv in filtered.items():
                    if "depth" in sk:
                        continue
                    sv_np = sv.cpu().numpy() if hasattr(sv, 'cpu') else np.asarray(sv)
                    if sv_np.ndim < 3:
                        continue
                    frame = observations_to_image({sk: sv}, {})
                    _rgb_split[env_idx].setdefault(sk, []).append(frame)

        # ── Top-down view: save first frame ──
        if _topdown_view:
            _td_init_eps = envs.current_episodes()
            topdown_imgs = envs.call(
                ["render_topdown"] * envs.num_envs,
                [{"height": _topdown_height}] * envs.num_envs,
            )
            for env_idx in range(_actual_num_envs):
                if topdown_imgs[env_idx] is not None:
                    _topdown_feed(env_idx, topdown_imgs[env_idx], _td_init_eps[env_idx].episode_id)

        # ── Camera params: save first frame (t=0) ──
        # camera_params are skipped in the main loop when done=True (env already
        # reset to next episode), so the initial frame must be recorded here
        # explicitly; otherwise 1-step episodes would have empty camera_params.
        if _save_camera_params:
            _init_cam_eps = envs.current_episodes()
            _init_cam_all = envs.call(
                ["get_camera_params"] * envs.num_envs
            )
            for env_idx in range(_actual_num_envs):
                cam_data = _init_cam_all[env_idx]
                frame_extrinsics = {}
                for sensor_uuid, params in cam_data.items():
                    if not any(p in sensor_uuid for p in _CAMERA_PARAM_PATTERNS):
                        continue
                    if sensor_uuid not in camera_intrinsics_cache:
                        camera_intrinsics_cache[sensor_uuid] = params["intrinsics"]
                    frame_extrinsics[sensor_uuid] = params["extrinsics"]
                camera_params_buffers[env_idx].append(frame_extrinsics)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        agent.eval()
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    **space_lengths,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            for env_i in range(len(dones)):
                _step_counters[env_i] += 1

            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            # ── Trajectory recording: per-step agent positions + collision info ──
            if _save_trajectory:
                n_agents = len(config.habitat.simulator.agents)
                all_positions = envs.call(
                    ["get_agent_positions"] * envs.num_envs
                )
                for env_i in range(len(dones)):
                    step_record = {"agents": []}
                    positions = all_positions[env_i]

                    if dones[env_i] and len(trajectory_buffers[env_i]) > 0:
                        # env already reset; positions are from the next
                        # episode. Reuse the previous step's positions so
                        # that collision info (which IS valid) is preserved.
                        prev_step = trajectory_buffers[env_i][-1]
                        for a in prev_step.get("agents", []):
                            step_record["agents"].append({
                                "agent": a["agent"],
                                "position": a["position"][:],
                            })
                    else:
                        for agent_idx in range(n_agents):
                            agent_name = config.habitat.simulator.agents_order[agent_idx]
                            if agent_idx < len(positions):
                                step_record["agents"].append({
                                    "agent": agent_name,
                                    "position": positions[agent_idx],
                                })

                    info_i = infos[env_i]
                    did_collide_human = bool(info_i.get("did_collide", False))
                    rc = info_i.get("robot_collisions", {})
                    cur_obj = int(rc.get("robot_obj_colls", 0))
                    cur_scene = int(rc.get("robot_scene_colls", 0))
                    prev = _prev_robot_colls[env_i]
                    step_record["collisions"] = {
                        "did_collide_human": did_collide_human,
                        "robot_obj_colls": cur_obj - prev["robot_obj_colls"],
                        "robot_scene_colls": cur_scene - prev["robot_scene_colls"],
                    }
                    _prev_robot_colls[env_i] = {
                        "robot_obj_colls": cur_obj,
                        "robot_scene_colls": cur_scene,
                    }

                    trajectory_buffers[env_i].append(step_record)

            # ── Camera parameter recording (every step, aligned with depth PNG frame indices) ──
            if _save_camera_params:
                all_cam_params = envs.call(
                    ["get_camera_params"] * envs.num_envs
                )
                for env_i in range(len(dones)):
                    if dones[env_i]:
                        continue
                    cam_data = all_cam_params[env_i]
                    frame_extrinsics = {}
                    for sensor_uuid, params in cam_data.items():
                        if not any(p in sensor_uuid for p in _CAMERA_PARAM_PATTERNS):
                            continue
                        if sensor_uuid not in camera_intrinsics_cache:
                            camera_intrinsics_cache[sensor_uuid] = params["intrinsics"]
                        frame_extrinsics[sensor_uuid] = params["extrinsics"]
                    camera_params_buffers[env_i].append(frame_extrinsics)

            # ── Depth PNG saving (every step) ──
            if _depth_png_mode:
                for env_i in range(len(dones)):
                    if not dones[env_i]:
                        _save_depth_obs(
                            observations[env_i],
                            env_i,
                            current_episodes_info[env_i].episode_id,
                        )

            # ── RGB full-frame saving (every step, unaffected by frame_skip) ──
            if _save_rgb_frames:
                for env_i in range(len(dones)):
                    if not dones[env_i]:
                        _save_rgb_obs(
                            observations[env_i],
                            env_i,
                            current_episodes_info[env_i].episode_id,
                        )

            # ── Top-down view (every step, subject to frame_skip, streaming encoding) ──
            if _topdown_view:
                need_topdown = any(
                    not dones[ei] and _should_capture(ei)
                    for ei in range(len(dones))
                )
                if need_topdown:
                    topdown_imgs = envs.call(
                        ["render_topdown"] * envs.num_envs,
                        [{"height": _topdown_height}] * envs.num_envs,
                    )
                    for env_i in range(len(dones)):
                        if not dones[env_i] and _should_capture(env_i):
                            if topdown_imgs[env_i] is not None:
                                _topdown_feed(
                                    env_i, topdown_imgs[env_i],
                                    current_episodes_info[env_i].episode_id,
                                )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(config.habitat_baselines.eval.video_option) > 0 and not _skip_main_video:
                    is_done = not not_done_masks[i].any().item()
                    if _should_capture(i) or is_done:
                        frame = observations_to_image(
                            _filter_obs({k: v[i] for k, v in batch.items()}), disp_info
                        )
                        if is_done:
                            final_frame = observations_to_image(
                                _filter_obs({k: v[i] * 0.0 for k, v in batch.items()}),
                                disp_info,
                            )
                            rgb_frames[i].append(final_frame)
                            rgb_frames[i].append(frame)
                        else:
                            rgb_frames[i].append(frame)

                # ── Per-sensor RGB: collect frame each step (RGB image sensors only) ──
                if _split_rgb:
                    is_done = not not_done_masks[i].any().item()
                    if _should_capture(i) or is_done:
                        filtered = _filter_obs(
                            {k: v[i] for k, v in batch.items()}
                        )
                        for sk, sv in filtered.items():
                            if "depth" in sk:
                                continue
                            sv_np = sv.cpu().numpy() if hasattr(sv, 'cpu') else np.asarray(sv)
                            if sv_np.ndim < 3:
                                continue
                            frame_s = observations_to_image({sk: sv}, {})
                            if is_done:
                                black_s = observations_to_image(
                                    {sk: sv * 0.0}, {}
                                )
                                _rgb_split[i].setdefault(sk, []).append(black_s)
                                _rgb_split[i][sk].append(frame_s)
                            else:
                                _rgb_split[i].setdefault(sk, []).append(frame_s)

                # episode ended
                if not not_done_masks[i].any().item():
                    _step_counters[i] = 0
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    # ==========================================
                    # Video saving logic (based on did_collide)
                    # ==========================================
                    if len(config.habitat_baselines.eval.video_option) > 0:
                        # 1. Check whether a collision occurred
                        is_collision = episode_stats.get("did_collide", 0) > 0
                        
                        # 2. Build filename ID
                        base_id = f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}"
                        
                        if is_collision:
                            custom_episode_id = f"COLLISION_{base_id}"
                        else:
                            custom_episode_id = base_id
                        
                        # 3. Generate tiled video (skipped in split/depth_png modes)
                        if not _split_rgb and not _depth_png_mode:
                            generate_video(
                                video_option=config.habitat_baselines.eval.video_option,
                                video_dir=config.habitat_baselines.video_dir,
                                images=rgb_frames[i][:-1],
                                episode_id=custom_episode_id,
                                checkpoint_idx=checkpoint_index,
                                metrics=extract_scalars_from_info(disp_info),
                                fps=config.habitat_baselines.video_fps,
                                tb_writer=writer,
                                keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                            )
                        
                        rgb_frames[i] = rgb_frames[i][-1:]

                    # ── Per-sensor RGB video saving ──
                    if _split_rgb and _rgb_split[i]:
                        ep_id = current_episodes_info[i].episode_id
                        ep_rgb_dir = os.path.join(
                            _rgb_video_dir, f"rgb_split_ep{ep_id}"
                        )
                        os.makedirs(ep_rgb_dir, exist_ok=True)
                        for sk, frames in _rgb_split[i].items():
                            if len(frames) > 1:
                                images_to_video(
                                    frames[:-1],
                                    ep_rgb_dir,
                                    sk,
                                    fps=config.habitat_baselines.video_fps,
                                    verbose=False,
                                )
                            _rgb_split[i][sk] = frames[-1:]

                    # ── Top-down video saving (close ffmpeg pipe) ──
                    if _topdown_view and _topdown_procs[i] is not None:
                        _topdown_finalize(_topdown_procs[i])
                        _topdown_procs[i] = None

                    # ── Trajectory saving ──
                    if _save_trajectory and len(trajectory_buffers[i]) > 0:
                        traj_steps = trajectory_buffers[i]

                        # Build collision summary
                        human_coll_steps = []
                        obj_coll_steps = []
                        scene_coll_steps = []
                        total_obj = 0
                        total_scene = 0
                        for si, sr in enumerate(traj_steps):
                            c = sr.get("collisions", {})
                            if c.get("did_collide_human"):
                                human_coll_steps.append(si)
                            oc = c.get("robot_obj_colls", 0)
                            sc = c.get("robot_scene_colls", 0)
                            if oc > 0:
                                obj_coll_steps.append(si)
                                total_obj += oc
                            if sc > 0:
                                scene_coll_steps.append(si)
                                total_scene += sc

                        traj_data = {
                            "episode_id": current_episodes_info[i].episode_id,
                            "scene_id": current_episodes_info[i].scene_id,
                            "num_steps": len(traj_steps),
                            "episode_metrics": {
                                k: v for k, v in episode_stats.items()
                                if isinstance(v, (int, float, bool))
                            },
                            "collision_summary": {
                                "did_collide_human": len(human_coll_steps) > 0,
                                "human_collision_steps": human_coll_steps,
                                "total_robot_obj_colls": total_obj,
                                "robot_obj_coll_steps": obj_coll_steps,
                                "total_robot_scene_colls": total_scene,
                                "robot_scene_coll_steps": scene_coll_steps,
                            },
                            "trajectory": traj_steps,
                        }
                        traj_dir = config.habitat_baselines.video_dir
                        os.makedirs(traj_dir, exist_ok=True)
                        traj_filename = f"trajectory_ep{current_episodes_info[i].episode_id}.json"
                        traj_path = os.path.join(traj_dir, traj_filename)
                        _write_json_file(traj_path, traj_data)
                        trajectory_buffers[i] = []
                        _prev_robot_colls[i] = {
                            "robot_obj_colls": 0,
                            "robot_scene_colls": 0,
                        }

                    # ── Camera params saving + first frame of new episode ──
                    if _save_camera_params and len(camera_params_buffers[i]) > 0:
                        cam_data = {
                            "episode_id": current_episodes_info[i].episode_id,
                            "scene_id": current_episodes_info[i].scene_id,
                            "num_frames": len(camera_params_buffers[i]),
                            "intrinsics": camera_intrinsics_cache,
                            "frames": camera_params_buffers[i],
                        }
                        cam_dir = config.habitat_baselines.video_dir
                        os.makedirs(cam_dir, exist_ok=True)
                        cam_filename = f"camera_params_ep{current_episodes_info[i].episode_id}.json"
                        cam_path = os.path.join(cam_dir, cam_filename)
                        _write_json_file(cam_path, cam_data)
                        camera_params_buffers[i] = []
                        next_cam = envs.call_at(i, "get_camera_params")
                        frame_ext = {}
                        for s_uuid, s_params in next_cam.items():
                            if not any(p in s_uuid for p in _CAMERA_PARAM_PATTERNS):
                                continue
                            if s_uuid not in camera_intrinsics_cache:
                                camera_intrinsics_cache[s_uuid] = s_params["intrinsics"]
                            frame_ext[s_uuid] = s_params["extrinsics"]
                        camera_params_buffers[i].append(frame_ext)

                    # ── Depth PNG metadata saving ──
                    if _depth_png_mode and _depth_frame_counters[i]:
                        ep_depth_dir = os.path.join(
                            _depth_video_dir,
                            f"depth_png_ep{current_episodes_info[i].episode_id}",
                        )
                        depth_meta = {
                            "episode_id": current_episodes_info[i].episode_id,
                            "scene_id": current_episodes_info[i].scene_id,
                            "depth_format": "uint8" if _depth_8bit else "uint16",
                            "depth_scale": _depth_scale,
                            "depth_unit": "normalized" if _depth_8bit else "mm",
                            "max_depth_m": _depth_max,
                            "sensors": {
                                k: {"num_frames": v}
                                for k, v in _depth_frame_counters[i].items()
                            },
                        }
                        os.makedirs(ep_depth_dir, exist_ok=True)
                        _write_json_file(
                            os.path.join(ep_depth_dir, "depth_meta.json"),
                            depth_meta,
                        )
                        _depth_frame_counters[i] = {}
                        _save_depth_obs(
                            observations[i],
                            i,
                            next_episodes_info[i].episode_id,
                        )

                    # ── RGB full frames: write metadata + first frame of new episode ──
                    if _save_rgb_frames and _rgb_frame_counters[i]:
                        ep_rgb_all_dir = os.path.join(
                            _rgb_frames_dir,
                            f"rgb_all_ep{current_episodes_info[i].episode_id}",
                        )
                        rgb_meta = {
                            "episode_id": current_episodes_info[i].episode_id,
                            "scene_id": current_episodes_info[i].scene_id,
                            "jpeg_quality": _rgb_jpeg_quality,
                            "sensors": {
                                k: {"num_frames": v}
                                for k, v in _rgb_frame_counters[i].items()
                            },
                        }
                        os.makedirs(ep_rgb_all_dir, exist_ok=True)
                        _write_json_file(
                            os.path.join(ep_rgb_all_dir, "rgb_meta.json"),
                            rgb_meta,
                        )
                        _rgb_frame_counters[i] = {}
                        _save_rgb_obs(
                            observations[i],
                            i,
                            next_episodes_info[i].episode_id,
                        )

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

            if len(envs_to_pause) > 0:
                _keep = sorted(set(range(n_envs)) - set(envs_to_pause))
                _step_counters = [_step_counters[j] for j in _keep]
                if _save_trajectory:
                    trajectory_buffers = [trajectory_buffers[j] for j in _keep]
                    _prev_robot_colls = [_prev_robot_colls[j] for j in _keep]
                if _save_camera_params:
                    camera_params_buffers = [camera_params_buffers[j] for j in _keep]
                if _split_rgb:
                    _rgb_split = [_rgb_split[j] for j in _keep]
                if _topdown_view:
                    _topdown_procs = [_topdown_procs[j] for j in _keep]
                if _depth_png_mode:
                    _depth_frame_counters = [_depth_frame_counters[j] for j in _keep]
                if _save_rgb_frames:
                    _rgb_frame_counters = [_rgb_frame_counters[j] for j in _keep]
                agent.actor_critic.on_envs_pause(envs_to_pause)

        pbar.close()

        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)