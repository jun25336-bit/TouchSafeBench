#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@habitat.registry.register_env(name="myEnv")` for reusability
"""

import importlib
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import gym
import magnum as mn
import numpy as np

import habitat
from habitat import Dataset
from habitat.gym.gym_wrapper import HabGymWrapper

if TYPE_CHECKING:
    from omegaconf import DictConfig


RLTaskEnvObsType = Union[np.ndarray, Dict[str, np.ndarray]]


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return habitat.registry.get_env(env_name)


class RLTaskEnv(habitat.RLEnv):
    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        super().__init__(config, dataset)
        self._reward_measure_name = self.config.task.reward_measure
        self._success_measure_name = self.config.task.success_measure
        self._slack_reward = self.config.task.slack_reward
        self._success_reward = self.config.task.success_reward
        self._end_on_success = self.config.task.end_on_success
        assert (
            self._reward_measure_name is not None
        ), "The key task.reward_measure cannot be None"
        assert (
            self._success_measure_name is not None
        ), "The key task.success_measure cannot be None"

    def reset(
        self, *args, return_info: bool = False, **kwargs
    ) -> Union[RLTaskEnvObsType, Tuple[RLTaskEnvObsType, Dict]]:
        return super().reset(*args, return_info=return_info, **kwargs)

    def step(
        self, *args, **kwargs
    ) -> Tuple[RLTaskEnvObsType, float, bool, dict]:
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        # We don't know what the reward measure is bounded by
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        current_measure = self._env.get_metrics()[self._reward_measure_name]
        reward = self._slack_reward

        reward += current_measure

        if self._episode_success():
            reward += self._success_reward

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        if self._end_on_success and self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self._env.get_metrics()


@habitat.registry.register_env(name="GymRegistryEnv")
class GymRegistryEnv(gym.Wrapper):
    """
    A registered environment that wraps a gym environment to be
    used with habitat-baselines
    """

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        for dependency in config["env_task_gym_dependencies"]:
            importlib.import_module(dependency)
        env_name = config["env_task_gym_id"]
        gym_env = gym.make(env_name)
        super().__init__(gym_env)


@habitat.registry.register_env(name="GymHabitatEnv")
class GymHabitatEnv(gym.Wrapper):
    """
    A registered environment that wraps a RLTaskEnv with the HabGymWrapper
    to use the default gym API.
    """

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        base_env = RLTaskEnv(config=config, dataset=dataset)
        env = HabGymWrapper(env=base_env)
        super().__init__(env)

    def get_agent_positions(self):
        """Get 3D world-frame positions for all agents.

        Reads base_pos directly from RearrangeSim's agents_mgr,
        bypassing obs_keys / localization_sensor.

        Returns:
            List[List[float]]: [x, y, z] coordinates per agent.
        """
        sim = self.env.env._env._sim
        positions = []
        for agent_data in sim.agents_mgr._all_agent_data:
            pos = agent_data.articulated_agent.base_pos
            positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
        return positions

    def get_camera_params(self) -> Dict[str, Any]:
        """Get camera intrinsics and extrinsics for all sensors.

        Intrinsics: computed from hfov / width / height, fixed per sensor.
        Extrinsics: 4x4 camera-to-world transform, varies per frame.

        Returns:
            dict: {
                sensor_uuid: {
                    "intrinsics": {
                        "width": int, "height": int,
                        "hfov_deg": float,
                        "fx": float, "fy": float,
                        "cx": float, "cy": float,
                        "K": [[fx,0,cx],[0,fy,cy],[0,0,1]]
                    },
                    "extrinsics": {
                        "camera_to_world": [[4x4 float]],
                        "position": [x, y, z],
                        "rotation": [w, x, y, z]  (quaternion)
                    }
                }
            }
        """
        sim = self.env.env._env._sim
        result: Dict[str, Any] = {}

        for sensor_uuid, sensor in sim._sensors.items():
            sensor_obj = sensor._sensor_object
            node = sensor_obj.node

            cam2world = node.absolute_transformation()

            c2w_list = [
                [float(cam2world[i][j]) for j in range(4)]
                for i in range(4)
            ]

            pos = cam2world.translation
            rot_mat = cam2world.rotation()
            quat = mn.Quaternion.from_matrix(rot_mat)

            spec = sensor_obj.specification()
            w = spec.resolution[1]
            h = spec.resolution[0]
            hfov_deg = float(getattr(spec, "hfov", 90.0))
            hfov_rad = math.radians(hfov_deg)
            fx = w / (2.0 * math.tan(hfov_rad / 2.0))
            fy = fx
            cx = w / 2.0
            cy = h / 2.0

            result[sensor_uuid] = {
                "intrinsics": {
                    "width": w,
                    "height": h,
                    "hfov_deg": hfov_deg,
                    "fx": fx, "fy": fy,
                    "cx": cx, "cy": cy,
                    "K": [
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ],
                },
                "extrinsics": {
                    "camera_to_world": c2w_list,
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "rotation_wxyz": [
                        float(quat.scalar),
                        float(quat.vector[0]),
                        float(quat.vector[1]),
                        float(quat.vector[2]),
                    ],
                },
            }

        return result

    def _compute_topdown_height(
        self,
        scene_extent_x: float,
        scene_extent_z: float,
        hfov_deg: float,
        img_w: int,
        img_h: int,
        margin: float = 1.10,
    ) -> float:
        """Compute minimum perspective top-down camera height from scene extent and sensor FOV."""
        hfov_rad = math.radians(hfov_deg)
        vfov_rad = 2.0 * math.atan(
            math.tan(hfov_rad / 2.0) * (img_h / img_w)
        )
        h_for_x = (scene_extent_x / 2.0) / math.tan(hfov_rad / 2.0)
        h_for_z = (scene_extent_z / 2.0) / math.tan(vfov_rad / 2.0)
        return max(h_for_x, h_for_z) * margin

    def _get_topdown_scene_info(self) -> Tuple[
        str, float, float, float, float, float, float,
    ]:
        """Get scene top-down parameters from the navmesh or agent positions.

        Returns:
            (sensor_key, cx, cz, floor_y, ceiling_y, extent_x, extent_z)
            sensor_key is None when no usable sensor is found.
        """
        sim = self.env.env._env._sim

        sensor_key = None
        for k in sim._sensors:
            if "third_rgb" in k:
                sensor_key = k
                break

        pf = sim.pathfinder
        if pf.is_loaded:
            bounds = pf.get_bounds()
            cx = (bounds[0][0] + bounds[1][0]) / 2.0
            cz = (bounds[0][2] + bounds[1][2]) / 2.0
            floor_y = float(bounds[0][1])
            ceiling_y = float(bounds[1][1])
            extent_x = float(bounds[1][0] - bounds[0][0])
            extent_z = float(bounds[1][2] - bounds[0][2])
        else:
            positions = []
            for ad in sim.agents_mgr._all_agent_data:
                pos = ad.articulated_agent.base_pos
                positions.append(pos)
            cx = sum(float(p[0]) for p in positions) / len(positions)
            cz = sum(float(p[2]) for p in positions) / len(positions)
            floor_y = 0.0
            ceiling_y = 3.0
            extent_x = 10.0
            extent_z = 10.0

        return sensor_key, cx, cz, floor_y, ceiling_y, extent_x, extent_z

    def render_topdown(self, height: float = 0.0) -> Optional[np.ndarray]:
        """Render a perspective-projection top-down RGB image from a fixed overhead position.

        When height <= 0, automatically computes the optimal height to
        cover the entire scene based on scene extent and sensor FOV.

        Temporarily borrows the third_rgb sensor for rendering and
        restores its original transform afterwards.

        Args:
            height: Camera height above the floor (meters). <=0 for auto.

        Returns:
            RGB numpy array (H, W, 3) uint8, or None.
        """
        sim = self.env.env._env._sim
        sensor_key, cx, cz, floor_y, _, extent_x, extent_z = (
            self._get_topdown_scene_info()
        )
        if sensor_key is None:
            return None

        sensor_obj = sim._sensors[sensor_key]._sensor_object
        old_transform = mn.Matrix4(sensor_obj.node.transformation)

        if height <= 0:
            spec = sensor_obj.specification()
            hfov_deg = float(getattr(spec, "hfov", 90.0))
            img_w = spec.resolution[1]
            img_h = spec.resolution[0]
            height = self._compute_topdown_height(
                extent_x, extent_z, hfov_deg, img_w, img_h
            )

        eye = mn.Vector3(cx, floor_y + height, cz)
        target = mn.Vector3(cx, floor_y, cz)
        cam_world = mn.Matrix4.look_at(eye, target, mn.Vector3(0, 0, -1))

        agent_T = sim._default_agent.scene_node.transformation
        sensor_obj.node.transformation = agent_T.inverted() @ cam_world

        sim_obs = sim.get_sensor_observations()
        sensor_obj.node.transformation = old_transform

        img = sim_obs.get(sensor_key)
        if img is None:
            return None
        return img[:, :, :3]

    def render_topdown_ortho(self, height: float = 0.0) -> Optional[np.ndarray]:
        """Render an orthographic-projection top-down RGB image.

        All projection rays are parallel to the floor normal (straight down),
        so objects maintain true scale regardless of distance.

        ortho_scale is auto-computed from scene extent and image resolution
        to ensure the entire scene is visible (with 10% margin).

        Temporarily borrows the third_rgb sensor, switches its projection
        matrix to orthographic, renders, then restores perspective params.

        Args:
            height: Camera height above the floor (meters). <=0 for auto.

        Returns:
            RGB numpy array (H, W, 3) uint8, or None.
        """
        sim = self.env.env._env._sim
        sensor_key, cx, cz, floor_y, ceiling_y, extent_x, extent_z = (
            self._get_topdown_scene_info()
        )
        if sensor_key is None:
            return None

        sensor_obj = sim._sensors[sensor_key]._sensor_object
        old_transform = mn.Matrix4(sensor_obj.node.transformation)

        if height <= 0:
            height = (ceiling_y - floor_y) + 5.0

        spec = sensor_obj.specification()
        img_w = spec.resolution[1]
        img_h = spec.resolution[0]

        margin = 1.10
        ortho_scale = 1.0 / max(extent_x * margin, extent_z * margin)

        eye = mn.Vector3(cx, floor_y + height, cz)
        target = mn.Vector3(cx, floor_y, cz)
        cam_world = mn.Matrix4.look_at(eye, target, mn.Vector3(0, 0, -1))

        agent_T = sim._default_agent.scene_node.transformation
        sensor_obj.node.transformation = agent_T.inverted() @ cam_world

        render_camera = sensor_obj.render_camera
        render_camera.set_orthographic_projection_matrix(
            img_w, img_h,
            0.01,
            height + 5.0,
            ortho_scale,
        )

        sim_obs = sim.get_sensor_observations()

        render_camera.set_projection_matrix(
            img_w, img_h,
            spec.near,
            spec.far,
            getattr(spec, "hfov", mn.Deg(90.0)),
        )
        sensor_obj.node.transformation = old_transform

        img = sim_obs.get(sensor_key)
        if img is None:
            return None
        return img[:, :, :3]
