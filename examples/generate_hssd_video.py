#!/usr/bin/env python3
"""
Generate simulation videos in HSSD scenes
Uses habitat-lab's high-level API, similar to shortest_path_follower_example.py
Output includes first-person RGB view + top-down map
"""

import os
import random
import shutil
import time

import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Use current time as random seed to ensure different results each run
RANDOM_SEED = int(time.time()) % 10000
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"🎲 Random seed: {RANDOM_SEED}")

import habitat
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video


IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


class SimpleRLEnv(habitat.RLEnv):
    """Simple RL environment wrapper, same as shortest_path_follower_example.py"""
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def draw_top_down_map(info, output_size):
    """Draw top-down map, same as shortest_path_follower_example.py"""
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def generate_hssd_video_with_topdown():
    """
    Randomly explore HSSD scenes and generate videos with first-person view and top-down map.
    Follows the style of shortest_path_follower_example.py.
    """
    
    # Check if HSSD scene dataset exists
    hssd_scenes_dir = "data/scene_datasets/hssd-hab/scenes"
    if not os.path.exists(hssd_scenes_dir):
        print(f"⚠️ HSSD scene directory not found: {hssd_scenes_dir}")
        print("Please download the HSSD dataset first")
        return
    
    # Get all available scenes
    scene_files = [f for f in os.listdir(hssd_scenes_dir) if f.endswith('.scene_instance.json')]
    if not scene_files:
        print("No scene files found!")
        return
    
    # Randomly select a scene
    selected_scene = random.choice(scene_files)
    scene_path = os.path.join(hssd_scenes_dir, selected_scene)
    print(f"🏠 Randomly selected scene: {selected_scene}")
    
    # Use pointnav config with top_down_map measurement
    # Same configuration approach as shortest_path_follower_example.py
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            # Add top_down_map measurement
            "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map",
            # Set scene
            f"habitat.simulator.scene={scene_path}",
            "habitat.simulator.scene_dataset=data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json",
            # Random settings
            f"habitat.seed={RANDOM_SEED}",
            "habitat.environment.iterator_options.shuffle=True",
            # Step limit
            "habitat.environment.max_episode_steps=200",
        ],
    )
    
    # Increase resolution
    with habitat.config.read_write(config):
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 480
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 480
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = 480
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = 480
    
    print("Initializing environment...")
    
    # Use SimpleRLEnv, same as shortest_path_follower_example.py
    with SimpleRLEnv(config=config) as env:
        print("✅ Environment created successfully!")
        
        # Run multiple episodes
        num_episodes = 3
        
        for episode_idx in range(num_episodes):
            env.reset()
            
            # Create output directory
            dirname = os.path.join(
                IMAGE_DIR, "hssd_exploration", f"{episode_idx:02d}"
            )
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)
            
            print(f"\n=== Episode {episode_idx + 1}/{num_episodes} ===")
            print("Agent randomly exploring HSSD scene...")
            
            images = []
            step_count = 0
            max_steps = 100
            
            while not env.habitat_env.episode_over and step_count < max_steps:
                # Randomly select action (excluding stop=0)
                # 1: move_forward, 2: turn_left, 3: turn_right
                action = random.randint(1, 3)
                
                # Execute action, get observations and info
                observations, reward, done, info = env.step(action)
                
                # Get first-person RGB image
                im = observations["rgb"]
                
                # Draw top-down map, resize to match RGB image height
                top_down_map = draw_top_down_map(info, im.shape[0])
                
                # Concatenate RGB and top_down_map (horizontal)
                output_im = np.concatenate((im, top_down_map), axis=1)
                images.append(output_im)
                
                step_count += 1
                
                if step_count % 20 == 0:
                    print(f"  Steps: {step_count}/{max_steps}")
            
            # Save video
            if images:
                images_to_video(images, dirname, "exploration")
                print(f"✅ Episode {episode_idx + 1} complete!")
                print(f"   Saved to: {dirname}/exploration.mp4")
                print(f"   Total frames: {len(images)}")
            else:
                print("⚠️ No frames captured")
        
        print("\n" + "=" * 60)
        print("🎉 All done!")
        print(f"   Scene: {selected_scene}")
        print(f"   Output directory: {os.path.join(IMAGE_DIR, 'hssd_exploration')}")


def main():
    """Main function"""
    
    print("=" * 60)
    print("HSSD Scene Video Generator")
    print("Output: First-person RGB + Top-down Map")
    print("=" * 60)
    
    try:
        generate_hssd_video_with_topdown()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
