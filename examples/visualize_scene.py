#!/usr/bin/env python3
"""
Render scenes directly using habitat_sim and save images
"""

import habitat_sim
import numpy as np
import os

# Test scene path
SCENE_PATH = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"

def main():
    # Check if scene exists
    if not os.path.exists(SCENE_PATH):
        print(f"Scene file not found: {SCENE_PATH}")
        print("Please download test scenes first:")
        print("  python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/")
        return

    # Configure simulator
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = SCENE_PATH

    # Configure RGB sensor
    sensor_cfg = habitat_sim.CameraSensorSpec()
    sensor_cfg.uuid = "color_sensor"
    sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
    sensor_cfg.resolution = [512, 512]
    sensor_cfg.position = [0.0, 1.5, 0.0]  # Sensor height

    # Configure Agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg]

    # Create simulator
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    print("Simulator created successfully!")
    print(f"Scene: {SCENE_PATH}")

    # Get Agent
    agent = sim.initialize_agent(0)
    
    # Collect multiple frames
    frames = []
    
    # Define actions to move viewpoint
    actions = [
        "turn_left", "turn_left", "move_forward",
        "turn_right", "move_forward", "move_forward",
        "turn_left", "move_forward"
    ]
    
    # Get initial observations
    obs = sim.get_sensor_observations()
    frames.append(obs["color_sensor"])
    
    for action in actions:
        # Execute action
        agent.act(action)
        # Get observations
        obs = sim.get_sensor_observations()
        frames.append(obs["color_sensor"])

    # Save images
    try:
        import imageio
        
        # Save GIF animation
        print("Saving animation to scene_tour.gif...")
        imageio.mimsave("scene_tour.gif", frames, fps=2)
        print("Done!")
        
        # Also save a single image
        from PIL import Image
        img = Image.fromarray(frames[0][:, :, :3])  # Remove alpha channel
        img.save("scene_snapshot.png")
        print("Saved screenshot to scene_snapshot.png")
        
    except ImportError:
        print("Please install imageio: pip install imageio")
        # At least save with numpy
        np.save("scene_frames.npy", np.array(frames))
        print("Frame data saved to scene_frames.npy")

    sim.close()
    print("\nView images:")
    print("  - scene_tour.gif (animation)")
    print("  - scene_snapshot.png (screenshot)")


if __name__ == "__main__":
    main()

