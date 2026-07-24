#!/usr/bin/env python3
"""
Example with visualization - save rendered images
"""

import gym
import habitat.gym  # noqa: F401
import imageio
import numpy as np

def example():
    with gym.make("HabitatRenderPick-v0") as env:
        print("Environment creation successful")
        observations = env.reset()
        
        frames = []
        print("Agent acting inside environment...")
        
        count_steps = 0
        terminal = False
        while not terminal and count_steps < 50:  # Only run 50 steps
            observations, reward, terminal, info = env.step(
                env.action_space.sample()
            )
            
            # Collect rendered images
            if "robot_third_rgb" in observations:
                frames.append(observations["robot_third_rgb"])
            elif "rgb" in observations:
                frames.append(observations["rgb"])
            
            count_steps += 1
        
        print(f"Episode finished after {count_steps} steps.")
        
        # Save as GIF
        if frames:
            print("Saving animation to output.gif...")
            imageio.mimsave("output.gif", frames, fps=10)
            print("Done! Open output.gif to see the visualization.")
        else:
            print("No frames captured. Available observation keys:", list(observations.keys()))


if __name__ == "__main__":
    example()

