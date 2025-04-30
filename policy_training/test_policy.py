import time
import numpy as np
from stable_baselines3 import PPO
from SpotmicroEnv import SpotmicroEnv

# Load environment in GUI mode
env = SpotmicroEnv(use_gui=True)
obs, _ = env.reset()

# Load your trained model
model = PPO.load("ppo_stand10k.debug1")  # or path to your .zip

# Run rollout
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
    
    time.sleep(1/240.)  # Match simulation step time for real-time playback

env.close()