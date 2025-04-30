from stable_baselines3 import PPO
from SpotmicroEnv import SpotmicroEnv
from stable_baselines3.common.env_checker import check_env

TOTAL_STEPS = 4096

env = SpotmicroEnv(use_gui=False)
check_env(env, warn=True) #optional

model = PPO("MlpPolicy", env, verbose = 1)
model.learn(total_timesteps=TOTAL_STEPS)
model.save("ppo_stand10k.debug1")