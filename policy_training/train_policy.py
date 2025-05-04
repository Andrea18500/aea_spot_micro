from stable_baselines3 import PPO
from SpotmicroEnv import SpotmicroEnv
from stable_baselines3.common.env_checker import check_env

TOTAL_STEPS = 10_000

env = SpotmicroEnv(use_gui=False)
check_env(env, warn=True) #optional

model = PPO(
    "MlpPolicy", 
    env, 
    verbose = 1, 
    learning_rate=2.5e-4,
    ent_coef=0.0015, 
    clip_range=0.1,
    tensorboard_log="./logs"
)
model.learn(total_timesteps=TOTAL_STEPS)
model.save("ppo_stand10k")
