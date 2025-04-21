import pytest
import numpy as np
from SpotmicroEnv import SpotmicroEnv

@pytest.fixture
def env():
    env = SpotmicroEnv()
    yield env
    env.close()  # Clean up PyBullet

def test_env_initialization(env):
    assert env is not None
    assert isinstance(env.observation_space.shape, tuple)
    assert isinstance(env.action_space.shape, tuple)

def test_reset_returns_valid_obs(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)
    assert len(env._motor_joints_id) == 12

def test_step_returns_valid_output(env):
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_observation_within_space(env):
    env.reset()
    obs, *_ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)

def test_observation_shape_matches_space(env):
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape, \
        f"Expected shape {env.observation_space.shape}, got {obs.shape}"

def test_multiple_steps_dont_crash(env):
    env.reset()
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape

def test_episode_terminates_on_timeout(env):
    env.reset()
    done = False
    for _ in range(env._MAX_EPISODE_LEN):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if truncated:
            done = True
            break
    assert done