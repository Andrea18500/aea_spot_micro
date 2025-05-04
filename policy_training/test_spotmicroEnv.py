import pytest
import numpy as np
from SpotmicroEnv import SpotmicroEnv

@pytest.fixture(scope="module")
def env():
    env = SpotmicroEnv(use_gui=False)
    yield env
    env.close()

def test_env_instantiates(env):
    assert env is not None
    assert hasattr(env, "reset")
    assert hasattr(env, "step")

def test_env_reset(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs), "Observation outside space"
    assert isinstance(info, dict)

def test_step_returns_valid(env):
    obs, _ = env.reset()
    action = env.action_space.sample()

    next_obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == env.observation_space.shape
    assert env.observation_space.contains(next_obs), "Next obs outside space"
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_action_bounds_clipped(env):
    obs, _ = env.reset()

    overbound_action = np.ones(env.action_space.shape) * 10.0
    obs, reward, terminated, truncated, info = env.step(overbound_action)

    # PyBullet will clip or saturate at limits, but your env should not crash
    assert isinstance(obs, np.ndarray)
    assert np.isfinite(obs).all()

def test_environment_termination_condition(env):
    obs, _ = env.reset()
    env._agent_state["base_position"] = (0.0, 0.0, 0.05)  # below threshold
    assert env._is_target_state(env._agent_state) is True
