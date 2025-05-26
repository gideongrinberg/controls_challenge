import os
import gym
import glob
import random
import numpy as np
import pandas as pd

from gym import spaces
from tqdm import trange
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from tinyphysics import (
    TinyPhysicsModel,
    TinyPhysicsSimulator,
    download_dataset,
    CONTEXT_LENGTH,
    STEER_RANGE,
    MAX_ACC_DELTA,
)

import warnings
warnings.filterwarnings("ignore")

class TinyPhysicsEnv(gym.Env):
    def __init__(
        self,
        model_path: str,
        data_path: str,
        debug: bool = False,
    ):
        super().__init__()
        self.model = TinyPhysicsModel(model_path, debug=debug)
        self.data_path = str(data_path)
        self.debug = debug

        tmp = TinyPhysicsSimulator(
            self.model, self.data_path, controller=None, debug=False
        )
        self.data = tmp.data

        self.action_space = spaces.Box(
            low=np.array([STEER_RANGE[0]], dtype=np.float32),
            high=np.array([STEER_RANGE[1]], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([-0.16868178, -0.10009988, -12.86107018], dtype=np.float32),
            high=np.array([0.17329173, 42.86994681, 8.9678448], dtype=np.float32),
            dtype=np.float32,
        )

        self._reset_internal()

    def _reset_internal(self):
        self.sim = TinyPhysicsSimulator(
            self.model, self.data_path, controller=None, debug=self.debug
        )
        self.done = False

    def reset(self):
        self._reset_internal()
        return self._get_obs()

    def step(self, action):
        a = float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))
        idx = self.sim.step_idx

        state, target, futureplan = self.sim.get_state_target_futureplan(idx)

        self.sim.state_history.append(state)
        self.sim.target_lataccel_history.append(target)
        self.sim.futureplan = futureplan
        self.sim.action_history.append(a)

        pred = self.model.get_current_lataccel(
            sim_states=self.sim.state_history[-CONTEXT_LENGTH:],
            actions=self.sim.action_history[-CONTEXT_LENGTH:],
            past_preds=self.sim.current_lataccel_history[-CONTEXT_LENGTH:],
        )
        pred = np.clip(
            pred,
            self.sim.current_lataccel - MAX_ACC_DELTA,
            self.sim.current_lataccel + MAX_ACC_DELTA,
        )
        self.sim.current_lataccel = pred
        self.sim.current_lataccel_history.append(pred)
        self.sim.step_idx += 1

        obs = self._get_obs()
        reward = -((target - pred) ** 2) * 100
        done = self.sim.step_idx >= len(self.data)
        info = {}
        return obs, reward, done, info

    def _get_obs(self):
        state = self.sim.state_history[-1]
        return np.array(
            [state.roll_lataccel, state.v_ego, state.a_ego], dtype=np.float32
        )

    def render(self, mode="human"):
        pass

    def close(self):
        pass

def make_env(data_path):
    return TinyPhysicsEnv("./models/tinyphysics.onnx", data_path)


def train():
    if not os.path.isdir("data/"):
        download_dataset()

    files = glob.glob("data/**.csv")
    random.shuffle(files)

    model = None
    for idx in trange(625):
        env = SubprocVecEnv(
            [partial(make_env, path) for path in files[idx * 32 : (idx + 1) * 32]]
        )

        if model is None:
            model = PPO("MlpPolicy", env, device="cuda", verbose=1)
        else:
            model.set_env(env)

        model.learn(total_timesteps=598)

    model.save("controls_model")


if __name__ == "__main__":
    train()
