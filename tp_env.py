import os
import gym
import glob
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from gym import spaces
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from tinyphysics import (
    TinyPhysicsModel,
    TinyPhysicsSimulator,
    download_dataset,
    CONTEXT_LENGTH,
    FUTURE_PLAN_STEPS,
    LATACCEL_RANGE,
    STEER_RANGE,
    MAX_ACC_DELTA,
)

import warnings
warnings.filterwarnings("ignore")

files = glob.glob("data/*.csv")
random.shuffle(files)

class TinyPhysicsEnv(gym.Env):
    def __init__(
        self,
        model_path: str,
        data_path: str,
    ):
        super().__init__()
        self.model = TinyPhysicsModel(model_path, debug=False)
        self.data_path = data_path
        self.data = self._get_data(self.data_path)

        self.action_space = spaces.Box(
            low=np.array([STEER_RANGE[0]], dtype=np.float32),
            high=np.array([STEER_RANGE[1]], dtype=np.float32),
            dtype=np.float32,
        )

        low_obs = np.concatenate([
            np.array([
                self.data["roll_lataccel"].min(),
                self.data["v_ego"].min(),
                self.data["a_ego"].min(),
            ], dtype=np.float32),
            np.full(FUTURE_PLAN_STEPS, LATACCEL_RANGE[0], dtype=np.float32),
        ])
        high_obs = np.concatenate([
            np.array([
                self.data["roll_lataccel"].max(),
                self.data["v_ego"].max(),
                self.data["a_ego"].max(),
            ], dtype=np.float32),
            np.full(FUTURE_PLAN_STEPS, LATACCEL_RANGE[1], dtype=np.float32),
        ])
        self.observation_space = spaces.Box(
            low=low_obs,
            high=high_obs,
            dtype=np.float32,
        )

        self._reset_internal()

    def _get_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        return pd.DataFrame({
            "roll_lataccel": np.sin(df["roll"].values) * 9.81,
            "v_ego": df["vEgo"].values,
            "a_ego": df["aEgo"].values,
            "target_lataccel": df["targetLateralAcceleration"].values,
            "steer_command": -df["steerCommand"].values,
        })

    def _reset_internal(self):
        self.sim = TinyPhysicsSimulator(
            self.model, self.data_path, controller=None, debug=False
        )
        _, _, future = self.sim.get_state_target_futureplan(self.sim.step_idx)
        self.sim.futureplan = future
        self.done = False

    def reset(self):
        self._reset_internal()
        return self._get_obs()

    def step(self, action):
        a = float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))
        idx = self.sim.step_idx

        state, target, future = self.sim.get_state_target_futureplan(idx)
        self.sim.state_history.append(state)
        self.sim.target_lataccel_history.append(target)
        self.sim.futureplan = future
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
        return obs, reward, done, {}

    def _get_obs(self):
        s = self.sim.state_history[-1]
        base = np.array([s.roll_lataccel, s.v_ego, s.a_ego], dtype=np.float32)
        future_list = list(self.sim.futureplan.lataccel)
        if len(future_list) < FUTURE_PLAN_STEPS:
            pad_val = future_list[-1] if future_list else 0.0
            future_list += [pad_val] * (FUTURE_PLAN_STEPS - len(future_list))
        future_arr = np.array(future_list, dtype=np.float32)
        return np.concatenate([base, future_arr])

    def render(self, mode="human"):
        pass

    def close(self):
        pass


def make_env(file_path):
    return TinyPhysicsEnv("./models/tinyphysics.onnx", file_path)

if __name__ == "__main__":
    envs = SubprocVecEnv([partial(make_env, f) for f in files[:10]])
    model = PPO("MlpPolicy", envs, verbose=1)
    model.learn(total_timesteps=200_000, progress_bar=True)
    model.save("controls_model")
