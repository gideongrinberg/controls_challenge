import numpy as np
from . import BaseController
from stable_baselines3 import PPO


class Controller(BaseController):
    def __init__(self):
        self.model = PPO.load("controls_model.zip")
        super().__init__()

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = np.array(
            [state.roll_lataccel, state.v_ego, state.a_ego], dtype=np.float32
        )
        action, _ = self.model.predict(obs, deterministic=True)

        steer = float(np.clip(action, -2, 2))
        return steer
