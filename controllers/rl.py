import numpy as np
from . import BaseController
from stable_baselines3 import PPO
from tinyphysics import STEER_RANGE, FUTURE_PLAN_STEPS

class Controller(BaseController):
    """
    A controller that queries a trained SB3 PPO policy for steering actions,
    including future_plan in the observation.
    """
    def __init__(self, model_path: str = "controls_model.zip"):
        self.model = PPO.load(model_path)
        super().__init__()

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = np.array([
            state.roll_lataccel,
            state.v_ego,
            state.a_ego,
        ], dtype=np.float32)

        future_arr = np.array(future_plan.lataccel, dtype=np.float32)
        if future_arr.shape[0] >= FUTURE_PLAN_STEPS:
            future_arr = future_arr[:FUTURE_PLAN_STEPS]
        else:
            pad_val = float(future_arr[-1]) if future_arr.size > 0 else 0.0
            future_arr = np.pad(
                future_arr,
                (0, FUTURE_PLAN_STEPS - future_arr.shape[0]),
                constant_values=pad_val,
            )
        obs = np.concatenate([obs, future_arr])

        action, _ = self.model.predict(obs, deterministic=True)
        steer = float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))
        return steer
