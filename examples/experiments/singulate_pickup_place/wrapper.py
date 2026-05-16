from typing import OrderedDict
import numpy as np
import requests
import copy
import gymnasium as gym
import time

from serl_robot_infra.so101_env.envs.so101_env import So101Env
from serl_robot_infra.so101_env.camera.usb_capture import USBCapture
from serl_robot_infra.so101_env.camera.video_capture import VideoCapture
from serl_robot_infra.franka_env.utils.rotations import euler_2_quat


class SingulatePickupPlaceEnv(So101Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """
        Move the robot to the goal position without any interpolation.
        """
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        self._send_pos_command(goal)
        time.sleep(timeout)
        self._update_currpos()
        
class GoalPoseWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        state_spaces = dict(env.observation_space["state"].spaces)

        state_spaces["goal_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(3,)
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_spaces),
                "images": env.observation_space["images"],
            }
        )

    def observation(self, obs):
        obs["state"]["goal_pose"] = self.env.goal_pose
        return obs


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info
