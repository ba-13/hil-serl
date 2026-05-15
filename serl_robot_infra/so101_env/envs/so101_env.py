"""Gym interface for a SO101 arm.

This is a policy-facing scaffold, not a robot-driver implementation.
It preserves the observation and action contract expected by the SERL
wrappers and training configs:

- action: 7D vector [dx, dy, dz, droll, dpitch, dyaw, gripper]
- observation['state']:
  - tcp_pose: xyz + quat, shape (7,)
  - tcp_vel: shape (6,)
  - gripper_pose: shape (1,)
  - tcp_force: shape (3,)
  - tcp_torque: shape (3,)
- observation['images']: dict of camera images, optionally empty

Replace the transport hooks with your ROS publishers/subscribers.
"""

from __future__ import annotations

import copy
import os
import queue
import threading
import time
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.rotations import euler_2_quat


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()
            if img_array is None:
                break

            frame = np.concatenate(
                [
                    cv2.resize(v, (128, 128))
                    for k, v in img_array.items()
                    if "full" not in k
                ],
                axis=1,
            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


class DefaultSO101EnvConfig:
    """Default configuration for `So101Env`.

    Fill these in for your robot, camera setup, and task.
    """

    CAMERAS: Dict = {"cam_wrist": {"id": "/dev/cam_wrist", "size": (480, 640)}}
    IMAGE_CROP: dict[str, callable] = {}

    TARGET_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.array([1.0, 1.0, 1.0])
    RESET_POSE = np.zeros((6,))

    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.0
    RANDOM_RZ_RANGE = 0.0

    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))

    DISPLAY_IMAGE: bool = False
    GRIPPER_SLEEP: float = 0.2
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0

    # Optional transport metadata. These are only documentation hooks for your ROS side.
    ROS_NAMESPACE: str = "/so101"
    POSE_COMMAND_TOPIC: str = "/so101/cmd_pose"
    GRIPPER_COMMAND_TOPIC: str = "/so101/cmd_gripper"
    STATE_TOPIC: str = "/so101/state"


class So101Env(gym.Env):
    """Policy-facing SO101 environment.

    The expected implementation pattern is:
    - read robot state in `_update_currpos`
    - send pose commands in `_send_pos_command`
    - send gripper commands in `_send_gripper_command`
    - optionally implement `_recover`, `_reset_robot_joints`, and `init_cameras`
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        hz: int = 10,
        fake_env: bool = False,
        save_video: bool = False,
        config: Optional[DefaultSO101EnvConfig] = None,
    ):
        self.config = config or DefaultSO101EnvConfig()
        self.action_scale = np.asarray(self.config.ACTION_SCALE)
        self._TARGET_POSE = np.asarray(self.config.TARGET_POSE)
        self._RESET_POSE = np.asarray(self.config.RESET_POSE)
        self._REWARD_THRESHOLD = np.asarray(self.config.REWARD_THRESHOLD)
        self.max_episode_length = self.config.MAX_EPISODE_LENGTH
        self.display_image = self.config.DISPLAY_IMAGE
        self.gripper_sleep = self.config.GRIPPER_SLEEP
        self.randomreset = self.config.RANDOM_RESET
        self.random_xy_range = self.config.RANDOM_XY_RANGE
        self.random_rz_range = self.config.RANDOM_RZ_RANGE
        self.hz = hz

        self.resetpos = np.concatenate(
            [self._RESET_POSE[:3], euler_2_quat(self._RESET_POSE[3:])]
        )

        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.cycle_count = 0
        self.curr_path_length = 0
        self.terminate = False

        self.save_video = save_video
        self.recording_frames = [] if self.save_video else None

        self.xyz_bounding_box = gym.spaces.Box(
            self.config.ABS_POSE_LIMIT_LOW[:3],
            self.config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            self.config.ABS_POSE_LIMIT_LOW[3:],
            self.config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )

        #### Define Action Space ####
        self.action_space = gym.spaces.Box(
            low=-np.ones((7,), dtype=np.float32),
            high=np.ones((7,), dtype=np.float32),
            dtype=np.float32,
        )

        #### Define Observation Space ####
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz,quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "goal_pose": gym.spaces.Box(-np.inf, np.inf, shape=(3,)), # goal
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            0,
                            255,
                            shape=(*val["size"], 3),
                            dtype=np.uint8,
                        )
                        for key, val in self.config.CAMERAS
                    }
                ),
                "masks": gym.spaces.Dict(
                    {
                        "target_mask": gym.spaces.Box(
                            0,
                            1,
                            shape=(*self.config.CAMERAS["cam_wrist"]["size"], 1),
                            dtype=np.uint8,
                        )
                    }
                ),
            }
        )

        if fake_env:
            self._init_fake_state()
            self.cap = None
            return

        self._init_state()
        self.cap = None
        self.init_cameras(self.config.CAMERAS)

        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, "so101")
            self.displayer.start()

    def _init_fake_state(self):
        self.currpos = self.resetpos.copy()
        self.currvel = np.zeros((6,), dtype=np.float64)
        self.currforce = np.zeros((3,), dtype=np.float64)
        self.currtorque = np.zeros((3,), dtype=np.float64)
        self.curr_gripper_pos = np.zeros((1,), dtype=np.float64)
        self.q = np.zeros((7,), dtype=np.float64)
        self.dq = np.zeros((7,), dtype=np.float64)

    def _init_state(self):
        self._update_currpos()
        if not hasattr(self, "currpos"):
            self.currpos = self.resetpos.copy()
        if not hasattr(self, "currvel"):
            self.currvel = np.zeros((6,), dtype=np.float64)
        if not hasattr(self, "currforce"):
            self.currforce = np.zeros((3,), dtype=np.float64)
        if not hasattr(self, "currtorque"):
            self.currtorque = np.zeros((3,), dtype=np.float64)
        if not hasattr(self, "curr_gripper_pos"):
            self.curr_gripper_pos = np.zeros((1,), dtype=np.float64)

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        pose = np.array(pose, copy=True)
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        sign = np.sign(euler[0])
        euler[0] = sign * np.clip(
            np.abs(euler[0]),
            self.rpy_bounding_box.low[0],
            self.rpy_bounding_box.high[0],
        )
        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()
        return pose

    def step(self, action: np.ndarray) -> tuple:
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
        self.nextpos[3:] = (
            Rotation.from_rotvec(action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        gripper_action = action[6] * self.action_scale[2]
        self._send_gripper_command(gripper_action)
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.hz) - dt))

        self._update_currpos()
        obs = self._get_obs()
        reward = self.compute_reward(obs)
        done = (
            self.curr_path_length >= self.max_episode_length or reward or self.terminate
        )
        return obs, int(reward), done, False, {"succeed": bool(reward)}

    def compute_reward(self, obs) -> bool:
        """Default sparse success check.

        Override this if the task reward comes from a classifier or a different signal.
        """
        current_pose = obs["state"]["tcp_pose"]
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(
            np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler])
        )
        return bool(np.all(delta < self._REWARD_THRESHOLD))

    def get_im(self) -> Dict[str, np.ndarray]:
        if not getattr(self, "cap", None):
            return {}

        images = {}
        display_images = {}
        full_res_images = {}
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = (
                    self.config.IMAGE_CROP[key](rgb)
                    if key in self.config.IMAGE_CROP
                    else rgb
                )
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.CAMERAS)
                return self.get_im()

        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        steps = max(1, int(timeout * self.hz))
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = path[-1]
        self._update_currpos()

    def go_to_reset(self, joint_reset: bool = False):
        """Default reset flow.

        Replace this if your robot needs a special homing or recovery sequence.
        """
        if joint_reset:
            self._reset_robot_joints()

        if self.randomreset:
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
        else:
            reset_pose = self.resetpos.copy()

        self.interpolate_move(reset_pose, timeout=1.0)

    def reset(self, joint_reset: bool = False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if (
            self.config.JOINT_RESET_PERIOD
            and self.cycle_count % self.config.JOINT_RESET_PERIOD == 0
        ):
            self.cycle_count = 0
            joint_reset = True

        self._recover()
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if self.recording_frames and len(self.recording_frames):
                if not os.path.exists("./videos"):
                    os.makedirs("./videos")

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                for camera_key in self.recording_frames[0].keys():
                    video_path = f"./videos/so101_{camera_key}_{timestamp}.mp4"
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")

            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        """Initialize cameras.

        Replace this if your camera stack is not RealSense-based.
        """
        if self.cap is not None:
            self.close_cameras()

        self.cap = OrderedDict()
        if not name_serial_dict:
            return

        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
            self.cap[cam_name] = cap

    def close_cameras(self):
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def close(self):
        if hasattr(self, "listener"):
            self.listener.stop()
        if getattr(self, "cap", None):
            self.close_cameras()
        if self.display_image and hasattr(self, "img_queue"):
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()

    def _recover(self):
        """Robot-specific error recovery hook."""
        return None

    def _reset_robot_joints(self):
        """Optional joint reset hook for robots that need it."""
        return None

    def _send_pos_command(self, pos: np.ndarray):
        """Send a pose command to the robot.

        Expected pose format: [x, y, z, qx, qy, qz, qw].
        """
        raise NotImplementedError("Implement the SO101 pose command transport here.")

    def _send_gripper_command(self, pos: float, mode: str = "binary"):
        """Send a gripper command to the robot.

        Keep the action convention compatible with the SERL policy:
        negative values close, positive values open.
        """
        raise NotImplementedError("Implement the SO101 gripper command transport here.")

    def _update_currpos(self):
        """Read the latest robot state.

        Populate:
        - currpos: shape (7,)
        - currvel: shape (6,)
        - currforce: shape (3,)
        - currtorque: shape (3,)
        - curr_gripper_pos: shape (1,)
        """
        raise NotImplementedError(
            "Implement the SO101 state subscription/polling here."
        )

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))
