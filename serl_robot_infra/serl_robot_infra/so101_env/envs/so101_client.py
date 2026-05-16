"""ROS2 client for SO101 robot communication.

Handles all ROS2 communication including:
- Publishing pose and gripper commands to topics
- Subscribing to state topics (pose, velocity, gripper)
- Joint reset via action server
- Parameter updates via ROS2 parameter service
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import Float32

# Placeholder for custom action message (update with your actual action type)
# from so101_msgs.action import ResetJoints


class So101RosClient(Node):
    """ROS2 client for SO101 arm communication.

    Manages publishers, subscribers, and service/action clients.
    Runs ROS2 executor in a background thread.

    Topic naming convention:
    - {namespace}/cmd_pose: publish desired pose (PoseStamped)
    - {namespace}/cmd_gripper: publish gripper angle (Float32)
    - {namespace}/state/pose: subscribe to current pose (PoseStamped)
    - {namespace}/state/twist: subscribe to velocity (TwistStamped)
    - {namespace}/state/gripper: subscribe to gripper state (Float32)

    Action:
    - {namespace}/reset_joints: action to trigger joint reset

    Parameters:
    - Update via ROS2 parameter service on target node
    """

    def __init__(
        self, namespace: str = "so101", controller_name: str = "so101_controller"
    ):
        """Initialize ROS2 client.

        Args:
            namespace: ROS2 namespace (e.g., "/so101")
            controller_name: Name of the robot controller node for parameter updates
        """
        self.namespace = namespace
        self.controller_name = controller_name

        # Create node
        self.node = rclpy.create_node("so101_client", namespace=namespace)

        # Create executor and run in background thread
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        # Cache for latest state messages
        self._pose_msg: Optional[PoseStamped] = None
        self._twist_msg: Optional[TwistStamped] = None
        self._gripper_msg: Optional[Float32] = None
        self._state_lock = threading.Lock()

        # Publishers
        self.pose_pub = self.node.create_publisher(PoseStamped, "cmd_pose", 10)
        self.gripper_pub = self.node.create_publisher(Float32, "cmd_gripper", 10)

        # Subscribers (cache latest message)
        self.node.create_subscription(
            PoseStamped,
            "state/pose",
            self._pose_callback,
            10,
        )
        self.node.create_subscription(
            TwistStamped,
            "state/twist",
            self._twist_callback,
            10,
        )
        self.node.create_subscription(
            Float32,
            "state/gripper",
            self._gripper_callback,
            10,
        )

        # Service clients
        self.param_client = self.node.create_client(
            SetParameters,
            f"/{controller_name}/set_parameters",
        )

        # Action client (will be set up when action type is available)
        # self.reset_joints_client = ActionClient(
        #     self.node, ResetJoints, f"{namespace}/reset_joints"
        # )

        self.node.get_logger().info(
            f"So101RosClient initialized with namespace={namespace}"
        )

    def _pose_callback(self, msg: PoseStamped) -> None:
        """Cache latest pose message."""
        with self._state_lock:
            self._pose_msg = msg

    def _twist_callback(self, msg: TwistStamped) -> None:
        """Cache latest twist (velocity) message."""
        with self._state_lock:
            self._twist_msg = msg

    def _gripper_callback(self, msg: Float32) -> None:
        """Cache latest gripper state message."""
        with self._state_lock:
            self._gripper_msg = msg

    def send_pose(self, pose: np.ndarray) -> None:
        """Publish desired pose command.

        Args:
            pose: [x, y, z, qx, qy, qz, qw] shape (7,)
        """
        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base"
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        msg.pose.position.z = float(pose[2])
        msg.pose.orientation.x = float(pose[3])
        msg.pose.orientation.y = float(pose[4])
        msg.pose.orientation.z = float(pose[5])
        msg.pose.orientation.w = float(pose[6])
        self.pose_pub.publish(msg)

    def send_gripper(self, angle: float) -> None:
        """Publish gripper angle command.

        Args:
            angle: Gripper angle (convention: negative = close, positive = open)
        """
        msg = Float32()
        msg.data = float(angle)
        self.gripper_pub.publish(msg)

    def get_state(self) -> dict:
        """Read current robot state from cached messages.

        Returns:
            dict with keys:
            - "pose": [x, y, z, qx, qy, qz, qw] shape (7,)
            - "vel": [vx, vy, vz, wx, wy, wz] shape (6,)
            - "gripper": [angle] shape (1,)
        """
        with self._state_lock:
            state = {
                "pose": np.zeros(7, dtype=np.float64),
                "vel": np.zeros(6, dtype=np.float64),
                "gripper": np.zeros(1, dtype=np.float64),
            }

            if self._pose_msg is not None:
                state["pose"] = np.array(
                    [
                        self._pose_msg.pose.position.x,
                        self._pose_msg.pose.position.y,
                        self._pose_msg.pose.position.z,
                        self._pose_msg.pose.orientation.x,
                        self._pose_msg.pose.orientation.y,
                        self._pose_msg.pose.orientation.z,
                        self._pose_msg.pose.orientation.w,
                    ],
                    dtype=np.float64,
                )

            if self._twist_msg is not None:
                state["vel"] = np.array(
                    [
                        self._twist_msg.twist.linear.x,
                        self._twist_msg.twist.linear.y,
                        self._twist_msg.twist.linear.z,
                        self._twist_msg.twist.angular.x,
                        self._twist_msg.twist.angular.y,
                        self._twist_msg.twist.angular.z,
                    ],
                    dtype=np.float64,
                )

            if self._gripper_msg is not None:
                state["gripper"] = np.array([self._gripper_msg.data], dtype=np.float64)

        return state

    def reset_joints(self, timeout: float = 30.0) -> bool:
        """Trigger joint reset action on the robot.

        Args:
            timeout: Timeout in seconds to wait for action completion

        Returns:
            True if successful, False otherwise

        Note:
            Update this method once the ResetJoints action type is defined.
            Current implementation is a placeholder.
        """
        self.node.get_logger().warn("Joint reset action not yet implemented")
        return False

        # Once action type is available, uncomment:
        # goal = ResetJoints.Goal()
        # future = self.reset_joints_client.send_goal_async(goal)
        # rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        # result = future.result()
        # return result is not None

    def update_params(self, params: dict) -> bool:
        """Update ROS2 parameters on the robot controller node.

        Args:
            params: Dictionary mapping parameter names to values
                   (e.g., {"compliance_x": 100.0, "precision_threshold": 0.01})

        Returns:
            True if successful, False otherwise

        Example:
            client.update_params({
                "compliance_param.x": 100.0,
                "precision_param.threshold": 0.01,
            })
        """
        if not params:
            return True

        try:
            # Wait for service
            while not self.param_client.wait_for_service(timeout_sec=1.0):
                self.node.get_logger().warn(
                    "Parameter service not available, retrying..."
                )

            # Convert dict to Parameter messages
            param_list = []
            for name, value in params.items():
                param = Parameter(
                    name=name,
                    value=self._python_to_parameter_value(value),
                )
                param_list.append(param)

            # Call service
            request = SetParameters.Request()
            request.parameters = param_list
            future = self.param_client.call_async(request)

            # Wait for response
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
            response = future.result()

            if response is None:
                self.node.get_logger().error("Parameter service call timed out")
                return False

            # Check if all parameters were set successfully
            all_successful = all(result.successful for result in response.results)
            if not all_successful:
                self.node.get_logger().error(
                    f"Some parameters failed to set: {response.results}"
                )
            return all_successful

        except Exception as e:
            self.node.get_logger().error(f"Failed to update parameters: {e}")
            return False

    def _python_to_parameter_value(self, value: any) -> ParameterValue:
        """Convert Python value to ROS2 ParameterValue.

        Args:
            value: Python value (int, float, str, bool, list, etc.)

        Returns:
            ParameterValue message
        """
        pv = ParameterValue()

        if isinstance(value, bool):
            pv.type = 1  # PARAMETER_BOOL
            pv.bool_value = value
        elif isinstance(value, int):
            pv.type = 2  # PARAMETER_INTEGER
            pv.integer_value = value
        elif isinstance(value, float):
            pv.type = 3  # PARAMETER_DOUBLE
            pv.double_value = value
        elif isinstance(value, str):
            pv.type = 4  # PARAMETER_STRING
            pv.string_value = value
        elif isinstance(value, (list, tuple)):
            if all(isinstance(v, int) for v in value):
                pv.type = 5  # PARAMETER_INTEGER_ARRAY
                pv.integer_array_value = list(value)
            elif all(isinstance(v, float) for v in value):
                pv.type = 6  # PARAMETER_DOUBLE_ARRAY
                pv.double_array_value = list(value)
            elif all(isinstance(v, str) for v in value):
                pv.type = 7  # PARAMETER_STRING_ARRAY
                pv.string_array_value = list(value)
            else:
                raise ValueError(f"Unsupported array type: {value}")
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")

        return pv

    def close(self) -> None:
        """Shutdown ROS2 client and cleanup resources."""
        try:
            self.executor.shutdown()
            self.executor_thread.join(timeout=5.0)
            self.node.destroy_node()
        except Exception as e:
            self.node.get_logger().error(f"Error during shutdown: {e}")
