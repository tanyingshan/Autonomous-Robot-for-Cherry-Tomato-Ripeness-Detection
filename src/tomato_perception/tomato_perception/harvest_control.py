import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
import time

class HarvestControlNode(Node):
    def __init__(self):
        super().__init__('harvest_control_node')
        
        # --- CONFIGURATION ---
        self.group_name = "arm_group"
        self.base_frame = "base_link"
        self.ee_link = "tcp_link"
        # ---------------------

        self.target_sub = self.create_subscription(
            PoseStamped, '/harvest_target/pose', self.target_callback, 10)
        
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')
        
        # STATE FLAG: Keeps track if we are already busy
        self.mission_status = "IDLE" # Options: IDLE, MOVING, DONE

        self.get_logger().info("Waiting for MoveIt...")
        self.move_group_client.wait_for_server()
        self.get_logger().info("READY. Show me a tomato, and I will move ONCE.")

    def target_callback(self, msg):
        # 1. If we already moved or are moving, IGNORE the camera
        if self.mission_status != "IDLE":
            return

        self.get_logger().info(f"Target Locked: X={msg.pose.position.x:.2f} Y={msg.pose.position.y:.2f} Z={msg.pose.position.z:.2f}")

        # 2. Change status so we don't process new camera frames
        self.mission_status = "MOVING"

        # 3. Setup the MoveIt Goal
        goal_msg = MoveGroup.Goal()
        goal_msg.request.workspace_parameters.header.frame_id = self.base_frame
        goal_msg.request.workspace_parameters.min_corner.x = -1.0
        goal_msg.request.workspace_parameters.min_corner.y = -1.0
        goal_msg.request.workspace_parameters.min_corner.z = -1.0
        goal_msg.request.workspace_parameters.max_corner.x = 1.0
        goal_msg.request.workspace_parameters.max_corner.y = 1.0
        goal_msg.request.workspace_parameters.max_corner.z = 1.0
        
        goal_msg.request.group_name = self.group_name
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.1 # Keep it slow
        goal_msg.request.max_acceleration_scaling_factor = 0.1

        # 4. Position Constraint
        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.ee_link 
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0
        pc.constraint_region.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01]))
        pc.constraint_region.primitive_poses.append(msg.pose)
        pc.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pc)
        goal_msg.request.goal_constraints.append(constraints)

        # 5. Send Command
        self.get_logger().info("Sending command to arm...")
        future = self.move_group_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("MoveIt REJECTED the plan. (Target might be unreachable)")
            self.mission_status = "IDLE" # Reset to try again
            return

        self.get_logger().info("Plan Accepted! Executing movement...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == 1: # 1 means SUCCESS
            self.get_logger().info("MISSION COMPLETE. I have reached the tomato.")
            self.mission_status = "DONE"
            # To reset, you would need to restart the node or add a reset button
        else:
            self.get_logger().error(f"Movement Failed with error code: {result.error_code.val}")
            self.mission_status = "IDLE" # Retry

def main(args=None):
    rclpy.init(args=args)
    node = HarvestControlNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()