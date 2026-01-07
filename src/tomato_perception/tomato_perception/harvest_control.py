import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, DisplayTrajectory
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import PositionConstraint, OrientationConstraint
from std_srvs.srv import Empty
import math
import copy
import time 

class HarvestControlNode(Node):
    def __init__(self):
        super().__init__('harvest_control_node')
        
        # --- GROUPS ---
        self.arm_group_name = "arm_group"
        self.gripper_group_name = "gripper_group"
        self.cutter_group_name = "cutter_group"  # <--- NEW GROUP
        
        self.base_frame = "base_link"
        self.ee_link = "tcp_link"
        self.max_reach = 0.65 
        
        # --- JOINTS ---
        self.gripper_joints = ["left_gripper_joint", "right_gripper_joint"]
        self.cutter_joints = ["left_cutter_joint", "right_cutter_joint"]
        
        # --- ANGLES (Radians) ---
        # Gripper
        self.gripper_open_val = math.radians(17.0)
        self.gripper_closed_val = math.radians(5.0) # Grip (Hold)

        # Cutter (Predefined States)
        # Open: Left -40, Right 40
        self.cutter_open_left = math.radians(-40.0)
        self.cutter_open_right = math.radians(40.0)
        # Close: Both 0
        self.cutter_closed_val = math.radians(0.0)

        # Rest (Arm)
        self.rest_joints_deg = [0.0, 0.0, 0.0, 0.0, 0.0] 
        self.rest_joints_rad = [math.radians(a) for a in self.rest_joints_deg]
        self.arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]

        self.target_sub = self.create_subscription(PoseStamped, '/harvest_target/pose', self.target_callback, 10)
        
        # Publishers
        self.display_path_pub = self.create_publisher(DisplayTrajectory, '/move_group/display_planned_path', 10)
        
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')
        self.clear_octomap_client = self.create_client(Empty, '/clear_octomap')

        self.state = "IDLE" 
        self.latest_target = None

        self.get_logger().info("Waiting for MoveIt...")
        self.move_group_client.wait_for_server()
        self.get_logger().info("READY. Using Cutter Group for Simulation.")

    def target_callback(self, msg):
        if self.state != "IDLE": return
        dist = math.sqrt(msg.pose.position.x**2 + msg.pose.position.y**2 + msg.pose.position.z**2)
        if dist > self.max_reach: return

        self.latest_target = msg
        self.get_logger().info("Tomato Detected. Preparing Tools...")
        # Step 1: Open BOTH Gripper and Cutter (Real)
        self.execute_tool_command("BOTH_OPEN", "PRE_APPROACH", plan_only=False)

    def execute_tool_command(self, command_type, next_state, plan_only=False):
        self.state = next_state
        goal_msg = MoveGroup.Goal()
        
        # Select Group based on command
        if command_type == "GRIPPER_CLOSE":
            goal_msg.request.group_name = self.gripper_group_name
            target_val = self.gripper_closed_val
            joints = self.gripper_joints
        elif command_type == "GRIPPER_OPEN":
            goal_msg.request.group_name = self.gripper_group_name
            target_val = self.gripper_open_val
            joints = self.gripper_joints
        elif command_type == "CUTTER_CLOSE":
            goal_msg.request.group_name = self.cutter_group_name
            # Handled specially below due to asymmetric joints
            joints = self.cutter_joints
        elif command_type == "CUTTER_OPEN":
            goal_msg.request.group_name = self.cutter_group_name
            joints = self.cutter_joints
        elif command_type == "BOTH_OPEN":
            # For simplicity, we just open the Cutter first here, Gripper logic loops
            goal_msg.request.group_name = self.cutter_group_name
            joints = self.cutter_joints

        # Options
        goal_msg.planning_options.plan_only = plan_only
        goal_msg.planning_options.look_around = False
        goal_msg.planning_options.replan = False

        constraints = Constraints()
        
        # --- JOINT LOGIC ---
        if command_type == "CUTTER_OPEN" or command_type == "BOTH_OPEN":
            # Asymmetric Open: Left -40, Right 40
            jc1 = JointConstraint(joint_name="left_cutter_joint", position=self.cutter_open_left, tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            jc2 = JointConstraint(joint_name="right_cutter_joint", position=self.cutter_open_right, tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            constraints.joint_constraints.append(jc1)
            constraints.joint_constraints.append(jc2)
            
        elif command_type == "CUTTER_CLOSE":
            # Close: Both 0
            for j in self.cutter_joints:
                jc = JointConstraint(joint_name=j, position=self.cutter_closed_val, tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
                constraints.joint_constraints.append(jc)
                
        else: # Gripper Commands
            for j in self.gripper_joints:
                jc = JointConstraint(joint_name=j, position=float(target_val), tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
                constraints.joint_constraints.append(jc)

        goal_msg.request.goal_constraints.append(constraints)
        
        future = self.move_group_client.send_goal_async(goal_msg)
        future.add_done_callback(self.tool_response_callback)

    def tool_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.state = "IDLE"
            return
        goal_handle.get_result_async().add_done_callback(self.tool_result_callback)

    def tool_result_callback(self, future):
        result = future.result().result
        
        # --- SEQUENCE LOGIC ---

        # 1. CUTTER OPENED -> NOW OPEN GRIPPER
        if self.state == "PRE_APPROACH":
            self.get_logger().info("Cutter Open. Opening Gripper...")
            # We recursively call to open gripper, but change state to APPROACH
            self.execute_tool_command("GRIPPER_OPEN", "APPROACHING", plan_only=False)

        # 2. GRIPPER OPENED -> MOVE ARM
        elif self.state == "APPROACHING":
            self.clear_octomap()
            self.get_logger().info("Tools Ready. Moving Arm...")
            self.execute_arm_move()

        # 3. ARM ARRIVED -> GRIP (REAL)
        elif self.state == "GRIPPING":
            self.get_logger().info("At Target. Gripping...")
            self.execute_tool_command("GRIPPER_CLOSE", "SIMULATING_CUT", plan_only=False)

        # 4. GRIPPED -> SIMULATE CUTTER (GHOST)
        elif self.state == "SIMULATING_CUT":
            self.get_logger().info("Gripped! Simulating Cutter Action...")
            # Plan ONLY (True) for Cutter Close
            self.execute_tool_command("CUTTER_CLOSE", "DISPLAY_CUT", plan_only=True)

        # 5. DISPLAY CUT ANIMATION
        elif self.state == "DISPLAY_CUT":
            if result.error_code.val == 1:
                traj = result.planned_trajectory
                display_msg = DisplayTrajectory()
                display_msg.model_id = "cocoabot"
                display_msg.trajectory.append(traj)
                
                self.get_logger().info("Visualizing Cut in RViz...")
                for i in range(5): # Loop 5 times
                    self.display_path_pub.publish(display_msg)
                    time.sleep(0.8)
            
            self.get_logger().info("Cut Done. Releasing...")
            self.execute_tool_command("GRIPPER_OPEN", "RELEASING", plan_only=False)

        # 6. RELEASED -> RETREAT
        elif self.state == "RELEASING":
            self.get_logger().info("Released. Retreating...")
            self.execute_linear_pullback()

    def clear_octomap(self):
        if self.clear_octomap_client.service_is_ready():
            self.clear_octomap_client.call_async(Empty.Request())

    def execute_arm_move(self):
        self.state = "MOVING_ARM"
        msg = self.latest_target
        
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.arm_group_name
        goal_msg.request.num_planning_attempts = 15
        goal_msg.request.allowed_planning_time = 10.0
        goal_msg.request.max_velocity_scaling_factor = 0.2
        goal_msg.request.max_acceleration_scaling_factor = 0.2

        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.ee_link
        pc.constraint_region.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.04]))
        pc.constraint_region.primitive_poses.append(msg.pose)
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.ee_link
        oc.orientation = msg.pose.orientation
        oc.absolute_x_axis_tolerance = 3.14 
        oc.absolute_y_axis_tolerance = 3.14 
        oc.absolute_z_axis_tolerance = 1.0 
        oc.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)
        goal_msg.request.goal_constraints.append(constraints)

        self.move_group_client.send_goal_async(goal_msg).add_done_callback(self.arm_response_callback)

    def arm_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.state = "IDLE"
            return
        goal_handle.get_result_async().add_done_callback(self.arm_result_callback)

    def arm_result_callback(self, future):
        if future.result().result.error_code.val == 1:
            # Arm arrived. Trigger the Gripping sequence in the callback chain
            self.state = "GRIPPING"
            self.tool_result_callback(future) # Manually call next step
        else:
            self.state = "IDLE"

    # --- RETREAT LOGIC ---
    def execute_linear_pullback(self):
        self.state = "PULLING_BACK"
        retreat_pose = copy.deepcopy(self.latest_target)
        retreat_pose.pose.position.x -= 0.12 
        
        goal