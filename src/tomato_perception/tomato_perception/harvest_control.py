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
        self.cutter_group_name = "cutter_group"
        
        self.base_frame = "base_link"
        self.ee_link = "tcp_link"
        self.max_reach = 0.65 
        
        # --- JOINTS ---
        self.gripper_joints = ["left_gripper_joint", "right_gripper_joint"]
        self.cutter_joints = ["left_cutter_joint", "right_cutter_joint"]
        
        # --- ANGLES (Radians) ---
        # Gripper
        self.gripper_open_val = math.radians(17.0)
        self.gripper_hold_val = math.radians(5.0)   # Hold Tomato
        self.gripper_closed_val = math.radians(0.0) # Fully Closed (Rest)

        # Cutter
        self.cutter_open_left = math.radians(-40.0)
        self.cutter_open_right = math.radians(40.0)
        self.cutter_closed_val = math.radians(0.0)  # Cut/Rest

        # Arm Rest (0 deg)
        self.rest_joints_deg = [0.0, 0.0, 0.0, 0.0, 0.0] 
        self.rest_joints_rad = [math.radians(a) for a in self.rest_joints_deg]
        self.arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]

        self.target_sub = self.create_subscription(PoseStamped, '/harvest_target/pose', self.target_callback, 10)
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')
        self.clear_octomap_client = self.create_client(Empty, '/clear_octomap')

        self.state = "IDLE" 
        self.latest_target = None

        self.get_logger().info("Connecting to MoveIt Server...")
        self.move_group_client.wait_for_server()
        
        # --- UI: STARTUP BANNER ---
        print("\n" + "="*50)
        print("   HARVEST CONTROLLER ONLINE")
        print("   [MODE] : LIVE ACTION (REAL CUTTING)")
        print("   [STATUS]: Waiting for target...")
        print("="*50 + "\n")

    def target_callback(self, msg):
        if self.state != "IDLE": return
        dist = math.sqrt(msg.pose.position.x**2 + msg.pose.position.y**2 + msg.pose.position.z**2)
        if dist > self.max_reach: return

        self.latest_target = msg
        
        # --- UI: NEW CYCLE HEADER ---
        print("\n" + "-"*40)
        self.get_logger().info(">>> NEW HARVEST CYCLE DETECTED")
        self.get_logger().info(f"[PLAN] Target Distance: {dist:.2f} meters")
        
        # Step 1: Open Everything
        self.execute_tool_command("BOTH_OPEN", "PRE_APPROACH", plan_only=False)

    def execute_tool_command(self, command_type, next_state, plan_only=False):
        self.state = next_state
        goal_msg = MoveGroup.Goal()
        
        # --- COMMAND SELECTOR ---
        if command_type == "GRIPPER_HOLD":
            goal_msg.request.group_name = self.gripper_group_name
            joints = self.gripper_joints
            target_val = self.gripper_hold_val
            
        elif command_type == "GRIPPER_CLOSE": # Fully closed
            goal_msg.request.group_name = self.gripper_group_name
            joints = self.gripper_joints
            target_val = self.gripper_closed_val

        elif command_type == "CUTTER_CLOSE": # The Cut
            goal_msg.request.group_name = self.cutter_group_name
            joints = self.cutter_joints
            
        elif command_type == "BOTH_OPEN":
            goal_msg.request.group_name = self.cutter_group_name # We drive cutter here, gripper logic handled via iteration if needed
            joints = self.cutter_joints
            
        elif command_type == "BOTH_CLOSE": # For Rest
            goal_msg.request.group_name = self.cutter_group_name
            joints = self.cutter_joints

        # Options
        goal_msg.planning_options.plan_only = plan_only
        
        constraints = Constraints()
        
        # --- JOINT CONSTRAINTS ---
        if command_type == "BOTH_OPEN":
            # 1. Cutter Open
            jc1 = JointConstraint(joint_name="left_cutter_joint", position=self.cutter_open_left, tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            jc2 = JointConstraint(joint_name="right_cutter_joint", position=self.cutter_open_right, tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            constraints.joint_constraints.append(jc1)
            constraints.joint_constraints.append(jc2)
            # (We will open gripper in the callback)
            
        elif command_type == "BOTH_CLOSE" or command_type == "CUTTER_CLOSE":
            # Cutter Closed (0)
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
        # --- STATE MACHINE SEQUENCE ---

        # 1. CUTTER OPENED -> OPEN GRIPPER
        if self.state == "PRE_APPROACH":
            self.get_logger().info("[ACTION] Tools Opening...")
            # Reuse the single Open command but change state
            self.execute_single_gripper("OPEN", "APPROACHING")

        # 2. TOOLS READY -> MOVE ARM
        elif self.state == "APPROACHING":
            self.clear_octomap()
            self.get_logger().info("[ACTION] Moving Arm to Tomato...")
            self.execute_arm_move()

        # 3. ARM ARRIVED -> GRIP (HOLD)
        elif self.state == "GRIPPING":
            self.get_logger().info("[ACTION] Arrived at Target. Gripping...")
            self.execute_tool_command("GRIPPER_HOLD", "CUTTING", plan_only=False)

        # 4. GRIPPED -> PERFORM CUT (REAL)
        elif self.state == "CUTTING":
            time.sleep(0.5) # Small pause before cut
            self.get_logger().info("[ACTION] >>> EXECUTING CUT <<<")
            self.execute_tool_command("CUTTER_CLOSE", "RELEASING", plan_only=False)

        # 5. CUT DONE -> OPEN EVERYTHING
        elif self.state == "RELEASING":
            time.sleep(0.5) 
            self.get_logger().info("[ACTION] Releasing...")
            # We open cutter first
            self.execute_tool_command("BOTH_OPEN", "FINAL_OPEN", plan_only=False)
            
        # 6. CUTTER OPEN -> OPEN GRIPPER
        elif self.state == "FINAL_OPEN":
             self.execute_single_gripper("OPEN", "RETREATING")

        # 7. OPEN -> MOVE TO REST
        elif self.state == "RETREATING":
            self.get_logger().info("[ACTION] Retreating to Rest Position...")
            self.execute_move_to_rest()
            
        # 8. AT REST -> CLOSE EVERYTHING (RESET)
        elif self.state == "RESETTING":
            self.get_logger().info("[ACTION] Resetting Tools to Closed State...")
            self.execute_tool_command("BOTH_CLOSE", "FINAL_RESET", plan_only=False)

        # 9. CUTTER CLOSED -> CLOSE GRIPPER
        elif self.state == "FINAL_RESET":
            self.execute_single_gripper("CLOSE", "IDLE")
            self.get_logger().info("[SUCCESS] Cycle Complete. Robot Ready.")
            print("-" * 40 + "\n") # End separator

    # --- HELPER: SINGLE GRIPPER COMMAND ---
    def execute_single_gripper(self, action, next_state):
        self.state = next_state
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.gripper_group_name
        
        target = self.gripper_open_val if action == "OPEN" else self.gripper_closed_val
        
        constraints = Constraints()
        for j in self.gripper_joints:
            jc = JointConstraint(joint_name=j, position=float(target), tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            constraints.joint_constraints.append(jc)
        goal_msg.request.goal_constraints.append(constraints)
        
        self.move_group_client.send_goal_async(goal_msg).add_done_callback(self.tool_response_callback)


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
            self.state = "GRIPPING"
            self.tool_result_callback(future) 
        else:
            self.state = "IDLE"

    def execute_move_to_rest(self):
        self.state = "MOVING_TO_REST"
        self.clear_octomap()
        
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.arm_group_name
        goal_msg.request.max_velocity_scaling_factor = 0.4
        
        constraints = Constraints()
        for i, joint_name in enumerate(self.arm_joint_names):
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = self.rest_joints_rad[i]
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
            
        goal_msg.request.goal_constraints.append(constraints)

        self.move_group_client.send_goal_async(goal_msg).add_done_callback(self.rest_response_callback)

    def rest_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.state = "IDLE"
            return
        goal_handle.get_result_async().add_done_callback(self.rest_result_callback)

    def rest_result_callback(self, future):
        self.get_logger().info("[ACTION] Robot Home. Closing Tools...")
        self.state = "RESETTING"
        self.tool_result_callback(future)

def main(args=None):
    rclpy.init(args=args)
    node = HarvestControlNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()