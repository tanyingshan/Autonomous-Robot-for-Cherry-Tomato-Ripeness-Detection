import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import PositionConstraint, OrientationConstraint
from std_srvs.srv import Empty as EmptySrv
from std_msgs.msg import Empty, String
import math
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
        
        self.gripper_joints = ["left_gripper_joint", "right_gripper_joint"]
        self.cutter_joints = ["left_cutter_joint", "right_cutter_joint"]
        
        # Angles
        self.gripper_open_val = math.radians(17.0)
        self.gripper_hold_val = math.radians(5.0)   
        self.gripper_closed_val = math.radians(0.0) 
        self.cutter_open_left = math.radians(-40.0)
        self.cutter_open_right = math.radians(40.0)
        self.cutter_closed_val = math.radians(0.0)  
        
        self.rest_joints_rad = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
        
        self.target_sub = self.create_subscription(PoseStamped, '/harvest_target/pose', self.target_callback, 10)
        
        # Publishers
        self.log_pub = self.create_publisher(Empty, '/harvest_trigger', 10)
        self.status_pub = self.create_publisher(String, '/harvest_status', 10)
        
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')
        self.clear_octomap_client = self.create_client(EmptySrv, '/clear_octomap')
        
        self.state = "IDLE" 
        self.latest_target = None
        
        self.get_logger().info("Connecting to MoveIt Server...")
        self.move_group_client.wait_for_server()
        
        print("\n" + "="*60)
        print("   HARVEST CONTROLLER ONLINE")
        print("   [MODE] : LIVE ACTION (REAL CUTTING)")
        print("   [LOGS] : Will trigger save on success")
        print("="*60 + "\n")
        
        # Publish initial status
        self.publish_status("IDLE")
    
    def publish_status(self, status):
        """Publish current harvest status"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f"[STATUS] {status}")
    
    def target_callback(self, msg):
        if self.state != "IDLE": 
            return
        dist = math.sqrt(msg.pose.position.x**2 + msg.pose.position.y**2 + msg.pose.position.z**2)
        if dist > self.max_reach: 
            return
        
        self.latest_target = msg
        print("-" * 50)
        self.get_logger().info(">>> NEW HARVEST CYCLE STARTED")
        self.get_logger().info(f"[PLAN] Target Valid. Distance: {dist:.2f} meters")
        
        self.execute_tool_command("BOTH_OPEN", "PRE_APPROACH", plan_only=False)
    
    def execute_tool_command(self, command_type, next_state, plan_only=False):
        self.state = next_state
        self.publish_status(next_state)
        
        goal_msg = MoveGroup.Goal()
        
        if command_type == "GRIPPER_HOLD":
            goal_msg.request.group_name = self.gripper_group_name
            target_val = self.gripper_hold_val
        elif command_type == "GRIPPER_CLOSE": 
            goal_msg.request.group_name = self.gripper_group_name
            target_val = self.gripper_closed_val
        elif command_type == "GRIPPER_OPEN":
            goal_msg.request.group_name = self.gripper_group_name
            target_val = self.gripper_open_val
        elif command_type == "CUTTER_CLOSE": 
            goal_msg.request.group_name = self.cutter_group_name
        elif command_type == "BOTH_OPEN":
            goal_msg.request.group_name = self.cutter_group_name 
        elif command_type == "BOTH_CLOSE": 
            goal_msg.request.group_name = self.cutter_group_name
        
        goal_msg.planning_options.plan_only = plan_only
        constraints = Constraints()
        
        if command_type == "BOTH_OPEN":
            jc1 = JointConstraint(joint_name="left_cutter_joint", position=self.cutter_open_left, tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            jc2 = JointConstraint(joint_name="right_cutter_joint", position=self.cutter_open_right, tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            constraints.joint_constraints.append(jc1)
            constraints.joint_constraints.append(jc2)
        elif command_type == "BOTH_CLOSE" or command_type == "CUTTER_CLOSE":
            for j in self.cutter_joints:
                jc = JointConstraint(joint_name=j, position=self.cutter_closed_val, tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
                constraints.joint_constraints.append(jc)
        elif command_type in ["GRIPPER_HOLD", "GRIPPER_CLOSE", "GRIPPER_OPEN"]:
            for j in self.gripper_joints:
                jc = JointConstraint(joint_name=j, position=float(target_val), tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
                constraints.joint_constraints.append(jc)
        
        goal_msg.request.goal_constraints.append(constraints)
        self.move_group_client.send_goal_async(goal_msg).add_done_callback(self.tool_response_callback)
    
    def tool_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("[ERROR] Goal not accepted!")
            self.state = "IDLE"
            self.publish_status("IDLE")
            return
        goal_handle.get_result_async().add_done_callback(self.tool_result_callback)
    
    def tool_result_callback(self, future):
        if self.state == "PRE_APPROACH":
            self.get_logger().info("[ACTION] Tools Opened. Opening Gripper...")
            self.execute_tool_command("GRIPPER_OPEN", "APPROACHING", plan_only=False)
            
        elif self.state == "APPROACHING":
            self.clear_octomap()
            self.get_logger().info("[ACTION] Moving Arm to Tomato...")
            self.execute_arm_move()
            
        elif self.state == "GRIPPING":
            self.get_logger().info("[ACTION] Arrived. Gripping Tomato...")
            self.execute_tool_command("GRIPPER_HOLD", "CUTTING", plan_only=False)
            
        elif self.state == "CUTTING":
            time.sleep(0.5) 
            self.get_logger().info("[ACTION] >>> EXECUTING CUT <<<")
            self.execute_tool_command("CUTTER_CLOSE", "CUTTER_RELEASE", plan_only=False)
            
        elif self.state == "CUTTER_RELEASE":
            time.sleep(0.5)
            self.get_logger().info("[ACTION] Opening Cutter Only...")
            self.execute_cutter_only("OPEN", "RETREATING")
            
        elif self.state == "RETREATING":
            self.get_logger().info("[ACTION] Retreating to Rest Position (Holding Tomato)...")
            self.execute_move_to_rest()
            
        elif self.state == "AT_REST":
            time.sleep(0.3)
            self.get_logger().info("[ACTION] At Rest Position. Releasing Tomato into Basket...")
            self.execute_tool_command("GRIPPER_OPEN", "CLOSING_GRIPPER", plan_only=False)
            
        elif self.state == "CLOSING_GRIPPER":
            time.sleep(0.5)
            self.get_logger().info("[ACTION] Closing Gripper...")
            self.execute_tool_command("GRIPPER_CLOSE", "CLOSING_CUTTER", plan_only=False)
            
        elif self.state == "CLOSING_CUTTER":
            time.sleep(0.3)
            self.get_logger().info("[ACTION] Closing Cutter to Complete Reset...")
            self.execute_cutter_only("CLOSE", "IDLE")
            
            # Trigger log saving
            self.get_logger().info("[SUCCESS] Cycle Complete. Tomato in Basket!")
            self.log_pub.publish(Empty())
            print("-" * 50 + "\n")
    
    def execute_cutter_only(self, action, next_state):
        """Open or close only the cutter while keeping gripper state"""
        self.state = next_state
        self.publish_status(next_state)
        
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.cutter_group_name
        
        constraints = Constraints()
        if action == "OPEN":
            jc1 = JointConstraint(joint_name="left_cutter_joint", position=self.cutter_open_left, tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            jc2 = JointConstraint(joint_name="right_cutter_joint", position=self.cutter_open_right, tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            constraints.joint_constraints.append(jc1)
            constraints.joint_constraints.append(jc2)
        else:
            for j in self.cutter_joints:
                jc = JointConstraint(joint_name=j, position=self.cutter_closed_val, tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
                constraints.joint_constraints.append(jc)
        
        goal_msg.request.goal_constraints.append(constraints)
        self.move_group_client.send_goal_async(goal_msg).add_done_callback(self.tool_response_callback)
    
    def clear_octomap(self):
        if self.clear_octomap_client.service_is_ready():
            self.clear_octomap_client.call_async(EmptySrv.Request())
    
    def execute_arm_move(self):
        self.state = "MOVING_ARM"
        self.publish_status("MOVING_ARM")
        
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
            self.get_logger().error("[ERROR] Arm movement not accepted!")
            self.state = "IDLE"
            self.publish_status("IDLE")
            return
        goal_handle.get_result_async().add_done_callback(self.arm_result_callback)
    
    def arm_result_callback(self, future):
        if future.result().result.error_code.val == 1:
            self.get_logger().info("[SUCCESS] Arm reached target!")
            self.state = "GRIPPING"
            self.publish_status("GRIPPING")
            self.tool_result_callback(future) 
        else:
            self.get_logger().error(f"[ERROR] Arm movement failed with code: {future.result().result.error_code.val}")
            self.state = "IDLE"
            self.publish_status("IDLE")
    
    def execute_move_to_rest(self):
        self.state = "MOVING_TO_REST"
        self.publish_status("MOVING_TO_REST")
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
            self.get_logger().error("[ERROR] Rest position not accepted!")
            self.state = "IDLE"
            self.publish_status("IDLE")
            return
        goal_handle.get_result_async().add_done_callback(self.rest_result_callback)
    
    def rest_result_callback(self, future):
        if future.result().result.error_code.val == 1:
            self.get_logger().info("[SUCCESS] Reached rest position!")
            self.state = "AT_REST"
            self.publish_status("AT_REST")
            self.tool_result_callback(future)
        else:
            self.get_logger().error("[ERROR] Failed to reach rest position!")
            self.state = "IDLE"
            self.publish_status("IDLE")

def main(args=None):
    rclpy.init(args=args)
    node = HarvestControlNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()