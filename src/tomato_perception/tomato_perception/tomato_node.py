import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory
# --- NEW IMPORT ---
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class TomatoPerceptionNode(Node):
    def __init__(self):
        super().__init__('tomato_perception_node')
        
        # 1. Initialize YOLO
        package_share_directory = get_package_share_directory('tomato_perception')
        model_path = os.path.join(package_share_directory, 'models', 'cherry_tomato_ripeness_detection_model.pt')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        
        self.camera_matrix = None
        self.depth_image = None

        # --- 2. DEFINE THE SENSOR QoS PROFILE ---
        # This matches what the Orbbec Gemini is sending
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 3. GUI Setup
        cv2.namedWindow("Cherry Tomato Detection", cv2.WINDOW_NORMAL)
        cv2.startWindowThread()

        # 4. ROS Subscribers (Updated with QoS)
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 10)
        
        # We apply the Best Effort QoS here specifically for depth
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, self.sensor_qos)
        
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        
        self.target_pub = self.create_publisher(PoseStamped, '/harvest_target/pose', 10)

        self.get_logger().info("Node Started! Looking for Depth via Best Effort QoS...")

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.get_logger().info("Camera Intrinsics Captured.")

    def depth_callback(self, msg):
        # Once this runs, the warning in Terminal 2 will disappear!
        if self.depth_image is None:
            self.get_logger().info("Depth connection established!")
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def rgb_callback(self, msg):
        if self.depth_image is None or self.camera_matrix is None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(cv_image, conf=0.5, verbose=False)
        res = results[0]
        
        # Display Window
        annotated_frame = res.plot()
        cv2.imshow("Cherry Tomato Detection", annotated_frame)
        cv2.waitKey(1) 

        # 3D Calculation logic
        if len(res.boxes) > 0:
            # Get the first box (closest or highest confidence)
            box = res.boxes[0]
            
            # Get center coordinates of the bounding box (u, v)
            # box.xywh returns [center_x, center_y, width, height]
            u = int(box.xywh[0][0]) 
            v = int(box.xywh[0][1])
            
            self.get_logger().info(f"Targeting Box Center: Pixel ({u}, {v})")

            try:
                z_mm = float(self.depth_image[v, u])
                
                if z_mm > 0:
                    # Camera intrinsics
                    fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
                    cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

                    # Convert to 3D Meters
                    tx = (u - cx) * z_mm / fx / 1000.0
                    ty = (v - cy) * z_mm / fy / 1000.0
                    tz = z_mm / 1000.0

                    # Create Pose message
                    target = PoseStamped()
                    target.header.stamp = self.get_clock().now().to_msg()
                    target.header.frame_id = "camera_link"
                    target.pose.position.x = tx
                    target.pose.position.y = ty
                    target.pose.position.z = tz
                    target.pose.orientation.w = 1.0
                    
                    self.target_pub.publish(target)
                    self.get_logger().info(f"!!! PUBLISHED !!! X:{tx:.2f} Y:{ty:.2f} Z:{tz:.2f}")
                else:
                    self.get_logger().warn(f"Depth at ({u},{v}) is 0.0. Move further away.")
            except Exception as e:
                self.get_logger().error(f"Calculation Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TomatoPerceptionNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()