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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# TF2 Imports
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

class TomatoPerceptionNode(Node):
    def __init__(self):
        super().__init__('tomato_perception_node')
        
        # 1. Initialize Models
        package_share_directory = get_package_share_directory('tomato_perception')
        ripeness_path = os.path.join(package_share_directory, 'models', 'cherry_tomato_ripeness_detection_model.pt')
        self.model_ripeness = YOLO(ripeness_path)
        
        stem_path = os.path.join(package_share_directory, 'models', 'stem_segmentation_model.pt')
        self.model_stem = YOLO(stem_path)
        
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.depth_image = None
        self.frame_count = 0  # <--- Restore Frame Counter

        # TF Buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 2. QoS Profile
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        cv2.namedWindow("Cherry Tomato Ripeness Detection", cv2.WINDOW_NORMAL)
        cv2.startWindowThread()

        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, self.sensor_qos)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        self.target_pub = self.create_publisher(PoseStamped, '/harvest_target/pose', 10)

        self.get_logger().info("System Ready. Logs + TF Correction Enabled.")

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def rgb_callback(self, msg):
        if self.depth_image is None or self.camera_matrix is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return
        
        # 1. Detection
        results_ripeness = self.model_ripeness(cv_image, conf=0.3, verbose=False)
        display_frame = results_ripeness[0].plot()
        
        # --- LOGGING COUNTERS RESTORED ---
        total_tomatoes = len(results_ripeness[0].boxes)
        ripe_count = 0
        semi_count = 0
        unripe_count = 0
        stems_segmented = 0

        for box in results_ripeness[0].boxes:
            cls_id = int(box.cls[0])
            label = self.model_ripeness.names[cls_id].lower()
            
            # Count Classes
            if 'semi' in label:
                semi_count += 1
            elif 'unripe' in label:
                unripe_count += 1
            elif 'ripe' in label:
                ripe_count += 1
                
                # 2. ROI Processing (Only for Ripe)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pad_top, pad_side, pad_bottom = 70, 40, 10
                
                roi_x1 = max(0, x1 - pad_side)
                roi_y1 = max(0, y1 - pad_top)
                roi_x2 = min(cv_image.shape[1], x2 + pad_side)
                roi_y2 = min(cv_image.shape[0], y2 + pad_bottom)
                
                roi_img = cv_image[roi_y1:roi_y2, roi_x1:roi_x2]
                results_stem = self.model_stem(roi_img, conf=0.15, verbose=False, retina_masks=True)
                
                if len(results_stem[0].boxes) > 0:
                    stem_res = results_stem[0]
                    if stem_res.masks is None: continue

                    mask_binary = stem_res.masks.data[0].cpu().numpy()
                    mask_binary = cv2.resize(mask_binary, (roi_img.shape[1], roi_img.shape[0]))
                    mask_coords = np.argwhere(mask_binary > 0.5)

                    if len(mask_coords) > 0:
                        stems_segmented += 1
                        
                        v_roi, u_roi = np.mean(mask_coords, axis=0).astype(int)
                        u_full = u_roi + roi_x1
                        v_full = v_roi + roi_y1
                        
                        # Visualization
                        cv2.circle(display_frame, (u_full, v_full), 6, (0, 0, 255), -1)
                        cv2.putText(display_frame, "STEM", (u_full+10, v_full-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        # 3. 3D Math & Publish
                        try:
                            if 0 <= v_full < self.depth_image.shape[0] and 0 <= u_full < self.depth_image.shape[1]:
                                z_mm = float(self.depth_image[v_full, u_full])
                                if z_mm > 0:
                                    fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
                                    cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
                                    
                                    # Optical Frame Math (Z = Forward, X = Right)
                                    x_opt = (u_full - cx) * z_mm / fx / 1000.0
                                    y_opt = (v_full - cy) * z_mm / fy / 1000.0
                                    z_opt = z_mm / 1000.0

                                    pose_optical = PoseStamped()
                                    pose_optical.header.stamp = self.get_clock().now().to_msg()
                                    pose_optical.header.frame_id = "camera_depth_optical_frame"
                                    pose_optical.pose.position.x = x_opt
                                    pose_optical.pose.position.y = y_opt
                                    pose_optical.pose.position.z = z_opt
                                    pose_optical.pose.orientation.w = 1.0

                                    # Transform to Robot Base
                                    try:
                                        transform = self.tf_buffer.lookup_transform(
                                            'base_link', 
                                            'camera_depth_optical_frame',
                                            rclpy.time.Time())
                                        
                                        pose_base = do_transform_pose(pose_optical.pose, transform)
                                        
                                        target_msg = PoseStamped()
                                        target_msg.header.stamp = self.get_clock().now().to_msg()
                                        target_msg.header.frame_id = "base_link"
                                        target_msg.pose = pose_base
                                        
                                        self.target_pub.publish(target_msg)
                                        
                                    except Exception:
                                        pass
                        except Exception:
                            pass

        # --- LOG THROTTLING RESTORED ---
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            if total_tomatoes > 0:
                self.get_logger().info(
                    f"Status: Detected {total_tomatoes} | Ripe: {ripe_count} | Semi: {semi_count} | Unripe: {unripe_count} | Stems: {stems_segmented}"
                )
            else:
                self.get_logger().info("Searching for tomatoes...", throttle_duration_sec=2.0)

        cv2.imshow("Cherry Tomato Ripeness Detection", display_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = TomatoPerceptionNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()