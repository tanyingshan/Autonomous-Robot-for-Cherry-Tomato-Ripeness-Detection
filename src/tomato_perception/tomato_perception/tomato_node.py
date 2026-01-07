import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Quaternion
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math

# TF2 Imports
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

class TomatoPerceptionNode(Node):
    def __init__(self):
        super().__init__('tomato_perception_node')
        
        # Models
        package_share_directory = get_package_share_directory('tomato_perception')
        ripeness_path = os.path.join(package_share_directory, 'models', 'cherry_tomato_ripeness_detection_model.pt')
        self.model_ripeness = YOLO(ripeness_path)
        stem_path = os.path.join(package_share_directory, 'models', 'stem_segmentation_model.pt')
        self.model_stem = YOLO(stem_path)
        
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.depth_image = None
        self.frame_count = 0 

        # TF Buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        cv2.namedWindow("Cherry Tomato Ripeness Detection", cv2.WINDOW_NORMAL)
        cv2.startWindowThread()

        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, self.sensor_qos)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        self.target_pub = self.create_publisher(PoseStamped, '/harvest_target/pose', 10)

        # --- UI: STARTUP BANNER ---
        print("\n" + "="*60)
        print("   TOMATO PERCEPTION NODE | STATUS: ONLINE")
        print("   [VISUALS]: Bounding Box + Green Stem Mask Overlay")
        print("   [LOGIC]  : SMART DEPTH SEARCH (Spiral Scan)")
        print("="*60 + "\n")

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_quaternion_from_angle(self, angle_rad):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(angle_rad / 2.0)
        q.w = math.cos(angle_rad / 2.0)
        return q

    # --- HELPER: SPIRAL SEARCH FOR VALID DEPTH ---
    def get_valid_depth(self, u, v, max_radius=10):
        """Searches neighbors if center pixel is 0/invalid"""
        if self.depth_image is None: return 0.0
        
        h, w = self.depth_image.shape
        # Check center first
        z = float(self.depth_image[v, u])
        if z > 100 and z < 2000: return z
        
        # Spiral search
        for r in range(1, max_radius + 1):
            # Check 8 neighbors at radius r
            neighbors = [
                (u-r, v), (u+r, v), (u, v-r), (u, v+r),
                (u-r, v-r), (u+r, v+r), (u-r, v+r), (u+r, v-r)
            ]
            valid_depths = []
            for nu, nv in neighbors:
                if 0 <= nu < w and 0 <= nv < h:
                    val = float(self.depth_image[nv, nu])
                    if val > 100 and val < 2000:
                        valid_depths.append(val)
            
            if valid_depths:
                return np.mean(valid_depths) # Return average of valid neighbors
        
        return 0.0 # Failed

    # --- UI HELPER: Calculate 3D Point for Logging ---
    def calculate_3d_point(self, u, v, depth_val):
        if depth_val <= 0: return None
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        x = (u - cx) * depth_val / fx / 1000.0
        y = (v - cy) * depth_val / fy / 1000.0
        z = depth_val / 1000.0
        return (x, y, z)

    def rgb_callback(self, msg):
        if self.depth_image is None or self.camera_matrix is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return
        
        results_ripeness = self.model_ripeness(cv_image, conf=0.5, verbose=False)
        display_frame = results_ripeness[0].plot() 
        
        total_tomatoes = len(results_ripeness[0].boxes)
        ripe_count = 0
        stems_segmented = 0

        for box in results_ripeness[0].boxes:
            cls_id = int(box.cls[0])
            label = self.model_ripeness.names[cls_id].lower()
            
            if 'ripe' in label and not 'semi' in label:
                ripe_count += 1
                
                # --- TOMATO CENTER ---
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tomato_cx = (x1 + x2) // 2
                tomato_cy = (y1 + y2) // 2
                cv2.circle(display_frame, (tomato_cx, tomato_cy), 5, (0, 0, 255), -1)

                pad_top, pad_side, pad_bottom = 70, 40, 10
                roi_x1 = max(0, x1 - pad_side)
                roi_y1 = max(0, y1 - pad_top)
                roi_x2 = min(cv_image.shape[1], x2 + pad_side)
                roi_y2 = min(cv_image.shape[0], y2 + pad_bottom)
                
                roi_img = cv_image[roi_y1:roi_y2, roi_x1:roi_x2]
                results_stem = self.model_stem(roi_img, conf=0.2, verbose=False, retina_masks=True)
                
                if len(results_stem[0].boxes) > 0:
                    stem_res = results_stem[0]
                    if stem_res.masks is None: continue

                    mask_binary = stem_res.masks.data[0].cpu().numpy()
                    mask_binary = cv2.resize(mask_binary, (roi_img.shape[1], roi_img.shape[0]))
                    
                    # --- UI: GREEN MASK SHADE ---
                    roi_display = display_frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    green_mask = np.zeros_like(roi_display)
                    mask_indices = mask_binary > 0.5
                    
                    if np.any(mask_indices):
                        green_mask[mask_indices] = [0, 255, 0] 
                        cv2.addWeighted(roi_display, 0.6, green_mask, 0.4, 0, roi_display)

                    mask_coords = np.argwhere(mask_binary > 0.5) 

                    if len(mask_coords) > 0:
                        stems_segmented += 1
                        v_roi, u_roi = np.mean(mask_coords, axis=0).astype(int)
                        u_stem = u_roi + roi_x1
                        v_stem = v_roi + roi_y1

                        cv2.circle(display_frame, (u_stem, v_stem), 6, (0, 255, 0), -1)
                        cv2.line(display_frame, (tomato_cx, tomato_cy), (u_stem, v_stem), (255, 255, 0), 1)

                        y_coords = mask_coords[:, 0]
                        x_coords = mask_coords[:, 1]
                        if len(x_coords) > 5: 
                            cov_mat = np.cov(x_coords, y_coords)
                            eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
                            major_axis = eig_vecs[:, 1] 
                            stem_angle = np.arctan2(major_axis[1], major_axis[0])
                            if major_axis[1] > 0: stem_angle += np.pi
                        else:
                            stem_angle = -1.57 

                        try:
                            # --- SMART DEPTH SEARCH ---
                            # 1. Get Reliable Tomato Depth (Spiral Search if center is bad)
                            z_tomato = self.get_valid_depth(tomato_cx, tomato_cy, max_radius=15)
                            
                            # 2. Get Raw Stem Depth
                            z_stem_raw = self.get_valid_depth(u_stem, v_stem, max_radius=5)

                            # 3. LOGIC: Force Stem Z to match Tomato Z
                            # Stems are too thin for depth cameras. Trust the tomato body.
                            if z_tomato > 0:
                                z_final = z_tomato # Use TOMATO depth for everything
                            elif z_stem_raw > 0:
                                z_final = z_stem_raw # Fallback
                            else:
                                continue # Both invalid, skip

                            # Calculate 3D
                            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
                            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
                            
                            # CRITICAL: Use STEM (u,v) but TOMATO (z)
                            x_opt = (u_stem - cx) * z_final / fx / 1000.0
                            y_opt = (v_stem - cy) * z_final / fy / 1000.0
                            z_opt = z_final / 1000.0

                            pose_optical = PoseStamped()
                            pose_optical.header.stamp = self.get_clock().now().to_msg()
                            pose_optical.header.frame_id = "camera_depth_optical_frame"
                            pose_optical.pose.position.x = x_opt
                            pose_optical.pose.position.y = y_opt
                            pose_optical.pose.position.z = z_opt
                            pose_optical.pose.orientation = self.get_quaternion_from_angle(stem_angle)

                            try:
                                transform = self.tf_buffer.lookup_transform(
                                    'base_link', 'camera_depth_optical_frame', rclpy.time.Time())
                                
                                pose_base = do_transform_pose(pose_optical.pose, transform)
                                
                                # --- LOGGING (Consistent Data) ---
                                grip_pt = self.calculate_3d_point(tomato_cx, tomato_cy, z_final)
                                
                                grip_str = f"({grip_pt[0]:6.2f}, {grip_pt[1]:6.2f}, {grip_pt[2]:6.2f})" if grip_pt else "   N/A   "
                                cut_str =  f"({pose_base.position.x:6.2f}, {pose_base.position.y:6.2f}, {pose_base.position.z:6.2f})"
                                
                                self.get_logger().info(
                                    f"\n[VISION] TARGET LOCK ACQUIRED:\n"
                                    f"   > GRIP (Tomato) : {grip_str}\n"
                                    f"   > CUT  (Stem)   : {cut_str}\n"
                                    f"   > APPROACH ANGLE: {np.degrees(stem_angle):.1f} deg"
                                )

                                target_msg = PoseStamped()
                                target_msg.header.stamp = self.get_clock().now().to_msg()
                                target_msg.header.frame_id = "base_link"
                                target_msg.pose = pose_base
                                self.target_pub.publish(target_msg)
                                
                            except Exception:
                                pass
                        except Exception:
                            pass

        self.frame_count += 1
        if self.frame_count % 60 == 0:
             if total_tomatoes > 0:
                self.get_logger().info(f"[STATUS] Visible: {total_tomatoes} | Ripe: {ripe_count} | Segments: {stems_segmented}")

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