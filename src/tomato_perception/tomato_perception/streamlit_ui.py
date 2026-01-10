import streamlit as st
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from threading import Thread, Lock
import time
import json
import atexit

class TomatoUINode(Node):
    def __init__(self):
        super().__init__('tomato_ui_node')
        self.bridge = CvBridge()
        
        # Data storage
        self.latest_display_image = None
        self.total_tomatoes = 0
        self.ripe_tomatoes = 0
        self.stems_detected = 0
        self.latest_target = None
        self.harvest_status = "IDLE"
        self.stem_cut_point = None
        self.tomato_grip_point = None
        self.last_update = time.time()
        
        # Subscribers
        self.display_image_sub = self.create_subscription(
            Image, 
            '/tomato/display_image', 
            self.display_image_callback, 
            10
        )
        
        self.stats_sub = self.create_subscription(
            String,
            '/tomato/stats',
            self.stats_callback,
            10
        )
        
        self.target_sub = self.create_subscription(
            PoseStamped,
            '/harvest_target/pose',
            self.target_callback,
            10
        )
        
        self.grip_point_sub = self.create_subscription(
            String,
            '/tomato/grip_point',
            self.grip_point_callback,
            10
        )
        
        self.status_sub = self.create_subscription(
            String,
            '/harvest_status',
            self.status_callback,
            10
        )
        
    def display_image_callback(self, msg):
        try:
            self.latest_display_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.last_update = time.time()
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
    
    def stats_callback(self, msg):
        try:
            stats = json.loads(msg.data)
            self.total_tomatoes = stats.get('total', 0)
            self.ripe_tomatoes = stats.get('ripe', 0)
            self.stems_detected = stats.get('stems', 0)
        except Exception as e:
            self.get_logger().error(f"Stats parsing error: {e}")
    
    def target_callback(self, msg):
        self.latest_target = msg
        # Extract stem cut point (this is the stem position with orientation)
        self.stem_cut_point = {
            'x': msg.pose.position.x,
            'y': msg.pose.position.y,
            'z': msg.pose.position.z,
            'qz': msg.pose.orientation.z,
            'qw': msg.pose.orientation.w
        }
    
    def grip_point_callback(self, msg):
        try:
            grip_data = json.loads(msg.data)
            self.tomato_grip_point = grip_data
        except Exception as e:
            self.get_logger().error(f"Grip point parsing error: {e}")
    
    def status_callback(self, msg):
        self.harvest_status = msg.data

# Initialize ROS2 in a separate thread
def ros_spin(node):
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except Exception as e:
        print(f"ROS spin error: {e}")
    finally:
        executor.shutdown()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Cherry Tomato Harvest System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3.5rem;
            font-weight: bold;
            color: #2E7D32;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .status-online {
            color: #4CAF50;
            font-weight: bold;
            font-size: 1.2rem;
        }
        .status-offline {
            color: #F44336;
            font-weight: bold;
            font-size: 1.2rem;
        }
        .harvest-status {
            font-size: 1.3rem;
            font-weight: bold;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            margin: 1rem 0;
        }
        .status-idle {
            background-color: #E3F2FD;
            color: #1976D2;
        }
        .status-active {
            background-color: #FFF3E0;
            color: #F57C00;
        }
        .status-success {
            background-color: #E8F5E9;
            color: #388E3C;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.2rem;
            border-radius: 0.8rem;
            border-left: 4px solid #2E7D32;
            margin: 0.5rem 0;
        }
        .coord-box {
            background-color: #263238;
            color: #00E676;
            padding: 1rem;
            border-radius: 0.5rem;
            font-family: 'Courier New', monospace;
            font-size: 1rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize ROS2
    if 'ros_initialized' not in st.session_state:
        # Check if rclpy is already initialized
        if not rclpy.ok():
            rclpy.init()
        st.session_state.node = TomatoUINode()
        st.session_state.ros_thread = Thread(target=ros_spin, args=(st.session_state.node,), daemon=True)
        st.session_state.ros_thread.start()
        st.session_state.ros_initialized = True
        
        # Register cleanup
        def cleanup():
            try:
                if hasattr(st.session_state, 'node'):
                    st.session_state.node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except:
                pass
        
        atexit.register(cleanup)
    
    node = st.session_state.node
    
    # Header
    st.markdown('<p class="main-header">🍅 Cherry Tomato Autonomous Harvest System</p>', unsafe_allow_html=True)
    
    # Main Layout
    col_left, col_right = st.columns([1.2, 1])
    
    # LEFT COLUMN - Camera Feed
    with col_left:
        st.markdown("### 📹 Live Detection Feed")
        
        # System Status Badge
        time_since_update = time.time() - node.last_update
        if time_since_update < 2.0:
            st.markdown('<p class="status-online">● CAMERA ONLINE</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-offline">● CAMERA OFFLINE</p>', unsafe_allow_html=True)
        
        image_placeholder = st.empty()
        
        if node.latest_display_image is not None:
            # Convert BGR to RGB for display
            rgb_image = cv2.cvtColor(node.latest_display_image, cv2.COLOR_BGR2RGB)
            image_placeholder.image(rgb_image, use_container_width=True, channels="RGB")
        else:
            image_placeholder.info("⏳ Waiting for camera feed...")
    
    # RIGHT COLUMN - Information Panel
    with col_right:
        # Harvest Status
        st.markdown("### 🤖 Harvest Status")
        status_class = "status-idle"
        if node.harvest_status in ["MOVING_ARM", "APPROACHING", "GRIPPING", "CUTTING", "CUTTER_RELEASE"]:
            status_class = "status-active"
        elif node.harvest_status in ["AT_REST", "CLOSING_GRIPPER", "CLOSING_CUTTER"]:
            status_class = "status-success"
        elif node.harvest_status in ["PRE_APPROACH", "RETREATING", "MOVING_TO_REST"]:
            status_class = "status-active"
        
        st.markdown(f'<div class="harvest-status {status_class}">{node.harvest_status}</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detection Statistics
        st.markdown("### 📊 Detection Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🍅 Total", node.total_tomatoes)
        with col2:
            st.metric("✅ Ripe", node.ripe_tomatoes)
        with col3:
            st.metric("🌿 Stems", node.stems_detected)
        
        st.markdown("---")
        
        # Target Cherry Tomato (Grip Point)
        st.markdown("### 🎯 Target Cherry Tomato")
        if node.tomato_grip_point:
            st.markdown(f"""
                <div class="coord-box">
                <strong>Position:</strong><br>
                X: {node.tomato_grip_point['x']:7.3f} m<br>
                Y: {node.tomato_grip_point['y']:7.3f} m<br>
                Z: {node.tomato_grip_point['z']:7.3f} m
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No target detected")
        
        st.markdown("---")
        
        # Stem Cut Point
        st.markdown("### ✂️ Stem Cut Point")
        if node.stem_cut_point:
            st.markdown(f"""
                <div class="coord-box">
                <strong>Position:</strong><br>
                X: {node.stem_cut_point['x']:7.3f} m<br>
                Y: {node.stem_cut_point['y']:7.3f} m<br>
                Z: {node.stem_cut_point['z']:7.3f} m<br>
                <br>
                <strong>Orientation:</strong><br>
                Z: {node.stem_cut_point['qz']:7.3f}<br>
                W: {node.stem_cut_point['qw']:7.3f}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No cut point detected")
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        refresh_rate = st.slider("Refresh Rate (Hz)", 1, 30, 15, 
                                 help="Adjust UI update frequency")
        st.caption(f"⏱️ Update interval: {1000/refresh_rate:.0f}ms")
    
    # Auto-refresh
    time.sleep(1.0 / refresh_rate)
    st.rerun()

if __name__ == '__main__':
    main()