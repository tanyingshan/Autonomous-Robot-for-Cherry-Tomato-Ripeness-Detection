from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    # Find package shares
    cocoabot_bringup_share = FindPackageShare('cocoabot_bringup')
    tomato_perception_share = FindPackageShare('tomato_perception')
    
    # Launch cocoabot
    cocoabot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                cocoabot_bringup_share,
                'launch',
                'cocoabot.launch.py'
            ])
        ])
    )
    
    # Tomato perception node
    tomato_node = Node(
        package='tomato_perception',
        executable='tomato_node',
        name='tomato_perception_node',
        output='screen'
    )
    
    # Harvest control node
    harvest_control_node = Node(
        package='tomato_perception',
        executable='harvest_control',
        name='harvest_control_node',
        output='screen'
    )
    
    # Streamlit UI (as a process)
    streamlit_ui = ExecuteProcess(
        cmd=[
            'streamlit', 'run',
            PathJoinSubstitution([
                tomato_perception_share,
                'streamlit_ui.py'
            ])
        ],
        output='screen'
    )
    
    return LaunchDescription([
        cocoabot_launch,
        tomato_node,
        harvest_control_node,
        streamlit_ui
    ])