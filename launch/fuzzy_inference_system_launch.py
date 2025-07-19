from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Path to the scene_params.yaml file
    config_file_path = os.path.join(
        '/home/marco/UR5_ws/src/common_config/config', 'scene_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='fuzzy_inference_system',
            executable='ee_distance_metrics_node',
            name='ee_distance_metrics_node',
            output='screen',
            parameters=[config_file_path]  # Add the config file
        ),
        Node(
            package='fuzzy_inference_system',
            executable='fuzzy_safety_node',
            name='fuzzy_safety_node',
            output='screen',
            parameters=[config_file_path]  # Add the config file
        ),
    ])