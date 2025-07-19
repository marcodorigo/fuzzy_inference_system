from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='fuzzy_inference_system',
            executable='ee_distance_metrics_node',
            name='ee_distance_metrics_node',
            output='screen'
        ),
        Node(
            package='fuzzy_inference_system',
            executable='fuzzy_safety_node',
            name='fuzzy_safety_node',
            output='screen'
        ),
    ])