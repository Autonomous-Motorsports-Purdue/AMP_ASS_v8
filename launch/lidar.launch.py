
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='2368'),
        DeclareLaunchArgument('bind_address', default_value='0.0.0.0'),
        Node(
            package='ros2_parts',
            executable='lidar',
            name='lidar',
            output='screen',
            parameters=[{
                'port': ParameterValue(LaunchConfiguration('port'), value_type=int),
                'bind_address': LaunchConfiguration('bind_address'),
                'resolution': 0.1,
                'grid_range': 29.0,
                'height_min': -0.25,
                'height_max': 2.0,
                'rate_hz': 10.0,
                'frame_id': 'velodyne',
            }],
        ),
    ])
