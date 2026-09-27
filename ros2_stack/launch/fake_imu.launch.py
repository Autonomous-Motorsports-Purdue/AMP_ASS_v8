"""Run the fake IMU on its own, for testing without hardware.

    ros2 launch ros2_parts fake_imu.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros2_parts',
            executable='fake_imu',
            name='fake_imu',
            output='screen',
            parameters=[{
                'imu_rate': 100.0,
                'speed_mps': 2.0,
                'turn_radius_m': 8.0,
            }],
        ),
    ])
