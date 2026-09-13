"""Bring up the GPS PID path-following stack.

The ROS 2 equivalent of ``gps_pid_runner.py``.

    ros2 launch ros2_parts gps_pid.launch.py path_csv:=/abs/path/to/path.csv
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

#: Executables to launch, in the order ``gps_pid_runner.py`` adds them.
NODES = [
    'health_check',
    'uart_backup',
    'imu',
    'gps',
    'gps_to_xy',
    'gps_pid',
]

#: Nodes that need the waypoint file passed through.
PATH_CONSUMERS = ['gps_pid']


def generate_launch_description() -> LaunchDescription:
    """Describe the full GPS PID stack."""
    default_params = PathJoinSubstitution([
        get_package_share_directory('ros2_parts'), 'config', 'gps_pid.yaml',
    ])

    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='YAML file of parameters for every node in the stack.',
    )
    path_csv = DeclareLaunchArgument(
        'path_csv',
        default_value='',
        description='Waypoint CSV with latitude and longitude columns.',
    )

    nodes = [
        Node(
            package='ros2_parts',
            executable=executable,
            name=executable,
            output='screen',
            parameters=(
                [LaunchConfiguration('params_file'),
                 {'path_csv': LaunchConfiguration('path_csv')}]
                if executable in PATH_CONSUMERS
                else [LaunchConfiguration('params_file')]
            ),
        )
        for executable in NODES
    ]

    return LaunchDescription([params_file, path_csv] + nodes)
