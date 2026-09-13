"""Bring up the cross-track-error driving stack.

The ROS 2 equivalent of ``cte_runner.py``: the same parts, in the same wiring,
each as its own node. Every node reads ``config/cte.yaml``.

    ros2 launch ros2_parts cte.launch.py path_csv:=/abs/path/to/path_xy_throttle.csv
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

#: Executables to launch, in the order ``cte_runner.py`` adds them.
NODES = [
    'loop_clock',
    'gstreamer_video_sync',
    'health_check',
    'uart_backup',
    'gps',
    'bno086',
    'gps_to_xy',
    'xy_to_gps',
    'heading_fusion',
    'cte_controller',
    'threaded_socket_pub',
    'logger_gps',
]

#: Nodes that need the waypoint file passed through.
PATH_CONSUMERS = ['cte_controller']


def generate_launch_description() -> LaunchDescription:
    """Describe the full CTE stack."""
    default_params = PathJoinSubstitution([
        get_package_share_directory('ros2_parts'), 'config', 'cte.yaml',
    ])

    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='YAML file of parameters for every node in the stack.',
    )
    path_csv = DeclareLaunchArgument(
        'path_csv',
        default_value='',
        description='Waypoint CSV in local XY with a throttle column.',
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
