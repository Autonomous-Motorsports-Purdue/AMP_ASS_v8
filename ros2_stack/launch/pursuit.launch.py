"""Bring up the camera-based pure pursuit stack.

Segmentation comes from the ``onnx`` node, which publishes the masks consumed
by ``curve_fit``.

    ros2 launch ros2_parts pursuit.launch.py
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

#: Executables to launch in dependency order.
NODES = [
    'health_check',
    'frame_publisher',
    'onnx',
    'curve_fit',
    'translate',
    'pure_pursuit',
    'control_mux',
    'logger',
    'uart_backup',
]

#: The segmentation model reads the left ZED frame, not a plain camera.
REMAPPINGS = {
    'onnx': [('camera/image_raw', 'camera/left/image_raw')],
}


def generate_launch_description() -> LaunchDescription:
    """Describe the full camera pure pursuit stack."""
    default_params = PathJoinSubstitution([
        get_package_share_directory('ros2_parts'), 'config', 'pursuit.yaml',
    ])

    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='YAML file of parameters for every node in the stack.',
    )

    nodes = [
        Node(
            package='ros2_parts',
            executable=executable,
            name=executable,
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
            remappings=REMAPPINGS.get(executable, []),
        )
        for executable in NODES
    ]

    return LaunchDescription([params_file] + nodes)
