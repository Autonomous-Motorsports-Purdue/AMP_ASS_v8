"""Build configuration for the ros2_parts package.

Every Donkeycar part in ``parts/`` becomes one executable here. The names match
the part module they came from, so ``ros2 run ros2_parts gps_to_xy`` runs the
port of ``parts/gps_to_xy.py``.
"""

from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'ros2_parts'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cole Byrne',
    maintainer_email='colewbyrne@gmail.com',
    description=(
        'ROS 2 nodes for the AMP autonomous software stack, ported one-for-one '
        'from the Donkeycar parts of the same name.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # -- Sensors ---------------------------------------------------
            'bno086 = ros2_parts.bno086_node:main',
            'frame_publisher = ros2_parts.frame_publisher_node:main',
            'gps = ros2_parts.gps_node:main',
            'gps_csv = ros2_parts.gps_csv_node:main',
            'image_publisher = ros2_parts.image_publisher_node:main',
            'imu = ros2_parts.imu_node:main',
            'zed_frame_publisher = ros2_parts.zed_frame_publisher_node:main',

            # -- Simulated sensors -----------------------------------------
            'fake_gps = ros2_parts.fake_gps_node:main',
            'fake_imu = ros2_parts.fake_imu_node:main',
            'test = ros2_parts.part_test_node:main',

            # -- Localization ----------------------------------------------
            'ekf_localizer = ros2_parts.ekf_localizer_node:main',
            'gps_to_xy = ros2_parts.gps_to_xy_node:main',
            'heading_fusion = ros2_parts.heading_fusion_node:main',
            'predictive_model = ros2_parts.predictive_model_node:main',
            'xy_to_gps = ros2_parts.xy_to_gps_node:main',

            # -- Perception ------------------------------------------------
            'curve_fit = ros2_parts.curve_fit_node:main',
            'image_cv = ros2_parts.image_cv_node:main',
            'lane_detect = ros2_parts.lane_detect_node:main',
            'onnx = ros2_parts.onnx_node:main',
            'preprocessor = ros2_parts.preprocessor_node:main',
            'segment_model = ros2_parts.segment_model_node:main',
            'translate = ros2_parts.translate_node:main',

            # -- Control ---------------------------------------------------
            'closed_loop_controller = ros2_parts.controller_node:main_closed_loop',
            'control_mux = ros2_parts.control_mux_node:main',
            'cte_controller = ros2_parts.cte_controller_node:main',
            'gps_pid = ros2_parts.gps_pid_node:main',
            'mpc_part = ros2_parts.controller_node:main',
            'pure_pursuit = ros2_parts.pure_pursuit_node:main',
            'pure_pursuit_controller = ros2_parts.pure_pursuit_controller_node:main',

            # -- Actuation -------------------------------------------------
            'uart = ros2_parts.uart_node:main',
            'uart_backup = ros2_parts.uart_backup_node:main',
            'uart_backup2 = ros2_parts.uart_backup2_node:main',

            # -- Telemetry, logging and visualization ----------------------
            'gps_visualizer = ros2_parts.gps_visualizer_node:main',
            'gstreamer_video_sync = ros2_parts.gstreamer_video_sync_node:main',
            'health_check = ros2_parts.health_check_node:main',
            'imu_visualizer = ros2_parts.imu_visualizer_node:main',
            'log = ros2_parts.log_node:main',
            'logger = ros2_parts.logger_node:main',
            'logger2 = ros2_parts.logger2_node:main',
            'logger_gps = ros2_parts.logger_gps_node:main',
            'loop_clock = ros2_parts.loop_clock_node:main',
            'threaded_socket_pub = ros2_parts.threaded_socket_pub_node:main',
        ],
    },
)
