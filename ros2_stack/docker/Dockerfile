# Dev container for ros2_parts. Ubuntu 22.04 / ROS 2 Humble, two targets:
#
#   TARGET=laptop  official multi-arch ros:humble. No CUDA, no ZED. The fake
#                  sensors, CSV replay, controllers, rviz2 and the tests. This
#                  is the default, and it builds on any machine (amd64 or an
#                  M-series Mac) in a few minutes.
#   TARGET=jetson  L4T/JetPack 6 base with CUDA, plus the ZED SDK and the
#                  CUDA build of onnxruntime. Build this on the Jetson.
#
#   docker compose -f docker/docker-compose.yml run --rm laptop
#   docker compose -f docker/docker-compose.yml run --rm amp      # on the kart
#
ARG TARGET=laptop

FROM ros:humble AS base-laptop
ENV AMP_TARGET=laptop
RUN apt update && apt install -y --no-install-recommends \
        ros-humble-desktop ros-humble-rqt ros-humble-rqt-graph \
        xvfb x11vnc openbox xterm novnc websockify \
    && rosdep update --rosdistro humble

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 AS base-jetson
ENV AMP_TARGET=jetson LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
RUN apt update && apt install -y --no-install-recommends \
        curl gnupg2 lsb-release locales sudo git build-essential cmake udev zstd \
        python3-pip python3-dev \
    && locale-gen en_US.UTF-8

RUN V="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F tag_name | cut -d '"' -f 4)" \
    && curl -fsSL -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${V}/ros2-apt-source_${V}.jammy_all.deb" \
    && apt update && apt install -y /tmp/ros2-apt-source.deb \
    && apt update && apt install -y ros-humble-desktop ros-dev-tools \
    && rosdep init && rosdep update --rosdistro humble

# ---------------------------------------------------------------------------
# Everything both targets share.
# ---------------------------------------------------------------------------
FROM base-${TARGET} AS common
ENV DEBIAN_FRONTEND=noninteractive

# ROS Humble's C extensions -- cv_bridge, cv2, matplotlib, skimage, pandas --
# are compiled against numpy 1.x, and onnxruntime pulls numpy 2 in as a
# dependency, which breaks all of them at import.
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
RUN echo 'numpy<1.25' > /etc/pip-constraints.txt


COPY package.xml /tmp/deps/ros2_parts/package.xml
RUN apt update && rosdep install --from-paths /tmp/deps --ignore-src -r -y --rosdistro humble

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY docker/entrypoint.sh /usr/local/bin/amp-entrypoint

RUN apt update && apt install -y --no-install-recommends \
        python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
        gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
        python3-tk libgl1-mesa-glx libglib2.0-0 libxkbcommon-x11-0 sudo


RUN if getent passwd 1000 >/dev/null; then userdel -r "$(getent passwd 1000 | cut -d: -f1)" 2>/dev/null || true; fi \
    && useradd -u 1000 -m -s /bin/bash amp \
    && usermod -aG sudo,dialout,tty,video,plugdev amp \
    && echo 'amp ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/amp \
    && mkdir -p /amp_ws/src /amp_ws/build /amp_ws/install /amp_ws/log \
    && chown -R amp:amp /amp_ws

FROM common AS extras-laptop
RUN pip3 install --no-cache-dir onnxruntime

FROM common AS extras-jetson

RUN pip3 install --no-cache-dir --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu \
    && python3 -c "import onnxruntime as o; p=o.get_available_providers(); print(p); assert 'CUDAExecutionProvider' in p"
RUN curl -fsSL -o /tmp/zed.run https://download.stereolabs.com/zedsdk/5.0/l4t36.4/jetsons \
    && chmod +x /tmp/zed.run && /tmp/zed.run -- silent skip_cuda skip_drivers skip_tools \
    && python3 /usr/local/zed/get_python_api.py \
    && rm -f /tmp/zed.run && rm -rf /usr/local/zed/resources/*

FROM extras-${TARGET} AS final
ENV QT_X11_NO_MITSHM=1 MPLBACKEND=TkAgg

USER amp
WORKDIR /amp_ws
RUN printf '%s\n' 'source /opt/ros/humble/setup.bash' \
      '[ -f /amp_ws/install/setup.bash ] && source /amp_ws/install/setup.bash' >> ~/.bashrc


ENTRYPOINT ["/bin/bash", "/usr/local/bin/amp-entrypoint"]
CMD ["bash"]
