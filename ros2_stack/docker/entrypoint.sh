#!/usr/bin/env bash

source /opt/ros/humble/setup.bash
if [ -f /amp_ws/install/setup.bash ]; then
    source /amp_ws/install/setup.bash
fi

wait_for_port() {
    local port="$1" pid="$2" service="$3" logfile="$4"
    for _ in {1..50}; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$service exited during startup; see $logfile" >&2
            return 1
        fi
        sleep 0.1
    done
    echo "Timed out waiting for $service on port $port; see $logfile" >&2
    return 1
}

if [ "${AMP_TARGET:-}" = "laptop" ]; then
    export DISPLAY=:1
    novnc_port="${NOVNC_PORT:-6080}"

    Xvfb "$DISPLAY" -screen 0 1280x800x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
    xvfb_pid=$!
    for _ in {1..50}; do
        [ -S /tmp/.X11-unix/X1 ] && break
        if ! kill -0 "$xvfb_pid" 2>/dev/null; then
            echo "Xvfb exited during startup; see /tmp/xvfb.log" >&2
            exit 1
        fi
        sleep 0.1
    done
    if [ ! -S /tmp/.X11-unix/X1 ]; then
        echo "Timed out waiting for Xvfb on $DISPLAY; see /tmp/xvfb.log" >&2
        exit 1
    fi

    DISPLAY="$DISPLAY" openbox --sm-disable >/tmp/openbox.log 2>&1 &
    openbox_pid=$!
    sleep 0.2
    if ! kill -0 "$openbox_pid" 2>/dev/null; then
        echo "Openbox exited during startup; see /tmp/openbox.log" >&2
        exit 1
    fi

    x11vnc -display "$DISPLAY" -rfbport 5901 -localhost -forever -shared -nopw \
        >/tmp/x11vnc.log 2>&1 &
    vnc_pid=$!
    wait_for_port 5901 "$vnc_pid" x11vnc /tmp/x11vnc.log || exit 1

    # The laptop service uses host networking for LiDAR UDP. Bind noVNC only
    # to host loopback so its passwordless desktop is never LAN-accessible.
    websockify --web /usr/share/novnc "127.0.0.1:${novnc_port}" 127.0.0.1:5901 \
        >/tmp/novnc.log 2>&1 &
    novnc_pid=$!
    wait_for_port "$novnc_port" "$novnc_pid" noVNC /tmp/novnc.log || exit 1

    xterm -display "$DISPLAY" -geometry 100x30+20+20 -e /bin/bash --login &
fi

exec "$@"
