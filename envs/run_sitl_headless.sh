#!/usr/bin/env bash
# Headless PX4 SITL + Gazebo rc_cessna for automated checks, isolated from an interactive session:
# own PX4 instance (MAVLink offboard port 14540 + N), own Gazebo partition, no GUI, no pxh shell.
#   envs/run_sitl_headless.sh [N]        start instance N (default 1), log in build/px4_sitl_default/instance_N/px4.log
#   envs/run_sitl_headless.sh stop [N]   stop it (only this instance and its Gazebo server)
set -e
if [ "$1" = stop ]; then STOP=1; N=${2:-1}; else STOP=0; N=${1:-1}; fi
source "$(dirname "${BASH_SOURCE[0]}")/activate_gz.sh" >/dev/null
B=$PX4_DIR/build/px4_sitl_default
W=$B/instance_$N
PART=fwverify_$N

if [ "$STOP" = 1 ]; then
    [ -f "$W/px4.pid" ] && kill "$(cat "$W/px4.pid")" 2>/dev/null || true
    for p in $(pgrep -f "gz sim"); do
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "GZ_PARTITION=$PART" && kill "$p" || true
    done
    rm -f "/tmp/px4_lock-$N" "/tmp/px4-sock-$N" "$W/px4.pid"
    echo "Stopped instance $N"
    exit 0
fi

mkdir -p "$W"
cd "$W"
rm -f "/tmp/px4_lock-$N" "/tmp/px4-sock-$N"
source "$B/rootfs/gz_env.sh"
export GZ_PARTITION=$PART HEADLESS=1 PX4_SYS_AUTOSTART=4003 PX4_SIM_MODEL=gz_rc_cessna
if ! gz topic -l | grep -q "^/world/default/clock"; then
    nohup gz sim --verbose=1 -r -s "$PX4_GZ_WORLDS/default.sdf" > "$W/gz.log" 2>&1 < /dev/null &
    for _ in $(seq 120); do gz topic -l | grep -q "^/world/default/clock" && break; sleep 1; done
fi
nohup "$B/bin/px4" -i "$N" -d "$B/etc" > "$W/px4.log" 2>&1 < /dev/null &
echo $! > "$W/px4.pid"
echo "PX4 instance $N started (pid $(cat "$W/px4.pid")), MAVLink offboard udp port $((14540 + N)), log $W/px4.log"
