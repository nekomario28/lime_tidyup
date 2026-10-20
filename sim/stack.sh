#!/usr/bin/env bash
# In-container half of bin/cccsim. See that file for why each step exists.
set -Eeo pipefail

cube_distance="1.0"
tree=""
record_dir=""
start_moveit=true
keep=false
while (($# > 0)); do
    case "$1" in
        --cube-distance) cube_distance="$2"; shift ;;
        --tree) tree="$2"; shift ;;
        --record) record_dir="$2"; shift ;;
        --keep) keep=true ;;
        --no-moveit) start_moveit=false ;;
    esac
    shift
done

source /opt/ros/humble/setup.bash
source /project/lib_ws/install/setup.bash
source /root/turtlebot3_ws/install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/root/repo/bin/cyclone.xml
export GAZEBO_MODEL_DATABASE_URI=""
export LIBGL_ALWAYS_SOFTWARE=1
export DISPLAY="${DISPLAY:-:99}"
export GAZEBO_PLUGIN_PATH=/project/lib_ws/install/ros2_linkattacher/lib:${GAZEBO_PLUGIN_PATH:-}
export PYTHONPATH=/root/repo/yolo_ws/src/barcode_detector:${PYTHONPATH:-}

# Rendering preflight. gzserver with camera sensors *segfaults* when no X display
# is available (measured: fresh container, cameras in the world, no Xvfb), so the
# display must exist before Gazebo starts.
if ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    echo "[cccsim] starting Xvfb on ${DISPLAY}"
    (nohup Xvfb "${DISPLAY}" -screen 0 1280x960x24 -nolisten tcp \
        >/tmp/xvfb.log 2>&1 &)
    sleep 3
fi
if ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    echo "[cccsim] no X display on ${DISPLAY}; install xvfb or export DISPLAY" >&2
    echo "[cccsim] (Gazebo camera sensors crash the server without rendering)" >&2
    exit 1
fi
echo "[cccsim] display ok: ${DISPLAY}"

repo=/root/repo
ev="${CCCSIM_EVIDENCE:-/root/repo/yolo_ws/log/cccsim/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$ev"
echo "[cccsim] evidence: $ev"

pids=()
cleanup() {
    if ${keep}; then
        echo "[cccsim] stack kept running (--keep)"
        return
    fi
    # only the processes this run started, by PID
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    sleep 1
    for pid in "${pids[@]:-}"; do
        kill -9 "${pid}" 2>/dev/null || true
    done
    echo "[cccsim] stopped"
}
trap cleanup EXIT

# 1. world with cameras
echo "[cccsim] gazebo"
nohup gzserver --verbose -s libgazebo_ros_init.so \
    -s libgazebo_ros_factory.so "$repo/sim/world/cube_world.world" \
    >"$ev/gzserver.log" 2>&1 &
pids+=("$!")
for _ in $(seq 1 60); do
    ros2 service list 2>/dev/null | grep -Fxq /spawn_entity && break
    sleep 1
done

# 2. robot description with use_sim:=true
echo "[cccsim] robot description"
nohup ros2 launch "$repo/sim/rsp_sim.launch.py" >"$ev/rsp.log" 2>&1 &
pids+=("$!")
for _ in $(seq 1 30); do
    ros2 topic list 2>/dev/null | grep -Fxq /robot_description && break
    sleep 1
done
sleep 2

# 3. robot into the world
echo "[cccsim] spawn robot"
python3 "$repo/sim/robot_spawn.py" >"$ev/spawn_robot.log" 2>&1 \
    || echo "[cccsim] robot spawn reported a problem"
sleep 8

# 4. controllers
for controller in joint_state_broadcaster arm_controller gripper_controller; do
    timeout 60 ros2 run controller_manager spawner "$controller" \
        >"$ev/spawn_$controller.log" 2>&1 || echo "[cccsim] $controller spawn issue"
done
ros2 control list_controllers >"$ev/controllers.log" 2>&1 || true
grep -c active "$ev/controllers.log" | xargs -I{} echo "[cccsim] active controllers: {}"

# 5. static TF for the sim camera, computed from the same offset
echo "[cccsim] camera transform"
python3 "$repo/sim/camera_tf.py" >"$ev/camera_tf.txt"
mapfile -t tf <"$ev/camera_tf.txt"
nohup ros2 run tf2_ros static_transform_publisher "${tf[@]}" \
    base_link sim_camera_color_optical_frame >"$ev/static_tf.log" 2>&1 &
pids+=("$!")

# 6. move_group
if ${start_moveit}; then
    echo "[cccsim] move_group"
    nohup ros2 launch turtlebot3_lime_moveit_config move_group.launch.py \
        use_sim:=true start_rviz:=false >"$ev/move_group.log" 2>&1 &
    pids+=("$!")
    for _ in $(seq 1 60); do
        ros2 service list 2>/dev/null | grep -Fxq /compute_ik && break
        sleep 2
    done
fi

# 7. perception (no detector: bbox from the cube's colour region)
echo "[cccsim] perception"
nohup python3 -m barcode_detector.yolo_pose_node --ros-args \
    -p perception_backend:=color_frontend \
    -p bbox_source:=colour_blob \
    -p debug_image_path:="$ev/perception_debug.jpg" \
    >"$ev/perception.log" 2>&1 &
pids+=("$!")
sleep 10
timeout 20 ros2 topic echo --once --full-length /cube_pose_result \
    >"$ev/cube_pose_result.txt" 2>&1 || true

# 8. the cube
echo "[cccsim] cube at ${cube_distance} m"
timeout 10 ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity \
    "{name: rubik_cube}" >/dev/null 2>&1 || true
python3 "$repo/sim/spawn_cube.py" "${cube_distance}" 0.0 \
    >"$ev/spawn_cube.log" 2>&1 || tail -2 "$ev/spawn_cube.log"

# 9. recorder
if [[ -n "${record_dir}" ]]; then
    mkdir -p "$record_dir"
    nohup python3 "$repo/sim/record_frames.py" "$record_dir" 900 10 \
        >"$ev/record.log" 2>&1 &
    pids+=("$!")
    echo "[cccsim] recording to $record_dir"
fi

# 10. register cm1 once
echo "[cccsim] registering cm1"
cd /root/pytwb_ws
printf 'create cm1\nY\nexit\n' | timeout 120 pytwb >"$ev/register.log" 2>&1 \
    || echo "[cccsim] cm1 registration reported a problem (see $ev/register.log)"

# 11. one long-lived cm1 server, stdin kept open
cd /root/pytwb_ws/src
fifo="$ev/cm1.in"
mkfifo "$fifo"
nohup python3 cm1 <"$fifo" >"$ev/cm1.log" 2>&1 &
pids+=("$!")
exec 3>"$fifo"

if [[ -n "${tree}" ]]; then
    echo "[cccsim] running tree ${tree}"
    echo "${tree}" >&3
    for _ in $(seq 1 240); do
        grep -q "mission complete" "$ev/cm1.log" 2>/dev/null && break
        sleep 1
    done
    grep -vE "WARN|^ +|^/" "$ev/cm1.log" | tail -12 || true

    echo "[cccsim] attach at the fingertips"
    python3 "$repo/sim/attach_cube.py" turtlebot3_lime_system \
        gripper_left_link rubik_cube body >"$ev/attach.log" 2>&1 || true
    tail -1 "$ev/attach.log" || true
    sleep 2

    echo "[cccsim] lift"
    echo 'cchair' >&3
    for _ in $(seq 1 60); do
        grep -q "mission complete" <(sed -n '/running tree/,$p' "$ev/cm1.log") 2>/dev/null \
            && break
        sleep 1
    done
    sleep 4
    gz model --model-name rubik_cube --info 2>/dev/null \
        | grep -m1 -A3 position | tr -d ' \n' >"$ev/cube_pose_final.txt" || true
    echo "[cccsim] cube: $(cat "$ev/cube_pose_final.txt" 2>/dev/null)"
else
    echo "[cccsim] cm1 prompt: type a tree name (e.g. ccsim, ccgrasp, ccprod2), Ctrl-D to leave"
    exec python3 cm1 <&3
fi
