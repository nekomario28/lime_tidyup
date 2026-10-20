#!/usr/bin/env bash
# In-container stages for bin/cccsim.
#
# Each invocation is ONE stage, launched from the host with `docker compose exec`
# (see bin/cccsim). That mirrors bin/ccc: a stage either starts and stays up, or
# fails where the host can see it, with its own log in the evidence directory.
#
# Usage: sim/stack.sh <stage> [args]
#
#   gazebo | rsp | tf | moveit | perception | record DIR   long-lived
#   spawn | controllers | cube M | attach | register         short
#   tree NAME                                                runs a tree end to end
#   prompt                                                   cm1 in the foreground
set -Eeo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ev="${CCCSIM_EVIDENCE_DIR:-/root/cccsim_evidence}"
mkdir -p "${ev}"

source /opt/ros/humble/setup.bash
source /project/lib_ws/install/setup.bash
source /root/turtlebot3_ws/install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-/root/bin/cyclone.xml}"
export GAZEBO_PLUGIN_PATH="/project/lib_ws/install/ros2_linkattacher/lib:${GAZEBO_PLUGIN_PATH:-}"
export GAZEBO_MODEL_DATABASE_URI=""

stage="${1:-}"
shift || true

# The arm/gripper controllers are useless until this is active, and the
# upstream launch hands gazebo_ros2_control the hardware plugin class unless
# `use_sim` is passed as a xacro mapping (sim/rsp_sim.launch.py does that).
spawn_controllers() {
    for controller in joint_state_broadcaster arm_controller gripper_controller; do
        timeout 90 ros2 run controller_manager spawner "${controller}" \
            >"${ev}/spawn_${controller}.log" 2>&1 \
            || echo "[stage] ${controller} failed to spawn (see spawn_${controller}.log)" >&2
    done
    ros2 control list_controllers >"${ev}/controllers.log" 2>&1 || true
    grep -c active "${ev}/controllers.log" 2>/dev/null || echo 0
}

case "${stage}" in
    gazebo)
        exec gzserver --verbose -s libgazebo_ros_init.so \
            -s libgazebo_ros_factory.so "${repo}/sim/world/cube_world.world"
        ;;
    rsp)
        exec ros2 launch "${repo}/sim/rsp_sim.launch.py"
        ;;
    spawn)
        # the camera sensor is injected here, so the image rides on base_link
        python3 "${repo}/sim/robot_spawn.py"
        ;;
    controllers)
        spawn_controllers
        ;;
    tf)
        # the same offset the injected sensor uses; images and TF must agree or
        # a pose expressed in the image frame points somewhere else
        read -r -a camera_tf <<<"$(python3 "${repo}/sim/camera_tf.py")"
        exec ros2 run tf2_ros static_transform_publisher "${camera_tf[@]}" \
            base_link sim_camera_color_optical_frame
        ;;
    moveit)
        exec ros2 launch turtlebot3_lime_moveit_config move_group.launch.py \
            use_sim:=true start_rviz:=false
        ;;
    perception)
        source /root/yolo_ws/install/setup.bash
        # No learned model: the cube's colour region gives the box and the box
        # is all the approach needs. depth_blob cannot be used here (the depth
        # image has two publishers and invalid near rows).
        exec ros2 run barcode_detector yolo_pose_node --ros-args \
            -p perception_backend:=color_frontend \
            -p bbox_source:=colour_blob \
            -p publish_debug_image:=true \
            -p debug_image_path:="${ev}/perception_debug.jpg"
        ;;
    cube)
        distance="${1:-1.0}"
        timeout 10 ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity \
            "{name: rubik_cube}" >/dev/null 2>&1 || true
        python3 "${repo}/sim/spawn_cube.py" "${distance}" 0.0
        ;;
    record)
        exec python3 "${repo}/sim/record_frames.py" "${1:?record needs a directory}" 900 10
        ;;
    attach)
        python3 "${repo}/sim/attach_cube.py" turtlebot3_lime_system \
            gripper_left_link rubik_cube body
        ;;
    register)
        cd /root/pytwb_ws
        printf 'create cm1\nY\nexit\n' | timeout 180 pytwb
        ;;
    prompt)
        # cm1 reads trees from stdin; ccc opens it this way too
        cd /root/pytwb_ws/src
        exec python3 cm1
        ;;
    tree)
        name="${1:?tree needs a name}"
        fifo="${ev}/cm1.in"
        rm -f "${fifo}"
        mkfifo "${fifo}"
        cd /root/pytwb_ws/src
        # One long-lived cm1 fed by a FIFO: restarting it per tree re-registers
        # the subsystems and the next tree never runs.
        python3 cm1 <"${fifo}" >"${ev}/cm1.log" 2>&1 &
        cm1_pid=$!
        exec 3>"${fifo}"
        trap 'kill -TERM "${cm1_pid}" 2>/dev/null || true' EXIT
        echo "${name}" >&3

        for _ in $(seq 1 "${CCCSIM_TREE_TIMEOUT:-300}"); do
            grep -q 'mission complete' "${ev}/cm1.log" 2>/dev/null && break
            sleep 1
        done
        grep -vE 'WARN|^ +|^/' "${ev}/cm1.log" | tail -8 || true
        if ! grep -q 'mission complete' "${ev}/cm1.log" 2>/dev/null; then
            echo "[stage] tree ${name} did not complete (see ${ev}/cm1.log)" >&2
            exit 1
        fi

        # The tree reaches the cube; closing the fingers on it is not something
        # this build can do without friction (gazebo_grasp_fix is absent), so the
        # cube is attached at the fingertip link and then lifted by cchair.
        python3 "${repo}/sim/attach_cube.py" turtlebot3_lime_system \
            gripper_left_link rubik_cube body >"${ev}/attach.log" 2>&1 || true
        tail -1 "${ev}/attach.log" || true
        echo 'cchair' >&3
        for _ in $(seq 1 90); do
            grep -q 'mission complete' <(sed -n '/^cchair$/,$p' "${ev}/cm1.log") \
                2>/dev/null && break
            sleep 1
        done
        sleep 4
        gz model --model-name rubik_cube --info 2>/dev/null \
            | grep -m1 -A3 position | tr -d ' \n' >"${ev}/cube_pose.txt" || true
        echo "[stage] cube: $(cat "${ev}/cube_pose.txt" 2>/dev/null)"
        ;;
    *)
        echo "usage: $0 {gazebo|rsp|spawn|controllers|tf|moveit|perception|cube M|record DIR|attach|register|prompt|tree NAME}" >&2
        exit 2
        ;;
esac
