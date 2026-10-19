#!/usr/bin/env bash
# Run the real perception node against Gazebo with no detector.
set -Eeo pipefail
ev=/work/.sim/evidence/nodector
mkdir -p "$ev"
export ROS_DOMAIN_ID=91 GAZEBO_MASTER_URI=http://127.0.0.1:11401
export GAZEBO_MODEL_DATABASE_URI="" LIBGL_ALWAYS_SOFTWARE=1
source /opt/ros/humble/setup.bash
export PYTHONPATH=/dev2/yolo_ws/src/barcode_detector:$PYTHONPATH

gzserver --verbose -s libgazebo_ros_init.so -s libgazebo_ros_factory.so \
  /work/.sim/rubik_world.world >"$ev/gazebo.log" 2>&1 &
gz=$!
for _ in $(seq 1 120); do ros2 service list 2>/dev/null | grep -Fxq /spawn_entity && break; sleep 0.5; done
PYTHONPATH=/work/yolo_ws/src/barcode_detector:$PYTHONPATH \
  python3 /work/.sim/rubik_spawn.py >"$ev/spawn.log" 2>&1
for _ in $(seq 1 160); do
  ros2 topic list 2>/dev/null | grep -Fxq /cube8_sim/depth/depth/image_raw && break
  sleep 0.5
done
sleep 2

echo "[run] starting the perception node (no detector, depth bbox)"
python3 -m barcode_detector.yolo_pose_node --ros-args \
  -p perception_backend:=color_frontend \
  -p bbox_source:=depth_blob \
  -p depth_topic:=/cube8_sim/depth/depth/image_raw \
  -p depth_scale_m:=1.0 \
  -p cube_side_length_m:=0.057 \
  -p debug_image_path:="$ev/debug_from_node.jpg" \
  -r /camera/camera/color/image_raw:=/cube8_sim/camera/image_raw \
  -r /camera/camera/color/camera_info:=/cube8_sim/camera/camera_info \
  >"$ev/node.log" 2>&1 &
node=$!

for _ in $(seq 1 120); do ros2 topic list 2>/dev/null | grep -Fxq /cube_pose_result && break; sleep 0.5; done
timeout 60 ros2 topic echo --once --full-length /cube_pose_result >"$ev/result.yaml" 2>"$ev/echo.err" || true
python3 /work/.sim/save_debug_image.py "$ev/overlay_from_topic.jpg" || true

kill "$node" 2>/dev/null || true; sleep 1; kill -9 "$node" "$gz" 2>/dev/null || true
echo "[run] done"
