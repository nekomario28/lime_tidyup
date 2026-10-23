# `cccsim` — one command for the cube-grasp simulation

`bin/cccsim` is the sim counterpart of `bin/ccc`: it brings up Gazebo with the
cube, the robot, MoveIt, perception and `cm1`, then drops into the same prompt.
With `--tree NAME` it runs a tree, attaches the cube and lifts it, and with
`--record` it also captures both camera views.

    ./bin/cccsim --tree ccprod2 --record      # full mission + recording
    ./bin/cccsim                              # interactive cm1 prompt

Flags follow `bin/ccc`: `--dry-run`, `--keep`, `--evidence-dir PATH`, `-h`. The
sim-specific ones are `--tree NAME`, `--record`, `--cube-distance M`,
`--no-moveit` and `--xvfb` (headless host). `--grasp` runs the grasp after the
tree; it implies `--no-attach`, because holding the cube by contact is what the
grasp is being tested for. `--record` captures four views: the camera, the third
person observer, the perception overlay, and the grasp overlay (the recognised
box and the aim point) — see [the grasp survey](../research/2026-10-20-grasp-oss-survey.md)
for what the last two are made of.

At the prompt, use `ccprod2` or `ccprod`: they approach by contact and close the
fingers, so a bounding box is enough. `ccsim` and `ccgrasp` ask for a `pose_3d`,
which the model-free route deliberately does not produce, and stop with
"mission failure" if you pick them.

## Why it is structured this way

The harness exists to make one thing true: a run must either complete or say
which step failed. Every oddity below was found by a run that appeared to hang.

**The evidence directory is the report.** Each stage runs as its own
`docker compose exec` and writes its own log into
`yolo_ws/log/cccsim/<timestamp>-<sha>/` (gzserver, rsp, controllers, tf, moveit,
perception, cm1, attach, cube pose), mounted at `/root/cccsim_evidence`. A stage
is only accepted when its process is actually running, otherwise the log tail is
printed and the run fails — the same shape as `bin/ccc`.

**The camera belongs to the robot.** The robot description carries no camera and
the RealSense Gazebo plugin cannot find one, so `sim/robot_spawn.py` injects a
camera sensor into the description before spawning. The two obvious alternatives
both fail: a camera model placed in the world is left behind (and then driven
over) as soon as the robot moves, and moving a world model needs
`libgazebo_ros_state.so`, which segfaults gzserver in this world.

**Rendering must exist before Gazebo starts.** A camera sensor with no X display
takes gzserver down with a segmentation fault rather than a message, so
`stack.sh` starts Xvfb if needed and refuses to continue without a display.

**One `cm1` process, fed on stdin.** `cm1` is registered once through `pytwb` and
then runs as a single long-lived process reading a FIFO. Starting it per phase
leaves it waiting for a tree that has already finished.

**Cleanup is PID-scoped.** `pkill -x gzserver` in one run's cleanup kills another
run's server; with several runs in flight that looks like controllers vanishing
mid-mission.

## Limits worth knowing

- The grasp holds by contact, not by attachment: `gazebo_grasp_fix` is built and
  loaded, so the cube is held because the fingers are touching it. `--grasp`
  therefore runs with the cube unattached — that is the test.
- The first-person view carries the gripper into the frame during the final
  approach, so the cube is seen through the fingers at the moment of grasping.
  That is not only cosmetic: the arm occludes the face the pose is measured from,
  which is why the last centimetres are searched by touch rather than aimed.
- The tree drives fixed distances to approach; closing the range is the grasp
  script's job, because it is the first thing that has a measured cube pose.
