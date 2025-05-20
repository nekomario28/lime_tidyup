# Turtlebot3 Lime application sample with Behavior Tree and ros_actor
This repository contains demo code for Lime, the next generation TurtleBot3 with a Jetson GPU and a 6-DOF manipulator, running in a single PC environment using Docker.

## Installation
- Go to working directory of any location
- Make sure the DISPLAY environment variable is set properly

Misconfiguration of X windows can be a source of trouble. The X server must be configured to accept access via tcpip (e.g. xhost +).  
Value of the DISPLAY environment variable should be:  
'IP address of host os':0.0  

```
$ printenv DISPLAY # for checking DISPLAY environment variable
$ git pull https://github.com/momoiorg-repository/LimeSimulDemo.git
$ cd LimeSimulDemo
$ docker compose build
$ docker compose up
```

By these operations, docker will now start and an X window will appear. From now on, we will work within this window.  The first thing you need to do is register the package you'll be running here. The package sources are in ~/pytwb_ws, and the following operations register the 'cm1' package there, which contains various sample codes you can run.
```
# cd ~/pytwb_ws
# pytwb
> create cm1
Y
```

The installation is now complete.

## Demo Code Execution
[searching Coke Can](doc/demo.mp4)

- Launching the simulation environment

```
# run_all
```

You will now see one Gazebo screen and two Rviz screens.

As a next step, run sample codes which are in Behavior Tree or ros_actor actor format. They can be executed from the command interpreter provided by ros_actor. The source code for these is located under ~/pytwb_ws/src/cm1.

- Launch vscode, attach to 'limesimuldemo' docker from Remote Explorer, and open the /root/pytwb_ws directory.

- When you run '~/pytwb_ws/src/cm1/\_\_main\_\_.py from vscode, a command prompt will appear.

- From the command prompt, type 'bt_search' to start the demo.  The operation executes the 'bt_search' Behavior Tree.

To run the demo, you need to add Coke Can
from Gazebo's Insert tab.  Any Coke Can position should work.

##  Grasping Coke Can
You can also try to get a robot to grasp Coke Cans by running 'bt_catch' Behavior Tree.
The original Coke Can model is too heavy to pick as it is, so edit the model and set the weight to around 0.04kg.
- Place the Coke Can in any position, select it with the mouse and select 'Edit model' from the operation list displayed by right-clicking.
- The entire screen will turn white, so select the Coke Can again and double left-click to display the editing screen.
- From there, change the Mass value from 0.39 kg to 0.039 kg and click 'ok'

![Edit Coke Can properties](doc/weight.png)

- Exit model editing mode with cntl-X and save the edited model with a different name

- Place the reduced weight Coke Can in a position directly visible to the robot and start 'bt_catch'.

## Picking Operation
The picking operation, which combines bt_search and bt_catch, can be executed by the 'bt_pick' Behavior Tree.

## Internal structure
ros_actor is heavily used to implement each Behavior Tree. As the main thing,
- Nav2 operation,
- MoveIt2 operation,
- Analysis of camera information,

etc.

[Detailed description](doc/app.pdf)

## ros_actor mechanism documentation
- [Principle of ros_actor](doc/principle.pdf)

## Related repositories and sites
- [ros_actor](https://github.com/momoiorg-repository/ros_actor)
- [pytwb](https://github.com/momoiorg-repository/pytwb)
- [vector_map](https://github.com/RobotSpatialCognition/vector)

## Dependencies  
Special thanks to the following works:  
- [“turtlebot3_behavior_demos”](https://github.com/sea-bass/turtlebot3_behavior_demos) by sea-bass  
Gazebo world is derived from this repository.
- [lecture page of OpenCV and Python](https://demura.net/education/22777.html) by Demura Kosei  

## About us
[momoi.org](https://momoi.org/?yada_wiki=ros-related-projects)
