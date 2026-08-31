FROM osrf/ros:humble-desktop-full
ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# If you get a gpg error during docker build, uncomment the following three lines:
RUN rm -f /etc/apt/sources.list.d/ros*.list \ /etc/apt/sources.list.d/openrobotics.list
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y --no-install-recommends \
 git python3-pip vim eog xterm less wget terminator

RUN apt-get update && apt install -y python3-colcon-common-extensions

RUN pip3 uninstall -y numpy
RUN pip3 install numpy==1.26.4
RUN pip3 install pyquaternion matplotlib transforms3d simple-pid \
 numpy-quaternion pyrealsense2

# Node.js のインストール
RUN apt-get update && apt-get install -y nodejs npm && \
    npm install n -g && \
    n stable && \
    apt purge -y nodejs npm && \
    apt autoremove -y && \
    hash -r

WORKDIR /root
RUN git clone https://github.com/yulat214/OneStageROS.git
WORKDIR /root/OneStageROS
RUN npm install

# --- Webots本体のインストール（cyberbotics公式リポジトリ） ---
RUN mkdir -p /etc/apt/keyrings && \
    wget -qO- https://cyberbotics.com/Cyberbotics.asc | gpg --dearmor -o /etc/apt/keyrings/cyberbotics.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/cyberbotics.gpg] https://cyberbotics.com/debian/ binary-amd64/" \
      > /etc/apt/sources.list.d/cyberbotics.list && \
    apt-get update && apt-get install -y --no-install-recommends webots

# Create Colcon workspace with external dependencies
WORKDIR /
RUN mkdir -p /project/lib_ws/src
WORKDIR /project/lib_ws/src
COPY dependencies.repos .
RUN vcs import < dependencies.repos

# patch downloaded modules before build
WORKDIR /project/lib_ws/src/pymoveit2
COPY ./project/resource/pymoveit2_setup.py setup.py

# Build the base Colcon workspace, installing dependencies first.
WORKDIR /project/lib_ws
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
 && apt-get update -y \
 && rosdep install --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y \
 && colcon build --symlink-install

WORKDIR /project
COPY ./project .
WORKDIR /root
COPY ./bin bin

WORKDIR /root

RUN echo "source /opt/ros/humble/setup.bash" >> .bashrc
RUN echo "source /project/lib_ws/install/setup.bash" >> .bashrc
RUN echo "source ~/turtlebot3_ws/install/setup.bash" >> .bashrc
RUN echo "source /root/webots_ws/install/setup.bash" >> .bashrc
RUN echo "export ROS_LOCALHOST_ONLY=1" >> .bashrc
RUN echo "export CYCLONEDDS_URI=/project/resource/cyclonedds.xml" >> .bashrc
RUN echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> .bashrc
RUN echo "export WEBOTS_HOME=/usr/local/webots" >> .bashrc
RUN echo 'export PATH=$PATH:$WEBOTS_HOME' >> .bashrc
RUN echo 'export USER=$(whoami)' >> .bashrc
RUN echo 'PATH=$PATH:/root/bin' >> .bashrc


RUN apt-get update && apt-get install -y --no-install-recommends \
 ros-humble-rmw-cyclonedds-cpp \
 ros-humble-navigation2 \
 ros-humble-nav2-bringup

RUN apt-get update && apt-get install -y --no-install-recommends \
 ros-humble-dynamixel-sdk ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-gripper-controllers \
 ros-humble-moveit ros-humble-moveit-servo ros-humble-cartographer \
 ros-humble-realsense2-description \
 ros-humble-cartographer-ros ros-humble-gripper-controllers \
 ros-humble-tf-transformations ros-humble-rosbridge-suite

RUN apt-get update && apt-get install -y --no-install-recommends \
 ros-humble-webots-ros2 \
 ros-humble-webots-ros2-driver \
 ros-humble-webots-ros2-control \
 ros-humble-webots-ros2-importer
RUN sed -i \
    "s/return 'microsoft-standard' in uname().release\$/return 'microsoft-standard' in uname().release and shutil.which('wslpath') is not None/" \
    /opt/ros/humble/local/lib/python3.10/dist-packages/webots_ros2_driver/utils.py

RUN mkdir -p /root/turtlebot3_ws/src
WORKDIR /root/turtlebot3_ws
RUN git clone -b humble-devel https://github.com/ROBOTIS-JAPAN-GIT/turtlebot3_lime.git
RUN git clone https://github.com/ldrobotSensorTeam/ldlidar_stl_ros2.git

# --- turtlebot3_lime本体へのWebots対応パッチ ---
COPY ./project/resource/turtlebot3_lime_webots.patch /root/turtlebot3_ws/turtlebot3_lime/
WORKDIR /root/turtlebot3_ws/turtlebot3_lime
RUN git apply turtlebot3_lime_webots.patch && rm turtlebot3_lime_webots.patch
WORKDIR /root/turtlebot3_ws

RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
&& colcon build --symlink-install

# --- webots_ws のビルド ---
RUN mkdir -p /root/webots_ws/src
WORKDIR /root/webots_ws/src
RUN git clone https://github.com/yulat214/turtlebot3_lime_webots
WORKDIR /root/webots_ws
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
 && rosdep install --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y \
 && colcon build --symlink-install

WORKDIR /root
