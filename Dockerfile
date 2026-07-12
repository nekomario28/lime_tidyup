FROM osrf/ros:humble-desktop-full
ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# If you get a gpg error during docker build, uncomment the following three lines:
# RUN rm -f /etc/apt/sources.list.d/ros*.list \ /etc/apt/sources.list.d/openrobotics.list
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-pip vim eog xterm less wget portaudio19-dev \
    alsa-utils pulseaudio-utils libasound2-plugins

RUN apt-get update && apt install -y python3-colcon-common-extensions

RUN pip3 uninstall -y numpy
RUN pip3 install numpy==1.26.4
RUN pip3 install pyquaternion matplotlib transforms3d simple-pid \
    numpy-quaternion pyrealsense2 SpeechRecognition pyaudio sympy

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg espeak-ng
# Whisper is optional: torch/openai-whisper can fail on some environments due to size/network.
# Keep the image buildable when the optional Whisper installation is unavailable.
RUN pip3 install --no-cache-dir torch \
    && pip3 install --no-cache-dir openai-whisper \
    && python3 -c "import whisper; whisper.load_model('base')" \
    || echo "[voice] WARNING: openai-whisper install skipped; build continues"
RUN pip install "opencv-python<4.10" "opencv-contrib-python<4.10" "numpy==1.26.4"

RUN python3 -m pip install --no-cache-dir --ignore-installed \
    numpy==1.26.4 \
    pyquaternion \
    matplotlib \
    transforms3d \
    simple-pid \
    numpy-quaternion \
    pyrealsense2 \
    "opencv-python<4.10" \
    "opencv-contrib-python<4.10"

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
RUN source /opt/ros/humble/setup.bash \
 && npm ci \
 && npx generate-ros-messages

WORKDIR /root
RUN git clone https://github.com/yulat214/sdf_building_editor.git
WORKDIR /root/sdf_building_editor
RUN pip install -r requirements.txt

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

#WORKDIR /project/lib_ws/src/gazebo-pkgs/gazebo_grasp_plugin
#COPY ./project/resource/grasp/CMakeLists.txt CMakeLists.txt
#COPY ./project/resource/grasp/package.xml package.xml
#WORKDIR /project/lib_ws/src/gazebo-pkgs/
#RUN rm -r gazebo_test_tools gazebo_state_plugins gazebo_world_plugin_loader

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
RUN echo "source /usr/share/gazebo/setup.sh" >> .bashrc
RUN echo "source ~/turtlebot3_ws/install/setup.bash" >> .bashrc
RUN echo "export CYCLONEDDS_URI=~/bin/cyclone.xml" >> .bashrc
RUN echo "export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/project/lib_ws/build/IFRA_LinkAttacher:/opt/ros/humble/lib" >> .bashrc
RUN echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> .bashrc
RUN echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models:/root/practice_ws/worlds' >> .bashrc
RUN echo 'PATH=$PATH:/root/bin' >> .bashrc
RUN echo "export ROS_DOMAIN_ID=20" >> .bashrc


RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-gazebo-* ros-humble-navigation2 \
    ros-humble-nav2-bringup

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-dynamixel-sdk ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-gripper-controllers \
    ros-humble-moveit ros-humble-moveit-servo ros-humble-cartographer \
    ros-humble-realsense2-description \
    ros-humble-cartographer-ros ros-humble-gripper-controllers \
    ros-humble-tf-transformations

RUN mkdir -p /root/turtlebot3_ws/src
WORKDIR /root/turtlebot3_ws
RUN git clone -b humble-devel https://github.com/ROBOTIS-JAPAN-GIT/turtlebot3_lime.git
RUN git clone https://github.com/ldrobotSensorTeam/ldlidar_stl_ros2.git

# --- turtlebot3_lime本体へのWebots対応パッチ ---
COPY ./project/resource/turtlebot3_lime_webots.patch /root/turtlebot3_ws/turtlebot3_lime/
WORKDIR /root/turtlebot3_ws/turtlebot3_lime
RUN git apply turtlebot3_lime_webots.patch && rm turtlebot3_lime_webots.patch

RUN sed -i -E 's/^(\s*Frame Rate:\s*)[0-9]+/\110/' \
      turtlebot3_lime_navigation2/rviz/navigation2.rviz \
      turtlebot3_lime_moveit_config/config/moveit.rviz

RUN sed -i -E 's/^([[:space:]]*laser_max_range:[[:space:]]*)[0-9.]+/\13.4/' \
    turtlebot3_lime_navigation2/param/turtlebot3*.yaml

RUN sed -i 's/^\(\s*base_frame_id:\s*\)"base_footprint"/\1"base_link"/' \
      /root/turtlebot3_ws/turtlebot3_lime/turtlebot3_lime_navigation2/param/turtlebot3.yaml \
 && grep -q 'base_frame_id: "base_link"' \
      /root/turtlebot3_ws/turtlebot3_lime/turtlebot3_lime_navigation2/param/turtlebot3.yaml

WORKDIR /root/turtlebot3_ws

RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --symlink-install
WORKDIR /root/turtlebot3_ws/install
COPY ./project/resource/turtlebot3_lime.urdf.xacro turtlebot3_lime_description/share/turtlebot3_lime_description/urdf
# COPY ./project/resource/gazebo2.launch.py turtlebot3_lime_bringup/share/turtlebot3_lime_bringup/launch
# COPY ./project/resource/moveit_gazebo2.launch.py turtlebot3_lime_moveit_config/share/turtlebot3_lime_moveit_config/launch
COPY ./project/resource/sim_house.world turtlebot3_lime_bringup/share/turtlebot3_lime_bringup/worlds

WORKDIR /root/.gazebo
RUN mkdir models

WORKDIR /root/.gazebo/models
COPY ./project/resource/model_editor_models  .

WORKDIR /root
