# Guide to use DOPE with ROS

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Install ROS Noetic](#1-install-ros-noetic)
   1. [Setup your sources.list](#11-setup-your-sourceslist)
   2. [Set up your keys](#12-set-up-your-keys)
   3. [Installation](#13-installation)
   4. [Environment setup](#14-environment-setup)
3. [Create a catkin workspace](#2-create-a-catkin-workspace)
4. [Download the DOPE code](#3-download-the-dope-code)
5. [Install Python dependencies](#4-install-python-dependencies)
6. [Install ROS dependencies](#5-install-ros-dependencies)
7. [Build](#6-build)
8. [Set up the weights](#7-set-up-the-weights)
9. [Running](#running)
10. [Debugging](#debugging)
11. [In Case](#in-case )
12. [Changelog](#changelog)

This directory and its subdirectories contain code for running DOPE with ROS Noetic.

## Prerequisites

Before beginning the installation, ensure you have one of the following operating systems installed:

- Ubuntu 20.04.6 LTS (Focal Fossa) - [Download Link](https://releases.ubuntu.com/focal/)
- Kubuntu 20.04.6 LTS (Focal Fossa) - [Download Link](https://cdnjs.cloudflare.com/kubuntu/releases/20.04/release/)

## 1. Install ROS Noetic

Follow these detailed instructions to install ROS Noetic on your system:

### 1.1 Setup your sources.list

Set up your computer to accept software from packages.ros.org:

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```

### 1.2 Set up your keys

```bash
sudo apt install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```

### 1.3 Installation

First, make sure your Debian package index is up-to-date:

```bash
sudo apt update
```

Now install ROS Desktop, which includes ROS-Base plus tools like rqt and rviz:

```bash
sudo apt install ros-noetic-desktop
```

### 1.4 Environment setup

You must source the ROS setup script in every bash terminal you use ROS in:

```bash
source /opt/ros/noetic/setup.bash
```

To automatically source this script every time a new shell is launched, use one of the following commands based on your shell:

For Bash:
```bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

For Zsh:
```bash
echo "source /opt/ros/noetic/setup.zsh" >> ~/.zshrc
source ~/.zshrc
```

## 2. Create a catkin workspace

If you do not already have a catkin workspace, create one:

```bash
mkdir -p ~/catkin_ws/src   # Replace `catkin_ws` with the name of your workspace
cd ~/catkin_ws/
catkin_make
```

## 3. Download the DOPE code

```bash
mkdir ~/src
cd ~/src
git clone https://github.com/NVlabs/Deep_Object_Pose.git
ln -s ~/src/Deep_Object_Pose/ros1/dope ~/catkin_ws/src/dope
```

## 4. Install Python dependencies

```bash
cd ~/catkin_ws/src/dope
python3 -m pip install -r ~/src/Deep_Object_Pose/requirements.txt
```

## 5. Install ROS dependencies

```bash
cd ~/catkin_ws
rosdep install --from-paths src -i --rosdistro noetic
sudo apt-get install ros-noetic-rosbash ros-noetic-ros-comm
```

## 6. Build

```bash
cd ~/catkin_ws
catkin_make
```

## 7. Set up the weights

Save the weights to the `weights` folder, i.e., `~/catkin_ws/src/dope/weights/`.

With these steps completed, you should have DOPE with ROS Noetic installed and configured on your system.

## Running

**Note:** Each step below should be run in a separate terminal window. Remember to source the setup file in each new terminal window you open:

```bash
cd ~/catkin_ws
source devel/setup.bash
```

1. **Start ROS master**
    ```bash
    roscore
    ```
    Keep this running in its own terminal window.

2. **Start camera node** (in a new terminal window)
    ```bash
    roslaunch dope camera.launch  # Publishes RGB images to `/dope/webcam_rgb_raw`
    ```

    The camera must publish a correct `camera_info` topic to enable DOPE to compute the correct poses. Basically all ROS drivers have a `camera_info_url` parameter where you can set the calibration info (but most ROS drivers include a reasonable default).

    For details on calibration and rectification of your camera see the [camera tutorial](doc/camera_tutorial.md).

3. **Edit config info** (if desired) in `~/catkin_ws/src/dope/config/config_pose.yaml`
    * `topic_camera`: RGB topic to listen to
    * `topic_camera_info`: camera info topic to listen to
    * `topic_publishing`: topic namespace for publishing
    * `input_is_rectified`: Whether the input images are rectified. It is strongly suggested to use a rectified input topic.
    * `downscale_height`: If the input image is larger than this, scale it down to this pixel height. Very large input images eat up all the GPU memory and slow down inference. Also, DOPE works best when the object size (in pixels) has appeared in the training data (which is downscaled to 400 px). For these reasons, downscaling large input images to something reasonable (e.g., 400-500 px) improves memory consumption, inference speed *and* recognition results.
    * `weights`: dictionary of object names and there weights path name, **comment out any line to disable detection/estimation of that object**
    * `dimensions`: dictionary of dimensions for the objects  (key values must match the `weights` names)
    * `class_ids`: dictionary of class ids to be used in the messages published on the `/dope/detected_objects` topic (key values must match the `weights` names)
    * `draw_colors`: dictionary of object colors (key values must match the `weights` names)
    * `model_transforms`: dictionary of transforms that are applied to the pose before publishing (key values must match the `weights` names)
    * `meshes`: dictionary of mesh filenames for visualization (key values must match the `weights` names)
    * `mesh_scales`: dictionary of scaling factors for the visualization meshes (key values must match the `weights` names)
    * `overlay_belief_images`: whether to overlay the input image on the belief images published on /dope/belief_[obj_name]
    * `thresh_angle`: undocumented
    * `thresh_map`: undocumented
    * `sigma`: undocumented
    * `thresh_points`: Thresholding the confidence for object detection; increase this value if you see too many false positives, reduce it if objects are not detected.

4. **Start DOPE node** (in a new terminal window)
    ```bash
    roslaunch dope dope.launch [config:=/path/to/my_config.yaml]  # Config file is optional; default is `config_pose.yaml`
    ```

## Debugging

* The following ROS topics are published (assuming `topic_publishing == 'dope'`):
    ```
    /dope/belief_[obj_name]    # belief maps of object
    /dope/dimension_[obj_name] # dimensions of object
    /dope/pose_[obj_name]      # timestamped pose of object
    /dope/rgb_points           # RGB images with detected cuboids overlaid
    /dope/detected_objects     # vision_msgs/Detection3DArray of all detected objects
    /dope/markers              # RViz visualization markers for all objects
    ```
    *Note:* `[obj_name]` is in {brick_duplo_2x4_bleu, brick_duplo_2x4_jaune, brick_duplo_2x4_rouge, brick_duplo_2x4_vert}

* To debug in RViz (in a new terminal window):
    1. Run `rviz`
    2. Add one or more of the following displays:
        * `Add > Image` to view the raw RGB image or the image with cuboids overlaid
        * `Add > Pose` to view the object coordinate frame in 3D.
        * `Add > MarkerArray` to view the cuboids, meshes etc. in 3D.
        * `Add > Camera` to view the RGB Image with the poses and markers from above.

**Remember, each of these components (roscore, camera node, DOPE node, and RViz) should be run in separate terminal windows, and you need to source the setup file in each window:**

```bash
cd ~/catkin_ws
source devel/setup.bash
```

This ensures that each component has access to the necessary ROS environment variables and packages.


## In Case

in case of a camera info manager dependncy error pops up do this inside **src** catkin workspace:
```
git clone https://github.com/ros-perception/camera_info_manager_py.git
```


## ChangeLog

#### Added

- **Enhanced Incomplete Cuboid Handling**
  - The object detection process now estimates missing cuboid points instead of skipping detections when some points are missing.
  - A new `estimate_missing_points` function has been implemented (starting from line 304), which attempts to fill in the missing points by leveraging the geometric properties of the cuboid.
  - The estimation uses a combination of symmetry, vector-based extrapolation, and the center point as a reference to predict the positions of missing points.
  - If enough points (at least 4 corner points plus the center) are available, the algorithm estimates the missing points using known cuboid dimensions and validates the estimations.
  - The `validate_estimated_point` function, located in the same file, checks the estimated points against expected cuboid dimensions and adjusts them to ensure they are consistent with the geometric constraints.

#### Changed

- **Modified `find_object_poses` Method ([Line 304 in `detector.py`](./dope/src/dope/inference/detector.py#L304)):**
  - The `find_object_poses` method now incorporates the `estimate_missing_points` function (starting from line 304) to handle incomplete cuboids more robustly.
  - Instead of skipping detections with missing points, the method attempts to estimate and reconstruct the full cuboid.
  - This change allows the system to handle partial detections more effectively and reduces the likelihood of missed detections.

#### Fixed

- **Improved Robustness of Object Detection:**
  - By implementing the estimation of missing points, the detection process is now more resilient to incomplete data, resulting in more consistent and reliable object detection.
