# common Folder

This folder contains common utility methods and classes used throughout the DEEP_OBJECT_POSE code repository. These files provide essential functions for tasks such as solving the Perspective-n-Point (PnP) problem, drawing cuboids, and performing object detection.

## Files Overview

### 1. `cuboid_pnp_solver.py`
   This script provides methods to solve the Perspective-n-Point (PnP) problem, which is essential for determining the pose of a cuboid from 2D image points.

### 2. `cuboid.py`
   Contains methods for drawing and handling cuboids. This file is used to generate 3D cuboid representations in the context of object pose estimation.

### 3. `debug.py`
   This file includes various debugging utilities used to track and log information during the execution of the object detection and pose estimation processes.

### 4. `detector.py`
   Implements object detection methods. This file is crucial for identifying objects in images and serves as a backbone for the object detection pipeline in the repository.

### 5. `models.py`
   Contains the definitions and utilities related to machine learning models used in the object detection and pose estimation tasks.

### 6. `utils.py`
   A collection of utility functions that are widely used across the repository. These functions include general-purpose methods that assist with tasks such as data processing, mathematical operations, and more.

## Usage

These files are intended to be imported and used by other scripts in the repository. They provide fundamental building blocks that enable object detection and pose estimation tasks. Each file is designed to be modular, allowing it to be easily integrated into various parts of the repository.