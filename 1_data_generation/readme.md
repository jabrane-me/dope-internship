# Synthetic Data Generation

## Table of Contents
1. [Introduction](#introduction)
2. [Objects for Training](#1-objects-for-training)
3. [Backgrounds](#2-backgrounds)
4. [Google Scanned Models](#3-google-scanned-models)
5. [Blender Data Generation](#4-blender-data-generation)
   1. [Image Output Size](#image-output-size)
   2. [Camera Settings](#camera-settings)
   3. [Object Folders](#object-folders)
   4. [Object Settings](#object-settings)
   5. [Background Settings](#background-settings)
   6. [Number of Objects](#number-of-objects)
   7. [Frames](#frames)
   8. [Example of Usage](#example-of-usage)
6. [Validation](#5-validation)
7. [JSON Fields](#json-fields)

This directory contains code for data generation (both
images and associated JSON files) for training DOPE.
two variations of this code are provided, one that uses [NVISII](https://github.com/owl-project/NVISII) *(unmaintained)* for rendering, and another that
uses [Blenderproc](https://github.com/DLR-RM/BlenderProc) *(still maintained)*.

**This codes has been modified to work on a single GPU.**

Here is a possible README file for the `data_generation` folder that incorporates the details you've provided:


# Data Generation

This folder contains the necessary scripts and resources to generate training data for object detection models. The primary components include 3D object models, backgrounds, and scripts for generating synthetic training data using Blender.

## 1. Objects for Training

The models that we will train our detection model on need to be created in Blender and placed inside the `models` folder. In our case, the following objects were used:

- `brick_duplo_2x4_bleu`
- `brick_duplo_2x4_jaune`
- `brick_duplo_2x4_rouge`
- `brick_duplo_2x4_vert`

Ensure that the models are appropriately scaled and textured before use.

## 2. Backgrounds

Backgrounds are essential for rendering the objects in different environments. We recommend using HDRIs (High Dynamic Range Images), which can be downloaded for free from [Polyhaven](https://polyhaven.com/hdris). HDRIs provide realistic lighting and reflections by capturing the full range of lighting information from real-world environments.

The code takes a frame from the HDRI background and renders the objects onto it. To control the rotation of the selected frame, refer to `./blenderproc_data_gen/generate_training_data.py` at line 310 in the method `randomize_background()`.

## 3. Google Scanned Models

The `google_scanned_models` folder contains 3D object models created by Google. These models can be used as distractors in your dataset to make the object detection model more robust. The folder contains over 1000 3D objects. You can download these models using the following command:

```bash
python download_google_scanned_objects.py
```

## 4. Blender Data Generation

To generate data using Blender, you can control various options through the script. Below are the primary options you can set:

- **Image Output Size**:
  - `--width`: Image output width (default: 960).
  - `--height`: Image output height (default: 540).

- **Camera Settings**:
  - `--focal-length`: Focal length of the camera.

- **Object Folders**:
  - `--distractors_folder`: Folder containing distraction objects (default: `../google_scanned_models/`).
  - `--objs_folder`: Folder containing training objects (default: `../models/`).
  - `--path_single_obj`: Path to a single object file if using one.

- **Object Settings**:
  - `--object_class`: The class name of the objects. If not provided, the directory name is used.
  - `--scale`: Scaling to apply to the target objects (default: 50).

- **Background Settings**:
  - `--backgrounds_folder`: Folder containing background images. Supported formats: .jpeg, .png, .hdr (default: `../dome_hdri_haven/`).

- **Number of Objects**:
  - `--nb_objects`: Number of objects to include (default: 1).
  - `--nb_distractors`: Number of distractor objects (default: 1).
  - `--distractor_scale`: Scaling to apply to distractor objects (default: 50).

- **Frames**:
  - `--nb_frames`: Number of frames to generate (default: 200).

These options allow you to customize the data generation process to fit your needs.

### Example of Usage

For generating data for a single object:

```bash
python ./run_blenderproc_datagen.py --nb_runs 10 --outf ./output2/dataset/ --path_single_obj ../models/brick_duplo_2x4_rouge/google_16k/textured.obj
```

This command will generate 10 runs of data, outputting the results to the specified folder.

## 5. Validation

The `validate_data.py` script can be used to validate the generated data to ensure it meets the required quality and structure.


## JSON Fields

Each generated image is accompanied by a JSON file. This JSON file **must** contain the following fields to be used for training:

* `objects`: An array, containing one entry for each object instance, with:
    - `class`: class name. This name is referred to in configuration files.
    - `location` and `quaternion_xyzw`: position and orientation of the object in the *camera* coordinate system
    - `projected_cuboid`: 2D coordinates of the projection of the the vertices of the 3D bounding cuboid (in pixels) plus the centroid. See the above section "Projected Cuboid Corners" for more detail.
    - `visibility`: The visible fraction of the object silhouette (= `px_count_visib`/`px_count_all`). 
      Note that if NVISII is used, the object may still not be fully visible when `visibility == 1.0` because it may extend beyond the borders of the image.
      

### Optional Fields
These fields are not required for training, but are used for debugging and numerical evaluation of the results.  We recommend generating this data if possible.

* `camera_data`
    - `camera_view_matrix`: 4×4 transformation matrix from the world to the camera coordinate system.
    - `height` and `width`: dimensions of the image in pixels
    - `intrinsics`: the camera intrinsics


* `objects`
    - `local_cuboid`: 3D coordinates of the vertices of the 3D bounding cuboid (in meters); currently always `null`
    - `local_to_world_matrix`: 4×4 transformation matrix from the object to the world coordinate system
    - `name`: a unique string that identifies the object instance internally
