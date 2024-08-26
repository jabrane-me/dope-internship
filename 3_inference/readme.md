# Deep Object Pose Estimation (DOPE) - Inference

This directory contains a simple example of running inference with DOPE (Deep Object Pose Estimation). The provided script enables you to use a pre-trained DOPE model to perform object pose estimation on images.

## Setup

Before running inference, ensure that you have installed the necessary dependencies. If you haven't done so, install them by running the following command:

```bash
pip install -r ../requirements.txt
```

## Running Inference

The `inference.py` script allows you to perform inference using a trained model without relying on ROS components. You need to specify the path to the images on which you want to run the inference.

### Basic Usage

To run inference, use the following command:

```bash
python inference.py --data ../data_generation/blenderproc_data_gen/output/testset --outf out_experiment/test
```

### Configuration Files

There are two important configuration files that you may need to adjust before running inference:

1. **camera_info.yaml**: This file contains information about the camera used to capture the images. Ensure that this file is correctly configured based on your camera settings. You can extract relevant information from the JSON files in the generated datasets.

2. **config_pose.yaml**: This file contains object-specific parameters such as dimensions, class IDs, and weights. Make sure that the values in this file are correct for the object you wish to detect. Alternatively, you can define a new configuration file and specify it using the `--config` flag.

### Important Notes

- Ensure that the `camera_info.yaml` file is set correctly before running inference. The camera parameters should match the ones used during data generation or real-world data capture.
- The object-specific parameters such as `dimensions`, `class_ids`, and weights in the `config_pose.yaml` file (or any custom config file) should be correctly set for accurate detection.

## Script Options

The `inference.py` script provides several options to customize the inference process. Below is a list of the available command-line arguments:

- `--pause`: Time (in seconds) to pause between processing images. Default is 0 (no pause).
- `--showbelief`: Show the belief maps during inference. Use this flag to visualize the belief maps.
- `--dontshow`: Enable headless mode (no display during inference).
- `--outf`: Specify the output folder where the results will be stored. Default is `./output2/BLUE4/`.
- `--data`: Specify the folder containing the images to run inference on. Supported formats: `.png`, `.jpeg`, `.jpg`. Default is `../data_generation/blenderproc_data_gen/output2/test44/`.
- `--config`: Specify the path to the configuration file for inference. Default is `config_inference2/config_pose.yaml`.
- `--camera`: Specify the path to the camera info file. Default is `config_inference2/camera_info.yaml`.
- `--realsense`: Use this flag if you are running inference with a RealSense camera.

### Example Command

Here's an example of running inference on a dataset with custom configuration files:

```bash
python inference.py --data ../data_generation/blenderproc_data_gen/output/testset --outf out_experiment/test --config custom_configs/config_pose.yaml --camera custom_configs/camera_info.yaml --showbelief
```

In this example, inference will run on the images in the specified `testset` folder, using custom configuration files for the pose and camera information, and the belief maps will be displayed during inference.

## Additional Notes

- You can modify the output folder path using the `--outf` flag to organize results from different experiments.
- Make sure your camera and object parameters in the configuration files are well-defined to achieve accurate inference results.