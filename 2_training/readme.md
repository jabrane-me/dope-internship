# Deep Object Pose Estimation (DOPE) - Training

This repository contains a simplified version of the training pipeline for DOPE.

## Table of Contents
1. [Installing Dependencies](#installing-dependencies)
2. [Training Script Options](#training-script-options)
3. [Example Usage](#example-usage)
4. [Plotting Loss Graphs](#plotting-loss-graphs)

## Installing Dependencies

**Note:**  
It is highly recommended to install these dependencies in a virtual environment. You can create and activate a virtual environment by running:

```bash
python -m venv ./output/dope_training
source ./output/dope_training/bin/activate
```

To install the required dependencies, run:

```bash
pip install -r ../requirements.txt
```

## Training Script Options

The training script offers several options that can be controlled via command-line arguments:

- `--data`: Path to training data (required).
- `--val_data`: Path to validation data (required).
- `--object`: Object(s) to train the network for (required).
- `--workers`: Number of data loading workers (default: 8).
- `--batchsize`: Input batch size (default: 32).
- `--imagesize`: The height/width of the input image to the network (default: 512).
- `--lr`: Learning rate (default: 0.0001).
- `--net_path`: Path to the network for continuing training.
- `--epochs`: Number of epochs to train for (default: 60).
- `--exts`: Extensions for images to use (default: ["png"]).
- `--outf`: Folder to output images and model checkpoints (default: "output2/").
- `--pretrained`: Use pretrained weights. Must also specify `--net_path`.

## Example Usage

To run the training script, at minimum, the `--data` and `--object` flags must be specified if training with data that is stored locally. For example:

```bash
python train_.py --data ../data_generation/blenderproc_data_gen/output2/dataset --val_data ../data_generation/blenderproc_data_gen/output2/valset --object brick_duplo_2x4_jaune --epochs 200 --batchsize 32
```

The training script uses the Adam optimizer by default, which can be changed in the code (line 107).

## Plotting Loss Graphs

A mechanism has been added to plot loss graphs for each epoch during training. The training script automatically saves loss graphs for training and validation loss, as well as specific loss components like Train Affinities Loss and Train Belief Loss.

Additionally, it tracks and plots the average loss across all epochs. These graphs are saved in the `graphs` directory within the output folder specified by the `--outf` option. The plots provide valuable insights into the training progress and help monitor the model's performance.