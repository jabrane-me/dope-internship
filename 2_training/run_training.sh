#!/bin/bash

# Default Values
TRAIN_DATA=("../data/training_set/")  # Path to training data (can be multiple paths, space-separated)
VAL_DATA="../data/validation_set/"  # Path to validation data
OBJECTS=("brick_duplo_2x4_vert")  # Specify objects to train network for
EPOCHS=100
WORKERS=8
BATCHSIZE=32
IMAGESIZE=512
LEARNING_RATE=0.0001
NET_PATH=""  # Path to pre-trained network (leave blank if not resuming)
NAMEFILE="epoch"
EXTS=("png")  # File extensions to use (e.g., png, jpg)
OUTF="output_real/"
PRETRAINED=false

# Parsing command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data) IFS=" " read -r -a TRAIN_DATA <<< "$2"; shift ;;
        --val_data) VAL_DATA="$2"; shift ;;
        --object) IFS=" " read -r -a OBJECTS <<< "$2"; shift ;;
        --workers) WORKERS="$2"; shift ;;
        --batchsize | --batch_size) BATCHSIZE="$2"; shift ;;
        --imagesize) IMAGESIZE="$2"; shift ;;
        --lr) LEARNING_RATE="$2"; shift ;;
        --net_path) NET_PATH="$2"; shift ;;
        --namefile) NAMEFILE="$2"; shift ;;
        --epochs | --epoch | -e) EPOCHS="$2"; shift ;;
        --exts) IFS=" " read -r -a EXTS <<< "$2"; shift ;;
        --outf) OUTF="$2"; shift ;;
        --pretrained) PRETRAINED=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Print configuration
echo "Starting training with the following configuration:"
echo "Training Data: ${TRAIN_DATA[@]}"
echo "Validation Data: $VAL_DATA"
echo "Objects: ${OBJECTS[@]}"
echo "Workers: $WORKERS"
echo "Batch Size: $BATCHSIZE"
echo "Image Size: $IMAGESIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Net Path: $NET_PATH"
echo "Pretrained: $PRETRAINED"
echo "Number of Epochs: $EPOCHS"
echo "Output Folder: $OUTF"
echo "File Extensions: ${EXTS[@]}"

# Construct the training command
TRAIN_CMD="python train.py --data ${TRAIN_DATA[@]} --val_data $VAL_DATA --object ${OBJECTS[@]} \
    --workers $WORKERS --batchsize $BATCHSIZE --imagesize $IMAGESIZE --lr $LEARNING_RATE \
    --namefile $NAMEFILE --epochs $EPOCHS --exts ${EXTS[@]} --outf $OUTF"

# Add optional arguments if specified
if [ ! -z "$NET_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --net_path $NET_PATH"
fi

if [ "$PRETRAINED" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --pretrained"
fi

# Execute the training command
echo "Executing: $TRAIN_CMD"
eval $TRAIN_CMD
