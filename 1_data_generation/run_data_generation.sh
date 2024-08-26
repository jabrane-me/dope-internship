#!/bin/bash

# Default Values (change these if needed)
NB_RUNS=1
RUN_ID=0
WIDTH=960
HEIGHT=540
FOCAL_LENGTH=35.0  # Example default focal length, change if needed
DISTRACTORS_FOLDER="./google_scanned_models/"
OBJS_FOLDER="./models/"
PATH_SINGLE_OBJ=""
SCALE=0.01
BACKGROUNDS_FOLDER="./dome_hdri_haven/"
MIN_OBJECTS=1  # Minimum number of objects
MAX_OBJECTS=5  # Maximum number of objects
MIN_DISTRACTORS=1  # Minimum number of distractors
MAX_DISTRACTORS=10  # Maximum number of distractors
DISTRACTOR_SCALE=100
NB_FRAMES=2
MIN_PIXELS=20
OUTF="output/"
DEBUG=false

# Parsing command-line arguments (optional, you can pass values directly)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nb_runs) NB_RUNS="$2"; shift ;;
        --run_id) RUN_ID="$2"; shift ;;
        --width) WIDTH="$2"; shift ;;
        --height) HEIGHT="$2"; shift ;;
        --focal_length) FOCAL_LENGTH="$2"; shift ;;
        --distractors_folder) DISTRACTORS_FOLDER="$2"; shift ;;
        --objs_folder) OBJS_FOLDER="$2"; shift ;;
        --path_single_obj) PATH_SINGLE_OBJ="$2"; shift ;;
        --object_class) OBJECT_CLASS="$2"; shift ;;
        --scale) SCALE="$2"; shift ;;
        --backgrounds_folder) BACKGROUNDS_FOLDER="$2"; shift ;;
        --min_objects) MIN_OBJECTS="$2"; shift ;;
        --max_objects) MAX_OBJECTS="$2"; shift ;;
        --min_distractors) MIN_DISTRACTORS="$2"; shift ;;
        --max_distractors) MAX_DISTRACTORS="$2"; shift ;;
        --distractor_scale) DISTRACTOR_SCALE="$2"; shift ;;
        --nb_frames) NB_FRAMES="$2"; shift ;;
        --min_pixels) MIN_PIXELS="$2"; shift ;;
        --outf) OUTF="$2"; shift ;;
        --debug) DEBUG=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Function to generate a random number within a given range
generate_random_number() {
    local min=$1
    local max=$2
    echo $(( ( RANDOM % (max - min + 1) ) + min ))
}

# Generate random numbers for objects and distractors
NB_OBJECTS=$(generate_random_number $MIN_OBJECTS $MAX_OBJECTS)
NB_DISTRACTORS=$(generate_random_number $MIN_DISTRACTORS $MAX_DISTRACTORS)

# Print generated values
echo "Randomly selected number of objects: $NB_OBJECTS"
echo "Randomly selected number of distractors: $NB_DISTRACTORS"

# Run the Blender data generation
for (( i=1; i<=NB_RUNS; i++ ))
do
    echo "Running data generation - Iteration $i"
    python generate_training_data.py \
        --run_id "$i" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --focal-length "$FOCAL_LENGTH" \
        --distractors_folder "$DISTRACTORS_FOLDER" \
        --objs_folder "$OBJS_FOLDER" \
        --path_single_obj "$PATH_SINGLE_OBJ" \
        --scale "$SCALE" \
        --backgrounds_folder "$BACKGROUNDS_FOLDER" \
        --nb_objects "$NB_OBJECTS" \
        --nb_distractors "$NB_DISTRACTORS" \
        --distractor_scale "$DISTRACTOR_SCALE" \
        --nb_frames "$NB_FRAMES" \
        --min_pixels "$MIN_PIXELS" \
        --outf "$OUTF" \
        $( [ "$DEBUG" = true ] && echo "--debug" )
done

# Run the BlenderProc data generation
echo "Running BlenderProc data generation"
python run_blenderproc_datagen.py --some_option "$YOUR_OPTION_HERE"  # Add your specific options here

echo "Data generation completed."
