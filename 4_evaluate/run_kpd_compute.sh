#!/bin/bash

total_accuracy=0
count=0

prediction_dir="../train2/output2/GREEN4"
output_directory="results_kpd/GREEN4_200"
ground_truth_dir="../data_generation/blenderproc_data_gen/output2/test44"
output_file="$output_directory/accuracy_results.txt"

mkdir -p "$output_directory"

> "$output_file"

for batch_dir in $prediction_dir/*; do
    batch_num=$(basename $batch_dir)
    
    # Check if the corresponding ground truth directory exists
    if [ -d "$ground_truth_dir/$batch_num" ]; then
        # Run the Python script and capture its output
        output=$(python kpd_compute.py --data_prediction "$prediction_dir/$batch_num" --data "$ground_truth_dir/$batch_num" --outf "$output_directory/$batch_num")
        
        # Extract the accuracy from the output
        accuracy=$(echo "$output" | python extract_accuracy.py)
        
        # Check
        if [[ $accuracy != "Accuracy not found" ]]; then
            echo "Batch $batch_num: $accuracy"
            echo "Batch $batch_num: $accuracy" >> "$output_file"
            total_accuracy=$(echo "$total_accuracy + $accuracy" | bc)
            count=$((count + 1))
        else
            echo "Batch $batch_num: Accuracy not found"
            echo "Batch $batch_num: Accuracy not found" >> "$output_file"
        fi
    else
        echo "Ground truth for batch $batch_num not found"
        echo "Ground truth for batch $batch_num not found" >> "$output_file"
    fi
done

# Calculate the average accuracy
if [[ $count -gt 0 ]]; then
    average_accuracy=$(echo "scale=10; $total_accuracy / $count" | bc)
    echo "Total accuracy: $average_accuracy"
    echo "Total accuracy: $average_accuracy" >> "$output_file"
else
    echo "No valid accuracy values found"
    echo "No valid accuracy values found" >> "$output_file"
fi
