# Evaluation Metrics - ADD and KPD

This directory provides scripts for evaluating object pose estimation results using two metrics:

1. **ADD (Average Distance of Model Points)**: A 3D pose evaluation metric.
2. **KPD (KeyPoint Distance)**: A 2D image-based pose evaluation metric.

## 1. ADD Metrics Computation

The ADD metric measures the average distance between corresponding points on the predicted 3D model and the ground truth 3D model. This metric is suitable when a 3D model is available.

### How to Run ADD Metrics

To compute the ADD metric, you need the following:

- Predictions for the object poses in the image.
- Ground truth poses for the same object.
- The corresponding 3D model for the object.

Once you have the necessary data, you can compute the ADD metric by running the following command:

```bash
python add_compute.py
```

### Available Options

The script provides several options to customize the evaluation:

- `--data_prediction`: Path to the folder containing the predicted pose data. Default is `data/table_dope_results/`.
- `--data`: Path to the folder containing the ground truth data. Default is `data/table_ground_truth/`.
- `--models`: Path to the folder containing the 3D models. Default is `content/`.
- `--outf`: Folder where the evaluation results will be saved. Default is `results/`.
- `--adds`: Run ADDS (symmetric objects). This option might take longer to compute.
- `--cuboid`: Use cuboid shape to compute ADD.
- `--show`: Display the evaluation graph at the end of the computation.

### Example Command

```bash
python add_compute.py --data_prediction path/to/predictions --data path/to/ground_truth --models path/to/3d_models --outf output_results/ --adds --show
```

## 2. KPD Metrics Computation

The KPD (KeyPoint Distance) metric is a 2D image-based evaluation metric that calculates the Euclidean (L2) distance between predicted and ground truth keypoints. This is useful when a 3D model is not available, and you want to evaluate the quality of the detections based on keypoints.

### How to Run KPD Metrics

To compute the KPD metric, you can run the following command:

```bash
python kpd_compute.py
```

### Available Options

The script offers similar options to the ADD computation:

- `--data_prediction`: Path to the folder containing the predicted keypoint data. Default is `data/table_dope_results/`.
- `--data`: Path to the folder containing the ground truth keypoint data. Default is `data/table_ground_truth/`.
- `--outf`: Folder where the evaluation results will be saved. Default is `results_kpd/`.
- `--show`: Display the evaluation graph at the end of the computation.

### Example Command

```bash
python kpd_compute.py --data_prediction path/to/predictions --data path/to/ground_truth --outf output_results_kpd/ --show
```

## Notes

- Both scripts support displaying graphs of the evaluation results using the `--show` option.
- Ensure that the paths to your data, ground truth, and models are correctly set for accurate evaluation.