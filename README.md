# Entrance Behavior Vision Pipeline

A Vision pipeline that classifies normal activities and anomalous behaviors in entrance camera footage on a block-by-block basis.

## Classification Labels

| Label | Meaning                                                                           |
| ----- | --------------------------------------------------------------------------------- |
| N1    | Normal entrance activity: resident entry/exit, regular delivery, waiting, leaving |
| A17   | Repeatedly pressing the doorbell                                                  |
| A18   | Repeated attempts to open the door                                                |
| A19   | Repeatedly kicking the door                                                       |
| A20   | Attempting to look inside the door                                                |
| A21   | Prolonged or repeated knocking on the door                                        |

## Data Split

| Split      | Data Used                                                                            |
| ---------- | ------------------------------------------------------------------------------------ |
| Train      | AIHub TL/TS, mydata, and augmented data derived from the original train data         |
| Validation | Original AIHub VL/VS videos only                                                     |
| Test       | Separate exhibition camera footage or held-out field footage to be constructed later |

`normal` and `delivery` from mydata are used as N1, while `lookingInside` is used as A20.

## Processing Pipeline

```text
Video
→ YOLO person detection + BoT-SORT tracking
→ MediaPipe Pose extraction
→ Aggregation of behavior and pose features by annotation block
→ XGBoost classification
→ N1 / A17~A21 predictions
```

* Tracking and pose extraction are performed only once per video.
* Feature extraction and training use the `start_frame` and `end_frame` ranges of each annotation block.
* Augmented training data uses the original block boundaries.

## Execution Order

Activate the virtual environment and run the following commands from the project root.

```powershell
python .\utils\load_annotation.py
python .\utils\generate_annotations.py
python .\tracking\tracker.py
python .\tracking\extract_behavior_features.py
python .\tracking\pose_extractor.py
python .\tracking\merge_features.py
python .\models\train_XGBoost.py
```

`tracker.py` and `pose_extractor.py` skip processing when per-video CSV files already exist, allowing the pipeline to be safely re-run after interruption.

## Generated Outputs

| Path                                           | Contents                                                          |
| ---------------------------------------------- | ----------------------------------------------------------------- |
| `splits/annotations/train_annotations.csv`     | AIHub train block manifest based on TL/TS                         |
| `splits/annotations/valid_annotations.csv`     | Validation block manifest based on VL/VS                          |
| `splits/annotations/train_annotations_all.csv` | Combined train manifest including AIHub + mydata + augmented data |
| `results/tracking/`                            | Per-video person tracking results                                 |
| `results/pose/`                                | Per-video frame-level pose features                               |
| `results/features/`                            | Merged block-level behavior and pose features                     |
| `results/models/xgboost_validation/`           | Validation report, confusion matrix, and model files              |

## Current Baseline Results

AIHub VL/VS hold-out validation results:

* Accuracy: **86.36%**
* Macro F1-score: **68.82%**
* Weighted F1-score: **86.20%**

Since N1 accounts for a large proportion of the data, Macro F1-score is evaluated alongside Accuracy rather than relying on Accuracy alone. These results represent validation performance, and final generalization performance should be further evaluated using exhibition camera footage or separate field test videos.

## Git Management

Original and augmented videos, execution results, automatically generated manifests, and model weights are excluded from Git using `.gitignore`. Only code and configuration files required for reproducibility are tracked in Git.

