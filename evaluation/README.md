# Evaluation — Human Review Pipeline

This folder contains the tools used to **manually evaluate** the indicator detection pipeline
(SAM3 segmentation → vehicle association → classification → light ON/OFF detection)
across day and night sequences.

---

## Folder Structure

```
Evaluation/
├── generate_evaluation_sample.py   # Step 1 — create evaluation CSVs from the full data
├── evaluation_tool.py              # Step 2 — interactive human review tool
│
├── Input/                          # Sampled CSVs (input to the evaluation tool)
│   ├── sam3_evaluation_sample_day_part_1.csv
│   ├── sam3_evaluation_sample_day_part_2.csv
│   └── sam3_evaluation_sample_night_part_1.csv
│
├── Output/                         # Metric results per frame (output of the evaluation)
│   ├── results_sam3_evaluation_day.csv
│   └── results_sam3_evaluation_night.csv
```

---

## Step 1 — Generating the Evaluation Sample

**Script:** [`generate_evaluation_sample.py`](generate_evaluation_sample.py)

### What it does

The full camera-output CSV from the pipeline (`Day_camera_output.csv`, etc.) contains
hundreds of thousands of rows. Evaluating all of them manually is not feasible.
This script selects a **representative, stratified random subset** of frames.

**Stratified sampling strategy:**

| Pool | Condition | Target share |
|---|---|---|
| Lights ON | Pipeline predicts ≥ 1 indicator is ON | 50 % |
| Lights OFF | Pipeline predicts all indicators are OFF | 50 % |

This 50/50 balance ensures that both false positives (wrongly lit indicators) and
false negatives (missed blinks) are equally represented in the evaluation set,
avoiding a biased metric where everything looks correct because the dataset is
dominated by OFF frames.

Within each pool, frames are distributed **evenly across all cameras** (front/rear/left/right)
so that no single viewpoint dominates the sample.

**Filtering criteria applied before sampling:**

- Frame must have ≥ `MIN_TRACKS_PER_IMAGE` tracked vehicles (default: 3)
- `frame_valid_masks` > 0 (pipeline produced at least one valid mask)
- All valid masks must be associated (`total_assoc == frame_valid`)

After sampling, frames are randomly shuffled and split into chunks of
`IMAGES_PER_CSV` rows (default: 50) so evaluation can be done in multiple
short sessions.

### Configuration

Edit the block at the top of the script:

```python
CSV_PATH             = 'Data/day/Day_camera_output.csv'   # Full pipeline output
OUTPUT_PREFIX        = 'Evaluation/Input/sam3_evaluation_sample_day'

N_IMAGES_TOTAL       = 100   # Total frames to sample
MIN_TRACKS_PER_IMAGE = 3     # Minimum tracked vehicles per frame
IMAGES_PER_CSV       = 50    # Frames per output chunk
```

### Input / Output

| Direction | File |
|---|---|
| **Input** | `Data/day/Day_camera_output.csv` — full camera-output CSV from `process_all_valid_images_with_sam` |
| **Output** | `Evaluation/Input/sam3_evaluation_sample_day_part_N.csv` — one file per chunk |

---

## Step 2 — Running the Evaluation Tool

**Script:** [`evaluation_tool.py`](evaluation_tool.py)

### What it does

The tool loops over every sampled frame and, for each SAM3 mask detected in that frame,
presents a two-part visual display:

1. **Film strip (top row):** Five consecutive frames centred on the current frame
   (t−2, t−1, **t**, t+1, t+2) cropped tightly around the mask centroid.
   This provides temporal context — e.g. to confirm whether a light is actually
   blinking rather than a reflection.

2. **Full camera frame (bottom row):** The entire undistorted frame with:
   - All tracked vehicle bounding boxes (cyan = other vehicles, lime = the assigned vehicle)
   - A **red ring + yellow arrow** pointing to the specific mask being evaluated

For each mask the evaluator answers a **cascading series of yes/no questions**
that match the four pipeline stages:

```
Stage 1 — SAM3 Segmentation
  Q: Is the red ring on a real indicator? (1=Yes / 0=No)
    YES → sam3_tp += 1
      Stage 2 — Association
        Q: Is the car assignment (lime box) correct?
          YES → assoc_tp += 1
            Stage 3 — Classification
              Q: Is the indicator side/position label correct?
                YES → class_tp += 1
                  Stage 4 — Light ON/OFF
                    Q: Is the light actually ON? (check film strip)
                    → Fills light_tp / light_fp / light_tn / light_fn
                NO → class_fp += 1
          NO → assoc_fp += 1
    NO → sam_fp += 1

After all masks in a frame:
  Q: How many TRUE indicators are visible in the full image?
  → sam_fn = max(0, answer − sam_tp)   # missed masks
```

Results are written to the output CSV **after every frame** — so the session can be
interrupted and resumed at any time without losing progress (crash recovery).

### Configuration

```python
seq_dir    = 'Data/day/'                                        # Folder with images + calibration.json
INPUT_CSV  = 'Evaluation/Input/sam3_evaluation_sample_day_part_1.csv'
OUTPUT_CSV = 'Evaluation/Output/results_sam3_evaluation_day.csv'
```

### Input / Output

| Direction | File |
|---|---|
| **Input** | `Evaluation/Input/sam3_evaluation_sample_day_part_N.csv` — chunk CSV from Step 1 |
| **Input** | `Data/day/calibration.json` — camera calibration (undistort functions) |
| **Input** | `Data/day/{camera_name}/*.jpg` — per-camera image folders |
| **Output** | `Evaluation/Output/results_sam3_evaluation_day.csv` — one row per evaluated frame |

#### Output CSV columns

| Column | Description |
|---|---|
| `timestamp` | ROS2 timestamp of the frame |
| `camera` | Camera name |
| `true_total_indicators` | Human count of all visible indicators in the frame |
| `sam3_generated_masks` | Number of masks the pipeline produced |
| `sam3_tp / fp / fn` | SAM3 segmentation accuracy |
| `assoc_tp / fp` | Vehicle association accuracy |
| `class_tp / fp` | Indicator classification accuracy |
| `light_tp / fp / tn / fn` | Light ON/OFF detection accuracy |

---

## Example Outputs

### Correct association + blinking detection

![Association and blinking](Association-Blinking.png)

The lime bounding box shows the correctly associated vehicle. The red ring is on the
rear-left indicator which the pipeline correctly classified as BLINKING.

### Missed masks (False Negatives)

![Missing masks](MissingMasks.png)

SAM3 failed to segment the indicators on the distant vehicles. These are counted as
`sam3_fn` in the final metrics — correctly prompted by the end-of-frame question
*"How many TRUE indicators are visible in this image?"*
