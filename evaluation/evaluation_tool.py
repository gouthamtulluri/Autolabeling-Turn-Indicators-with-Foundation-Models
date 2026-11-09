"""
evaluation_tool.py
──────────────────
Interactive Jupyter-based human evaluation tool for the indicator detection pipeline.

For each sampled frame the tool:
  1. Shows a 5-frame film strip centred on the indicator of interest (temporal context).
  2. Shows the full undistorted camera frame with bounding boxes and a red marker on
     the specific mask being evaluated.
  3. Asks a cascading series of yes/no questions that mirror the pipeline stages:
       Stage 1 — SAM3 Segmentation:  is the red ring on a real indicator?
       Stage 2 — Association:        is the car assignment correct?
       Stage 3 — Classification:     is the indicator side/position label correct?
       Stage 4 — Light ON/OFF:       does the indicator match the predicted state?

Results are saved to a CSV after every frame so the session can be interrupted and
resumed without losing work (crash recovery is built in).

Usage
-----
Run inside a Jupyter notebook cell:
    %run Evaluation/evaluation_tool.py

Or copy-paste the content into a notebook cell after editing CONFIGURATION.

Inputs
------
  INPUT_CSV     One of the chunk CSVs produced by generate_evaluation_sample.py
                e.g. 'Evaluation/Input/sam3_evaluation_sample_day_part_1.csv'
  seq_dir       Root folder containing per-camera image folders and calibration.json

Outputs
-------
  OUTPUT_CSV    One row per evaluated frame, written to Evaluation/Output/
                e.g. 'Evaluation/Output/results_sam3_evaluation_day.csv'

                Columns:
                  timestamp, camera,
                  true_total_indicators,   # human count of visible indicators
                  sam3_generated_masks,
                  sam3_tp, sam3_fp, sam3_fn,
                  assoc_tp, assoc_fp,
                  class_tp, class_fp,
                  light_tp, light_fp, light_tn, light_fn
"""

import os
import cv2
import json
import ast
import time
import random

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, clear_output

import sys
sys.path.insert(0, '.')
from Functions.common import load_calibration_file

# ==========================================
# ⚙️  CONFIGURATION — edit only this block
# ==========================================
seq_dir    = 'Data/day/'
INPUT_CSV  = 'Evaluation/Input/sam3_evaluation_sample_day_part_1.csv'
OUTPUT_CSV = 'Evaluation/Output/results_sam3_evaluation_day.csv'
# ==========================================

calib_data = load_calibration_file(os.path.join(seq_dir, 'calibration.json'))

print(f"Loading {INPUT_CSV}...")
df      = pd.read_csv(INPUT_CSV)
grouped = df.groupby(['timestamp', 'camera', 'camera_timestamp'])

# ── Crash recovery: resume from last checkpoint ───────────────────────────────
if os.path.exists(OUTPUT_CSV):
    prev            = pd.read_csv(OUTPUT_CSV)
    results         = prev.to_dict('records')
    completed       = set(zip(prev['timestamp'], prev['camera']))
    print(f"  Resuming — {len(completed)} frames already evaluated.")
else:
    results   = []
    completed = set()

os.makedirs(os.path.dirname(OUTPUT_CSV) or '.', exist_ok=True)

# ── Main evaluation loop ──────────────────────────────────────────────────────
for (ts, cam, img_time), group in grouped:
    if (ts, cam) in completed:
        continue

    cam_dir = os.path.join(seq_dir, cam)

    # Find ±2 temporal frames for the film strip
    all_imgs = sorted(
        [f for f in os.listdir(cam_dir) if f.endswith('.jpg')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    current_img_name = f"{int(img_time)}.jpg"
    img_path         = os.path.join(cam_dir, current_img_name)

    if not os.path.exists(img_path):
        print(f"  Image not found: {img_path} — skipping.")
        continue

    try:
        idx            = all_imgs.index(current_img_name)
        temporal_paths = [
            os.path.join(cam_dir, all_imgs[i])
            for i in range(max(0, idx - 2), min(len(all_imgs), idx + 3))
        ]
    except ValueError:
        temporal_paths = [img_path]

    # Pre-load and undistort all temporal frames
    undistort_fn    = calib_data['cameras'][cam]['undistort_function']
    temporal_frames = []
    for tp in temporal_paths:
        raw = cv2.imread(tp)
        if raw is not None:
            flat = undistort_fn(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            temporal_frames.append((os.path.basename(tp), flat))

    flat_img_rgb = next(img for name, img in temporal_frames if name == current_img_name)

    # Build centroids list for this frame
    centroids = []
    for _, row in group.iterrows():
        if pd.isna(row.get('associated_masks')):
            continue
        try:
            masks = ast.literal_eval(row['associated_masks'])
        except Exception:
            continue

        for m in masks:
            classification = 'Unknown (Discarded / Not Assigned)'
            light_on_pred  = False

            for col in ['rear_left_indicator', 'rear_right_indicator',
                        'front_left_indicator', 'front_right_indicator']:
                val = row.get(col)
                if pd.notna(val):
                    try:
                        d = ast.literal_eval(val)
                        if d.get('mask_idx') == m['mask_idx']:
                            classification = (col.replace('_indicator', '')
                                                 .replace('_', ' ').title())
                            light_on_pred  = d.get('light_on', False)
                            break
                    except Exception:
                        pass

            centroids.append({
                'cx':             int(m['cx']),
                'cy':             int(m['cy']),
                'track_id':       row['track_id'],
                'classification': classification,
                'light_on_pred':  light_on_pred,
                'box_cx':         row.get('center_x'),
                'box_cy':         row.get('center_y'),
                'box_w':          row.get('box_width'),
                'box_h':          row.get('box_height'),
            })

    sam_tp = sam_fp = 0
    assoc_tp = assoc_fp = 0
    class_tp = class_fp = 0
    light_tp = light_fp = light_tn = light_fn = 0

    for i, c in enumerate(centroids):
        cx, cy = c['cx'], c['cy']
        crop_size = 150

        plt.close('all')
        clear_output(wait=False)
        time.sleep(0.2)

        fig = plt.figure(figsize=(24, 18))
        gs  = fig.add_gridspec(2, 5, height_ratios=[1, 3])

        # Film strip (top row)
        for j, (t_name, t_img) in enumerate(temporal_frames):
            ax_t = fig.add_subplot(gs[0, j])
            y1c = max(0, cy - crop_size); y2c = min(t_img.shape[0], cy + crop_size)
            x1c = max(0, cx - crop_size); x2c = min(t_img.shape[1], cx + crop_size)
            ax_t.imshow(t_img[y1c:y2c, x1c:x2c])
            ax_t.axis('off')
            if t_name == current_img_name:
                ax_t.set_title('CURRENT FRAME', color='lime', weight='bold',
                                fontsize=16, bbox=dict(facecolor='black', alpha=0.7))
                ax_t.add_patch(patches.Circle(
                    (cx - x1c, cy - y1c), radius=15,
                    linewidth=3, edgecolor='red', facecolor='none'))
            else:
                ax_t.set_title(t_name.replace('.jpg', ''), color='white',
                                fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

        # Full frame (bottom row)
        ax_main = fig.add_subplot(gs[1, :])
        ax_main.imshow(flat_img_rgb)
        ax_main.axis('off')
        ax_main.add_patch(patches.Circle(
            (cx, cy), radius=25, linewidth=4,
            edgecolor='red', facecolor='none', zorder=20))
        ax_main.annotate(
            '', xy=(cx, cy - 25), xytext=(cx, cy - 120),
            arrowprops=dict(facecolor='yellow', edgecolor='red', width=5, headwidth=15),
            zorder=20)

        for _, row in group.iterrows():
            bx = row['center_x'] - row['box_width']  / 2
            by = row['center_y'] - row['box_height'] / 2
            is_target = (row['track_id'] == c['track_id'])
            colour = 'lime' if is_target else 'cyan'
            lw     = 4      if is_target else 2
            ax_main.add_patch(patches.Rectangle(
                (bx, by), row['box_width'], row['box_height'],
                linewidth=lw, edgecolor=colour, facecolor='none',
                zorder=10 if is_target else 5))
            ax_main.text(
                bx, by - 10, f"Car {int(row['track_id'])}",
                color=colour, fontsize=18 if is_target else 14, weight='bold',
                bbox=dict(facecolor='black', alpha=0.5, pad=3),
                zorder=10 if is_target else 5)

        fig.suptitle(
            f"Evaluating Mask {i + 1} of {len(centroids)} | "
            f"Is the red ring directly on an indicator?",
            fontsize=22, weight='bold')
        plt.tight_layout()
        display(fig)
        plt.close(fig)

        # ── Stage 1: SAM3 segmentation ──────────────────────────────────────
        while True:
            ans = input(f"Mask {i + 1}: Is the red ring on a real indicator? (1=Yes / 0=No): ").strip()
            if ans in ('0', '1'): break

        if ans == '1':
            sam_tp += 1

            # ── Stage 2: Association ─────────────────────────────────────────
            while True:
                ans = input(f"  Assigned to Car {c['track_id']} (lime box). Correct? (1=Yes / 0=No): ").strip()
                if ans in ('0', '1'): break

            if ans == '1':
                assoc_tp += 1

                # ── Stage 3: Classification ──────────────────────────────────
                while True:
                    ans = input(f"  Classified as [{c['classification']}]. Correct? (1=Yes / 0=No): ").strip()
                    if ans in ('0', '1'): break

                if ans == '1':
                    class_tp += 1

                    # ── Stage 4: Light ON / OFF ──────────────────────────────
                    pred_on = c['light_on_pred']
                    while True:
                        ans = input(f"  Is the indicator light actually ON? (check film strip) (1=Yes / 0=No): ").strip()
                        if ans in ('0', '1'): break

                    actual_on = (ans == '1')
                    if actual_on and pred_on:          light_tp += 1
                    elif actual_on and not pred_on:    light_fn += 1
                    elif not actual_on and pred_on:    light_fp += 1
                    else:                              light_tn += 1
                else:
                    class_fp += 1
            else:
                assoc_fp += 1
        else:
            sam_fp += 1

    # ── Frame-level FN count: how many TRUE indicators did we miss entirely? ──
    clear_output(wait=False)
    time.sleep(0.2)
    fig_fn = plt.figure(figsize=(24, 10))
    plt.imshow(flat_img_rgb)
    plt.title(
        f"{cam}  |  ts={ts}\n"
        f"You verified {sam_tp} correct mask(s). "
        f"How many TRUE indicators are visible in the full image?",
        fontsize=20)
    plt.axis('off')
    display(fig_fn)
    plt.close(fig_fn)

    while True:
        user_in = input("How many TRUE indicators are visible on any target vehicle? ").strip()
        if user_in.isdigit():
            val = int(user_in)
            if input(f"  You entered {val}. Confirm? (y/n): ").strip().lower() == 'y':
                true_indicator_count = val
                break
        else:
            print("  Please enter a number.")

    sam_fn = max(0, true_indicator_count - sam_tp)

    results.append({
        'timestamp':             ts,
        'camera':                cam,
        'true_total_indicators': true_indicator_count,
        'sam3_generated_masks':  len(centroids),
        'sam3_tp':   sam_tp,   'sam3_fp':  sam_fp,  'sam3_fn':  sam_fn,
        'assoc_tp':  assoc_tp, 'assoc_fp': assoc_fp,
        'class_tp':  class_tp, 'class_fp': class_fp,
        'light_tp':  light_tp, 'light_fp': light_fp,
        'light_tn':  light_tn, 'light_fn': light_fn,
    })

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    plt.close('all')

clear_output(wait=False)
print(f"Evaluation complete.  Results saved to {OUTPUT_CSV}")
