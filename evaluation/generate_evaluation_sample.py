"""
generate_evaluation_sample.py
─────────────────────────────
Generates a stratified random sample of frames from the full camera-output CSV.

The sample is balanced 50/50 between:
  • Frames where the pipeline predicts at least one indicator is ON (active light)
  • Frames where the pipeline predicts all indicators are OFF

Frames are distributed evenly across all cameras and then split into smaller
CSV chunks ready for the human evaluation tool.

Usage
-----
Edit the CONFIGURATION block below and run:
    python generate_evaluation_sample.py

Inputs
------
  CSV_PATH        Full camera-output CSV from the pipeline
                  e.g. 'Data/day/Day_camera_output.csv'

Outputs
-------
  OUTPUT_PREFIX   One or more numbered CSVs, e.g.
                  'Evaluation/Input/sam3_evaluation_sample_day_part_1.csv'
"""

import pandas as pd
import ast
import os
import random

# ==========================================
# ⚙️  CONFIGURATION — edit only this block
# ==========================================
CSV_PATH        = 'Data/day/Day_camera_output.csv'
OUTPUT_PREFIX   = 'Evaluation/Input/sam3_evaluation_sample_day'

N_IMAGES_TOTAL       = 100   # Total frames to sample
MIN_TRACKS_PER_IMAGE = 3     # Only keep frames with at least this many tracked vehicles
IMAGES_PER_CSV       = 50    # Frames per output CSV chunk
# ==========================================


def count_masks(mask_str):
    """Count how many masks were associated in this row."""
    if pd.isna(mask_str):
        return 0
    try:
        return len(ast.literal_eval(mask_str))
    except Exception:
        return 0


def check_any_light_on(row):
    """Return 1 if any indicator on this vehicle is predicted ON, else 0."""
    for col in ['rear_left_indicator', 'rear_right_indicator',
                'front_left_indicator', 'front_right_indicator']:
        val = row.get(col)
        if pd.notna(val):
            try:
                if ast.literal_eval(val).get('light_on', False):
                    return 1
            except Exception:
                pass
    return 0


def sample_across_cameras(pool, target_total):
    """
    Sample `target_total` frames distributed evenly across all cameras.
    Any shortfall from one camera is backfilled from the rest.
    """
    cameras = pool['camera'].unique()
    if len(cameras) == 0:
        return []

    images_per_cam = target_total // len(cameras)
    extra_images   = target_total % len(cameras)

    pool = pool.sample(frac=1, random_state=42).reset_index(drop=True)
    sampled_rows = []

    for cam in cameras:
        target = images_per_cam + (1 if extra_images > 0 else 0)
        extra_images -= 1
        cam_pool = pool[pool['camera'] == cam]
        for _, row in cam_pool.head(target).iterrows():
            sampled_rows.append(row)

    if len(sampled_rows) < target_total:
        shortfall = target_total - len(sampled_rows)
        used = {r['camera_timestamp'] for r in sampled_rows}
        remaining = pool[~pool['camera_timestamp'].isin(used)]
        for _, row in remaining.head(shortfall).iterrows():
            sampled_rows.append(row)

    return sampled_rows


print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

print("Extracting mask counts and light status...")
df['num_assoc']        = df['associated_masks'].apply(count_masks)
df['has_active_light'] = df.apply(check_any_light_on, axis=1)

image_stats = (
    df.groupby(['timestamp', 'camera', 'camera_timestamp'])
    .agg(
        total_assoc   = ('num_assoc',        'sum'),
        frame_valid   = ('frame_valid_masks', 'first'),
        num_tracks    = ('track_id',          'nunique'),
        active_lights = ('has_active_light',  'sum'),
    )
    .reset_index()
)

pool_on = image_stats[
    (image_stats['num_tracks']    >= MIN_TRACKS_PER_IMAGE) &
    (image_stats['frame_valid']   >  0) &
    (image_stats['total_assoc']   == image_stats['frame_valid']) &
    (image_stats['active_lights'] >  0)
].copy()

pool_off = image_stats[
    (image_stats['num_tracks']    >= MIN_TRACKS_PER_IMAGE) &
    (image_stats['frame_valid']   >  0) &
    (image_stats['total_assoc']   == image_stats['frame_valid']) &
    (image_stats['active_lights'] == 0)
].copy()

print(f"  ON pool: {len(pool_on)} frames  |  OFF pool: {len(pool_off)} frames")

HALF = N_IMAGES_TOTAL // 2
sample_on  = sample_across_cameras(pool_on,  HALF)
sample_off = sample_across_cameras(pool_off, N_IMAGES_TOTAL - len(sample_on))
print(f"  Sampled  ON: {len(sample_on)}  |  OFF: {len(sample_off)}")

all_sampled = sample_on + sample_off
random.seed(42)
random.shuffle(all_sampled)
sampled_df = pd.DataFrame(all_sampled)

final_sample = pd.merge(
    df, sampled_df[['timestamp', 'camera', 'camera_timestamp']],
    on=['timestamp', 'camera', 'camera_timestamp']
)

review_cols = [
    'timestamp', 'camera', 'camera_timestamp', 'track_id', 'object_type',
    'frame_generated_masks', 'frame_valid_masks', 'associated_masks',
    'rear_left_indicator', 'rear_right_indicator',
    'front_left_indicator', 'front_right_indicator',
    'center_x', 'center_y', 'box_width', 'box_height',
]
cols_to_save = [c for c in review_cols if c in final_sample.columns]

unique_images = (
    final_sample[['timestamp', 'camera', 'camera_timestamp']]
    .drop_duplicates()
    .sample(frac=1, random_state=99)
    .reset_index(drop=True)
)
chunks = [unique_images[i:i + IMAGES_PER_CSV]
          for i in range(0, len(unique_images), IMAGES_PER_CSV)]

os.makedirs(os.path.dirname(OUTPUT_PREFIX) or '.', exist_ok=True)
print(f"\nSaving {len(chunks)} chunk(s):")
for idx, chunk in enumerate(chunks, 1):
    chunk_df = pd.merge(chunk, final_sample,
                        on=['timestamp', 'camera', 'camera_timestamp'], how='left')
    out = f"{OUTPUT_PREFIX}_part_{idx}.csv"
    chunk_df[cols_to_save].to_csv(out, index=False)
    print(f"  Saved {out}  ({len(chunk)} frames)")

print(f"\nDone. {len(unique_images)} frames across {len(chunks)} file(s).")
