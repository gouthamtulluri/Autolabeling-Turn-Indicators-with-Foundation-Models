"""
visualise.py
───────────────
Pipeline output visualiser.

Renders a full composite frame (6 cameras + LiDAR BEV) for a single master
timestamp using the CSV outputs of the processing pipeline:

  Stage 1 — process_all_valid_images_with_sam  (model_tools_v19)
             → df_camera  (one row per camera × track × frame)

  Stage 2 — detect_blinks_per_camera + fuse_to_global  (indicator_core_v3)
             → df_camera  (same rows, blink_state columns added)
             → df_fused   (one row per timestamp × track, left/right indicator)

Public API
----------
  show_pipeline_frame(timestamp, df_camera, df_fused, calib_data, seq_dir)
      → displays a matplotlib figure in the notebook

  save_pipeline_frame(timestamp, df_camera, df_fused, calib_data, seq_dir, out_path)
      → saves the composite as a JPEG/PNG (no screen display required)
"""

import os
import ast
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─── camera layout (left wing | BEV centre | right wing) ─────────────────────
# Keys must match the camera names used in calib_data and in df_camera['camera'].
_CAM_LAYOUT = [
    # (camera_name,          grid_row, grid_col)
    ('front_left_camera',    0,        0),
    ('left_camera',          1,        0),
    ('back_left_camera',     2,        0),
    ('front_right_camera',   0,        2),
    ('right_camera',         1,        2),
    ('back_right_camera',    2,        2),
]
# Grid column 1  is reserved for the LiDAR BEV panel.

# ─── indicator state → colour mapping ────────────────────────────────────────
_STATE_COLOR = {
    'BLINKING':   '#FFA500',   # amber
    'STEADY_ON':  '#FFFF00',   # yellow
    'OFF':        '#444444',   # dark grey
    None:         '#222222',
}

_LR_LABEL_COLOR = {
    'BLINKING':  '#FFA500',
    'STEADY_ON': '#FFD700',
    'OFF':       '#aaaaaa',
    None:        '#aaaaaa',
}


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_lidar_from_npz(npz_path):
    """
    Safely opens an .npz file, detects the internal array structure, 
    and returns a formatted point cloud array.
    """
    try:
        data = np.load(npz_path)
        
        possible_keys = ['pcd', 'points', 'data', 'xyz', 'arr_0']
        for key in possible_keys:
            if key in data:
                pcd = data[key]
                if len(pcd.shape) == 2 and pcd.shape[1] >= 3:
                    return pcd
        
        if 'x' in data and 'y' in data and 'z' in data:
            arrays = [data['x'].flatten(), data['y'].flatten(), data['z'].flatten()]
            if 'intensity' in data or 'i' in data:
                i_key = 'intensity' if 'intensity' in data else 'i'
                arrays.append(data[i_key].flatten())
            if 'time' in data or 't' in data:
                t_key = 'time' if 'time' in data else 't'
                arrays.append(data[t_key].flatten())
                
            return np.column_stack(arrays)
            
        print(f"⚠️ Warning: Could not identify point cloud keys in {npz_path}.")
        return None
        
    except Exception as e:
        print(f"❌ Error loading NPZ file {npz_path}: {e}")
        return None


def create_point_cloud_bev_image(pcd, left_right_range, forward_backward_range, res=0.15, z_clip=(-3, 5), annotation_data=None, background_color=30):
    assert pcd.shape[1] >= 3, "pcd must have at least 3 columns (x,y,z)"

    # TRUE-NORTH CORRECTION: Filter X against Left/Right, Y against Forward/Backward
    x_mask = (pcd[:, 0] >= left_right_range[0]) & (pcd[:, 0] <= left_right_range[1])
    y_mask = (pcd[:, 1] >= forward_backward_range[0]) & (pcd[:, 1] <= forward_backward_range[1])
    z_mask = (pcd[:, 2] >= z_clip[0]) & (pcd[:, 2] <= z_clip[1])
    mask = x_mask & y_mask & z_mask
    pcd = pcd[mask]

    theta = np.deg2rad(30)
    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,            1, 0           ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    pcd_rotated = (Ry @ pcd[:, :3].T).T

    width = int((left_right_range[1] - left_right_range[0]) / res)
    height = int((forward_backward_range[1] - forward_backward_range[0]) / res)

    # TRUE-NORTH CORRECTION: Map X to Width, map Y to Height
    x_img = ((pcd_rotated[:, 0] - left_right_range[0]) / res).astype(np.int32)
    y_img = ((forward_backward_range[1] - pcd_rotated[:, 1]) / res).astype(np.int32)

    x_img = np.clip(x_img, 0, width - 1)
    y_img = np.clip(y_img, 0, height - 1)

    intensity = pcd[:, 3] if pcd.shape[1] >= 4 else ((pcd[:, 2] - z_clip[0]) / (z_clip[1] - z_clip[0])) * 255.0

    bev = np.full((height, width), fill_value=background_color, dtype=np.uint8)
    bev[y_img, x_img] = intensity.astype(np.uint8)

    mask_bg = (bev == background_color)
    bev = cv2.applyColorMap(bev, cv2.COLORMAP_TURBO)
    bev[mask_bg] = [background_color, background_color, background_color]

    # Note: Removing draw_annotation_data_onto_bev as it's not needed since we draw df_fused tracks manually
    # if annotation_data:
    #     bev = draw_annotation_data_onto_bev(bev, annotation_data, left_right_range, forward_backward_range, res, Ry)

    return bev

def _load_flat(cam_name, cam_ts, seq_dir, calib_data):
    """Read, undistort and return an RGB image, or None on failure."""
    cam_dir = os.path.join(seq_dir, cam_name)
    img_path = os.path.join(cam_dir, f"{int(cam_ts)}.jpg")
    if not os.path.exists(img_path):
        return None
    raw = cv2.imread(img_path)
    if raw is None:
        return None
    img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    try:
        undistort = calib_data['cameras'][cam_name]['undistort_function']
        return undistort(img_rgb)
    except Exception:
        return img_rgb


def _label_for_track(row):
    """
    Build the overlay text for one bounding box on a camera panel.
    Uses indicator dict columns from df_camera (model_tools_v19 output).
    Returns (text, bg_color).
    """
    tid = int(row['track_id'])

    # Collect the four indicator columns; each is a dict-string or NaN
    side_labels = []
    is_active = False
    for col, short in [
        ('rear_left_indicator',   'RL'),
        ('rear_right_indicator',  'RR'),
        ('front_left_indicator',  'FL'),
        ('front_right_indicator', 'FR'),
    ]:
        val = row.get(col)
        if pd.isna(val):
            continue
        try:
            d = ast.literal_eval(val)
        except Exception:
            continue
        state = d.get('blink_state', 'OFF')
        if state in ('BLINKING', 'STEADY_ON'):
            is_active = True
            side_labels.append(f"{short}:{state[:4]}")   # e.g. FL:BLIN

    indicator_str = ' | '.join(side_labels) if side_labels else 'OFF'
    label = f"ID:{tid}  {indicator_str}"
    bg = '#d35400' if is_active else '#111111'
    return label, bg


def _render_camera_panel(ax, img, cam_name, df_cam_frame):
    """
    Draw the undistorted camera image with bounding boxes and indicator labels.

    Parameters
    ----------
    ax           : matplotlib Axes (imshow target)
    img          : H×W×3 RGB numpy array, or None if the image was unavailable
    cam_name     : camera name string (for the panel title)
    df_cam_frame : rows from df_camera for this camera & timestamp
    """
    if img is None:
        ax.set_facecolor('#111111')
        ax.text(0.5, 0.5, f'{cam_name}\n(image not found)',
                color='white', ha='center', va='center',
                fontsize=9, transform=ax.transAxes)
        ax.axis('off')
        return

    ax.imshow(img)
    ax.axis('off')

    img_h, img_w = img.shape[:2]

    for _, row in df_cam_frame.iterrows():
        # Bounding box from df_camera columns set by model_tools_v19
        cx  = row.get('center_x', np.nan)
        cy  = row.get('center_y', np.nan)
        bw  = row.get('box_width',  0)
        bh  = row.get('box_height', 0)
        if pd.isna(cx) or pd.isna(cy):
            continue

        x1, y1 = cx - bw / 2, cy - bh / 2

        label, bg = _label_for_track(row)
        is_active = bg != '#111111'

        edge  = '#FFA500' if is_active else '#00BFFF'
        lw    = 2.5       if is_active else 1.5

        rect = mpatches.FancyBboxPatch(
            (x1, y1), bw, bh,
            boxstyle='square,pad=0',
            linewidth=lw, edgecolor=edge, facecolor='none'
        )
        ax.add_patch(rect)

        # Text label
        fs = 7
        ax.text(x1, max(0, y1 - 4), label,
                color='white', fontsize=fs, fontweight='bold',
                bbox=dict(facecolor=bg, alpha=0.82, edgecolor='none', pad=2))

    # Camera name watermark
    ax.text(0.01, 0.97, cam_name, transform=ax.transAxes,
            color='white', fontsize=8, fontweight='bold', va='top',
            bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=2))


def _render_bev_panel(ax, timestamp, df_fused_ts, calib_data, seq_dir,
                      fwd_range=100, lr_range=50, res=0.10):
    """
    Render the Bird's-Eye View LiDAR panel into `ax`.

    Overlays df_fused track positions (x_3d, y_3d) with indicator state labels.

    Parameters
    ----------
    fwd_range : metres forward/backward  (±fwd_range centred on ego)
    lr_range  : metres left/right        (±lr_range  centred on ego)
    res       : metres per pixel
    """
    forward_backward_range = (-fwd_range, fwd_range)
    left_right_range       = (-lr_range,  lr_range)

    pcd_path = os.path.join(seq_dir, 'middle_lidar', f"{int(timestamp)}.npz")

    bev_img = None
    if os.path.exists(pcd_path):
        pcd = load_lidar_from_npz(pcd_path)
        if pcd is not None:
            bev_img = create_point_cloud_bev_image(
                pcd=pcd,
                left_right_range=left_right_range,
                forward_backward_range=forward_backward_range,
                res=res,
            )
            bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)

    if bev_img is None:
        bev_h = int((fwd_range * 2) / res)
        bev_w = int((lr_range  * 2) / res)
        bev_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

    ax.imshow(bev_img, origin='upper')
    ax.axis('off')

    bev_h, bev_w = bev_img.shape[:2]

    # Ego marker (centre of BEV)
    cx_ego, cy_ego = bev_w // 2, bev_h // 2
    ax.plot(cx_ego, cy_ego, 'o', color='#cc0000', markersize=8, zorder=10)
    ax.text(cx_ego + 6, cy_ego, 'EGO', color='#cc0000',
            fontsize=8, fontweight='bold', va='center')

    # Track overlays from df_fused
    for _, row in df_fused_ts.iterrows():
        x3d = row.get('x_3d')
        y3d = row.get('y_3d')
        if pd.isna(x3d) or pd.isna(y3d):
            continue

        # Map physical coords → BEV pixel coords (same convention as create_point_cloud_bev_image)
        px = int((x3d - left_right_range[0])       / res)
        py = int((forward_backward_range[1] - y3d)  / res)
        px = np.clip(px, 0, bev_w - 1)
        py = np.clip(py, 0, bev_h - 1)

        l_state = row.get('left_indicator')
        r_state = row.get('right_indicator')
        l_active = l_state in ('BLINKING', 'STEADY_ON')
        r_active = r_state in ('BLINKING', 'STEADY_ON')
        any_active = l_active or r_active

        dot_color = '#FFA500' if any_active else '#00BFFF'
        ax.plot(px, py, 'o', color=dot_color, markersize=9,
                markeredgecolor='white', markeredgewidth=1.2, zorder=9)

        lbl_parts = [f"ID:{int(row['track_id'])}"]
        if l_active:
            lbl_parts.append(f"L:{l_state[:4]}")
        if r_active:
            lbl_parts.append(f"R:{r_state[:4]}")
        label = '  '.join(lbl_parts)

        bg = '#d35400' if any_active else '#111111'
        ax.text(px + 8, py, label, color='white', fontsize=7,
                fontweight='bold',
                bbox=dict(facecolor=bg, alpha=0.80, edgecolor='none', pad=1.5))

    ax.text(0.01, 0.99, 'LiDAR BEV  (middle_lidar)', transform=ax.transAxes,
            color='white', fontsize=8, fontweight='bold', va='top',
            bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=2))
    ax.text(0.01, 0.95, f'ts: {timestamp}', transform=ax.transAxes,
            color='#aaaaaa', fontsize=7, va='top')


def _indicator_legend(fig):
    """Add a compact indicator state legend to the figure."""
    handles = [
        mpatches.Patch(color='#FFA500', label='BLINKING'),
        mpatches.Patch(color='#FFD700', label='STEADY_ON'),
        mpatches.Patch(color='#00BFFF', label='OFF'),
        mpatches.Patch(color='#cc0000', label='EGO vehicle'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=8, framealpha=0.85, edgecolor='#cccccc',
               bbox_to_anchor=(0.5, 0.005))


# ─── main public function ─────────────────────────────────────────────────────

def _build_composite(timestamp, df_camera, df_fused, calib_data, seq_dir,
                     fwd_range=100, lr_range=50):
    """
    Core: build the figure for one master timestamp.

    Parameters
    ----------
    timestamp  : int or str   master ROS2 timestamp (nanoseconds)
    df_camera  : pd.DataFrame output of process_all_valid_images_with_sam
                             (optionally with blink_state columns from detect_blinks_per_camera)
    df_fused   : pd.DataFrame output of fuse_to_global  (may be None)
    calib_data : dict         from load_calibration_file
    seq_dir    : str          root folder with per-camera image dirs + middle_lidar/
    fwd_range  : float        LiDAR BEV forward/backward extent in metres
    lr_range   : float        LiDAR BEV left/right extent in metres

    Returns
    -------
    matplotlib.figure.Figure
    """
    timestamp = int(timestamp)

    # Slice the two dataframes to this timestamp
    df_ts   = df_camera[df_camera['timestamp'] == timestamp]
    df_f_ts = (df_fused[df_fused['timestamp'] == timestamp]
               if df_fused is not None and len(df_fused) else pd.DataFrame())

    # ── Figure layout ──────────────────────────────────────────────────────────
    # 3 rows × 3 columns:
    #   col 0 — left  cameras  (front_left, left, back_left)
    #   col 1 — LiDAR BEV  (spans all 3 rows)
    #   col 2 — right cameras  (front_right, right, back_right)
    fig = plt.figure(figsize=(26, 14), facecolor='#1a1a1a')
    gs  = GridSpec(
        3, 3, figure=fig,
        width_ratios=[2, 1.5, 2],    # cameras wider than BEV
        hspace=0.04, wspace=0.04,
        left=0.02, right=0.98,
        top=0.94,  bottom=0.06
    )

    # ── Camera panels ──────────────────────────────────────────────────────────
    for cam_name, grow, gcol in _CAM_LAYOUT:
        ax = fig.add_subplot(gs[grow, gcol])
        ax.set_facecolor('#0d0d0d')

        # Find the camera_timestamp for this camera at this master timestamp
        df_cam = df_ts[df_ts['camera'] == cam_name]
        cam_ts = None
        if not df_cam.empty and 'camera_timestamp' in df_cam.columns:
            cam_ts = df_cam['camera_timestamp'].iloc[0]

        img = _load_flat(cam_name, cam_ts, seq_dir, calib_data) if cam_ts else None
        _render_camera_panel(ax, img, cam_name, df_cam)

    # ── LiDAR BEV (centre column, all 3 rows) ─────────────────────────────────
    ax_bev = fig.add_subplot(gs[:, 1])
    ax_bev.set_facecolor('#0d0d0d')
    _render_bev_panel(ax_bev, timestamp, df_f_ts, calib_data, seq_dir,
                      fwd_range=fwd_range, lr_range=lr_range)

    # ── Title & legend ─────────────────────────────────────────────────────────
    n_tracks = df_ts['track_id'].nunique() if not df_ts.empty else 0
    n_blink  = (
        (df_f_ts['left_indicator'].isin(['BLINKING', 'STEADY_ON']) |
         df_f_ts['right_indicator'].isin(['BLINKING', 'STEADY_ON'])).sum()
        if not df_f_ts.empty else 0
    )
    fig.suptitle(
        f"Pipeline Output  |  ts = {timestamp}  |  "
        f"tracked vehicles: {n_tracks}  |  "
        f"active indicators: {n_blink}",
        color='white', fontsize=12, fontweight='bold', y=0.975
    )
    _indicator_legend(fig)

    return fig


def show_pipeline_frame(timestamp, df_camera, df_fused, calib_data, seq_dir,
                        fwd_range=100, lr_range=50):
    """
    Display a composite pipeline-output frame in the notebook for one timestamp.

    Parameters
    ----------
    timestamp  : int    master ROS2 timestamp
    df_camera  : pd.DataFrame  model_tools_v19 output (optionally post-blink-detection)
    df_fused   : pd.DataFrame  indicator_core_v3.fuse_to_global output (or None)
    calib_data : dict   from load_calibration_file
    seq_dir    : str    root folder with camera image dirs + middle_lidar/
    fwd_range  : float  BEV longitudinal range in metres (default 100)
    lr_range   : float  BEV lateral range in metres (default 50)
    """
    fig = _build_composite(timestamp, df_camera, df_fused, calib_data, seq_dir,
                           fwd_range=fwd_range, lr_range=lr_range)
    plt.show()
