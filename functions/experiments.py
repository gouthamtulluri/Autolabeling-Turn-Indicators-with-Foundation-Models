"""
experiments.py — Reusable experiment functions for indicator color filter analysis.

Experiments
-----------
  1  run_single_sam3_inference()            Segmentation: single pass with tuned thresholds.
1.1  run_mask_threshold_ablation()          Segmentation: sweep SAM3 mask thresholds.
1.2  run_conf_threshold_ablation()          Segmentation: sweep SAM3 confidence thresholds.
  2  compare_raw_depth_vs_lidar()           Depth Estimation: compare DA3 depth vs LiDAR ground truth.
  3  visualize_final_dataframe_per_mask()   Association: plots for 3D mapping and vehicle association.
 4.1 run_max_dist_sweep()                   CIELAB Filter: sweep MAX_DIST with B_MIN fixed.
 4.2 run_bmin_sweep()                       CIELAB Filter: sweep B_MIN with MAX_DIST fixed.
 5.1 run_autocorr_sweep                     Sweep autocorr_min_peak  (autocorrelation sensitivity)
 5.2 run_detrend_window_sweep               Sweep detrend_window_sec  (drift removal window)

Typical usage in a notebook cell
---------------------------------
    from functions.experiments import run_max_dist_sweep, run_bmin_sweep

    run_max_dist_sweep(
        image_path      = "topic-17-files/.../frame.jpg",
        mask_path       = "topic-17-files/.../frame.npz",
        mask_idx        = 1,
        cam_name        = "front_right_camera",
        calib_data      = calib_data,
        fixed_b_min     = 135,
        max_dist_values = [50, 80, 120],
    )
"""

from PIL import Image
import ast
import sys
import pandas as pd
from scipy.signal import find_peaks
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches


#--------------------------------------------------------------------------------------
# 1 SEGMENTATION EXPERIMENTS (SAM 3 Inference & Ablations)

def run_single_sam3_inference(image_path, cam_name, calib_data, generate_sam3_masks, processor, model, device, text_prompt="car taillight", exclude_prompt=None, conf_threshold=0.5, mask_threshold=0.6):
    """
    Executes a single SAM 3 inference pass and renders the predicted masks 
    over the original undistorted image for visual validation.

    Args:
        image_path (str): The file path to the raw input image.
        cam_name (str): The specific camera identifier (e.g., 'front_left_camera') 
                        to retrieve the correct undistortion matrix.
        calib_data (dict): The parsed calibration dictionary containing sensor intrinsics.
        generate_sam3_masks (callable): The inference function pipeline.
        processor (object): The HuggingFace processor.
        model (object): The loaded SAM 3 model.
        device (str/torch.device): Hardware acceleration target.
        text_prompt (str, optional): The positive grounding prompt. Defaults to "car taillight".
        exclude_prompt (str, optional): The negative grounding prompt. Defaults to None.
        conf_threshold (float, optional): The object-level confidence threshold. Defaults to 0.5.
        mask_threshold (float, optional): The pixel-level binary mask threshold. Defaults to 0.6.
    """
    # 1. Image Preparation & Undistortion
    undistort_function = calib_data['cameras'][cam_name]['undistort_function']
    raw_img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    # Flatten lens distortion so masks align properly with 3D LiDAR projections later
    flat_img = undistort_function(img_rgb)
    pil_flat_img = Image.fromarray(flat_img)

    # 2. SAM 3 Inference
    target_masks, target_scores = generate_sam3_masks(
        image_path=pil_flat_img, 
        text_prompt=text_prompt, 
        exclude_prompt=exclude_prompt,
        processor=processor,
        model=model,
        device=device,
        conf_threshold=conf_threshold, 
        mask_threshold=mask_threshold
    )

    masks_cpu = target_masks.cpu().numpy()
    scores_cpu = target_scores.cpu().numpy()

    # 3. Visualization Setup
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig.patch.set_facecolor('white')

    ax.imshow(flat_img)
    ax.set_title(f"SAM 3 Masks | Prompt: '{text_prompt}'", fontsize=18, color='black', fontweight='bold')
    ax.axis('off')

    legend_elements = []
    
    if len(masks_cpu) == 0:
        print("No masks detected with the current thresholds.")
    else:
        # Overlay each valid mask with a unique random color
        for i, (mask, score) in enumerate(zip(masks_cpu, scores_cpu)):
            if mask.sum() == 0:
                continue
                
            # Create a 4-channel RGBA overlay for alpha blending
            solid_color = np.random.random(3)
            overlay_color = np.concatenate([solid_color, [0.8]]) 
            
            h, w = mask.shape
            color_overlay = np.zeros((h, w, 4))
            color_overlay[mask > 0] = overlay_color
            ax.imshow(color_overlay)
            
            # Create a custom legend patch to link colors with confidence scores
            patch = mpatches.Patch(
                color=solid_color, 
                label=f'Mask {i+1}: {score:.2f}'
            )
            legend_elements.append(patch)

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            title="Confidence Scores",
            title_fontsize=12,
            fontsize=10,
            facecolor='white',
            labelcolor='black',
            edgecolor='none',
            framealpha=0.8,
            borderpad=1
        )

    plt.tight_layout()
    plt.show()


#--------------------------------------------------------------------------------------
# 1.1 Mask Threshold Ablation

def run_mask_threshold_ablation(image_path, cam_name, calib_data, generate_sam3_masks, processor, model, device, text_prompt="car headlight", fixed_conf=0.30):
    """
    Performs a visual ablation study on the pixel-level mask threshold (0.1 to 0.9)
    while keeping the object-level confidence score fixed.

    Args:
        image_path (str): The file path to the raw input image.
        cam_name (str): The specific camera identifier for undistortion.
        calib_data (dict): The parsed calibration dictionary.
        generate_sam3_masks (callable): The inference function pipeline.
        processor (object): The HuggingFace processor.
        model (object): The loaded SAM 3 model.
        device (str/torch.device): Hardware acceleration target.
        text_prompt (str, optional): The positive grounding prompt. Defaults to "car headlight".
        fixed_conf (float, optional): The static confidence score to lock in. Defaults to 0.30.
    """
    undistort_function = calib_data['cameras'][cam_name]['undistort_function']
    raw_img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    flat_img = undistort_function(img_rgb)
    pil_image = Image.fromarray(flat_img)

    # Generate a linear sweep array from 0.1 to 0.9
    steps = np.arange(0.1, 1.0, 0.1)
    print("Starting Ablation Grid 1: Varying Mask Threshold...")

    # Create an ultra-tall 9x1 subplot to vertically stack the ablation results
    fig1, axs1 = plt.subplots(9, 1, figsize=(16, 60), facecolor='white')
    fig1.suptitle(
        f"SAM 3 Mask Threshold Sweep  |  conf_threshold fixed = {fixed_conf}  |  Prompt: '{text_prompt}'",
        fontsize=13, fontweight='bold', y=1.005
    )

    # Iteratively run inference across the threshold sweep
    for idx, m_thresh in enumerate(steps):
        m_val = round(m_thresh, 1)
        ax = axs1[idx]

        masks_t, _ = generate_sam3_masks(
            image_path=pil_image,
            text_prompt=text_prompt,
            exclude_prompt=None,
            processor=processor, 
            model=model, 
            device=device,
            conf_threshold=fixed_conf, 
            mask_threshold=m_val
        )

        ax.imshow(pil_image)
        ax.axis('off')
        ax.set_title(
            f"mask_threshold = {m_val}   |   masks found: {len(masks_t)}",
            fontsize=11, fontweight='bold', color='#111111', pad=4
        )
        ax.set_facecolor('#f7f7f7')

        # Render all valid masks in a uniform Cyan color for this sweep
        if len(masks_t) > 0:
            mask_overlay = np.zeros((pil_image.size[1], pil_image.size[0], 4), dtype=np.float32)
            for m in masks_t:
                mask_np = m.cpu().numpy()
                mask_overlay[mask_np > 0] = [0, 1, 1, 0.6]   # Cyan RGBA
            ax.imshow(mask_overlay)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


#--------------------------------------------------------------------------------------
# 1.2 Confidence Threshold Ablation

def run_conf_threshold_ablation(image_path, cam_name, calib_data, generate_sam3_masks, processor, model, device, text_prompt="car taillight", fixed_mask=0.60):
    """
    Performs a visual ablation study on the object-level confidence threshold (0.1 to 0.9)
    while keeping the pixel-level mask threshold fixed.

    Args:
        image_path (str): The file path to the raw input image.
        cam_name (str): The specific camera identifier for undistortion.
        calib_data (dict): The parsed calibration dictionary.
        generate_sam3_masks (callable): The inference function pipeline.
        processor (object): The HuggingFace processor.
        model (object): The loaded SAM 3 model.
        device (str/torch.device): Hardware acceleration target.
        text_prompt (str, optional): The positive grounding prompt. Defaults to "car taillight".
        fixed_mask (float, optional): The static mask threshold to lock in. Defaults to 0.60.
    """
    undistort_function = calib_data['cameras'][cam_name]['undistort_function']
    raw_img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    flat_img = undistort_function(img_rgb)
    pil_image = Image.fromarray(flat_img)

    # Generate a linear sweep array from 0.1 to 0.9
    steps = np.arange(0.1, 1.0, 0.1)
    print("Starting Ablation Grid 2: Varying Confidence Threshold...")

    # Create an ultra-tall 9x1 subplot to vertically stack the ablation results
    fig1, axs1 = plt.subplots(9, 1, figsize=(16, 60), facecolor='white')
    fig1.suptitle(
        f"SAM 3 Confidence Threshold Sweep  |  mask_threshold fixed = {fixed_mask}  |  Prompt: '{text_prompt}'",
        fontsize=13, fontweight='bold', y=1.005
    )

    # Iteratively run inference across the threshold sweep
    for idx, c_thresh in enumerate(steps):
        c_val = round(c_thresh, 1)
        ax = axs1[idx]

        masks_t, _ = generate_sam3_masks(
            image_path=pil_image,
            text_prompt=text_prompt,
            exclude_prompt=None,
            processor=processor, 
            model=model, 
            device=device,
            conf_threshold=c_val, 
            mask_threshold=fixed_mask
        )

        ax.imshow(pil_image)
        ax.axis('off')
        ax.set_title(
            f"conf_threshold = {c_val}   |   masks found: {len(masks_t)}",
            fontsize=11, fontweight='bold', color='#111111', pad=4
        )
        ax.set_facecolor('#f7f7f7')

        # Render all valid masks in a uniform Magenta color to distinguish from the mask sweep
        if len(masks_t) > 0:
            mask_overlay = np.zeros((pil_image.size[1], pil_image.size[0], 4), dtype=np.float32)
            for m in masks_t:
                mask_np = m.cpu().numpy()
                mask_overlay[mask_np > 0] = [1, 0, 1, 0.6]   # Magenta RGBA
            ax.imshow(mask_overlay)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


#--------------------------------------------------------------------------------------
# 2 DEPTH ESTIMATION EXPERIMENTS  (DA3 vs LiDAR ground truth)

def compare_raw_depth_vs_lidar(image_name, cam_name, df_cam, calib_data, depth_model, device, seq_dir):
    """
    Compare DA3 depth estimation against LiDAR ground truth using four scaling strategies.

    Scaling strategies (one row each in the output figure):
      1. RAW        — relative depth map (no metric calibration)
      2. GLOBAL     — single median scale fitted over all LiDAR-visible pixels
      3. ZONAL      — spatially-varying scale map (griddata interpolation)

    Each row shows:
      Left  — error-coloured scatter overlay on the undistorted camera frame
              (green = accurate, red = large error, normalised 0-100 %)
      Right — predicted depth vs true LiDAR depth scatter with ideal 1:1 line
              and MAE in the title

    Args:
        image_name  : str   filename of the camera image, e.g. '1750164055850.jpg'
        cam_name    : str   camera key in calib_data, e.g. 'front_right_camera'
        df_cam      : DataFrame  camera-output CSV from model_tools_v19
        calib_data  : dict  calibration dict (undistort functions, K, cTv, vTl)
        depth_model : object   loaded DA3 model with .infer_image() or .inference()
        device      : str   'cuda' or 'cpu'
        seq_dir     : str   root folder containing per-camera image dirs and LiDAR NPZ files
    """
    print(f"🔬 Launching Depth Accuracy Analysis for {image_name}")
    
    cam_ts = int(image_name.split('.')[0])
    frame_data = df_cam[df_cam['camera_timestamp'] == cam_ts].copy()
    if frame_data.empty:
        print("❌ No data for this frame.")
        return
        
    master_ts = int(frame_data['timestamp'].iloc[0])
    lidar_path = os.path.join(seq_dir, "middle_lidar", f"{master_ts}.npz")
    cam_dir = os.path.join(seq_dir, cam_name)
    full_image_path = os.path.join(cam_dir, image_name)

    undistort_function = calib_data['cameras'][cam_name]['undistort_function']
    raw_img = cv2.imread(full_image_path)
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    flat_img = undistort_function(img_rgb)
    img_h, img_w = flat_img.shape[:2]

    # 1. Get LiDAR points
    lidar_data = np.load(lidar_path)
    pts_raw = np.vstack([lidar_data['x'], lidar_data['y'], lidar_data['z']]).T
    pts_hom = np.hstack((pts_raw, np.ones((pts_raw.shape[0], 1))))
    
    vTl = np.array(calib_data['lidars']['middle_lidar']['vTl'], dtype=np.float64)
    cam_calib = calib_data['cameras'][cam_name]
    cTv = np.array(cam_calib['cTv'], dtype=np.float64)
    K = np.array(cam_calib['K_undistortion'], dtype=np.float64)
    
    cam_points = ((cTv @ vTl) @ pts_hom.T).T
    valid_z = cam_points[:, 2] > 0
    cam_points = cam_points[valid_z]
    
    uv_points = (K @ cam_points[:, :3].T).T
    u = uv_points[:, 0] / uv_points[:, 2]
    v = uv_points[:, 1] / uv_points[:, 2]
    
    valid_uv = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u_valid, v_valid = u[valid_uv].astype(int), v[valid_uv].astype(int)
    true_z = cam_points[valid_uv, 2]

    # 2. Get DA3 Raw Depth
    with torch.no_grad():
        if hasattr(depth_model, 'infer_image'):
            raw_depth = depth_model.infer_image(flat_img)
        elif hasattr(depth_model, 'inference'):
            pred = depth_model.inference([flat_img])
            raw_depth = pred.depth[0] if hasattr(pred, 'depth') else pred[0]
            
    if raw_depth.shape[:2] != (img_h, img_w):
        disparity = cv2.resize(raw_depth, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    else:
        disparity = raw_depth

    # 3. Apply a single Global Scale
    disp_sampled = disparity[v_valid, u_valid]
    valid_disp = disp_sampled > 1e-6
    true_z = true_z[valid_disp]
    disp_sampled = disp_sampled[valid_disp]
    u_val = u_valid[valid_disp]
    v_val = v_valid[valid_disp]
    
    global_scale = np.median(true_z / np.clip(disp_sampled, 1e-6, None))
    global_metric_depth = global_scale * disparity
    predicted_z_global = global_metric_depth[v_val, u_val]
    
    # 4. Apply Zonal Scale (Griddata Interpolation)
    sparse_scales = true_z / np.clip(disp_sampled, 1e-6, None)
    from scipy.interpolate import griddata
    grid_x, grid_y = np.meshgrid(np.arange(img_w), np.arange(img_h))
    points = np.vstack((u_val, v_val)).T
    
    zonal_scale_map = griddata(
        points, sparse_scales, (grid_x, grid_y), 
        method='linear', fill_value=global_scale
    )
    if np.any(np.isnan(zonal_scale_map)):
        zonal_scale_map = griddata(points, sparse_scales, (grid_x, grid_y), method='nearest', fill_value=global_scale)
        
    zonal_metric_depth = zonal_scale_map * disparity
    predicted_z_zonal = zonal_metric_depth[v_val, u_val]
    
    # 4.5 Apply Global Scale + Shift (Using f_x from Camera Projection Matrix)
    f_x = K[0, 0]
    # Solve: f_x / True_Z = s * disparity + t
    s, t = np.polyfit(disp_sampled, f_x / true_z, 1)
    
    # Apply to entire map
    global_shift_depth = f_x / (s * np.clip(disparity, 1e-6, None) + t)
    # Prevent negative depths if shift is large
    global_shift_depth = np.clip(global_shift_depth, 0.1, 200.0)
    predicted_z_shift = global_shift_depth[v_val, u_val]
    
    # 5. Calculate Errors
    raw_depth = 1.0 / disp_sampled
    error_pct_raw = (np.abs(raw_depth - true_z) / true_z) * 100
    mae_raw = np.mean(np.abs(raw_depth - true_z))
    
    error_pct_global = (np.abs(predicted_z_global - true_z) / true_z) * 100
    mae_global = np.mean(np.abs(predicted_z_global - true_z))
    
    error_pct_shift = (np.abs(predicted_z_shift - true_z) / true_z) * 100
    mae_shift = np.mean(np.abs(predicted_z_shift - true_z))
    
    error_pct_zonal = (np.abs(predicted_z_zonal - true_z) / true_z) * 100
    mae_zonal = np.mean(np.abs(predicted_z_zonal - true_z))
    
    # ── Shared style constants ────────────────────────────────────────────────
    TXT       = '#111111'
    GRID_KW   = dict(linestyle='--', color='#dddddd', alpha=0.7)
    SCATTER_S = 4
    lim_z     = max(true_z) * 1.05   # common axis limit for all scatter plots

    def _style_scatter(ax, xlabel='True Depth (m)', ylabel='Predicted Depth (m)'):
        """Apply uniform style to a predicted-vs-true scatter axis."""
        ax.set_facecolor('#f7f7f7')
        ax.set_xlabel(xlabel, color=TXT, fontsize=9)
        ax.set_ylabel(ylabel, color=TXT, fontsize=9)
        ax.set_xlim(0, lim_z); ax.set_ylim(0, lim_z)
        ax.grid(**GRID_KW)
        ax.tick_params(labelsize=8)
        # Ideal 1:1 line
        ax.plot([0, lim_z], [0, lim_z], color='#cc0000', lw=1.8,
                linestyle='--', label='ideal (y=x)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── Figure layout ─────────────────────────────────────────────────────────
    # 3 rows × 2 columns:
    #   col 0 — error-coloured overlay on camera frame (RdYlGn: green=good, red=bad)
    #   col 1 — predicted vs true scatter with 1:1 ideal line + MAE
    # Row 0 — RAW disparity map (left)
    # Row 1 — GLOBAL median scale error
    # Row 2 — ZONAL scale error
    fig, axs = plt.subplots(3, 2, figsize=(24, 24), facecolor='white')
    fig.suptitle(
        f"Depth Fusion Experiment — DA3 vs LiDAR  |  {cam_name}  |  {image_name}\n"
        f"Colour scale: green = low error, red = high error (0–100 %)",
        color=TXT, fontsize=13, fontweight='bold', y=1.005
    )

    # ── Row 0: RAW disparity map ───────────────────
    axs[0, 0].remove()
    axs[0, 1].remove()
    gs = axs[0, 0].get_gridspec()
    ax = fig.add_subplot(gs[0, :])
    
    im0 = ax.imshow(disparity, cmap='turbo')
    ax.set_title('RAW DA3 Disparity Map  (relative, no metric calibration)',
                 fontsize=11, fontweight='bold', color=TXT)
    plt.colorbar(im0, ax=ax, label='Relative disparity')
    ax.axis('off')

    # ── Row 1: GLOBAL median scale ───────────────────────────────────────────────────
    ax = axs[1, 0]
    ax.imshow(flat_img)
    sc = ax.scatter(u_val, v_val, c=error_pct_global, cmap='RdYlGn_r',
                    s=SCATTER_S, alpha=0.85, vmin=0, vmax=100)
    ax.set_title(f'GLOBAL Median Scale  |  Error overlay  (MAE = {mae_global:.2f} m)',
                 fontsize=11, fontweight='bold', color=TXT)
    plt.colorbar(sc, ax=ax, label='Absolute error (%)', shrink=0.9)
    ax.axis('off')

    ax = axs[1, 1]
    ax.scatter(true_z, predicted_z_global, s=SCATTER_S, alpha=0.30, color='#2255cc')
    _style_scatter(ax)
    ax.set_title(f'GLOBAL Median Scale Accuracy  |  MAE = {mae_global:.2f} m',
                 fontsize=11, fontweight='bold', color=TXT)

    # ── Row 2: ZONAL scale (spatially-varying interpolated map) ──────────────
    ax = axs[2, 0]
    ax.imshow(flat_img)
    sc = ax.scatter(u_val, v_val, c=error_pct_zonal, cmap='RdYlGn_r',
                    s=SCATTER_S, alpha=0.85, vmin=0, vmax=100)
    ax.set_title(f'ZONAL Scale  |  Error overlay  (MAE = {mae_zonal:.2f} m)',
                 fontsize=11, fontweight='bold', color=TXT)
    plt.colorbar(sc, ax=ax, label='Absolute error (%)', shrink=0.9)
    ax.axis('off')

    ax = axs[2, 1]
    ax.scatter(true_z, predicted_z_zonal, s=SCATTER_S, alpha=0.30, color='#007700')
    _style_scatter(ax)
    ax.set_title(f'ZONAL Scale Accuracy  |  MAE = {mae_zonal:.2f} m',
                 fontsize=11, fontweight='bold', color=TXT)

    plt.tight_layout()
    plt.show()


#--------------------------------------------------------------------------------------
# 3 ASSOCIATION EXPERIMENTS (Tracking & Depth Matching)

def visualize_final_dataframe_per_mask(df_frame, image_path, calib_data):
    """
    Renders a 3-panel visual dashboard (Candidates, Winner, and 3D Bird's Eye View) 
    for every identified indicator mask in a specific dataframe.

    This function reconstructs the object association pipeline by reverse-engineering 
    the bounding box intersections and 3D centroid mapping, allowing for visual 
    validation of the algorithm's indicator assignment logic.

    Args:
        df_frame (pd.DataFrame): The dataframe containing frame-level tracking and indicator data.
        image_path (str): The file path to the raw input image.
        calib_data (dict): The parsed calibration dictionary for camera undistortion.
    """
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"❌ Error: Could not load image at {image_path}")
        return

    # Undistort the image to ensure the plotted 3D points align geometrically with the 2D pixels
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    cam_name = df_frame['camera'].iloc[0]
    undistort_function = calib_data['cameras'][cam_name]['undistort_function']
    flat_img = undistort_function(img_rgb)
    img_h, img_w = flat_img.shape[:2]

    masks_dict = {}

    # Pass 1: Seed Centroids from Generic Associated Masks
    # First, collect all segmented masks that were successfully associated with 
    # ANY vehicle, even if their final indicator state wasn't finalized.
    for _, row in df_frame.iterrows():
        assoc_str = row.get('associated_masks', '[]')
        if pd.isna(assoc_str):
            continue
            
        try:
            cands = ast.literal_eval(str(assoc_str))
            for cand in cands:
                m_idx = cand.get('mask_idx')
                
                # Initialize a master record for this unique mask
                if m_idx is not None and m_idx not in masks_dict:
                    masks_dict[m_idx] = {
                        'mask_idx':    m_idx,
                        'cx':          cand.get('cx'),
                        'cy':          cand.get('cy'),
                        'cx_3d':       cand.get('cx_3d'),
                        'cy_3d':       cand.get('cy_3d'),
                        'state_tag':   cand.get('state_tag', -1),
                        'candidates':  [],
                        'winner':      None,
                        'winner_side': None,
                    }
        except Exception:
            pass

    # Pass 2: Assign Winners & Extract High-Precision 3D Data
    # Second, scan the specific indicator columns (e.g., 'rear_left_indicator'). 
    # If a mask made it into one of these columns, it is the official "Winner".
    for _, row in df_frame.iterrows():
        for side, col in [
            ('REAR_LEFT',   'rear_left_indicator'),
            ('REAR_RIGHT',  'rear_right_indicator'),
            ('FRONT_LEFT',  'front_left_indicator'),
            ('FRONT_RIGHT', 'front_right_indicator'),
        ]:
            raw_val = row.get(col, None)
            if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
                continue
                
            raw_str = str(raw_val).strip()
            if raw_str in ('None', 'nan', ''):
                continue
                
            try:
                ind = ast.literal_eval(raw_str)
                m_idx = ind.get('mask_idx')
                if m_idx is None:
                    continue

                # Seed if not present (catches masks that skipped Pass 1)
                if m_idx not in masks_dict:
                    masks_dict[m_idx] = {
                        'mask_idx':    m_idx,
                        'cx':          ind.get('mask_cx'),
                        'cy':          ind.get('mask_cy'),
                        'cx_3d':       ind.get('cx_3d'),
                        'cy_3d':       ind.get('cy_3d'),
                        'state_tag':   ind.get('state_tag', -1),
                        'candidates':  [],
                        'winner':      None,
                        'winner_side': None,
                    }

                m = masks_dict[m_idx]
                
                # Overwrite 3D coordinates from the specialized indicator dictionary, 
                # as these are generated post-depth-fusion and are highly precise.
                if ind.get('cx_3d') is not None:  m['cx_3d'] = ind.get('cx_3d')
                if ind.get('cy_3d') is not None:  m['cy_3d'] = ind.get('cy_3d')
                if m['cx'] is None:               m['cx']    = ind.get('mask_cx')
                if m['cy'] is None:               m['cy']    = ind.get('mask_cy')
                
                # Always keep the most confident state tag (1: Active > 0: Inactive > -1: Unknown)
                if ind.get('state_tag', -1) > m['state_tag']:
                    m['state_tag'] = ind.get('state_tag', -1)

                # Lock in the winner
                m['winner']      = row
                m['winner_side'] = side

            except Exception:
                pass

    # Pass 3: Strict Dynamic Candidate Backtracking
    # Geometric verification: Find every vehicle in the frame whose bounding box 
    # spatially encapsulated this mask's 2D centroid.
    for m_idx, m_data in masks_dict.items():
        m_cx, m_cy = m_data['cx'], m_data['cy']
        if m_cx is None or m_cy is None:
            continue

        for _, row in df_frame.iterrows():
            car_pts = []
            
            # Reconstruct the 2D bounding polygon from corner points
            for c_col in ['front_corners_2d', 'rear_corners_2d']:
                c_str = row.get(c_col, None)
                if type(c_str) is float or c_str is None:
                    continue
                try:
                    pts = (np.array(ast.literal_eval(c_str))
                           if isinstance(c_str, str) else np.array(c_str))
                    if len(pts) > 0:
                        car_pts.extend(pts)
                except Exception:
                    pass

            if not car_pts:
                continue

            car_pts = np.array(car_pts)
            x_min, y_min = np.min(car_pts, axis=0)
            x_max, y_max = np.max(car_pts, axis=0)

            # If the mask centroid is inside this vehicle's polygon, it is a valid candidate
            if x_min <= m_cx <= x_max and y_min <= m_cy <= y_max:
                already_added = any(int(c['track_id']) == int(row['track_id']) for c in m_data['candidates'])
                if not already_added:
                    m_data['candidates'].append(row)

    # Pass 4: Render the 3-Panel Dashboard Per Mask
    for m_idx, m_data in masks_dict.items():
        m_cx, m_cy = m_data['cx'], m_data['cy']

        all_pts    = []
        cand_boxes = []

        # Collect bounding boxes for all valid candidates to determine the necessary image crop
        for cand_row in m_data['candidates']:
            c_track_id = int(cand_row['track_id'])
            cand_pts   = []

            for c_col in ['front_corners_2d', 'rear_corners_2d']:
                c_str = cand_row.get(c_col, None)
                if type(c_str) is float or c_str is None:
                    continue
                try:
                    pts = (np.array(ast.literal_eval(c_str))
                           if isinstance(c_str, str) else np.array(c_str))
                    if len(pts) > 0:
                        # Prevent out-of-bounds drawing
                        pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)
                        cand_pts.extend(pts)
                        all_pts.extend(pts)
                except Exception:
                    pass

            if cand_pts:
                cand_pts = np.array(cand_pts)
                c_xmin, c_ymin = np.min(cand_pts, axis=0).astype(int)
                c_xmax, c_ymax = np.max(cand_pts, axis=0).astype(int)
                cand_boxes.append({
                    'track_id': c_track_id, 'row': cand_row,
                    'xmin': c_xmin, 'ymin': c_ymin,
                    'xmax': c_xmax, 'ymax': c_ymax,
                })

        if not cand_boxes:
            continue

        # Calculate a tight dynamic crop around the candidates + mask with an 80px pad
        all_pts = np.array(all_pts)
        x_min, y_min = np.min(all_pts, axis=0).astype(int)
        x_max, y_max = np.max(all_pts, axis=0).astype(int)

        if m_cx is not None:
            x_min, x_max = min(x_min, int(m_cx)), max(x_max, int(m_cx))
        if m_cy is not None:
            y_min, y_max = min(y_min, int(m_cy)), max(y_max, int(m_cy))

        pad = 80
        x1 = max(0,     x_min - pad);  y1 = max(0,     y_min - pad)
        x2 = min(img_w, x_max + pad);  y2 = min(img_h, y_max + pad)

        crop_img = flat_img[y1:y2, x1:x2]
        if crop_img.size == 0:
            continue

        winner = m_data['winner']
        cx3d = m_data['cx_3d']
        cy3d = m_data['cy_3d']
        state_tag = m_data['state_tag']

        # Determine the source of the 3D depth for logging
        if cx3d is not None and cy3d is not None and pd.notna(cx3d) and pd.notna(cy3d):
            depth_val = np.sqrt(float(cx3d)**2 + float(cy3d)**2)
            tag_str = "(3D Success)" if state_tag == 1 else "(2D Fallback)"
            plot_cx3d = float(cx3d)
            plot_cy3d = float(cy3d)
        elif winner is not None and pd.notna(winner.get('x_3d')) and pd.notna(winner.get('y_3d')):
            plot_cx3d = float(winner['x_3d'])
            plot_cy3d = float(winner['y_3d'])
            depth_val = np.sqrt(plot_cx3d**2 + plot_cy3d**2)
            tag_str = "(Approx — vehicle centre)"
        else:
            plot_cx3d = None
            plot_cy3d = None
            depth_val = 0.0
            tag_str = "(No 3D data)"

        # Initialize the 3-panel plotting figure
        fig, (ax_cand, ax_fin, ax_bev) = plt.subplots(1, 3, figsize=(24, 6))
        fig.suptitle(
            f"Mask ID: {m_idx}  |  "
            f"STRICT Contained By {len(cand_boxes)} Vehicles  |  "
            f"Est. Mask Depth: {depth_val:.1f} m",
            fontweight='bold', fontsize=16
        )

        # Panel 1: Target Candidates (2D Image Overlay)
        ax_cand.imshow(crop_img)
        ax_cand.set_title(f"Target Candidates ({len(cand_boxes)})", fontsize=12)
        ax_cand.axis('off')

        for box in cand_boxes:
            bw = box['xmax'] - box['xmin']
            bh = box['ymax'] - box['ymin']
            rect = patches.Rectangle(
                (box['xmin'] - x1, box['ymin'] - y1), bw, bh,
                linewidth=2, edgecolor='cyan', facecolor='none', linestyle='dashed')
            ax_cand.add_patch(rect)
            ax_cand.text(
                box['xmin'] - x1, max(0, box['ymin'] - y1 - 10),
                f"Target ID: {box['track_id']}",
                color='cyan', weight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

        if m_cx is not None and m_cy is not None:
            mcx_crop, mcy_crop = m_cx - x1, m_cy - y1
            ax_cand.plot(mcx_crop, mcy_crop, 'w+', markersize=14, markeredgewidth=3)
            ax_cand.text(
                mcx_crop + 10, mcy_crop, f"Mask {m_idx}", color='white', weight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))

        # Panel 2: Final Associated Winner (2D Image Overlay)
        ax_fin.imshow(crop_img)
        ax_fin.set_title("Final Associated Vehicle", fontsize=12)
        ax_fin.axis('off')

        if winner is not None:
            w_tid = int(winner['track_id'])
            w_box = next((b for b in cand_boxes if b['track_id'] == w_tid), None)

            if w_box:
                bw = w_box['xmax'] - w_box['xmin']
                bh = w_box['ymax'] - w_box['ymin']
                rect = patches.Rectangle(
                    (w_box['xmin'] - x1, w_box['ymin'] - y1), bw, bh,
                    linewidth=3, edgecolor='#00FF00', facecolor='none')
                ax_fin.add_patch(rect)
                ax_fin.text(
                    w_box['xmin'] - x1, max(0, w_box['ymin'] - y1 - 10),
                    f"WINNER ID: {w_tid}",
                    color='#00FF00', weight='bold',
                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

            # Color code based on the physical side of the vehicle
            m_color = 'red'
            if m_data['winner_side'] is not None:
                side = m_data['winner_side']
                if 'LEFT' in side:   m_color = 'gold'
                elif 'RIGHT' in side: m_color = 'orange'

            if m_cx is not None and m_cy is not None:
                mcx_crop, mcy_crop = m_cx - x1, m_cy - y1
                ax_fin.plot(mcx_crop, mcy_crop, marker='*', color=m_color, markersize=18, markeredgecolor='black')
                ax_fin.text(
                    mcx_crop + 15, mcy_crop,
                    f"{m_data['winner_side']} INDICATOR",
                    color=m_color, weight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
        else:
            # Explicit warning if no vehicle claimed the mask
            ax_fin.text(0.5, 0.5, "NO WINNER", color='red', fontsize=20, weight='bold', ha='center', va='center', transform=ax_fin.transAxes)
            if m_cx is not None and m_cy is not None:
                ax_fin.plot(m_cx - x1, m_cy - y1, 'rX', markersize=14)

        # Panel 3: Bird's Eye View (3D X-Y Plane)
        ax_bev.set_title("3D Bird's Eye View (X-Y Plane)", fontsize=12)
        ax_bev.grid(True, linestyle='--', alpha=0.5)
        ax_bev.set_xlabel("Lateral X (meters)")
        ax_bev.set_ylabel("Longitudinal Y (meters)")
        
        # Plot Ego-Vehicle at Origin (0,0)
        ax_bev.plot(0, 0, marker='^', color='black', markersize=15, label='Ego Vehicle')

        # Project Candidate Vehicles into 3D space
        for box in cand_boxes:
            c_row = box['row']
            c_tid = box['track_id']

            if pd.notna(c_row.get('x_3d')) and pd.notna(c_row.get('yaw_3d')):
                V_x, V_y = c_row['x_3d'], c_row['y_3d']
                yaw = c_row['yaw_3d']
                W   = c_row.get('width_3d',  2.0)
                L   = c_row.get('length_3d', 5.0)

                # Generate bounding box corners relative to vehicle center
                c_x = [-W/2,  W/2,  W/2, -W/2, -W/2]
                c_y = [ L/2,  L/2, -L/2, -L/2,  L/2]

                # Apply yaw rotation and global translation
                box_pts = []
                for bx, by in zip(c_x, c_y):
                    rx = np.cos(yaw) * bx - np.sin(yaw) * by
                    ry = np.sin(yaw) * bx + np.cos(yaw) * by
                    box_pts.append([V_x + rx, V_y + ry])
                box_pts = np.array(box_pts)

                # Differentiate visually between the Winner and Losers
                is_winner = (winner is not None and c_tid == int(winner['track_id']))
                b_color  = '#00FF00' if is_winner else 'cyan'
                l_width  = 3        if is_winner else 1
                l_style  = '-'      if is_winner else '--'
                b_label  = (f"Winner ID: {c_tid}" if is_winner else f"Candidate ID: {c_tid}")

                ax_bev.plot(box_pts[:, 0], box_pts[:, 1], color=b_color, linewidth=l_width, linestyle=l_style, label=b_label)

        # Plot Mask 3D Centroid
        if plot_cx3d is not None and plot_cy3d is not None:
            winner_side = m_data['winner_side']
            m_color = 'red'
            if winner_side is not None:
                if 'LEFT' in winner_side:    m_color = 'gold'
                elif 'RIGHT' in winner_side: m_color = 'orange'
                
            ax_bev.plot(plot_cx3d, plot_cy3d, marker='*', color=m_color, markersize=16, markeredgecolor='black', linestyle='None', label=f"Mask {m_idx} {tag_str} [{winner_side}]")
        else:
            ax_bev.text(0.5, 0.05, f"Mask {m_idx}: No 3D data", color='red', ha='center', transform=ax_bev.transAxes, fontsize=9)

        ax_bev.set_aspect('equal', adjustable='datalim')
        
        # Deduplicate Legend
        handles, labels = ax_bev.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_bev.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.tight_layout()
        plt.show()


#--------------------------------------------------------------------------------------
# 4 CIELAB COLOR FILTER EXPERIMENTS

# SHARED CONSTANTS  (ideal amber point — do not change)
IDEAL_L, IDEAL_A, IDEAL_B = 205, 153, 208
W_L, W_A, W_B             = 1.0, 1.0, 1.0

# HELPER FUNCTIONS

def _load_data(image_path, mask_path, mask_idx, cam_name,
               calib_data, expansion_ratio, zoom_pad, context_mult):
    """Load image + mask, undistort, expand vertically, extract pixel arrays."""
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"Cannot load image: {image_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        img_rgb = calib_data['cameras'][cam_name]['undistort_function'](img_rgb)
    except Exception as e:
        print(f"Undistortion skipped: {e}")
    H, W = img_rgb.shape[:2]

    with np.load(mask_path) as data:
        assert str(mask_idx) in data, f"mask_idx={mask_idx} not found in {mask_path}"
        mask_u8 = (data[str(mask_idx)] > 0).astype(np.uint8)

    if expansion_ratio > 0:
        y_idx, _ = np.where(mask_u8 > 0)
        if len(y_idx) > 0:
            k = max(1, int((y_idx.max() - y_idx.min()) * expansion_ratio))
            k = k if k % 2 == 1 else k + 1
            if k >= 3:
                kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
                mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

    y_coords, x_coords = np.where(mask_u8 > 0)
    pixels_rgb         = img_rgb[y_coords, x_coords]
    valid              = np.any(pixels_rgb > 0, axis=1)
    y_coords, x_coords, pixels_rgb = y_coords[valid], x_coords[valid], pixels_rgb[valid]
    assert len(pixels_rgb) > 0, "No non-black pixels inside the mask."

    lab                    = cv2.cvtColor(
                                 pixels_rgb.reshape(-1, 1, 3).astype(np.uint8),
                                 cv2.COLOR_RGB2LAB
                             ).reshape(-1, 3).astype(float)
    L_vals, A_vals, B_vals = lab[:, 0], lab[:, 1], lab[:, 2]
    scatter_rgb            = pixels_rgb / 255.0

    x0 = max(0, x_coords.min() - zoom_pad);  x1 = min(W, x_coords.max() + zoom_pad)
    y0 = max(0, y_coords.min() - zoom_pad);  y1 = min(H, y_coords.max() + zoom_pad)
    cx_local = x_coords - x0;  cy_local = y_coords - y0
    crop_h, crop_w = y1 - y0, x1 - x0

    ctx = int(max(crop_w, crop_h) * context_mult)
    zx0, zx1 = max(0, x0 - ctx), min(W, x1 + ctx)
    zy0, zy1 = max(0, y0 - ctx), min(H, y1 + ctx)

    return (img_rgb, pixels_rgb, L_vals, A_vals, B_vals, scatter_rgb,
            x0, y0, x1, y1, cx_local, cy_local, crop_h, crop_w,
            zx0, zy0, zx1, zy1)


def _classify(L_vals, A_vals, B_vals, max_dist, b_min):
    """Split pixels into: amber accepted / fail B_MIN floor / fail distance."""
    dist      = np.sqrt(W_L*(L_vals-IDEAL_L)**2 +
                        W_A*(A_vals-IDEAL_A)**2 +
                        W_B*(B_vals-IDEAL_B)**2)
    pass_b    = (B_vals >= b_min) if b_min > 0 else np.ones(len(B_vals), dtype=bool)
    pass_dist = dist < max_dist
    return pass_b & pass_dist, ~pass_b, pass_b & ~pass_dist


def _render_lab_bg(max_dist, b_min, mode):
    """
    Render a 256x256 CIELAB color-space swatch:
      accepted region -> true color,  rejected region -> darkened (x0.25).

    mode='lb'  X = b*, Y = L*   (fix A = IDEAL_A)
    mode='ab'  X = a*, Y = b*   (fix L = IDEAL_L)
    """
    arr    = np.arange(256)
    Xa, Ya = np.meshgrid(arr, arr)      # [row=y, col=x]

    L_g, A_g, B_g = (Ya, np.full_like(Xa, IDEAL_A), Xa) if mode == 'lb' \
                    else (np.full_like(Xa, IDEAL_L), Xa, Ya)

    rgb_img  = cv2.cvtColor(np.dstack((L_g, A_g, B_g)).astype(np.uint8),
                            cv2.COLOR_LAB2RGB)
    dist_bg  = np.sqrt(W_L*(L_g-IDEAL_L)**2 + W_A*(A_g-IDEAL_A)**2 + W_B*(B_g-IDEAL_B)**2)
    b_ok     = (B_g >= b_min) if b_min > 0 else np.ones_like(B_g, dtype=bool)
    accepted = (dist_bg <= max_dist) & b_ok
    dark     = (rgb_img.astype(float) * 0.25).astype(np.uint8)
    return np.where(np.expand_dims(accepted, 2), rgb_img, dark)


def _draw_full_image_panel(ax, img_rgb, zy0, zy1, zx0, zx1,
                            x0, y0, x1, crop_w, crop_h,
                            mask_idx, cam_name, fixed_label, txt):
    ax.imshow(img_rgb[zy0:zy1, zx0:zx1])
    ax.axis('off')
    ax.set_title(
        f"Indicator Region  |  {cam_name}  |  Mask {mask_idx}  |  {fixed_label}",
        color=txt, fontsize=13, fontweight='bold', pad=6)
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0-zx0, y0-zy0), crop_w, crop_h, boxstyle='square,pad=0',
        linewidth=2.5, edgecolor='#cc0000', facecolor='none'))
    cx = (x0-zx0 + x1-zx0) / 2
    ax.annotate(f"Mask {mask_idx}", xy=(cx, y0-zy0),
                xytext=(cx, max(2, y0-zy0 - (zy1-zy0)*0.07)),
                color='#cc0000', fontsize=11, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.8))


def _draw_cutout_panel(ax, img_rgb, y0, y1, x0, x1,
                        cy_local, cx_local, crop_h, crop_w,
                        amber, fail_bmin, fail_dist, row_label, txt):
    ax.set_facecolor('white')
    crop_base = img_rgb[y0:y1, x0:x1].copy()
    dimmed    = (crop_base.astype(float) * 0.25).astype(np.uint8)
    overlay   = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    overlay[cy_local[amber],     cx_local[amber]]     = [0,   210,  80]
    overlay[cy_local[fail_bmin], cx_local[fail_bmin]] = [30,  120, 255]
    overlay[cy_local[fail_dist], cx_local[fail_dist]] = [230,  50,  50]
    active  = amber | fail_bmin | fail_dist
    display = dimmed.copy()
    display[cy_local[active], cx_local[active]] = cv2.addWeighted(
        crop_base[cy_local[active], cx_local[active]], 0.25,
        overlay[cy_local[active],   cx_local[active]], 0.75, 0)
    ax.imshow(display);  ax.axis('off')
    ax.set_title(row_label, color=txt, fontsize=9, pad=5)
    ax.legend(handles=[
        mpatches.Patch(color='#00d250', label=f'Amber accepted  ({amber.sum()} px)'),
        mpatches.Patch(color='#1e78ff', label=f'Fail B_MIN floor  ({fail_bmin.sum()} px)'),
        mpatches.Patch(color='#e63232', label=f'Fail distance  ({fail_dist.sum()} px)'),
    ], loc='lower left', fontsize=8,
       facecolor='white', labelcolor=txt, framealpha=0.9, edgecolor='#ccc')
    for sp in ax.spines.values(): sp.set_edgecolor('#ccc')


def _draw_color_ax(ax, mode, bg_img, scatter_x, scatter_y, scatter_rgb,
                   b_min, semi_x, semi_y, txt):
    """Render one CIELAB color-space axes (L* vs b* or a* vs b*)."""
    ax.set_facecolor('#f0f0f0')
    ax.imshow(bg_img, extent=[0,255,0,255], origin='lower', aspect='auto', alpha=0.95)

    cx = IDEAL_B if mode == 'lb' else IDEAL_A
    cy = IDEAL_L if mode == 'lb' else IDEAL_B
    ax.add_patch(Ellipse(
        (cx, cy), width=2*semi_x, height=2*semi_y,
        fill=False, edgecolor='white', linewidth=2.5, linestyle='--',
        path_effects=[pe.withStroke(linewidth=4, foreground='black')]))
    ax.plot(cx, cy, 'w+', markersize=16, markeredgewidth=3,
            path_effects=[pe.withStroke(linewidth=4, foreground='black')])

    if b_min > 0:
        if mode == 'lb':
            ax.axvline(b_min, color='#00aaff', lw=2.5)
            ax.add_patch(Rectangle((0,0), b_min, 255,
                                   fill=True, color='black', alpha=0.25, hatch='//'))
        else:
            ax.axhline(b_min, color='#00aaff', lw=2.5)
            ax.add_patch(Rectangle((0,0), 255, b_min,
                                   fill=True, color='black', alpha=0.25, hatch='//'))
    else:
        ax.text(10, 230, 'B_MIN disabled', color='#00aaff', fontsize=9, fontweight='bold')

    ax.scatter(scatter_x, scatter_y, c=scatter_rgb, s=8, alpha=0.8, linewidths=0, zorder=5)

    if mode == 'lb':
        xlabel, ylabel, title = 'b*  (Blue <- -> Yellow)', 'L*  (Dark <- -> Bright)', 'L* vs b*'
    else:
        xlabel, ylabel, title = 'a*  (Green <- -> Red)', 'b*  (Blue <- -> Yellow)', 'a* vs b*'

    ax.set_xlim(0, 255);  ax.set_ylim(0, 255)
    ax.set_xlabel(xlabel, color=txt, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, color=txt, fontsize=10, fontweight='bold')
    ax.set_title(title, color=txt, fontsize=11, pad=5)
    ax.tick_params(colors=txt, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#aaa'); sp.set_linewidth(1.5)


#--------------------------------------------------------------------------------------
# 4.1 MAX_DIST SWEEP EXPERIMENT

def run_max_dist_sweep(
    image_path,
    mask_path,
    mask_idx,
    cam_name,
    calib_data,
    fixed_b_min     = 135,
    max_dist_values = (50, 80, 120),
    expansion_ratio = 0.20,
    zoom_pad        = 20,
    context_mult    = 2.5,
):
    """
    Experiment — MAX_DIST sweep (B_MIN fixed).

    One row per MAX_DIST value, 3 columns per row:
      Col 0 : indicator cutout  (green=amber, blue=fail B_MIN, red=fail distance)
      Col 1 : L* vs b* CIELAB plane with acceptance boundary + pixel scatter
      Col 2 : a* vs b* CIELAB plane with acceptance boundary + pixel scatter

    Args:
        image_path       : str   path to the camera image
        mask_path        : str   path to the NPZ mask file
        mask_idx         : int   key inside the NPZ (e.g. 1)
        cam_name         : str   camera name for undistort_function lookup
        calib_data       : dict  calibration dict with undistort functions
        fixed_b_min      : int   B* floor held constant (default 135)
        max_dist_values  : list  3 ellipsoid radii to sweep (default [50, 80, 120])
        expansion_ratio  : float vertical mask dilation (0 = exact SAM mask)
        zoom_pad         : int   pixel padding around tight mask crop
        context_mult     : float context window size relative to mask bounding box
    """

    TXT = '#111111'
    (img_rgb, pixels_rgb, L_vals, A_vals, B_vals, scatter_rgb,
     x0, y0, x1, y1, cx_local, cy_local, crop_h, crop_w,
     zx0, zy0, zx1, zy1) = _load_data(
         image_path, mask_path, mask_idx, cam_name,
         calib_data, expansion_ratio, zoom_pad, context_mult)

    N   = len(max_dist_values)
    fig = plt.figure(figsize=(20, 5 + 5.5*N), facecolor='white')
    gs  = GridSpec(N+1, 3, figure=fig, height_ratios=[1.8]+[1]*N,
                   hspace=0.30, wspace=0.18, top=0.94, bottom=0.04,
                   left=0.04, right=0.97)

    _draw_full_image_panel(
        fig.add_subplot(gs[0, :]), img_rgb, zy0, zy1, zx0, zx1,
        x0, y0, x1, crop_w, crop_h, mask_idx, cam_name,
        f"B_MIN fixed = {fixed_b_min}", TXT)

    for i, max_dist in enumerate(max_dist_values):
        amber, fail_bmin, fail_dist = _classify(L_vals, A_vals, B_vals, max_dist, fixed_b_min)
        n_amber   = int(amber.sum())
        n_total   = len(pixels_rgb)
        amber_pct = n_amber / n_total * 100
        label = (f"MAX_DIST = {max_dist}  |  "
                 f"amber_pct = {amber_pct:.1f}%  ({n_amber} / {n_total} px)")

        _draw_cutout_panel(fig.add_subplot(gs[i+1, 0]), img_rgb, y0, y1, x0, x1,
                            cy_local, cx_local, crop_h, crop_w,
                            amber, fail_bmin, fail_dist, label, TXT)
        _draw_color_ax(fig.add_subplot(gs[i+1, 1]), 'lb',
                       _render_lab_bg(max_dist, fixed_b_min, 'lb'),
                       B_vals, L_vals, scatter_rgb, fixed_b_min,
                       max_dist/np.sqrt(W_B), max_dist/np.sqrt(W_L), TXT)
        _draw_color_ax(fig.add_subplot(gs[i+1, 2]), 'ab',
                       _render_lab_bg(max_dist, fixed_b_min, 'ab'),
                       A_vals, B_vals, scatter_rgb, fixed_b_min,
                       max_dist/np.sqrt(W_A), max_dist/np.sqrt(W_B), TXT)

    fig.suptitle(
        f"Experiment A2 — MAX_DIST Sweep  |  B_MIN fixed = {fixed_b_min}  "
        f"|  Ideal = (L={IDEAL_L}, a={IDEAL_A}, b={IDEAL_B})",
        color=TXT, fontsize=13, fontweight='bold')
    plt.show()


#--------------------------------------------------------------------------------------
# 4.2 B_MIN SWEEP EXPERIMENT

def run_bmin_sweep(
    image_path,
    mask_path,
    mask_idx,
    cam_name,
    calib_data,
    fixed_max_dist  = 80,
    b_min_values    = (0, 150, 190),
    expansion_ratio = 0.20,
    zoom_pad        = 20,
    context_mult    = 2.5,
):
    """
    Experiment — B_MIN sweep (MAX_DIST fixed).

    One row per B_MIN value, 3 columns per row:
      Col 0 : indicator cutout  (green=amber, blue=fail B_MIN, red=fail distance)
      Col 1 : L* vs b* CIELAB plane with acceptance boundary + pixel scatter
      Col 2 : a* vs b* CIELAB plane with acceptance boundary + pixel scatter

    Args:
        image_path       : str   path to the camera image
        mask_path        : str   path to the NPZ mask file
        mask_idx         : int   key inside the NPZ (e.g. 1)
        cam_name         : str   camera name for undistort_function lookup
        calib_data       : dict  calibration dict with undistort functions
        fixed_max_dist   : int   ellipsoid radius held constant (default 80)
        b_min_values     : list  3 B* floor values to sweep (0 = disabled)
        expansion_ratio  : float vertical mask dilation (0 = exact SAM mask)
        zoom_pad         : int   pixel padding around tight mask crop
        context_mult     : float context window size relative to mask bounding box
    """

    TXT = '#111111'
    (img_rgb, pixels_rgb, L_vals, A_vals, B_vals, scatter_rgb,
     x0, y0, x1, y1, cx_local, cy_local, crop_h, crop_w,
     zx0, zy0, zx1, zy1) = _load_data(
         image_path, mask_path, mask_idx, cam_name,
         calib_data, expansion_ratio, zoom_pad, context_mult)

    semi = {
        'lb_x': fixed_max_dist/np.sqrt(W_B), 'lb_y': fixed_max_dist/np.sqrt(W_L),
        'ab_x': fixed_max_dist/np.sqrt(W_A), 'ab_y': fixed_max_dist/np.sqrt(W_B),
    }

    N   = len(b_min_values)
    fig = plt.figure(figsize=(20, 5 + 5.5*N), facecolor='white')
    gs  = GridSpec(N+1, 3, figure=fig, height_ratios=[1.8]+[1]*N,
                   hspace=0.30, wspace=0.18, top=0.94, bottom=0.04,
                   left=0.04, right=0.97)

    _draw_full_image_panel(
        fig.add_subplot(gs[0, :]), img_rgb, zy0, zy1, zx0, zx1,
        x0, y0, x1, crop_w, crop_h, mask_idx, cam_name,
        f"MAX_DIST fixed = {fixed_max_dist}", TXT)

    for i, b_min in enumerate(b_min_values):
        amber, fail_bmin, fail_dist = _classify(L_vals, A_vals, B_vals, fixed_max_dist, b_min)
        n_amber   = int(amber.sum())
        n_total   = len(pixels_rgb)
        amber_pct = n_amber / n_total * 100
        bmin_str  = f"B_MIN = {b_min}" if b_min > 0 else "B_MIN = disabled"
        label = (f"{bmin_str}  |  "
                 f"amber_pct = {amber_pct:.1f}%  ({n_amber} / {n_total} px)")

        _draw_cutout_panel(fig.add_subplot(gs[i+1, 0]), img_rgb, y0, y1, x0, x1,
                            cy_local, cx_local, crop_h, crop_w,
                            amber, fail_bmin, fail_dist, label, TXT)
        _draw_color_ax(fig.add_subplot(gs[i+1, 1]), 'lb',
                       _render_lab_bg(fixed_max_dist, b_min, 'lb'),
                       B_vals, L_vals, scatter_rgb, b_min,
                       semi['lb_x'], semi['lb_y'], TXT)
        _draw_color_ax(fig.add_subplot(gs[i+1, 2]), 'ab',
                       _render_lab_bg(fixed_max_dist, b_min, 'ab'),
                       A_vals, B_vals, scatter_rgb, b_min,
                       semi['ab_x'], semi['ab_y'], TXT)

    fig.suptitle(
        f"Experiment A3 — B_MIN Sweep  |  MAX_DIST fixed = {fixed_max_dist}  "
        f"|  Ideal = (L={IDEAL_L}, a={IDEAL_A}, b={IDEAL_B})",
        color=TXT, fontsize=13, fontweight='bold')
    plt.show()



#--------------------------------------------------------------------------------------
# 5 BLINK DETECTION EXPERIMENTS

# HELPER FUNCTIONS

def _load(path_or_df):
    """Accept either a file path string or an already-loaded DataFrame."""
    if isinstance(path_or_df, str):
        return pd.read_csv(path_or_df)
    return path_or_df.copy()


def _get_signal(raw_df, track_id, cam_name, indicator_col):
    """Return time-sorted amber_pct array for one (track, camera, indicator)."""
    grp = (raw_df[(raw_df['track_id'] == track_id) & (raw_df['camera'] == cam_name)]
           .sort_values('timestamp'))
    if grp.empty:
        raise ValueError(f"No rows for track_id={track_id}, camera={cam_name}")

    def _parse(val):
        if val is None: return 0.0
        try:
            if pd.isna(val): return 0.0
        except Exception: pass
        s = str(val).strip()
        if s in ('', 'None', 'nan'): return 0.0
        try:
            return float(ast.literal_eval(s).get('amber_pct', 0.0))
        except Exception:
            return 0.0

    return np.array([_parse(v) for v in grp[indicator_col].values], dtype=float)


def _get_gt_binary(gt_df, track_id, cam_name, indicator_col):
    """
    Return a frame-level binary array (1=BLINKING, 0=OFF) from the GT CSV.
    GT CSV must have a '{indicator_col}_blink_state' column.
    Returns None if the column or track is missing.
    """
    blink_col = f'{indicator_col}_blink_state'
    if blink_col not in gt_df.columns:
        return None
    grp = (gt_df[(gt_df['track_id'] == track_id) & (gt_df['camera'] == cam_name)]
           .sort_values('timestamp'))
    if grp.empty:
        return None
    states = grp[blink_col].fillna('OFF').values
    return (states == 'BLINKING').astype(int)


def _run_detect(signal, fps,
                autocorr_min_peak, min_detrended_range, detrend_window_sec):
    """Call indicator_core.detect_blink_signal with given params."""
    from functions.indicator_core import detect_blink_signal
    return detect_blink_signal(
        signal, fps,
        detrend_window_sec, 0.3,      # detrend_noise_factor fixed
        min_detrended_range,
        autocorr_min_peak,
        2, 2,                          # min_on_frames, min_off_frames fixed
        0.8, 2.5,                      # blink_min_hz, blink_max_hz fixed
        2, 40.0,                       # min_cycles, max_jitter_pct fixed
    )

def _draw_blink_row(axes, signal, res, gt_binary, fps, sweep_label, autocorr_threshold=None):
    """
    4-panel diagnostic row for one parameter value.

      Panel 0 : Raw amber_pct + rolling-median baseline  + GT shading
      Panel 1 : Detrended signal + threshold.
                Blue fill  = Stage 4 cleaned binary (before episode filter).
                Red fill   = Stage 7 frame_state == BLINKING (FINAL output,
                             after 1.5 x best_lag episode rejection).
                Green shading = GT BLINKING regions.
                Key insight: if a row has blue but no red, Stage 7 rejected
                the blinking due to irregular episode spacing.
      Panel 2 : Autocorrelation with valid-lag window + detected peak
      Panel 3 : Verdict vs GT using frame-level IoU overlap — NOT a track-level
                binary match. A wrong-timing BLINKING cannot fake a MATCH here.

    GT shading: semi-transparent GREEN band where GT = BLINKING.
    """
    TXT = '#111111'
    frames = np.arange(len(signal))
    n = len(signal)

    # ── Align GT to signal length ─────────────────────────────────────────────
    if gt_binary is not None:
        gt = np.zeros(n, dtype=int)
        gt[:min(n, len(gt_binary))] = gt_binary[:n]
    else:
        gt = None

    # ── Stage 7 frame_state → binary (the TRUE final output) ─────────────────
    frame_state = res.get('frame_state', None)
    if frame_state is not None:
        exp_final = (frame_state == 'BLINKING').astype(int)
    else:
        exp_final = res['cleaned'].astype(int) if res['is_blinking'] else np.zeros(n, dtype=int)

    def _shade_gt(ax):
        """Shade green columns where GT = BLINKING."""
        if gt is None:
            return
        in_block = False
        start = 0
        for f in range(n + 1):
            val = gt[f] if f < n else 0
            if val == 1 and not in_block:
                in_block = True; start = f
            elif val == 0 and in_block:
                ax.axvspan(start, f - 1, ymin=0, ymax=1,
                           color='#00cc44', alpha=0.18, zorder=0)
                in_block = False

    # ── Panel 0: Raw signal + baseline ───────────────────────────────────────
    ax = axes[0]
    ax.plot(frames, signal,          color='#555555', lw=1.2, alpha=0.9,  label='amber_pct')
    ax.plot(frames, res['baseline'], color='#cc0000', lw=2.0, linestyle='--', label='baseline')
    _shade_gt(ax)
    ax.set_ylabel('amber_pct %', color=TXT, fontsize=8)
    ax.set_title(sweep_label, color=TXT, fontsize=9, fontweight='bold', pad=3)
    if gt is not None:
        ax.plot([], [], color='#00cc44', alpha=0.5, lw=6, label='GT: BLINKING')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.set_facecolor('#f7f7f7'); ax.tick_params(labelsize=7)
    ax.set_xlim(0, n - 1)

    # ── Panel 1: Detrended + Stage 4 (cleaned) + Stage 7 (final) ─────────────
    ax = axes[1]
    ax.plot(frames, res['detrended'], color='#2255cc', lw=1.2, label='detrended')
    thresh = res['det_threshold']
    ax.axhline(thresh, color='#ff8800', lw=1.5, linestyle=':', label=f'threshold={thresh:.1f}')

    ymax_det = max(np.abs(res['detrended']).max() * 1.1, abs(thresh) * 1.5, 1.0)

    # Stage 4 cleaned binary — what edge-timing sees
    cleaned = res['cleaned'].astype(float)
    ax.fill_between(frames, 0, cleaned * ymax_det,
                    color='#2255cc', alpha=0.15, label='Stage 4 (cleaned)')

    # Stage 7 frame_state — FINAL output after 1.5 x best_lag episode filter
    ax.fill_between(frames, 0, exp_final * ymax_det,
                    color='#cc2200', alpha=0.35, label='Stage 7 (final BLINKING)')

    _shade_gt(ax)
    if gt is not None:
        ax.plot([], [], color='#00cc44', alpha=0.5, lw=6, label='GT: BLINKING')
    ax.set_ylabel('detrended %', color=TXT, fontsize=8)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.set_facecolor('#f7f7f7'); ax.tick_params(labelsize=7)
    ax.set_xlim(0, n - 1)

    # ── Panel 2: Autocorrelation ──────────────────────────────────────────────
    ax = axes[2]
    ac = res['autocorr']
    lags = np.arange(len(ac))
    ax.plot(lags, ac, color='#444444', lw=1.0)
    min_lag, max_lag = res['autocorr_lags']
    ax.axvspan(min_lag, max_lag, color='#ffe0a0', alpha=0.45,
               label=f'valid lags [{min_lag}-{max_lag}]')
    if res['best_lag'] is not None:
        bl = res['best_lag']
        ax.axvline(bl, color='#cc0000', lw=2.0, linestyle='--',
                   label=f"peak lag={bl}  ({res['detected_hz']:.2f} Hz)")
        ax.plot(bl, ac[bl], 'r*', markersize=12)
    # ── Minimum threshold line — shows exactly why a peak passes or is rejected
    if autocorr_threshold is not None:
        ax.axhline(autocorr_threshold, color='#8800cc', lw=2.0, linestyle='--',
                   label=f'min threshold = {autocorr_threshold:.2f}')
        # Shade the rejection zone below the threshold
        ax.axhspan(0, autocorr_threshold, color='#cc88ff', alpha=0.10, zorder=0)
    ax.set_ylabel('autocorr', color=TXT, fontsize=8)
    ax.set_xlabel('lag (frames)', color=TXT, fontsize=8)
    ax.set_xlim(0, min(len(ac) - 1, max_lag * 2))
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.set_facecolor('#f7f7f7'); ax.tick_params(labelsize=7)

    # ── Panel 3: Verdict vs GT — FRAME-LEVEL IoU ─────────────────────────────
    ax = axes[3]
    ax.axis('off')
    ax.set_facecolor('#f7f7f7')
    for sp in ax.spines.values(): sp.set_edgecolor('#cccccc')

    exp_verdict = 'BLINKING' if res['is_blinking'] else 'OFF'
    exp_color   = '#007700' if res['is_blinking'] else '#cc0000'

    if gt is not None:
        gt_verdict = 'BLINKING' if gt.any() else 'OFF'
        gt_color   = '#007700' if gt.any() else '#cc0000'

        # Frame overlap: mismatch > 0.2 → MISMATCH, else MATCH
        intersection = int(np.sum((gt == 1) & (exp_final == 1)))
        union        = int(np.sum((gt == 1) | (exp_final == 1)))
        iou          = intersection / union if union > 0 else 0.0
        mismatch     = 1.0 - iou

        if mismatch > 0.2:
            match_str = f'MISMATCH  (overlap={iou:.2f})'
            match_col = '#990000'; match_bg = '#ffcccc'
        else:
            match_str = f'MATCH  (overlap={iou:.2f})'
            match_col = '#005500'; match_bg = '#ccffcc'

        ax.text(0.5, 0.88, f'GT:  {gt_verdict}',  color=gt_color,  fontsize=11,
                fontweight='bold', ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.73, f'EXP: {exp_verdict}', color=exp_color, fontsize=11,
                fontweight='bold', ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.57, match_str, color=match_col, fontsize=11,
                fontweight='bold', ha='center', va='center', transform=ax.transAxes,
                bbox=dict(facecolor=match_bg, edgecolor=match_col, boxstyle='round,pad=0.3'))
        ax.text(0.5, 0.44, f'({intersection} / {union} frames overlap)', color='#555555',
                fontsize=8, ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.80, f'EXP: {exp_verdict}', color=exp_color, fontsize=12,
                fontweight='bold', ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.62, '(no GT loaded)', color='#888888', fontsize=9,
                ha='center', va='center', transform=ax.transAxes)

    freq_str = f"{res['detected_hz']:.2f} Hz" if res['detected_hz'] else "---"
    info = (
        f"det_range : {res['det_range']:.1f}%\n"
        f"autocorr  : {res['autocorr_strength']:.3f}\n"
        f"freq      : {freq_str}\n"
        f"cycles    : {res['n_cycles']}\n"
        f"reason: {res['reason'][:50]}"
    )
    ax.text(0.5, 0.24, info, color=TXT, fontsize=7.5,
            ha='center', va='center', transform=ax.transAxes,
            family='monospace', linespacing=1.6)

#--------------------------------------------------------------------------------------
# 5.1 Sweep autocorr_min_peak  (autocorrelation sensitivity)

def run_autocorr_sweep(
    raw_csv_path,
    gt_csv_path,
    track_id,
    cam_name,
    indicator_col,
    fps,
    autocorr_values      = (0.15, 0.30, 0.45),
    fixed_min_range      = 5.0,
    fixed_detrend_window = 2.5,
):
    """
    Experiment — sweep autocorr_min_peak (autocorrelation sensitivity).

    This is the single most important parameter.
      Low  → easier detection, risk of false positives.
      High → stricter, only strongly periodic amber signals pass.

    3 rows (one per value). Each row has 5 panels:
      [Parameter label] [Raw + baseline] [Detrended + binary] [Autocorrelation] [Verdict vs GT]

    The swept value is displayed large and centred in the leftmost panel so it
    is immediately obvious to the reader what changes between rows.

    GT blink regions (from indicator_core output) are shaded GREEN on
    panels 1 and 2 so the reader can see which value matches the ground truth.

    Args:
        raw_csv_path         : str or DataFrame  — model_tools output CSV
        gt_csv_path          : str or DataFrame  — indicator_core output CSV
                               (must contain {indicator_col}_blink_state column)
        track_id             : int/str  known-blinking track to analyse
        cam_name             : str      e.g. 'front_right_camera'
        indicator_col        : str      e.g. 'rear_left_indicator'
        fps                  : float    video frame rate (e.g. 10.0)
        autocorr_values      : tuple    exactly 3 values to compare
        fixed_min_range      : float    min_detrended_range held constant
        fixed_detrend_window : float    detrend_window_sec held constant
    """
    raw_df = _load(raw_csv_path)
    gt_df  = _load(gt_csv_path)

    signal    = _get_signal(raw_df, track_id, cam_name, indicator_col)
    gt_binary = _get_gt_binary(gt_df, track_id, cam_name, indicator_col)

    N = len(autocorr_values)
    fig = plt.figure(figsize=(24, 4.8 * N), facecolor='white')
    fig.suptitle(
        f"Experiment B1 — autocorr_min_peak Sweep  |  Track {track_id}  |  "
        f"{cam_name}  |  {indicator_col}\n"
        f"Fixed: min_detrended_range={fixed_min_range}%   detrend_window={fixed_detrend_window}s   fps={fps}"
        f"     ▌ Green shading = GT BLINKING",
        fontsize=11, fontweight='bold', y=1.01
    )

    # 5 columns: [label | raw+baseline | detrended+binary | autocorr | verdict]
    # label column is narrow
    gs = fig.add_gridspec(N, 5, width_ratios=[1.0, 3.2, 3.2, 3.2, 2.4],
                          hspace=0.55, wspace=0.35)

    for i, ac_val in enumerate(autocorr_values):
        res = _run_detect(signal, fps,
                          autocorr_min_peak=float(ac_val),
                          min_detrended_range=fixed_min_range,
                          detrend_window_sec=fixed_detrend_window)

        # ── Label panel ──────────────────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[i, 0])
        ax_lbl.axis('off')
        ax_lbl.set_facecolor('#eef2ff')
        for sp in ax_lbl.spines.values():
            sp.set_visible(True); sp.set_edgecolor('#aaaacc'); sp.set_linewidth(1.5)

        ax_lbl.text(0.5, 0.62, 'autocorr_min_peak', color='#333366',
                    fontsize=9, fontweight='bold',
                    ha='center', va='center', transform=ax_lbl.transAxes)
        ax_lbl.text(0.5, 0.35, f'{ac_val:.2f}', color='#111155',
                    fontsize=28, fontweight='bold',
                    ha='center', va='center', transform=ax_lbl.transAxes)

        data_axes = [fig.add_subplot(gs[i, j]) for j in range(1, 5)]
        _draw_blink_row(data_axes, signal, res, gt_binary, fps, sweep_label='',
                        autocorr_threshold=float(ac_val))

    plt.show()


#--------------------------------------------------------------------------------------
# 5.2 Sweep detrend_window_sec  (drift removal window)

def run_detrend_window_sweep(
    raw_csv_path,
    gt_csv_path,
    track_id,
    cam_name,
    indicator_col,
    fps,
    window_values    = (1.0, 2.5, 5.0),
    fixed_autocorr   = 0.30,
    fixed_min_range  = 5.0,
):
    """
    Experiment — sweep detrend_window_sec (rolling-median baseline window).

    Controls how aggressively slow illumination drift is removed.

      Too small → baseline tracks the blink itself → detrended signal is flat.
      Too large → slow drift not removed → flatness gate fails.
      Sweet spot → only trend removed, periodic blink survives.

    3 rows (one per value). Each row has 5 panels:
      [Parameter label] [Raw + baseline] [Detrended + binary] [Autocorrelation] [Verdict vs GT]

    GT blink regions shaded GREEN for direct comparison.

    Args:
        raw_csv_path     : str or DataFrame  — model_tools output CSV
        gt_csv_path      : str or DataFrame  — indicator_core output CSV
        track_id         : int/str  known-blinking track
        cam_name         : str
        indicator_col    : str
        fps              : float
        window_values    : tuple   exactly 3 detrend_window_sec values to compare
        fixed_autocorr   : float   autocorr_min_peak held constant
        fixed_min_range  : float   min_detrended_range held constant
    """
    raw_df = _load(raw_csv_path)
    gt_df  = _load(gt_csv_path)

    signal    = _get_signal(raw_df, track_id, cam_name, indicator_col)
    gt_binary = _get_gt_binary(gt_df, track_id, cam_name, indicator_col)

    N = len(window_values)
    fig = plt.figure(figsize=(24, 4.8 * N), facecolor='white')
    fig.suptitle(
        f"Experiment B3 — detrend_window_sec Sweep  |  Track {track_id}  |  "
        f"{cam_name}  |  {indicator_col}\n"
        f"Fixed: autocorr_min_peak={fixed_autocorr}   min_detrended_range={fixed_min_range}%   fps={fps}"
        f"     ▌ Green shading = GT BLINKING",
        fontsize=11, fontweight='bold', y=1.01
    )

    gs = fig.add_gridspec(N, 5, width_ratios=[1.0, 3.2, 3.2, 3.2, 2.4],
                          hspace=0.55, wspace=0.35)

    for i, win in enumerate(window_values):
        res = _run_detect(signal, fps,
                          autocorr_min_peak=fixed_autocorr,
                          min_detrended_range=fixed_min_range,
                          detrend_window_sec=float(win))

        # ── Label panel ──────────────────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[i, 0])
        ax_lbl.axis('off')
        ax_lbl.set_facecolor('#fff4e6')
        for sp in ax_lbl.spines.values():
            sp.set_visible(True); sp.set_edgecolor('#cc9944'); sp.set_linewidth(1.5)

        ax_lbl.text(0.5, 0.65, 'detrend_window_sec', color='#664400',
                    fontsize=9, fontweight='bold',
                    ha='center', va='center', transform=ax_lbl.transAxes)
        ax_lbl.text(0.5, 0.38, f'{win:.1f} s', color='#3b1f00',
                    fontsize=26, fontweight='bold',
                    ha='center', va='center', transform=ax_lbl.transAxes)
        ax_lbl.text(0.5, 0.14, f'({int(win * fps)} frames)', color='#664400',
                    fontsize=9, ha='center', va='center', transform=ax_lbl.transAxes)

        data_axes = [fig.add_subplot(gs[i, j]) for j in range(1, 5)]
        _draw_blink_row(data_axes, signal, res, gt_binary, fps, sweep_label='',
                        autocorr_threshold=fixed_autocorr)

    plt.show()