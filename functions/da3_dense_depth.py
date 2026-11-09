import os
import cv2
import torch
import numpy as np
from PIL import Image
from contextlib import redirect_stdout
from scipy.interpolate import griddata

def get_dense_metric_depth(image_np, lidar_u, lidar_v, lidar_cam_z, processor, model, device):
    """
    Generates a dense, metric depth map by fusing relative DA3 depth predictions 
    with sparse LiDAR ground-truth data.

    Args:
        image_np (np.ndarray): The 2D RGB input image array.
        lidar_u (np.ndarray): 1D array of horizontal (x) image coordinates for LiDAR hits.
        lidar_v (np.ndarray): 1D array of vertical (y) image coordinates for LiDAR hits.
        lidar_cam_z (np.ndarray): 1D array of metric depth (z) values for the LiDAR hits.
        processor (object or None): The HuggingFace processor. If None, assumes Depth Anything V3.
        model (object): The instantiated depth estimation model (HF or custom).
        device (str/torch.device): The hardware acceleration target (e.g., 'cuda' or 'cpu').

    Returns:
        np.ndarray: A 2D dense depth map (in meters) matching the original image dimensions.
    """
    H_img, W_img = image_np.shape[:2]
    
    # 1. Generate Relative Depth Map via DA3 Foundation Model
    if processor is None:
        # Depth Anything V3 Path (Custom Model)
        with torch.no_grad():
            
            # Suppress noisy [INFO] prints from the model internals during inference
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                if hasattr(model, 'infer_image'):
                    raw_depth = model.infer_image(image_np) # DA v2/v3 standard API
                elif hasattr(model, 'inference'):
                    pred = model.inference([image_np]) # Fallback for some wrapper APIs
                    raw_depth = pred.depth[0] if hasattr(pred, 'depth') else pred[0]
                else:
                    # Fallback for raw torch modules without wrappers
                    raw_depth = model(torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device))
                    raw_depth = raw_depth.squeeze().cpu().numpy()
                
        # Resize DA3 prediction to match original image shape if necessary
        if raw_depth.shape[:2] != (H_img, W_img):
            disparity = cv2.resize(raw_depth, (W_img, H_img), interpolation=cv2.INTER_CUBIC)
        else:
            disparity = raw_depth
            
    else:
        # HuggingFace Transformers Path
        pil_image = Image.fromarray(image_np) if not isinstance(image_np, Image.Image) else image_np
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
    
        # Interpolate HuggingFace output back to native resolution
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(H_img, W_img),
            mode="bicubic",
            align_corners=False,
        )
        disparity = prediction.squeeze().cpu().numpy()
    
    # 2. Filter Valid LiDAR Points
    H, W = disparity.shape
    
    # Keep only LiDAR points that are in front of the camera (z > 0)
    valid_mask = (lidar_u >= 0) & (lidar_u < W) & (lidar_v >= 0) & (lidar_v < H) & (lidar_cam_z > 0)
    
    u = lidar_u[valid_mask]
    v = lidar_v[valid_mask]
    z = lidar_cam_z[valid_mask]
    
    # Failsafe: If no valid LiDAR points exist to anchor the scale, return a static default depth
    if len(z) < 10:
        return np.ones((H, W)) * 10.0 
        
    # Extract DA3 predicted depth exactly where LiDAR hits exist
    disp_sampled = disparity[v, u]
    
    # Avoid zero division when calculating scale factors
    valid_disp = disp_sampled > 1e-6
    if not np.any(valid_disp):
        return np.ones((H, W)) * 10.0
        
    disp_sampled = disp_sampled[valid_disp]
    z = z[valid_disp]
    u = u[valid_disp]
    v = v[valid_disp]
    
    # 3. Compute and Interpolate Metric Scale
    # The DA3 outputs relative depth directly. 
    # Therefore: Z_metric = Scale * DA3_Depth -> Scale = Z_metric / DA3_Depth
    sparse_scales = z / np.clip(disp_sampled, 1e-6, None)
    global_median_scale = np.median(sparse_scales)
    
    # Create geometric coordinate grids to perform 2D spatial interpolation
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    points = np.vstack((u, v)).T
    
    # Interpolate the sparse scale factors across the entire 2D image.
    # Between LiDAR hit points: Linear method creates a smooth Delaunay triangulation.
    # Outside LiDAR hit points (e.g. the sky): global median scale is used.
    dense_scale_map = griddata(
        points, sparse_scales, (grid_x, grid_y), 
        method='linear', fill_value=global_median_scale
    )
    
    # Fallback to nearest-neighbor if linear triangulation fails (e.g., if all points are perfectly collinear)
    if np.any(np.isnan(dense_scale_map)):
        dense_scale_map = griddata(
            points, sparse_scales, (grid_x, grid_y), 
            method='nearest', fill_value=global_median_scale
        )
    
    # 4. Depth Fusion
    # Project the localized 2D scale map over the raw DA3 disparity to generate accurate dense metric depth
    metric_depth = dense_scale_map * disparity
    
    return metric_depth