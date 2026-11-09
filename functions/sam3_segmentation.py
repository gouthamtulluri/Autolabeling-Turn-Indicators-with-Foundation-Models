import torch
from PIL import Image


def generate_sam3_masks(image_path, text_prompt, processor, model, device, exclude_prompt=None, conf_threshold=0.15, mask_threshold=0.5):
    """
    Generates semantic masks using SAM3 and returns them strictly as PyTorch GPU Tensors.
    
    Args:
        image_path (str or PIL.Image): Path to the image or the loaded PIL Image.
        text_prompt (str): The primary text prompt for the target object (e.g., 'car taillight').
        processor: The SAM3 processor instance.
        model: The SAM3 model instance.
        device: The PyTorch device (e.g., 'cuda' or 'cpu').
        exclude_prompt (str, optional): A text prompt for regions to actively subtract/ignore.
        conf_threshold (float): Confidence threshold for instance detection.
        mask_threshold (float): Threshold for binarizing the segmentation masks.
        
    Returns:
        tuple: (target_masks, target_scores) both as raw GPU PyTorch Tensors.
    """
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path.convert("RGB")

    orig_w, orig_h = image.size

    prompts = [text_prompt]
    if exclude_prompt is not None:
        prompts.append(exclude_prompt)

    inputs = processor(images=image, text=prompts, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=torch.float16 if v.dtype == torch.float32 else None) for k, v in inputs.items()}

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    with torch.inference_mode(), torch.amp.autocast(device_type=device_type, dtype=torch.float16):
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=conf_threshold,
        mask_threshold=mask_threshold,
        target_sizes=[(orig_h, orig_w)] * len(prompts)
    )

    # Keep masks and scores on the GPU to minimize host transfer overhead
    target_masks = results[0]["masks"]
    target_scores = results[0]["scores"]

    if len(target_masks) == 0:
        return target_masks, target_scores

    # Filter out regions intersecting with the exclusion prompt
    if exclude_prompt is not None and len(results) > 1:
        exclude_masks = results[1]["masks"]
        if len(exclude_masks) > 0:
            forbidden_zone = torch.any(exclude_masks, dim=0)
            target_masks = target_masks & ~forbidden_zone
            valid_mask = target_masks.sum(dim=(1, 2)) > 10
            target_masks = target_masks[valid_mask]
            target_scores = target_scores[valid_mask]

    return target_masks, target_scores