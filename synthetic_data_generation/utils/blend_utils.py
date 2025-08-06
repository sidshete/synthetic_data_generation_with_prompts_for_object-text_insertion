import cv2
import numpy as np

def poisson_blend(scene_img, obj_img, top_left):
    """
    Blend obj_img (with alpha) onto scene_img at top_left coordinate using Poisson blending.

    Args:
        scene_img (np.ndarray): Background BGR image.
        obj_img (np.ndarray): Object BGRA image (with alpha).
        top_left (tuple): (x, y) coordinate on scene to place object top-left corner.

    Returns:
        blended_img (np.ndarray): Result image with object blended.
    """
    obj_rgb = obj_img[..., :3]
    alpha = obj_img[..., 3]

    # Create mask from alpha channel
    mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]

    h, w = obj_rgb.shape[:2]
    center = (top_left[0] + w // 2, top_left[1] + h // 2)

    # Make sure the region is within scene bounds
    H, W = scene_img.shape[:2]
    if top_left[0] < 0 or top_left[1] < 0 or top_left[0] + w > W or top_left[1] + h > H:
        raise ValueError("Object placement is out of scene bounds")

    # Clone object to the scene using seamlessClone
    blended = cv2.seamlessClone(obj_rgb, scene_img, mask, center, cv2.NORMAL_CLONE)

    return blended

def enhance_object(obj_img, alpha_scale=1.2, brightness_shift=30):
    """
    Enhance object image to make it more visible.

    Args:
        obj_img (np.ndarray): BGRA object image.
        alpha_scale (float): multiplier to alpha channel to increase opacity.
        brightness_shift (int): value to add to brightness (0-255 scale).

    Returns:
        np.ndarray: enhanced BGRA image.
    """
    obj = obj_img.copy()

    # Scale alpha channel (clamp max 255)
    obj[..., 3] = np.clip(obj[..., 3].astype(np.float32) * alpha_scale, 0, 255).astype(np.uint8)

    # Convert to HSV to increase brightness and saturation
    obj_rgb = obj[..., :3]
    hsv = cv2.cvtColor(obj_rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] + brightness_shift, 0, 255)  # increase value/brightness
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.1, 0, 255)               # increase saturation slightly

    enhanced_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    obj[..., :3] = enhanced_rgb

    return obj

def add_outline(obj_img, thickness=3, color=(255, 255, 255)):
    """
    Add an outline around the object using alpha mask.

    Args:
        obj_img (np.ndarray): BGRA object image.
        thickness (int): thickness of outline in pixels.
        color (tuple): BGR color of the outline.

    Returns:
        np.ndarray: object with outline.
    """
    alpha = obj_img[..., 3]
    # Find contours of the alpha mask
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask for the outline
    outline_mask = np.zeros_like(alpha)

    # Draw contours with thickness
    cv2.drawContours(outline_mask, contours, -1, 255, thickness)

    # Create color outline image
    outline_img = np.zeros_like(obj_img)
    outline_img[..., :3] = color
    outline_img[..., 3] = outline_mask

    # Composite outline behind the original object
    inv_alpha = cv2.bitwise_not(obj_img[..., 3])
    for c in range(3):
        outline_img[..., c] = cv2.bitwise_and(outline_img[..., c], inv_alpha)

    # Combine outline and original object
    combined = outline_img.copy()
    combined[..., :3] = cv2.add(combined[..., :3], obj_img[..., :3])
    combined[..., 3] = cv2.max(outline_img[..., 3], obj_img[..., 3])

    return combined
