import cv2
import numpy as np

def ensure_alpha_channel(image):
    """
    Ensure image has 4 channels (RGBA). If not, add fully opaque alpha.
    """
    if image.shape[2] == 4:
        return image
    elif image.shape[2] == 3:
        b, g, r = cv2.split(image)
        alpha = np.ones_like(b) * 255  # Fully opaque
        return cv2.merge((b, g, r, alpha))
    else:
        raise ValueError("Unsupported image format")

def resize_object(image, target_size):
    """Resize object image (with transparency) to target size (width, height)."""
    image = ensure_alpha_channel(image)
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def rotate_object(image, angle):
    """Rotate object image (with alpha) around center by angle degrees."""
    image = ensure_alpha_channel(image)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

def overlay_transparent(background, overlay, x, y):
    """
    Overlay transparent image (with alpha channel) at (x, y) on background.
    Assumes background is a standard BGR image (3 channels).
    """
    overlay = ensure_alpha_channel(overlay)
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    # Clip if overlay goes out of bounds
    # Clip if overlay goes out of bounds
    if x >= bw or y >= bh:
        return background
    if x + w > bw:
        overlay = overlay[:, :bw - x]
    if y + h > bh:
        overlay = overlay[:bh - y, :]

    h, w = overlay.shape[:2] 

    # Alpha blending
    alpha_mask = overlay[:, :, 3] / 255.0
    for c in range(3):  # BGR channels
        background[y:y+h, x:x+w, c] = (
            (1 - alpha_mask) * background[y:y+h, x:x+w, c] + alpha_mask * overlay[:, :, c]
        )

    return background

def convert_white_to_transparency(img_bgr):
    """
    Convert white background in a 3-channel image to transparent (RGBA)
    """
    # Convert to 4 channels
    bgr = img_bgr.astype(np.uint8)
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    
    # White threshold mask
    white = np.all(bgr > 240, axis=2)
    rgba[white, 3] = 0  # Set alpha to 0 where white

    return rgba

def segment_single_object(object_image: np.ndarray, sam_predictor):
    """
    Segments the main object from an object image using SAM.

    Args:
        object_image (np.ndarray): Input image (may or may not have alpha).
        sam_predictor (SamPredictor): Initialized SAM predictor.

    Returns:
        mask (np.ndarray): Binary mask of the object.
        rgba (np.ndarray): Object image with alpha based on segmentation.
    """
    print("[INFO] Segmenting object image with SAM...")

    h, w = object_image.shape[:2]

    # If it has alpha, convert to RGB over white
    if object_image.shape[2] == 4:
        print("[INFO] Image has alpha. Converting to RGB over white background for segmentation.")
        rgb = cv2.cvtColor(object_image, cv2.COLOR_BGRA2BGR)
        alpha = object_image[:, :, 3] / 255.0
        white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
        composite = (rgb * alpha[..., None] + white_bg * (1 - alpha[..., None])).astype(np.uint8)
    else:
        print("[INFO] No alpha. Using as-is for segmentation.")
        composite = object_image.copy()

    # Set image in SAM
    sam_predictor.set_image(composite)

    # Full-image bounding box
    input_box = np.array([[0, 0, w, h]])

    masks, scores, logits = sam_predictor.predict(
        box=input_box,
        multimask_output=False
    )

    best_mask = masks[0]

    # Apply mask to original RGB (or BGRA converted to RGB)
    if object_image.shape[2] == 4:
        rgba = object_image.copy()
        rgba[:, :, 3] = (best_mask * 255).astype(np.uint8)
    else:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = object_image[:, :, :3]
        rgba[:, :, 3] = (best_mask * 255).astype(np.uint8)

    return best_mask, rgba

#example usage
if __name__ == "__main__":
    object_img = cv2.imread("/home/dfki.uni-bremen.de/sshete/openai/smart_object_placement/data/objects/cap.jpg", cv2.IMREAD_UNCHANGED)
    object_img = ensure_alpha_channel(object_img)
    print("Object image shape:", object_img.shape)