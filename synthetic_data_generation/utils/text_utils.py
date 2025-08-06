import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def get_default_font_path(bold=False):
    import matplotlib
    base_path = os.path.join(matplotlib.get_data_path(), "fonts/ttf")
    if bold:
        bold_path = os.path.join(base_path, "DejaVuSans-Bold.ttf")
        if os.path.isfile(bold_path):
            return bold_path
    # fallback normal font
    return os.path.join(base_path, "DejaVuSans.ttf")

def create_text_image(text, font_path=None, font_size=None, font_color=(0, 0, 0), bold=False, vertical=False, padding=20):
    if font_path is None:
        font_path = get_default_font_path(bold)
    
    # Fallback for missing or invalid font_size
    if font_size is None or not isinstance(font_size, int) or font_size <= 0:
        print("[WARN] Invalid or missing font_size. Defaulting to 80.")
        font_size = 80

    font = ImageFont.truetype(font_path, font_size)

    # Create dummy image to compute text bounding box
    dummy_img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)  # (left, top, right, bottom)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Create final transparent image
    img_w = text_width + 2 * padding
    img_h = text_height + 2 * padding
    img = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # font_color is expected in BGR, convert to RGBA
    color_rgba = (font_color[2], font_color[1], font_color[0], 255)
    draw.text((padding, padding), text, font=font, fill=color_rgba)

    if vertical:
        img = img.rotate(90, expand=True)

    # Convert PIL image to OpenCV BGRA format
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


import pytesseract

def is_text_image(image):
    """
    Use OCR to determine if image contains readable text.
    Return True if any text detected, else False.
    """
    # Convert image to grayscale (OCR generally performs better)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to enhance text visibility for OCR
    thresh = cv2.adaptiveThreshold(gray, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Run OCR on thresholded image
    text = pytesseract.image_to_string(thresh).strip()

    return bool(text)  # True if any text detected, else False

def overlay_image_alpha(img, overlay, x, y):
    """
    Overlay `overlay` image onto `img` at position (x, y) considering alpha channel.
    Handles boundary conditions to avoid errors if overlay goes outside base image.
    """
    h, w = overlay.shape[:2]

    # Check for boundaries
    if x < 0:
        overlay = overlay[:, -x:]
        w += x
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h += y
        y = 0
    if x + w > img.shape[1]:
        overlay = overlay[:, :img.shape[1] - x]
        w = img.shape[1] - x
    if y + h > img.shape[0]:
        overlay = overlay[:img.shape[0] - y, :]
        h = img.shape[0] - y

    if w <= 0 or h <= 0:
        return img  # Nothing to overlay

    roi = img[y:y+h, x:x+w]

    overlay_color = overlay[..., :3]
    alpha_mask = overlay[..., 3:] / 255.0

    # Blend overlay with roi
    blended = (1.0 - alpha_mask) * roi + alpha_mask * overlay_color
    img[y:y+h, x:x+w] = blended.astype(img.dtype)

    return img

def rotate_text_image(img, orientation="horizontal"):
    """
    Rotate text image based on orientation. Supports 'horizontal' and 'vertical'.
    """
    if orientation.lower() == "vertical":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img  # No rotation for horizontal (default)
