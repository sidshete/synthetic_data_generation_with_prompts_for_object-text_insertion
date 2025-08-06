import cv2
import os
import numpy as np
import torch
import easyocr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from utils.image_utils import ensure_alpha_channel
from ultralytics import YOLO
import lpips
from rapidfuzz import fuzz
from scripts.prompt_parser import PromptParser  
import datetime
import shutil
from tabulate import tabulate
import pynvml

lpips_model = lpips.LPIPS(net='alex')

# CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# OCR and YOLO
ocr_reader = easyocr.Reader(['en'])
yolo_model = YOLO("yolov8x.pt")

def compute_lpips(obj_img_path, output_img_path, placement_coords):
    """
    Compute LPIPS perceptual similarity between the placed object and the corresponding
    region in the output image.

    Args:
        obj_img_path (str): Path to the original object image (PNG with alpha).
        output_img_path (str): Path to the composite output image.
        placement_coords (tuple): (x, y, w, h) region where object was placed in output image.

    Returns:
        lpips_score (float): LPIPS perceptual similarity score.
    """
    

    obj_img = Image.open(obj_img_path).convert('RGB')
    output_img = Image.open(output_img_path).convert('RGB')

    x, y, w, h = placement_coords

    placed_region = output_img.crop((x, y, x + w, y + h))

    obj_img = obj_img.resize((w, h))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    obj_tensor = transform(obj_img).unsqueeze(0)  
    placed_tensor = transform(placed_region).unsqueeze(0)

    with torch.no_grad():
        lpips_score = lpips_model(obj_tensor, placed_tensor).item()

    return lpips_score

def evaluate_text_object(prompt, output_image_path):
    """Evaluate and visualize OCR-based object/text placement."""
    img = cv2.imread(output_image_path)
    if img is None:
        raise ValueError(f"Could not load output image from {output_image_path}")

    results = ocr_reader.readtext(img)
    detected_texts = [r[1].lower() for r in results]
    print("[OCR] Detected text:", detected_texts)

    for bbox, text, _ in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(img, text, (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    debug_path = output_image_path.replace(".jpg", "_ocr_debug.jpg").replace(".png", "_ocr_debug.png")
    cv2.imwrite(debug_path, img)
    print(f"[Saved OCR Visualization] {debug_path}")

    prompt_text = prompt.lower()
    best_score = 0
    best_match = ""
    for dt in detected_texts:
        score = fuzz.ratio(prompt_text, dt)
        if score > best_score:
            best_score = score
            best_match = dt

    match = best_score >= 80
    print(f"[Fuzzy Match] Best match: '{best_match}' with score: {best_score}")


    image = Image.open(output_image_path).convert("RGB")
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        clip_score = outputs.logits_per_image[0][0].item()

    return {
        "type": "text",
        "ocr_match": match,
        "best_match": best_match,
        "match_score": best_score,
        "clip_score": clip_score
    }


def evaluate_image_object(prompt, object_image_path, output_image_path, object_class_name):
    """Evaluate and visualize object placement using YOLO + CLIP"""
    output_img = cv2.imread(output_image_path)
    obj_img = cv2.imread(object_image_path, cv2.IMREAD_UNCHANGED)

    obj_img = ensure_alpha_channel(obj_img)

    yolo_results = yolo_model(output_image_path)
    boxes = yolo_results[0].boxes
    class_ids = boxes.cls.cpu().numpy()
    coords = boxes.xyxy.cpu().numpy()

    detected_classes = [yolo_model.names[int(cls)] for cls in class_ids]
    print("[YOLO] Detected classes in output:", detected_classes)

    placement_coords = None


    for (cls_id, (x1, y1, x2, y2)) in zip(class_ids, coords):
        label = yolo_model.names[int(cls_id)]
        color = (0, 0, 255) if label.lower() == object_class_name.lower() else (0, 255, 255)
        cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(output_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if label.lower() == object_class_name.lower() and placement_coords is None:
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            placement_coords = (x, y, w, h)

    debug_path = output_image_path.replace(".jpg", "_yolo_debug.jpg").replace(".png", "_yolo_debug.png")
    cv2.imwrite(debug_path, output_img)
    print(f"[Saved YOLO Visualization] {debug_path}")

    detected = object_class_name.lower() in [cls.lower() for cls in detected_classes]

    image = Image.open(output_image_path).convert("RGB")
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        clip_score = outputs.logits_per_image[0][0].item()

    if placement_coords is None:
        placement_coords = (0, 0, obj_img.shape[1], obj_img.shape[0])

    lpips_score = compute_lpips(object_image_path, output_image_path, placement_coords)

    return {
        "type": "image",
        "object_detected": detected,
        "clip_score": clip_score,
        "lpips_score": lpips_score
    }



def evaluate_output(prompt, object_input, output_image_path, is_text=None):
    if is_text is None:
        if os.path.isfile(object_input) and object_input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
            is_text = False
        else:
            is_text = True

    if is_text:
        print("Detected: TEXT object")
        return evaluate_text_object(prompt, output_image_path)
    else:
        from scripts.prompt_parser import PromptParser 
        parser = PromptParser()
        parsed = parser.extract_relationship(prompt)
        class_name = parsed.get("placed_object") or parsed.get("target_object")

        if not class_name:
            raise ValueError("Could not determine the object to evaluate from the prompt.")

        print(f" Detected: IMAGE object to place ({class_name})")
        return evaluate_image_object(prompt, object_input, output_image_path, class_name)

def create_output_folder(base_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in base_name.strip().lower())
    folder_name = os.path.join("outputs", f"{safe_name}_{timestamp}")
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def save_evaluation_results(results, filepath, prompt=None, inference_time=None, gpu_util=None, gpu_mem=None):
    with open(filepath, "w") as f:
        if prompt:
            f.write(f"Prompt: {prompt}\n")
        if inference_time is not None:
            f.write(f"Inference Time (s): {inference_time:.3f}\n")
        if gpu_util and gpu_mem:
            f.write(f"GPU Utilization: {gpu_util}\n")
            f.write(f"GPU Memory Used: {gpu_mem}\n")
        f.write("\nEvaluation Metrics:\n")
        for key, val in results.items():
            f.write(f"{key}: {val}\n")


def copy_debug_images(composed_path, dest_folder):
    debug_files = [
        composed_path.replace(".jpg", "_ocr_debug.jpg").replace(".png", "_ocr_debug.png"),
        composed_path.replace(".jpg", "_yolo_debug.jpg").replace(".png", "_yolo_debug.png"),
    ]
    for debug_path in debug_files:
        if os.path.exists(debug_path):
            dest_path = os.path.join(dest_folder, os.path.basename(debug_path))
            if os.path.abspath(debug_path) != os.path.abspath(dest_path):
                shutil.copy(debug_path, dest_path)
            else:
                print(f" Skipping copy because source and destination are same: {debug_path}")



def get_gpu_usage():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_util = f"{util.gpu}%"
        gpu_mem = f"{mem.used // 1024**2} MiB"
        pynvml.nvmlShutdown()
        return gpu_util, gpu_mem
    except Exception:
        return "N/A", "N/A"

