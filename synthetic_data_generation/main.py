import warnings
warnings.filterwarnings("ignore")
import cv2
import os
import argparse
import sys
import time
import shutil
from scripts.scene_detection import SceneUnderstanding
from scripts.prompt_parser import PromptParser
from utils.placement_utils import *
from utils.image_utils import *
from utils.text_utils import create_text_image, rotate_text_image, is_text_image, overlay_image_alpha
from utils.blend_utils import *
from evaluation.evaluate_output import evaluate_output, get_gpu_usage, create_output_folder, save_evaluation_results, copy_debug_images
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main(scene_path, object_input, prompt, output_path="output.jpg", evaluate=False):
    scene_img = cv2.imread(scene_path)
    if scene_img is None:
        raise ValueError(f"Could not load scene image from {scene_path}")


    scene_model = SceneUnderstanding()
    parser = PromptParser()

    start_time = time.time()
    detections = scene_model.detect_objects(scene_img)

    parsed = parser.extract_relationship(prompt)
    relation = parsed.get("spatial_relation")
    ref_obj_name = parsed.get("reference_object")
    orientation = parsed.get("orientation", 0)

    size_modifier = parsed.get("size_modifier", None)
    scale_map = {
        "tiny": 0.3,
        "small": 0.5,
        "medium": 0.8,
        "large": 1.2,
        "huge": 1.5,
        "gigantic": 2.0
    }
    size_factor = scale_map.get(size_modifier, 1.0)


    is_text = not (os.path.isfile(object_input) and object_input.lower().endswith((".png", ".jpg", ".jpeg")))


    ref_bbox = find_reference_bbox(detections, ref_obj_name)
    if ref_bbox is None:
        raise ValueError(f"Reference object '{ref_obj_name}' not found in scene.")


    if is_text:
        print(f"[INFO] Generating text object: '{object_input}'")
        ref_width = ref_bbox[2] - ref_bbox[0]
        estimated_font_size = max(20, int(ref_width * 0.2))
        font_size = parsed.get("font_size", estimated_font_size)
        font_color = parsed.get("font_color", (0, 0, 0))
        bold = parsed.get("bold", False)
        orientation = parsed.get("orientation", "horizontal")

        obj_img = create_text_image(object_input, font_size=font_size, font_color=font_color, bold=bold)
        obj_img = rotate_text_image(obj_img, orientation)
        obj_img = ensure_alpha_channel(obj_img)
        resized_obj = obj_img
    else:
        obj_img = cv2.imread(object_input, cv2.IMREAD_UNCHANGED)
        if obj_img is None:
            raise ValueError(f"Could not load object image from {object_input}")
        if obj_img.shape[2] == 3:
            print("[WARN] Object image has no alpha channel. Converting white background to transparency.")
            obj_img = convert_white_to_transparency(obj_img)
        obj_img = ensure_alpha_channel(obj_img)
        target_size = recommend_scale(ref_bbox, obj_img.shape[:2][::-1], scale_factor=size_factor)
        resized_obj = resize_object(obj_img, target_size)


    x, y, angle = compute_placement_position(
    ref_bbox,
    resized_obj.shape[:2][::-1],
    relation,
    scene_img.shape,
    orientation=orientation
)
    rotated_obj = rotate_object(resized_obj, angle)
    enhanced_obj = enhance_object(rotated_obj)
    outlined_obj = add_outline(enhanced_obj, thickness=3, color=(0, 0, 0))


    if is_text or is_text_image(outlined_obj):
        composed = overlay_image_alpha(scene_img, outlined_obj, x, y)
    else:
        composed = poisson_blend(scene_img, outlined_obj, (x, y))

    if composed is None:
        raise RuntimeError("Composed image is invalid.")

    end_time = time.time()
    inference_time = end_time - start_time


    gpu_util, gpu_mem = get_gpu_usage()


    output_folder = create_output_folder(object_input)
    composed_path = os.path.join(output_folder, "composed.jpg")

    cv2.imwrite(composed_path, composed)
    print(f" Output image saved to {composed_path}")

    shutil.copy(scene_path, os.path.join(output_folder, "scene.jpg"))
    if is_text:
        text_img_path = os.path.join(output_folder, "generated_text.png")
        cv2.imwrite(text_img_path, outlined_obj)
    else:
        shutil.copy(object_input, os.path.join(output_folder, "object.png"))


    if evaluate:
        print(" Running evaluation...")
        results = evaluate_output(prompt, object_input, composed_path, is_text=is_text)

        save_evaluation_results(results,
                                os.path.join(output_folder, "results.txt"),
                                prompt=prompt,
                                inference_time=inference_time,
                                gpu_util=gpu_util,
                                gpu_mem=gpu_mem)


        table_data = [
            ["Prompt", prompt],
            ["Inference Time (s)", f"{inference_time:.3f}"],
            ["GPU Utilization", gpu_util],
            ["GPU Memory Used", gpu_mem],
        ]
        for k, v in results.items():
            table_data.append([k, v])

        print("\n Evaluation Summary:")
        print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))

        copy_debug_images(composed_path, output_folder)
        print(f"-------- All outputs saved in: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Place an object or text in a scene based on prompt")
    parser.add_argument("--scene", required=True, help="Path to scene image")
    parser.add_argument("--object", required=True, help="Path to object image or text string")
    parser.add_argument("--prompt", required=True, help="Natural language prompt describing placement")
    parser.add_argument("--output", default="output.jpg", help="Path to save composed output image")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after placement")
    args = parser.parse_args()

    main(args.scene, args.object, args.prompt, args.output, args.evaluate)
