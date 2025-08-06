import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO  
from segment_anything import sam_model_registry, SamPredictor  

class SceneUnderstanding:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        

        print("[INFO] Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8x.pt')  
        self.yolo_model.to(self.device)
        

        print("[INFO] Loading SAM model...")
        sam_checkpoint = "models/sam_vit_h_4b8939.pth"  
        sam_type = "vit_h" 
        
        self.sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.sam_predictor = SamPredictor(self.sam)
        

        print("[INFO] Loading MiDaS model...")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  
        self.midas.to(self.device)
        self.midas.eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        
    def detect_objects(self, image: np.ndarray):
        """Run YOLOv8 detection and return bounding boxes, class names, and confidences."""
        print("[INFO] Running YOLOv8 object detection...")
        results = self.yolo_model(image)[0]
        
        detections = []
        for *box, conf, cls in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "class_id": int(cls),
                "class_name": self.yolo_model.names[int(cls)]
            })
        return detections
    
    def segment_objects_with_boxes(self, image: np.ndarray, detections):
        """Run SAM segmentation on YOLO-detected objects using bounding boxes."""
        print("[INFO] Running SAM with YOLO bounding boxes...")
        self.sam_predictor.set_image(image)
        
        segmented_objects = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            input_box = np.array([x1, y1, x2, y2])
            
            masks, scores, logits = self.sam_predictor.predict(
                box=input_box[None, :],  
                multimask_output=False
            )
            
            segmented_objects.append({
                "class_name": det["class_name"],
                "bbox": det["bbox"],
                "mask": masks[0],
                "score": scores[0]
            })
        
        return segmented_objects

    
    def estimate_depth(self, image: np.ndarray):
        """Run MiDaS depth estimation and return depth map normalized to [0,255] uint8."""
        print("[INFO] Running MiDaS depth estimation...")
        input_batch = self.midas_transforms(image).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy()
        

        depth_min, depth_max = depth.min(), depth.max()
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
        depth_img = (depth_norm * 255).astype(np.uint8)
        return depth_img
    
    def save_debug_outputs(self, image: np.ndarray, detections, segmented_objects, depth_img, output_dir="debug_output"):
        os.makedirs(output_dir, exist_ok=True)


        cv2.imwrite(os.path.join(output_dir, "scene_original.jpg"), image)


        img_boxes = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            cv2.rectangle(img_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_boxes, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, "scene_bboxes.jpg"), img_boxes)


        img_masks = image.copy()
        for obj in segmented_objects:
            mask = obj["mask"]
            color_mask = np.zeros_like(image)
            color_mask[mask] = (0, 0, 255)  
            img_masks = cv2.addWeighted(img_masks, 1.0, color_mask, 0.5, 0)
        cv2.imwrite(os.path.join(output_dir, "scene_masks.jpg"), img_masks)


        cv2.imwrite(os.path.join(output_dir, "scene_depth.jpg"), depth_img)

        print(f"[INFO] Debug outputs saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scene Understanding with YOLOv8, SAM, and MiDaS")
    parser.add_argument("--image", required=True, help="Path to input scene image")
    parser.add_argument("--output_dir", default="debug_output", help="Directory to save debug outputs")
    args = parser.parse_args()
    

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Could not read image: {args.image}")
        exit(1)
        
    su = SceneUnderstanding()
    
    detections = su.detect_objects(img)
    masks = su.segment_objects_with_boxes(img, detections)

    depth_img = su.estimate_depth(img)
    
    su.save_debug_outputs(img, detections, masks, depth_img, output_dir=args.output_dir)
