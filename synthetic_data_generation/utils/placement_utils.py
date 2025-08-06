import numpy as np

SPATIAL_OFFSETS = {
    # Horizontal relations
    "left of": (-1, 0),
    "right of": (1, 0),
    "to the left of": (-1, 0),
    "to the right of": (1, 0),
    "beside": (1, 0),
    "next to": (1, 0),
    "near": (1, 0),
    "close to": (1, 0),
    "adjacent to": (1, 0),
    "across from": (1, 0),

    # Vertical relations
    "above": (0, -1),
    "below": (0, 1),
    "over": (0, -1),
    "under": (0, 1),
    "on top of": (0, -1),
    "on": (0, -1),
    "underneath": (0, 1),
    "inside": (0, 0),  # inside means within, so no offset

    # Depth / front-back relations
    "in front of": (0, -1),
    "behind": (0, 1),
    "in back of": (0, 1),

    # Diagonal / combined directions
    "upper left of": (-1, -1),
    "upper right of": (1, -1),
    "lower left of": (-1, 1),
    "lower right of": (1, 1),
    "top left": (-1, -1),
    "top right": (1, -1),
    "bottom left": (-1, 1),
    "bottom right": (1, 1),

    # Group or multi-object relations
    "between": (0, 0),  # needs special handling (between two reference objects)
    "among": (0, 0),    # similar to between, no single offset

    # Others
    "around": (0, 0),
    "close by": (1, 0),
    "facing": (0, -1),
    "nearby": (1, 0),
    "beside to": (1, 0),
}


def iou(boxA, boxB):
    """Compute Intersection over Union between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if boxAArea + boxBArea - interArea == 0:
        return 0
    
    return interArea / float(boxAArea + boxBArea - interArea)

def check_collision(new_bbox, existing_bboxes, iou_threshold=0.1):
    """
    Check if new_bbox collides with any bounding boxes in existing_bboxes.
    """
    for bbox in existing_bboxes:
        if iou(new_bbox, bbox) > iou_threshold:
            return True
    return False

def clamp_bbox(bbox, img_width, img_height):
    """
    Clamp bounding box coordinates to image boundaries.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    return (x1, y1, x2, y2)

def find_reference_bbox(detections, reference_class_name):
    """
    Find the first detection matching the reference class name.
    """
    for det in detections:
        if det["class_name"].lower() == reference_class_name.lower():
            return det["bbox"]
    return None

def recommend_scale(reference_bbox, base_object_size, scale_factor=1.5):
    """
    Scale the object to match the reference object's approximate size.
    """
    ref_w = reference_bbox[2] - reference_bbox[0]
    ref_h = reference_bbox[3] - reference_bbox[1]

    target_w = int(ref_w * scale_factor)
    target_h = int(ref_h * scale_factor)

    # Maintain aspect ratio based on base_object_size
    aspect_ratio = base_object_size[0] / base_object_size[1]
    if target_w / target_h > aspect_ratio:
        target_w = int(target_h * aspect_ratio)
    else:
        target_h = int(target_w / aspect_ratio)

    return target_w, target_h

def recommend_rotation(spatial_relation, orientation="horizontal"):
    rotations = {
        "on top of": 0,
        "under": 0,
        "left of": 0,
        "right of": 0,
        "in front of": 0,
        "behind": 0,
        "tilted": 15,
        "lying": 90,
        "upside down": 180,
        "angled": 30,
        "vertical": 90,
    }

    ORIENTATION_DEGREES = {
        "horizontal": 0,
        "vertical": 90,
        "rotated_90": 90,
        "rotated_180": 180,
        "rotated_270": 270,
        "angled": -30,
        "tilted": -15,
    }


    # Use orientation degree if available; otherwise fall back to spatial_relation-based rotation
    return ORIENTATION_DEGREES.get(orientation, rotations.get(spatial_relation.lower(), 0))

OVERLAP_ALLOWED_RELATIONS = {"on", "on top of", "under", "above", "below"}

def compute_placement_position(reference_bbox, object_size, spatial_relation, image_shape, existing_bboxes=None, max_attempts=10, orientation="horizontal"):

    ref_x1, ref_y1, ref_x2, ref_y2 = reference_bbox
    ref_cx = (ref_x1 + ref_x2) // 2
    ref_cy = (ref_y1 + ref_y2) // 2

    obj_w, obj_h = object_size
    H, W = image_shape[:2]

    rotation = recommend_rotation(spatial_relation, orientation)


    # Relations where object should be placed exactly centered on reference bbox
    centered_relations = {"on", "center", "center on", "on top of"}

    if spatial_relation.lower() in centered_relations:
        # Center object on reference bbox
        new_x = int(ref_cx - obj_w / 2)
        new_y = int(ref_cy - obj_h / 2)

        # Clamp to image boundaries
        new_x = max(0, min(W - obj_w, new_x))
        new_y = max(0, min(H - obj_h, new_y))

        # If collision checking requested, verify placement
        if existing_bboxes:
            overlap_allowed = spatial_relation in OVERLAP_ALLOWED_RELATIONS
            other_bboxes = [bbox for bbox in existing_bboxes if bbox != reference_bbox] if overlap_allowed else existing_bboxes
            new_bbox = (new_x, new_y, new_x + obj_w, new_y + obj_h)
            new_bbox = clamp_bbox(new_bbox, W, H)

            if not check_collision(new_bbox, other_bboxes):
                return new_x, new_y, rotation
            else:
                # If collision detected, try small offsets (optional)
                for attempt in range(max_attempts):
                    offset_x = np.random.randint(-15, 16)
                    offset_y = np.random.randint(-15, 16)
                    trial_x = max(0, min(W - obj_w, new_x + offset_x))
                    trial_y = max(0, min(H - obj_h, new_y + offset_y))
                    trial_bbox = (trial_x, trial_y, trial_x + obj_w, trial_y + obj_h)
                    trial_bbox = clamp_bbox(trial_bbox, W, H)
                    if not check_collision(trial_bbox, other_bboxes):
                        return trial_x, trial_y, rotation
                # No collision-free placement found centered
                return None
        else:
            # No collision check, just return centered placement
            return new_x, new_y, rotation

    # --- Fallback to your original offset placement logic ---

    dx, dy = SPATIAL_OFFSETS.get(spatial_relation, (1, 0))

    ref_w = ref_x2 - ref_x1
    ref_h = ref_y2 - ref_y1
    spacing = max(ref_w, ref_h) // 2 + max(obj_w, obj_h) // 2 + 10  # extra 10px gap

    base_cx = ref_cx + int(dx * spacing)
    base_cy = ref_cy + int(dy * spacing)

    overlap_allowed = spatial_relation in OVERLAP_ALLOWED_RELATIONS

    for attempt in range(max_attempts):
        offset_x = np.random.randint(-15, 16) if existing_bboxes else 0
        offset_y = np.random.randint(-15, 16) if existing_bboxes else 0

        new_cx = base_cx + offset_x
        new_cy = base_cy + offset_y

        new_x = int(new_cx - obj_w / 2)
        new_y = int(new_cy - obj_h / 2)

        new_x = max(0, min(W - obj_w, new_x))
        new_y = max(0, min(H - obj_h, new_y))

        new_bbox = (new_x, new_y, new_x + obj_w, new_y + obj_h)
        new_bbox = clamp_bbox(new_bbox, W, H)

        if existing_bboxes:
            if overlap_allowed:
                other_bboxes = [bbox for bbox in existing_bboxes if bbox != reference_bbox]
                if not check_collision(new_bbox, other_bboxes):
                    return new_x, new_y, rotation
            else:
                if not check_collision(new_bbox, existing_bboxes):
                    return new_x, new_y, rotation
        else:
            return new_x, new_y, rotation

    return None
