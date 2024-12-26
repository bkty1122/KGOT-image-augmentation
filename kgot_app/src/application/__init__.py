"""
Application package for the KGOT image processing pipeline.
"""

# Import specific functions or classes from utils.py
from .utils import (
    interpolate_centerline,
    calculate_tangents_and_normals,
    prepare_centerline,
    divide_mask_by_centerline,
    calculate_average_width,
    ensure_rgba,
    generate_reference_curve,
    load_and_preprocess_image,
    save_straightened_image,
)

# Import specific functions from image_processing.py
from .image_processing import (
    straighten_image,
    warp_straightened_image,
)

# Import specific functions from core_logic.py
from .core_logic import (
    load_coco_data,
    extract_reference_keypoints,
    process_image,
    save_output,
    circulate_image_from_coco,
)