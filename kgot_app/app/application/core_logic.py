import os
import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from .utils import ensure_rgba, divide_mask_by_centerline, generate_reference_curve
from .image_processing import warp_straightened_image, straighten_image


def load_coco_data(json_file_path):
    """Load and parse the COCO JSON file."""
    with open(json_file_path, 'r') as f:
        return json.load(f)


def extract_reference_keypoints(coco_data, ref_image_id):
    """Extract reference keypoints from the annotations."""
    ref_ann = next((ann for ann in coco_data['annotations'] if ann['image_id'] == ref_image_id), None)
    if ref_ann is None:
        return None
    return np.array(ref_ann['keypoints']).reshape(-1, 3)[:, :2]


def process_image(image_path, ann, reference_keypoints, output_width, output_height, scale):
    """Process an individual image: straighten, warp, and generate edge-detected frame."""
    if not os.path.exists(image_path):
        return None, None

    try:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None, None

        # Convert image to RGBA format
        image = ensure_rgba(image)

        # Extract keypoints and mask
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)[:, :2]
        rle = ann['segmentation']
        mask = mask_utils.decode(rle)

        # Calculate average width and total length
        _, widths = divide_mask_by_centerline(mask, keypoints)
        avg_width = int(np.mean(widths))
        total_length = np.sum(np.sqrt(np.sum(np.diff(keypoints, axis=0) ** 2, axis=1)))

        # Straighten the image
        straightened = straighten_image(image, keypoints, mask, total_length, avg_width)

        # Generate reference curve
        if scale:
            reference_curve, scale_factor = generate_reference_curve(
                reference_keypoints, num_points=10000, target_length=straightened.shape[1]
            )
        else:
            reference_curve, scale_factor = generate_reference_curve(reference_keypoints, num_points=10000)

        # Warp the straightened image
        warped, warped_filled, edge = warp_straightened_image(straightened, reference_curve, avg_width, scale_factor)

        # Resize if necessary to fit within the output dimensions
        if warped_filled.shape[0] > output_height or warped_filled.shape[1] > output_width:
            scale = min(output_height / warped_filled.shape[0], output_width / warped_filled.shape[1])
            new_size = (int(warped_filled.shape[1] * scale), int(warped_filled.shape[0] * scale))
            warped_filled = cv2.resize(warped_filled, new_size, interpolation=cv2.INTER_AREA)
            edge = cv2.resize(edge, new_size, interpolation=cv2.INTER_NEAREST)

        # Create the output canvas
        output = np.zeros((output_height, output_width, 4), dtype=np.uint8)

        # Calculate offsets
        y_offset = max(0, (output_height - warped_filled.shape[0]) // 2)
        x_offset = max(0, (output_width - warped_filled.shape[1]) // 2)

        # Assign warped_filled to output
        output[y_offset:y_offset + warped_filled.shape[0], x_offset:x_offset + warped_filled.shape[1]] = warped_filled

        return output, edge

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None


def save_output(output_folder, image_filename, ref_image_id, output, edge, frame_images):
    """Save the output image and edge-detected frame."""
    os.makedirs(output_folder, exist_ok=True)

    # Save the output image
    output_filename = f"{os.path.splitext(image_filename)[0]}_warped_ref_image{ref_image_id}_.png"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, output, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def circulate_image_from_coco(
    json_file_path, image_folder, output_folder, output_width=1024, output_height=1024, scale=True
):
    """
    Main function to orchestrate the processing of images using all keypoint references in the JSON, one by one.
    Avoids processing an image using keypoints extracted from itself.

    Args:
        json_file_path (str): Path to the COCO JSON file.
        image_folder (str): Path to the folder containing the images.
        output_folder (str): Path to the folder where outputs will be saved.
        output_width (int): Width of the output image.
        output_height (int): Height of the output image.
        scale (bool): Whether to scale the reference curve to the straightened image width.

    Returns:
        list: A list of edge-detected images for each processed frame.
    """
    # Load the COCO JSON file
    coco_data = load_coco_data(json_file_path)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in the folder
    image_filenames = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

    # Extract all annotations as keypoint references
    annotations = coco_data['annotations']
    if not annotations:
        raise ValueError("No annotations found in the JSON file.")

    # Initialize a list to store edge-detected images for each frame
    frame_images = []

    # Iterate over all image filenames in the folder
    for image_filename in image_filenames:
        image_path = os.path.join(image_folder, image_filename)

        # Find the corresponding image information in the COCO JSON
        image_info = next((img for img in coco_data['images'] if img['file_name'] == image_filename), None)
        if not image_info:
            print(f"No matching COCO image entry for file: {image_filename}")
            continue

        image_id = image_info['id']
        ann = next((a for a in coco_data['annotations'] if a['image_id'] == image_id), None)
        if not ann:
            print(f"No annotation found for image: {image_filename}")
            continue

        # Iterate through all annotations as reference keypoints
        for reference_annotation in annotations:
            # Extract reference keypoints from the current reference annotation
            reference_keypoints = np.array(reference_annotation['keypoints']).reshape(-1, 3)[:, :2]
            reference_image_id = reference_annotation['image_id']

            # Get the filename of the reference image
            reference_image_info = next((img for img in coco_data['images'] if img['id'] == reference_image_id), None)
            reference_image_filename = reference_image_info['file_name'] if reference_image_info else "Unknown"

            # **Constraint: Skip processing if the input image matches the reference keypoint's image**
            if image_filename == reference_image_filename:
                print(f"Skipping processing {image_filename} using its own keypoints.")
                continue

            # Generate the global reference curve for the current annotation
            reference_curve, scale_factor = generate_reference_curve(reference_keypoints, num_points=10000)

            # Process the image using the current reference keypoints
            output, edge = process_image(
                image_path,
                ann,
                reference_keypoints=reference_keypoints,  # Use the current reference keypoints
                output_width=output_width,
                output_height=output_height,
                scale=scale,
            )

            if output is None or edge is None:
                print(f"Skipping image due to processing failure: {image_filename}")
                continue

            # Save the processed output
            save_output(output_folder, image_filename, reference_image_id, output, edge, frame_images)

            print(f"Processed {image_filename} using reference keypoints from: {reference_image_filename}")

    return frame_images