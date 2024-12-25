import os
import json
import numpy as np
import cv2
from .utils import ensure_rgba, divide_mask_by_centerline
from .warp_straighten import warp_straightened_image
from .straighten import straighten_image
from .utils import generate_reference_curve
from pycocotools import mask as mask_utils


def circulate_image_from_coco(
    json_file_path, image_folder, output_folder, output_width=1024, output_height=1024, scale=True
):
    """
    Processes images and their annotations from a COCO JSON file to straighten and warp images.

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
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)

    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    image_ids = set(ann['image_id'] for ann in coco_data['annotations'])
    frame_images = []  # List to store edge-detected images for each frame

    for ref_image_id in image_ids:
        # Get the reference annotation
        ref_ann = next((ann for ann in coco_data['annotations'] if ann['image_id'] == ref_image_id), None)
        if ref_ann is None:
            continue  # Skip if no annotation found

        # Extract reference keypoints
        reference_keypoints = np.array(ref_ann['keypoints']).reshape(-1, 3)[:, :2]

        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)
            if image_info is None:
                continue  # Skip if no image info found

            image_filename = image_info['file_name']
            image_path = os.path.join(image_folder, image_filename)

            if not os.path.exists(image_path):
                continue  # Skip if image file is not found

            try:
                # Load the image
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue  # Skip if the image cannot be read

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

                # Save the output image
                output_filename = f"{os.path.splitext(image_filename)[0]}_warped_ref_image{ref_image_id}_.png"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, output, [cv2.IMWRITE_PNG_COMPRESSION, 9])

                # Save edge-detected frame
                frame = np.zeros((output_height, output_width), dtype=np.uint8)
                frame[y_offset:y_offset + warped_filled.shape[0], x_offset:x_offset + warped_filled.shape[1]] = edge
                frame_images.append(frame)

            except Exception:
                # Suppress error message and continue processing the next image
                pass

    return frame_images
# test program

json_file_path = r'D:\KGOT_github\KGOT-image-augmentation\kgot_app\example\example_annotation.json'
image_folder = r'D:\KGOT_github\KGOT-image-augmentation\kgot_app\example'
output_folder = r'D:\KGOT_github\KGOT-image-augmentation\kgot_app\example\output'

if __name__ == '__main__':
    frame_images = circulate_image_from_coco(json_file_path, image_folder, output_folder, output_width=1024, output_height=1024, scale=True)
    for i, frame in enumerate(frame_images):
        cv2.imwrite(os.path.join(output_folder, f"edge_{i}.png"), frame)