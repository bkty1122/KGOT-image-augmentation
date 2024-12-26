import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import cv2
from .utils import interpolate_centerline, calculate_tangents_and_normals, ensure_rgba

def straighten_image(image, keypoints, mask, output_length, output_width):
    """
    Straightens an image using a given set of keypoints and outputs a rectangular, straightened image.

    Args:
        image (ndarray): The input image to straighten.
        keypoints (ndarray): Keypoints (x, y) that define the centerline of the image.
        mask (ndarray): Binary mask indicating valid regions of the image.
        output_length (int): Desired length of the straightened output image.
        output_width (int): Desired width of the straightened output image.

    Returns:
        ndarray: The straightened image in RGBA format.
    """
    # Image dimensions
    h, w = image.shape[:2]
    
    # Calculate proportional height for the straightened image
    output_height = int(output_width * (h / w))
    
    # Initialize the straightened image (RGBA: 4 channels)
    straightened = np.zeros((output_height, int(output_length), 4), dtype=np.uint8)
    
    # Ensure the input image is in RGBA format
    image = ensure_rgba(image)
    
    # Interpolate the centerline from the given keypoints
    centerline = interpolate_centerline(keypoints, int(output_length))
    
    # Calculate tangents and normals for the centerline
    _, normals = calculate_tangents_and_normals(centerline)

    # Iterate along the centerline
    for i in range(int(output_length)):
        # Current centerline coordinates
        cx, cy = centerline[i]
        
        # Corresponding normal vector
        nx, ny = normals[i]

        # Iterate along the height of the straightened image
        for j in range(output_height):
            offset = (j - output_height / 2)  # Offset from the centerline
            sx, sy = int(cx + offset * nx), int(cy + offset * ny)  # Source coordinates

            # Check if the source coordinates are valid
            if 0 <= sx < w and 0 <= sy < h and mask[sy, sx]:
                # Copy pixel from the source to the straightened image
                straightened[j, i] = image[sy, sx]
            else:
                # Fill invalid pixels with transparent black
                straightened[j, i] = [0, 0, 0, 0]

    return straightened


def warp_straightened_image(straightened, reference_curve, avg_width, scale_factor=None):
    """
    Warps a straightened image back into a curved shape defined by the reference curve.

    Args:
        straightened (ndarray): The straightened image to warp.
        reference_curve (ndarray): The curve to warp the straightened image onto.
        avg_width (int): Average width of the image along the curve.
        scale_factor (float, optional): A scaling factor for the warping operation.

    Returns:
        tuple: The warped image, the filled warped image, and the mask.
    """
    # Dimensions of the straightened image
    height, width = straightened.shape[:2]
    
    # Extend the reference curve for smoother warping
    extended_curve = interpolate_centerline(reference_curve, int(len(reference_curve) * 1.2))
    
    # Get the bounds of the extended curve
    min_x, min_y = np.min(extended_curve, axis=0)
    max_x, max_y = np.max(extended_curve, axis=0)
    
    # Define padding for the warped image
    padding = avg_width
    
    # Calculate dimensions for the warped output
    output_width = int(max_x - min_x + 2 * padding)
    output_height = int(max_y - min_y + 2 * padding)
    
    # Initialize the warped image and a weight matrix
    warped = np.zeros((output_height, output_width, 4), dtype=np.uint8)  # RGBA
    weight = np.zeros((output_height, output_width), dtype=np.float32)
    
    # Calculate tangents and normals for the extended curve
    _, normals = calculate_tangents_and_normals(extended_curve)
    
    # Ensure the straightened image is in RGBA format
    straightened = ensure_rgba(straightened)
    
    # Create coordinate grids for the straightened image
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Normalize x-coordinates to match the curve
    t = x_coords / width
    
    # Map the normalized coordinates to indices on the curve
    curve_indices = np.minimum((t * (len(extended_curve) - 1)).astype(int), len(extended_curve) - 1)
    
    # Get the corresponding curve points and normals
    curve_points = extended_curve[curve_indices]
    normal_vectors = normals[curve_indices]
    
    # Calculate offsets for the warp (shift along the normal vectors)
    offsets = (y_coords / height - 0.5) * avg_width
    
    # Compute the warped coordinates
    warped_coords = curve_points - [min_x, min_y] + [padding, padding] + offsets[:, :, np.newaxis] * normal_vectors
    warped_coords = warped_coords.astype(int)  # Convert to integer pixel coordinates
    
    # Validate the warped coordinates
    valid_mask = (warped_coords[:, :, 0] >= 0) & (warped_coords[:, :, 0] < output_width) & \
                 (warped_coords[:, :, 1] >= 0) & (warped_coords[:, :, 1] < output_height)
    
    # Map valid pixels from the straightened image to the warped image
    warped[warped_coords[valid_mask, 1], warped_coords[valid_mask, 0]] = straightened[y_coords[valid_mask], x_coords[valid_mask]]
    weight[warped_coords[valid_mask, 1], warped_coords[valid_mask, 0]] += 1

    # Create a binary mask for transparent regions
    mask = (warped[:, :, 3] > 0).astype(np.uint8) * 255

    # Close gaps in the mask
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Detect and fill contours
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    fill_mask = np.zeros_like(mask)
    cv2.drawContours(fill_mask, contours, -1, 255, -1)

    # Dilate the filled mask
    kernel_dilate = np.ones((3, 3), np.uint8)
    fill_mask_dilated = cv2.dilate(fill_mask, kernel_dilate, iterations=2)

    # Identify transparent regions for inpainting
    transparent_mask = (fill_mask_dilated == 255) & (warped[:, :, 3] == 0)
    
    # Create a copy of the warped image for filling
    warped_filled = warped.copy()
    
    # Inpaint the transparent regions
    for i in range(3):  # Inpaint RGB channels
        warped_filled[:, :, i] = cv2.inpaint(warped[:, :, i], transparent_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    
    # Update the alpha channel of the filled image
    warped_filled[:, :, 3] = np.where(transparent_mask, 255, warped[:, :, 3])
    
    # Ensure output is in RGBA format
    warped = cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA)
    warped_filled = cv2.cvtColor(warped_filled, cv2.COLOR_BGRA2RGBA)

    return warped, warped_filled, mask