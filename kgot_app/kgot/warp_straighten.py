
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import cv2
from .utils import interpolate_centerline, calculate_tangents_and_normals, ensure_rgba


def warp_straightened_image(straightened, reference_curve, avg_width, scale_factor=None):
    height, width = straightened.shape[:2]
    
    extended_curve = interpolate_centerline(reference_curve, int(len(reference_curve) * 1.2))
    
    min_x, min_y = np.min(extended_curve, axis=0)
    max_x, max_y = np.max(extended_curve, axis=0)
    
    padding = avg_width
    output_width = int(max_x - min_x + 2 * padding)
    output_height = int(max_y - min_y + 2 * padding)
    
    warped = np.zeros((output_height, output_width, 4), dtype=np.uint8)
    weight = np.zeros((output_height, output_width), dtype=np.float32)
    
    _, normals = calculate_tangents_and_normals(extended_curve)
    
    straightened = ensure_rgba(straightened)
    
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    t = x_coords / width
    curve_indices = np.minimum((t * (len(extended_curve) - 1)).astype(int), len(extended_curve) - 1)
    
    curve_points = extended_curve[curve_indices]
    normal_vectors = normals[curve_indices]
    
    offsets = (y_coords / height - 0.5) * avg_width
    
    warped_coords = curve_points - [min_x, min_y] + [padding, padding] + offsets[:, :, np.newaxis] * normal_vectors
    warped_coords = warped_coords.astype(int)
    
    valid_mask = (warped_coords[:, :, 0] >= 0) & (warped_coords[:, :, 0] < output_width) & \
                 (warped_coords[:, :, 1] >= 0) & (warped_coords[:, :, 1] < output_height)
    
    warped[warped_coords[valid_mask, 1], warped_coords[valid_mask, 0]] = straightened[y_coords[valid_mask], x_coords[valid_mask]]
    weight[warped_coords[valid_mask, 1], warped_coords[valid_mask, 0]] += 1

    mask = (warped[:,:,3] > 0).astype(np.uint8) * 255

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    fill_mask = np.zeros_like(mask)
    cv2.drawContours(fill_mask, contours, -1, 255, -1)

    kernel_dilate = np.ones((3, 3), np.uint8)
    fill_mask_dilated = cv2.dilate(fill_mask, kernel_dilate, iterations=2)

    transparent_mask = (fill_mask_dilated == 255) & (warped[:,:,3] == 0)
    transparent_mask = (fill_mask == 255) & (warped[:,:,3] == 0)

    warped_filled = warped.copy()
    for i in range(3):
        warped_filled[:,:,i] = cv2.inpaint(warped[:,:,i], transparent_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    
    warped_filled[:,:,3] = np.where(transparent_mask, 255, warped[:,:,3])
    
    # Ensure output is in RGBA format
    warped = cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA)
    warped_filled = cv2.cvtColor(warped_filled, cv2.COLOR_BGRA2RGBA)

    return warped, warped_filled, warped