'''
General utility functions for the KGOT app
'''

import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import os
from skimage.draw import line
import time


def interpolate_centerline(keypoints, num_points):
    t = np.linspace(0, 1, len(keypoints))
    t_new = np.linspace(0, 1, num_points)
    fx = interp1d(t, keypoints[:, 0], kind='cubic', bounds_error=False, fill_value="extrapolate")
    fy = interp1d(t, keypoints[:, 1], kind='cubic', bounds_error=False, fill_value="extrapolate")
    return np.column_stack((fx(t_new), fy(t_new)))

def calculate_tangents_and_normals(points):
    tangents = np.gradient(points, axis=0)
    tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    return tangents, normals

def prepare_centerline(centerline, num_segments):
    centerline_interp = interpolate_centerline(centerline, num_segments * 10)
    _, normals = calculate_tangents_and_normals(centerline_interp)
    indices = np.linspace(0, len(centerline_interp) - 1, num_segments + 1, dtype=int)
    selected_points = centerline_interp[indices]
    return centerline_interp, normals, indices, selected_points

def calculate_width_and_fill_mask(mask, point, normal, segment_mask=None):
    width = 0
    for direction in [1, -1]:
        current_point = np.array(point, dtype=float)
        while True:
            x, y = np.round(current_point).astype(int)
            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0] or mask[y, x] == 0:
                break
            if segment_mask is not None:
                segment_mask[y, x] = 255
            current_point += direction * normal
            width += 1
    return width

def process_segment(mask, start_point, end_point, normal, create_mask=False):
    segment_mask = np.zeros_like(mask, dtype=np.uint8) if create_mask else None
    segment_widths = []
    rr, cc = line(int(start_point[1]), int(start_point[0]), int(end_point[1]), int(end_point[0]))
    
    for r, c in zip(rr, cc):
        width = calculate_width_and_fill_mask(mask, (c, r), normal, segment_mask)
        segment_widths.append(width)
    
    return segment_mask, np.mean(segment_widths)

def process_centerline(mask, centerline, num_segments, create_masks=False):
    centerline_interp, normals, indices, selected_points = prepare_centerline(np.array(centerline), num_segments)

    segment_masks = []
    widths = []

    for i in range(len(selected_points) - 1):
        start_point = selected_points[i]
        end_point = selected_points[i + 1]
        normal = normals[indices[i]]
        
        segment_mask, width = process_segment(mask, start_point, end_point, normal, create_masks)
        
        if create_masks:
            segment_masks.append(segment_mask)
        widths.append(width)

    return segment_masks, widths

def divide_mask_by_centerline(mask, centerline, num_segments=10):
    segment_masks, widths = process_centerline(mask, centerline, num_segments, create_masks=True)
    return segment_masks, widths

def calculate_average_width(mask, centerline, num_segments=10):
    _, widths = process_centerline(mask, centerline, num_segments, create_masks=False)
    return np.mean(widths)

def ensure_rgba(image):
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image

def generate_reference_curve(keypoints, num_points=1000, target_length=None):
    distances = np.linalg.norm(np.diff(keypoints, axis=0), axis=1)
    original_total_length = np.sum(distances)
    
    print(f"Original length: {original_total_length:.2f}")
    print(f"Target length: {target_length}")
    
    if target_length is not None:
        scale_factor = target_length / original_total_length
        scaled_distances = distances * scale_factor
        scaled_cumulative_distances = np.insert(np.cumsum(scaled_distances), 0, 0)
        
        scaled_keypoints = np.zeros_like(keypoints)
        scaled_keypoints[0] = keypoints[0]
        scaled_keypoints[1:] = keypoints[0] + np.cumsum(np.diff(keypoints, axis=0) * scale_factor, axis=0)
        
        t = np.linspace(0, 1, num_points)
        f = interp1d(scaled_cumulative_distances / scaled_cumulative_distances[-1], scaled_keypoints, axis=0, kind='linear')
        smooth_curve = f(t)
        
        sigma = 1 * scale_factor
        smooth_curve = gaussian_filter1d(smooth_curve, sigma, axis=0)
    else:
        scale_factor = 1
        smooth_curve = keypoints
    
    new_length = np.sum(np.linalg.norm(np.diff(smooth_curve, axis=0), axis=1))
    print(f"New length: {new_length:.2f}")
    print(f"Scale factor: {scale_factor:.2f}")
    
    return smooth_curve, scale_factor


def load_and_preprocess_image(image_path):
    # Read image in BGR format
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Convert to RGBA
    return ensure_rgba(image)

def save_straightened_image(image, output_path):
    cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])