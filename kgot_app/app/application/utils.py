import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from skimage.draw import line


def interpolate_centerline(keypoints, num_points):
    """
    Interpolates a smoother centerline from the given keypoints using cubic interpolation.

    Args:
        keypoints (ndarray): Array of shape (n, 2) containing (x, y) centerline points.
        num_points (int): Number of points for the interpolated centerline.

    Returns:
        ndarray: Interpolated (x, y) points for the centerline.
    """
    # Generate parameterized t values for the original keypoints
    t = np.linspace(0, 1, len(keypoints))
    t_new = np.linspace(0, 1, num_points)

    # Interpolate x and y values separately
    fx = interp1d(t, keypoints[:, 0], kind='cubic', bounds_error=False, fill_value="extrapolate")
    fy = interp1d(t, keypoints[:, 1], kind='cubic', bounds_error=False, fill_value="extrapolate")

    # Stack the interpolated x and y values
    return np.column_stack((fx(t_new), fy(t_new)))


def calculate_tangents_and_normals(points):
    """
    Computes tangents and normals for a given set of curve points.

    Args:
        points (ndarray): Array of shape (n, 2) representing curve points.

    Returns:
        tuple: Tangents and normals for each curve point.
    """
    # Compute tangents using gradient
    tangents = np.gradient(points, axis=0)
    tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]  # Normalize tangents

    # Compute normals as perpendicular vectors to tangents
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

    return tangents, normals


def prepare_centerline(centerline, num_segments):
    """
    Prepares the centerline by interpolating it and calculating normals.

    Args:
        centerline (ndarray): Array of (x, y) centerline points.
        num_segments (int): Number of segments to divide the centerline into.

    Returns:
        tuple: (interpolated centerline, normals, indices for segment boundaries, selected points).
    """
    # Smoothly interpolate the centerline
    centerline_interp = interpolate_centerline(centerline, num_segments * 10)

    # Calculate tangents and normals for the interpolated centerline
    _, normals = calculate_tangents_and_normals(centerline_interp)

    # Select segment boundary points
    indices = np.linspace(0, len(centerline_interp) - 1, num_segments + 1, dtype=int)
    selected_points = centerline_interp[indices]

    return centerline_interp, normals, indices, selected_points


def calculate_width_and_fill_mask(mask, point, normal, segment_mask=None):
    """
    Calculates the width of the mask along a normal direction and optionally fills the mask.

    Args:
        mask (ndarray): Binary mask indicating valid regions.
        point (tuple): Starting (x, y) point.
        normal (ndarray): Normal vector at the point.
        segment_mask (ndarray, optional): Mask to fill the segment (modified in-place).

    Returns:
        int: Width along the normal direction.
    """
    width = 0

    # Iterate in both directions along the normal vector
    for direction in [1, -1]:
        current_point = np.array(point, dtype=float)

        while True:
            x, y = np.round(current_point).astype(int)

            # Check bounds and mask validity
            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0] or mask[y, x] == 0:
                break

            # Optionally fill the segment mask
            if segment_mask is not None:
                segment_mask[y, x] = 255

            # Move along the normal
            current_point += direction * normal
            width += 1

    return width


def process_segment(mask, start_point, end_point, normal, create_mask=False):
    """
    Processes a segment of the centerline by calculating its width and creating a mask.

    Args:
        mask (ndarray): Binary mask indicating valid regions.
        start_point (tuple): Starting (x, y) point of the segment.
        end_point (tuple): Ending (x, y) point of the segment.
        normal (ndarray): Normal vector for the segment.
        create_mask (bool): Whether to create a mask for the segment.

    Returns:
        tuple: (segment mask, average width of the segment).
    """
    # Optionally create a segment mask
    segment_mask = np.zeros_like(mask, dtype=np.uint8) if create_mask else None
    segment_widths = []

    # Get all points along the line segment
    rr, cc = line(int(start_point[1]), int(start_point[0]), int(end_point[1]), int(end_point[0]))

    # Calculate width at each point
    for r, c in zip(rr, cc):
        width = calculate_width_and_fill_mask(mask, (c, r), normal, segment_mask)
        segment_widths.append(width)

    return segment_mask, np.mean(segment_widths)


def process_centerline(mask, centerline, num_segments, create_masks=False):
    """
    Processes a centerline by dividing it into segments and calculating widths.

    Args:
        mask (ndarray): Binary mask indicating valid regions.
        centerline (ndarray): Array of (x, y) points defining the centerline.
        num_segments (int): Number of segments to divide the centerline into.
        create_masks (bool): Whether to create masks for each segment.

    Returns:
        tuple: (list of segment masks, list of segment widths).
    """
    centerline_interp, normals, indices, selected_points = prepare_centerline(np.array(centerline), num_segments)

    segment_masks = []
    widths = []

    # Process each segment
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
    """
    Divides a mask into segments along the centerline.

    Args:
        mask (ndarray): Binary mask to divide.
        centerline (ndarray): Array of (x, y) points defining the centerline.
        num_segments (int): Number of segments to divide the centerline into.

    Returns:
        tuple: (list of segment masks, list of segment widths).
    """
    return process_centerline(mask, centerline, num_segments, create_masks=True)


def calculate_average_width(mask, centerline, num_segments=10):
    """
    Calculates the average width of a mask along the centerline.

    Args:
        mask (ndarray): Binary mask indicating valid regions.
        centerline (ndarray): Array of (x, y) points defining the centerline.
        num_segments (int): Number of segments to divide the centerline into.

    Returns:
        float: Average width along the centerline.
    """
    _, widths = process_centerline(mask, centerline, num_segments, create_masks=False)
    return np.mean(widths)


def ensure_rgba(image):
    """
    Ensures the input image is in RGBA format.

    Args:
        image (ndarray): Input image (BGR or BGRA).

    Returns:
        ndarray: Image in RGBA format.
    """
    if image.shape[2] == 3:  # If BGR, convert to BGRA
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    elif image.shape[2] == 4:  # If BGRA, convert to RGBA
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image


def generate_reference_curve(keypoints, num_points=1000, target_length=None):
    """
    Generates a smooth reference curve from keypoints.

    Args:
        keypoints (ndarray): Array of (x, y) points.
        num_points (int): Number of points for the reference curve.
        target_length (float, optional): Desired length of the reference curve.

    Returns:
        tuple: (smoothed reference curve, scale factor).
    """
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
    """
    Loads an image and converts it to RGBA format.

    Args:
        image_path (str): Path to the image file.

    Returns:
        ndarray: Preprocessed image in RGBA format.
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load image
    return ensure_rgba(image)  # Convert to RGBA


def save_straightened_image(image, output_path):
    """
    Saves the straightened image as a PNG file with compression.

    Args:
        image (ndarray): Straightened image.
        output_path (str): Output file path.

    Returns:
        None
    """
    cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
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