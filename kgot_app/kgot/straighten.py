import numpy as np
import cv2
from .utils import interpolate_centerline, calculate_tangents_and_normals, ensure_rgba

def straighten_image(image, keypoints, mask, output_length, output_width):
    h, w = image.shape[:2]
    output_height = int(output_width * (h / w))
    straightened = np.zeros((output_height, int(output_length), 4), dtype=np.uint8)

    image = ensure_rgba(image)
    centerline = interpolate_centerline(keypoints, int(output_length))
    _, normals = calculate_tangents_and_normals(centerline)

    for i in range(int(output_length)):
        cx, cy = centerline[i]
        nx, ny = normals[i]

        for j in range(output_height):
            offset = (j - output_height / 2)
            sx, sy = int(cx + offset * nx), int(cy + offset * ny)

            if 0 <= sx < w and 0 <= sy < h and mask[sy, sx]:
                straightened[j, i] = image[sy, sx]
            else:
                straightened[j, i] = [0, 0, 0, 0]

    return straightened