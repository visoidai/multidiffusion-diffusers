#!/usr/bin/env python3
from PIL import Image
import numpy as np


def get_mask_positions(image_path):
    """
    Identify mask positions (bounding boxes) from unique colors in an image.

    Loads an image and extracts the minimal bounding box for each unique color.
    Returns a dictionary mapping each unique color (as an (R, G, B) tuple) to a bounding box string.
    The bounding box is formatted as "x0:y0-x1:y1", where x1 and y1 are non-inclusive.
    """
    im = Image.open(image_path).convert("RGB")
    arr = np.array(im)

    positions = {}
    # Get unique colors from the image
    unique_colors = np.unique(arr.reshape(-1, 3), axis=0)

    for color in unique_colors:
        # Create a boolean mask where the pixel matches the current color
        mask = np.all(arr == color, axis=2)
        y_indices, x_indices = np.where(mask)
        if y_indices.size == 0 or x_indices.size == 0:
            continue
        # Calculate the bounding box coordinates
        top, left = int(y_indices.min()), int(x_indices.min())
        bottom, right = int(y_indices.max()), int(x_indices.max())

        # Format the bounding box as "x0:y0-x1:y1" (with x1 and y1 as non-inclusive)
        positions[tuple(color)] = f"{left}:{top}-{right+1}:{bottom+1}"

    return positions
