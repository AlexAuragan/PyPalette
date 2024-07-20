import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN


def extract_colors_color_thief(image: np.ndarray, n_colors=5, quality=10):
    """
    Extract dominant colors using a simplified Color Thief algorithm.

    :param image: NumPy array of the image in RGB format
    :param n_colors: Number of colors to extract
    :param quality: Determine the quality of the result. 1 is the highest quality, 10 is the default.
    :return: List of RGB tuples representing dominant colors
    """
    width, height, _ = image.shape
    pixels = image.reshape(-1, 3)

    # Sample the image to reduce computation
    pixels = pixels[::quality]

    # Use K-Means clustering to find the most common colors
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)

    # Get the colors
    colors = kmeans.cluster_centers_

    # Convert to integer RGB values
    colors = np.round(colors).astype(int)

    return [tuple(color) for color in colors]


def extract_colors_median_cut(image: np.ndarray, n_colors=5):
    """
    Extract dominant colors using the Median Cut algorithm.

    :param image: NumPy array of the image in RGB format
    :param n_colors: Number of colors to extract
    :return: List of RGB tuples representing dominant colors
    """

    def cut_box(box):
        channel = np.argmax(box[:, 1] - box[:, 0])
        sort_indices = np.argsort(box[:, channel])
        box = box[sort_indices]
        median = len(box) // 2
        return box[:median], box[median:]

    pixels = image.reshape(-1, 3)
    box = np.column_stack((np.min(pixels, axis=0), np.max(pixels, axis=0)))
    boxes = [box]

    while len(boxes) < n_colors:
        box = boxes.pop(0)
        box1, box2 = cut_box(box)
        boxes.extend([box1, box2])

    colors = []
    for box in boxes:
        colors.append(tuple(np.mean(box, axis=0).astype(int)))

    return colors


def _create_node(level, parent):
    return {
        'level': level,
        'parent': parent,
        'children': [None] * 8,
        'pixel_count': 0,
        'red': 0,
        'green': 0,
        'blue': 0,
        'next': None
    }


def extract_colors_octree(image: np.ndarray, n_colors=5):
    """
    Extract dominant colors using Octree Color Quantization with dictionary-based nodes.

    :param image: NumPy array of the image in RGB format
    :param n_colors: Number of colors to extract
    :return: List of RGB tuples representing dominant colors
    """

    def color_to_octree(color, level):
        if level > 7:
            return None
        bit = 7 - level
        index = ((color[0] & (1 << bit)) >> (bit - 2) |
                 (color[1] & (1 << bit)) >> (bit - 1) |
                 (color[2] & (1 << bit)) >> bit)
        return index

    def add_color(node, color, level):
        if level > 7:
            node['pixel_count'] += 1
            node['red'] += color[0]
            node['green'] += color[1]
            node['blue'] += color[2]
            return

        index = color_to_octree(color, level)
        if node['children'][index] is None:
            node['children'][index] = _create_node(level + 1, node)
        add_color(node['children'][index], color, level + 1)

    root = _create_node(0, None)
    pixels = image.reshape(-1, 3)

    for pixel in pixels:
        add_color(root, pixel, 0)

    def get_leaf_nodes(node):
        if all(child is None for child in node['children']):
            return [node]
        leaves = []
        for child in node['children']:
            if child is not None:
                leaves.extend(get_leaf_nodes(child))
        return leaves

    leaves = get_leaf_nodes(root)
    leaves.sort(key=lambda x: x['pixel_count'], reverse=True)

    colors = []
    for leaf in leaves[:n_colors]:
        if leaf['pixel_count'] > 0:
            r = leaf['red'] // leaf['pixel_count']
            g = leaf['green'] // leaf['pixel_count']
            b = leaf['blue'] // leaf['pixel_count']
            colors.append((r, g, b))

    return colors


def extract_colors_kmeans(image: np.ndarray, n_colors=5):
    """
    Extract dominant colors using K-Means clustering.

    :param image: NumPy array of the image in RGB format
    :param n_colors: Number of colors to extract
    :return: List of RGB tuples representing dominant colors
    """
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels,
                                    n_colors,
                                    None,
                                    criteria,
                                    10,
                                    cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    return [tuple(color) for color in centers]


def extract_colors_histogram(image: np.ndarray, n_colors=5):
    """
    Extract dominant colors using a histogram-based method.

    :param image: NumPy array of the image in RGB format
    :param n_colors: Number of colors to extract
    :return: List of RGB tuples representing dominant colors
    """
    pixels = image.reshape(-1, 3)
    hist, _ = np.histogramdd(pixels, bins=(8, 8, 8))

    colors = []
    while len(colors) < n_colors:
        idx = np.unravel_index(hist.argmax(), hist.shape)
        color = tuple((np.array(idx) * 32 + 16).astype(int))
        colors.append(color)
        hist[idx] = 0

    return colors


def extract_colors_dbscan(image: np.ndarray, eps=10, min_samples=5):
    """
    Extract dominant colors using DBSCAN clustering.

    :param image: NumPy array of the image in RGB format
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point
    :return: List of RGB tuples representing dominant colors
    """
    pixels = image.reshape(-1, 3)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixels)

    # noinspection PyUnresolvedReferences
    labels = db.labels_

    unique_labels = set(labels)

    colors = []
    for k in unique_labels:
        if k == -1:  # Skip noise
            continue
        class_member_mask = (labels == k)
        color = np.mean(pixels[class_member_mask], axis=0).astype(int)
        colors.append(tuple(color))

    return colors
