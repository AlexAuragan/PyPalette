from urllib.parse import urlparse

import numpy as np
import requests
import pathlib
import cv2


class CannotLoadImage(Exception):
    pass


def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except AttributeError:
        return False


def load_image(x: str) -> np.ndarray:
    """
    Load an image from either a URL or a local file path.

    :param x: Can either be a URL or a path to an image
    :return: Numpy array representing the image in RGB format
    :raises CannotLoadImage: If the image cannot be loaded
    """

    try:
        if uri_validator(x):
            # Handle URL
            response = requests.get(x)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        elif pathlib.Path(x).exists():
            # Handle local file path
            image = cv2.imread(str(x))
        else:
            raise CannotLoadImage(f"Invalid input: {x}")

        if image is None:
            raise CannotLoadImage(f"Failed to decode image from: {x}")

        # Convert to RGB (cv2 uses BGR by default)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    except Exception as e:
        raise CannotLoadImage(f"Error loading image from {x}: {str(e)}")
