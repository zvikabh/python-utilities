import os

import cv2
import numpy as np


def decode_image(encoded_img: bytes) -> np.ndarray:
    """Decodes an image into a NumPy array."""
    decoded_img = cv2.imdecode(
        np.frombuffer(encoded_img, np.uint8), flags=cv2.IMREAD_UNCHANGED)
    # OpenCV uses BGR encoding, while Matplotlib uses RGB.
    if decoded_img.ndim == 3:
        b_values = np.copy(decoded_img[:, :, 0])
        decoded_img[:, :, 0] = decoded_img[:, :, 2]
        decoded_img[:, :, 2] = b_values
    return decoded_img


def encode_image(image: np.ndarray, filetype: str) -> bytes:
    """Returns an encoded image with a specified file type.

    Args:
        image: 2D or 3D NumPy array. 3D arrays should have either RGB or RGBA
            encoding as the first channel.
        filetype: File extension supported by OpenCV, e.g., '.png'.
    """
    # OpenCV uses BGR encoding, while Matplotlib uses RGB.
    image = np.asarray(image)
    if image.ndim == 3:
        b_values = np.copy(image[:, :, 0])
        image[:, :, 0] = image[:, :, 2]
        image[:, :, 2] = b_values
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    success, b = cv2.imencode(filetype, image)
    assert success
    return b


def read_image(filename: str) -> np.ndarray:
    """Reads an image from disk and decodes it into a NumPy array."""
    with open(filename, 'rb') as f:
        return decode_image(f.read())


def write_image(image: np.ndarray, filename: str) -> None:
    """Write an image to a given file.

    Args:
        image: 2D or 3D NumPy array. 3D arrays should have either RGB or RGBA
            encoding as the first channel.
        filename: File to write to. Format is determined based on the file
            extension.
    """
    with open(filename, 'wb') as f:
        f.write(encode_image(image, os.path.splitext(filename)[1]))
