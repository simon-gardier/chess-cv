import cv2
import numpy as np

def in_list(array, list):
    """
    Tells if a np.array is in a list.
    Args:
        array: An np.array
        list: The list in which it may be.
    Returns:
        bool: True if the array is in the list.
    """
    return any(np.array_equal(array, center) for center in list)

def draw_piece_contours(frame, pieces_contours):
    """
    Draw the contours of the pieces.
    """
    if pieces_contours is not None:
        cv2.drawContours(frame, pieces_contours, -1, (0, 255, 0), 1)