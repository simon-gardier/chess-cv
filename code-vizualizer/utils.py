import cv2
import numpy as np
from line_profiler import profile

from constants import *

def handle_keys(program):
    """
    Wait for user input and modify the flow of the program based on those inputs.
    Args:
        program: The Program object.
    """
    if not DISPLAY_DEBUG_WINDOW and not DISPLAY_GAMESTATE_WINDOW:
        return
    key_pressed = cv2.waitKey(IMG_PER_SEC) & 0xFF

    if key_pressed == QUIT_KEY:
        program.set_finished()
    # if cv2.getWindowProperty(DEBUG_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
    #     program.set_finished()
    if key_pressed == RESUME_KEY:
        program.switch_pause()

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
