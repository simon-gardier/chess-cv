import cv2
import os
import sys

# Project constants
VIDEO_TYPE  = "fix"             # 'fix' or 'moving'.
GROUP_ID    = 5                 # Our group id.
STUDENT_ID  = 1                 # Our student id: 1: Eri, 2: Arthur, 3: Simon.
TASK_FOLDER = '.'               # The folder the code is in.

VIDEOS_FOLDER   = f'{TASK_FOLDER}/videos'
VIDEO_NAME  = f"{VIDEO_TYPE}_{GROUP_ID}_{STUDENT_ID}" # Videos must be in the format type_groupid_studentid.mp4
VIDEO_PATH_MP4  = f'{VIDEOS_FOLDER}/{VIDEO_NAME}.mp4'
VIDEO_PATH_MOV  = f'{VIDEOS_FOLDER}/{VIDEO_NAME}.mov'
DEBUG_WINDOW_NAME = "Debug"
DEBUG_WINDOW_SIZE = (1400, 800)
GAMESTATE_DEBUG_WINDOW_NAME = "Game state - Debug"
GAMESTATE_WINDOW_SIZE = (400, 400)

FRAME_SIZE = (1920, 1080)
FPS = 1                         # Must be 1 if you want the maximum FPS
IMG_PER_SEC = int(1000/FPS)
if FPS == 1:
    IMG_PER_SEC = 1             # May seems to not make sense like that but thats the value required by OpenCV if you want no delay on keypress

# Video tuning
FRAME_START = 0                 # The frame we start with.
FRAME_LIMIT = 0                 # The frame we stop at for testing purpose. (0 if we don't want to stop)
FRAME_INT = 25                  # The interval of the frame we consider.

# Tuning
MAX_CENTER_DIST = 275           # The number of pixel two adjacent square centers may be away from each other. sqrt(240² + 135²) = 275
MAX_BORDER_DIST = 20            # The number of pixel two adjacent square may be away from each other.

# Game
WHITE_PIECES = [ 1,  2,  3,  4,  5,  6]
BLACK_PIECES = [-1, -2, -3, -4, -5, -6]

# Constants representing piece types
PAWN = 1
ROOK = 2
KNIGHT = 3
BISHOP = 4
QUEEN = 5
KING = 6

# Structures indexes
W = 0
H = 1
WIDTH = 0
HEIGHT = 1
BLUE = 0
PINK = 1

# Keys
RESUME_KEY = ord('k')
QUIT_KEY = ord('q')

# Visual elements
DISPLAY_DEBUG_WINDOW = True     # Desactivate main window
DISPLAY_GAMESTATE_WINDOW = True  # Desactivate secondary window
SHOW_IF_FOUND = True            # Display the frame only if the corners are found
DRAW_OVERLAY = True             # DIisplay the overlay
DRAW_STICKERS = False           # Display the contour of the stickers found
DRAW_CORNERS = True             # Display the corners and the polygon
DRAW_SQUARES = False            # Display the black squares contours
DRAW_PIECES = True              # Display the pieces contours
BLACK_WHITE = False             # Display the black and white frame used to find the squares.
CORNER_RADIUS = 2               # The size of the corner dots.
CORNER_BGR = (255, 255, 0)      # The color of the corner dots.
BLUE_STICKER_CONTOUR_BGR = (255, 0, 0)
PINK_STICKER_CONTOUR_BGR = (0, 0, 255)
STICKERS_CENTER_RADIUS = 2      # The size of the sticker dots.
STICKERS_CENTER_POINT_COLOR_BGR = (0, 0, 0)
STICKERS_CONTOUR_THICKNESS = 2
POLYGON_COLOR = (0, 255, 0)     # The color the polygon is draw in.
POLYGON_THICKNESS = 2           # The thickness of the polygon.
FONT_SCALE = 0.8
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_THICKNESS = 2
FONT_COLOR = (255, 255, 255)
OVERLAY_POS = (20, 20)
LINE_TYPE = cv2.LINE_8

# Color spaces constants
MAX_H = 180
MAX_S = 255
MAX_V = 255
# Values for saturation equalized
LOWER_BLUE_HSV = [90, 110, 165] # 110 for 1_1 165 because 6_1
UPPER_BLUE_HSV = [105, 255, 255] # 105 is the upper limit for 5_1 and lower limit for 5_3 and 9_1
LOWER_PINK_HSV = [150, 160, 170]
UPPER_PINK_HSV = [170, 255, 255]

# Piece constants
PAWN = 1
ROOK = 2
KNIGHT = 3
BISHOP = 4
KING = 5
QUEEN = 6
