import cv2
import os
import shutil
import time

# Program title
TITLE = r"""   _____ _                               _                
  / ____| |                             | |               
 | |    | |__   ___  ___ ___   ___  ___ | |_   _____ _ __ 
 | |    | '_ \ / _ \/ __/ __| / __|/ _ \| \ \ / / _ \ '__|
 | |____| | | |  __/\__ \__ \ \__ \ (_) | |\ V /  __/ |   
  \_____|_| |_|\___||___/___/ |___/\___/|_| \_/ \___|_|

                                                       .::.
                                            _()_       _::_
                                  _O      _/____\_   _/____\_
           _  _  _     ^^__      / //\    \      /   \      /
          | || || |   /  - \_   {     }    \____/     \____/
          |_______| <|    __ |   \___/     (____)     (____)
    _     \__ ___ / <|    \      (___)      |  |       |  |
   (_)     |___|_|  <|     \      |_|       |__|       |__|
  (___)    |_|___|  <|______\    /   \     /    \     /    \
  _|_|_    |___|_|   _|____|_   (_____)   (______)   (______)
 (_____)  (_______) (________) (_______) (________) (________)
 /_____\  /_______\ /________\ /_______\ /________\ /________\

 Graillet Arthur, Gardier Simon, Van de Vyver Eri - 2024-2025
"""

# Project constants
VIDEO_TYPE      = "fix"             # 'fix' or 'moving'.
GROUP_ID        = 5                 # Our group id.
STUDENT_ID      = 1                 # Our student id: 1: Eri, 2: Arthur, 3: Simon.
TASK_FOLDER     = '.'               # The folder the code is in.
EXTENSION       = 'mp4'             # The extension of the video file.
VIDEOS_FOLDER   = f'{TASK_FOLDER}/videos'
VIDEO_NAME      = f"{VIDEO_TYPE}_{GROUP_ID}_{STUDENT_ID}"       # Videos must be in the format type_groupid_studentid.
VIDEO_PATH      = f'{VIDEOS_FOLDER}/{VIDEO_NAME}.{EXTENSION}'   # Add the folder path and file extension to the path.
FRAMES_FOLDER   = f'{TASK_FOLDER}/frames/{VIDEO_NAME}'          # The folder where the frames are saved.
NB_VIDEO_PROCESSING_WORKERS = os.cpu_count() - 2                # -2 because we still need one core for the main thread and one for the solver thread.
EXPERIMENT_MODE = False

# Frames save settings
SAVE_FRAMES     = False
CLEAR_FOLDER    = True
if CLEAR_FOLDER and os.path.exists(FRAMES_FOLDER) and os.path.isdir(FRAMES_FOLDER):
    shutil.rmtree(FRAMES_FOLDER)
if SAVE_FRAMES:
    os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Structures indexes
WIDTH_INDEX   = 0
HEIGHT_INDEX  = 1
BLUE_INDEX    = 0
PINK_INDEX    = 1

# Frame constants
FRAME_SIZE = (1920, 1080)                   # The size of the frame.
BOARD_FRAME_SIZE = (400, 400)               # The size of the cropped frame.
SQUARE_SIZE = int(BOARD_FRAME_SIZE[0] / 8)  # The size of a square on the cropped frame.

# Video processing tuning
FRAME_START = 0                   # The frame index we start with.
FRAME_INT = 25                    # The interval of the frame we consider.

# Tuning
MAX_BORDER_DIST = 20              # The number of pixel two adjacent square may be away from each other.
LOWER_BLUE_HSV = [90, 110, 165]   # 110 for 1_1 165 because 6_1
UPPER_BLUE_HSV = [105, 255, 255]  # 105 is the upper limit for 5_1 and lower limit for 5_3 and 9_1
LOWER_PINK_HSV = [150, 160, 170]
UPPER_PINK_HSV = [170, 255, 255]

# Pieces constants
PAWN = 1
ROOK = 2
KNIGHT = 3
BISHOP = 4
KING = 5
QUEEN = 6
UNKNOWN_TYPE = 7
WHITE_PIECES = [PAWN, ROOK, KNIGHT, BISHOP, KING, QUEEN]
BLACK_PIECES = [-PAWN, -ROOK, -KNIGHT, -BISHOP, -KING, -QUEEN]
WHITE = 1
BLACK = -1
TEMP_COLOR = 2
PIECES_TO_NUM = {
    "square": 0,
    "white_pawn": 1,
    "white_rook": 2,
    "white_knight": 3,
    "white_bishop": 4,
    "white_king": 5,
    "white_queen": 6,
    "white_unknown": 7,
    "black_pawn": -1,
    "black_rook": -2,
    "black_knight": -3,
    "black_bishop": -4,
    "black_king": -5,
    "black_queen": -6,
    "black_unknown": -7,
}

NUM_TO_PIECE = {
    0: "square",
    1: "white_pawn",
    2: "white_rook",
    3: "white_knight",
    4: "white_bishop",
    5: "white_king",
    6: "white_queen",
    7: "white_unknown",
    -1: "black_pawn",
    -2: "black_rook",
    -3: "black_knight",
    -4: "black_bishop",
    -5: "black_king",
    -6: "black_queen",
    -7: "black_unknown"
}

# Visual elements
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

t0 = time.time()
