import time

from constants import *
from draw import *
from utils import handle_keys
from line_profiler import profile

from chessboard import Chessboard
from game import Game

class Program:
    """Class that manage the program."""
    def __init__(self):
        self.chessboard = Chessboard()
        self.game = Game(self.chessboard)
        self.pause = False
        self.finished = False
        self.capture = None
        self.base_frame = None
        self.debug_frame = None
        self.gamestate_frame = None
        self.frame_nb = FRAME_START

    def init_video(self):
        """Initialize the video"""
        # Load the video
        self.capture = cv2.VideoCapture(VIDEO_PATH_MP4)
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(VIDEO_PATH_MOV)
            if not self.capture.isOpened():
                sys.exit(f"Unable to open {VIDEO_NAME}")
        # Resize
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[WIDTH])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[HEIGHT])
        if int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != FRAME_SIZE[WIDTH] or int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != FRAME_SIZE[HEIGHT]:
            print(f"Resolution conversion to {FRAME_SIZE[0]}x{FRAME_SIZE[1]} failed. Will resize frames separately.")
        else:
            print(f"Resolution conversion to {FRAME_SIZE[0]}x{FRAME_SIZE[1]} succeded.")
        # Setup the window for the user.
        if DISPLAY_DEBUG_WINDOW:
            cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(DEBUG_WINDOW_NAME, DEBUG_WINDOW_SIZE)
        if DISPLAY_GAMESTATE_WINDOW:
            cv2.namedWindow(GAMESTATE_DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(GAMESTATE_DEBUG_WINDOW_NAME, GAMESTATE_WINDOW_SIZE)
            

    @profile
    def read_frame(self):
        """Read the next frame and to_show versions."""
        # Select the frame number
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_nb)
        # Read the frame from the capture
        ret, self.base_frame = self.capture.read()
        if not ret:
            self.finished = True
            return
        # Resize the frame if needed.
        if (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != FRAME_SIZE[WIDTH] or int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != FRAME_SIZE[HEIGHT]):
            self.base_frame = cv2.resize(self.base_frame, FRAME_SIZE)
        # Get a copy for the user.
        if DISPLAY_GAMESTATE_WINDOW or DISPLAY_DEBUG_WINDOW:
            self.debug_frame = self.base_frame.copy()
    @profile
    def run(self):
        """Start the main loop"""
        # Initiate the test
        if FRAME_LIMIT > 0:
            t0 = time.time()
        self.read_frame()
        # Show first frame instead of empty window
        if DISPLAY_DEBUG_WINDOW:
            cv2.imshow(DEBUG_WINDOW_NAME, self.debug_frame)
        if DISPLAY_GAMESTATE_WINDOW:
            cv2.imshow(GAMESTATE_DEBUG_WINDOW_NAME, self.debug_frame)
        corners_found = False
        while not self.finished:
            handle_keys(self)
            # Finish if the test ended.
            if FRAME_LIMIT > 0 and self.frame_nb >= FRAME_LIMIT:
                self.set_finished()

            if self.pause:
                continue
            # print(f"Frame {self.frame_nb} processing...")
            corners_found = self.chessboard.update_corners(self.base_frame)
            if corners_found:
                self.chessboard.update_board(self.base_frame)
                self.game.update_game()

            if DISPLAY_DEBUG_WINDOW and (corners_found or not SHOW_IF_FOUND):
                if BLACK_WHITE:
                    self.debug_frame = self.chessboard.frame_black_square(self.base_frame)
                draw_stickers(self.debug_frame, self.chessboard.get_stickers())
                draw_corners(self.debug_frame, self.chessboard.get_corner_pos())
                draw_overlay(self.debug_frame, self.frame_nb)
                draw_squares_detection(self.base_frame, self.debug_frame, self.chessboard)
                cv2.imshow(DEBUG_WINDOW_NAME, self.debug_frame)

            if DISPLAY_GAMESTATE_WINDOW and corners_found:
                self.gamestate_frame = self.chessboard.get_cropped_frame()
                draw_piece_contours(self.gamestate_frame, self.chessboard)
                cv2.imshow(GAMESTATE_DEBUG_WINDOW_NAME, self.gamestate_frame)

            self.frame_nb += FRAME_INT
            self.read_frame()
        # Display the test result
        if FRAME_LIMIT > 0:
            t1 = time.time()
            print(f"Total: {t1-t0}\nAverage per frame: {(t1-t0)/FRAME_LIMIT}")

    def set_finished(self):
        """Change the attribute finished to true."""
        self.finished = True

    def switch_pause(self):
        """Set to pause if not on pause, remove the pause otherwise."""
        self.pause = not self.pause

    def close(self):
        """Releases the capture and destroys windows."""
        self.capture.release()
        cv2.destroyAllWindows()
