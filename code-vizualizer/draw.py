import cv2
import numpy as np
from constants import *

def draw_overlay(frame, frame_nb):
    """
    Draw the overlay.
    Args:
        frame: The frame to draw on.
        frame_nb: The index of the frame in the video.
    """
    if not DRAW_OVERLAY:
        return 
    cv2.putText(frame, f"Video processing | frame #{frame_nb} | Press {chr(RESUME_KEY)} to pause/resume processing", OVERLAY_POS,  FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, LINE_TYPE)

def draw_corners(frame, corners_pos):
    """
    Draw the corners.
    Args:
        frame: The frame to draw on.
        corners_pos: The position of the corners.
    """
    if DRAW_CORNERS:
        # Draw the polygon.
        cv2.polylines(frame, [np.array(corners_pos)], isClosed=True, color=POLYGON_COLOR, thickness=POLYGON_THICKNESS)
        # Draw the corner names.
        for i, corner_name in enumerate(['a1', 'a8', 'h8', 'h1']):
            current_pos = corners_pos[i]
            # Label shadow
            cv2.putText(frame, corner_name, current_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
            # Label text
            cv2.putText(frame, corner_name, current_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, POLYGON_COLOR, 2, cv2.LINE_AA)
            # Position shawdow
            pos_text = f"{current_pos}"
            cv2.putText(frame, pos_text, (current_pos[0], current_pos[1] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
            # Position text
            cv2.putText(frame, pos_text, (current_pos[0], current_pos[1] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, POLYGON_COLOR, 2, cv2.LINE_AA)

def draw_stickers(frame, stickers):
    """
    Draw the stickers.
    Args:
        frame: The frame to draw on.
        stickers: A list with the two Sticker objects.
    """
    if not DRAW_STICKERS:
        return
    # Draw the approximation of the contour of each sticker and a dot in the center
    if stickers[BLUE].get_contour() is not None:
        approx = cv2.approxPolyDP(stickers[BLUE].get_contour(), 0.02 * cv2.arcLength(stickers[BLUE].get_contour(), True), True)
        cv2.drawContours(frame, [approx], 0, BLUE_STICKER_CONTOUR_BGR, STICKERS_CONTOUR_THICKNESS)
        cv2.circle(frame, stickers[BLUE].get_pos(), STICKERS_CENTER_RADIUS, STICKERS_CENTER_POINT_COLOR_BGR, cv2.FILLED)
    if stickers[PINK].get_contour() is not None:
        approx = cv2.approxPolyDP(stickers[PINK].get_contour(), 0.02 * cv2.arcLength(stickers[PINK].get_contour(), True), True)
        cv2.drawContours(frame, [approx], 0, PINK_STICKER_CONTOUR_BGR, STICKERS_CONTOUR_THICKNESS)
        cv2.circle(frame, stickers[PINK].get_pos(), STICKERS_CENTER_RADIUS, STICKERS_CENTER_POINT_COLOR_BGR, cv2.FILLED)

def draw_squares_detection(base_frame, to_show, chessboard):
    """
    Find and draw the black squares.
    Args:
        base_frame: The frame to search on.
        to_show: The frame to draw on.
        chessboard: The chessboard of the game.
    """
    if not DRAW_SQUARES:
        return
    # Get squares from chessboard method
    black_squares = chessboard.find_black_squares(base_frame)
    # Draw the contours
    for square in black_squares:
        cv2.drawContours(to_show, [square], 0, (0, 0, 200), 2)

def draw_piece_contours(frame, chessboard):
    """
    Draw the contours of the pieces.
    """
    if not DRAW_PIECES:
        return
    contours = chessboard.find_pieces(frame)
    if contours is not None:
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
