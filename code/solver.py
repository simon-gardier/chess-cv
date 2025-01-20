import json
import queue
import numpy as np
from copy import deepcopy

from video_processing import VideoProcessing
from constants import *
from square import Square
from piece import Piece
from utils import *

class Solver:
    """Class responsible for analysing the chessboard and updating the game state."""
    def worker(chessboards_queue, nb_frames_to_process):
        int_counter = 0
        local_queue = queue.PriorityQueue()
        game_solver = Solver()
        while(int_counter < nb_frames_to_process):
            if local_queue.empty() or local_queue.queue[0][0] != int_counter:
                int_counter_from_queue, chessboard = chessboards_queue.get()
                while(int_counter_from_queue != int_counter):
                    local_queue.put((int_counter_from_queue, chessboard))
                    int_counter_from_queue, chessboard = chessboards_queue.get()
            else:
                int_counter_from_queue, chessboard = local_queue.get()
            if chessboard.is_identified():
                if SAVE_FRAMES:
                    print(f"💾 Save chessboard {int_counter} frame, containing {len(chessboard.get_pieces_contours())} pieces")
                    frame = chessboard.get_board_frame()
                    draw_piece_contours(frame, chessboard.get_pieces_contours())
                    cv2.imwrite(f"{FRAMES_FOLDER}/{int_counter}.png", frame)
                game_solver.update(chessboard)
            int_counter += 1
        game_solver.save_results()

    def __init__(self):
        self.chessboards = []   # Stores the chessboard that have been found in frames
        self.results = {
            "game_states":[]
        }
        self.results_analysis = {
            "game_state_unknown_pieces": []
        }
        self.pieces = []
        # 1 for white, -1 for black, 2 and -2 for unknown
        self.color_playing = TEMP_COLOR
        # Init squares
        self.squares = [[None for _ in range(8)] for _ in range(8)]
        for j in range(8):
            for i in range(8):
                pos = (i, j)
                self.squares[j][i] = Square((pos))

    def reset_logic(self):
        self.pieces = []
        # 1 for white, -1 for black, 2 and -2 for unknown
        self.color_playing = TEMP_COLOR
        # Init squares
        self.squares = [[None for _ in range(8)] for _ in range(8)]
        for j in range(8):
            for i in range(8):
                pos = (i, j)
                self.squares[j][i] = Square((pos))


    def update(self, chessboard):
        """
        Update the game knowledge based on a new chessboard seen in a new frame.
        """
        # If this is the first update, we create the pieces.
        print(f"🧩 Update for frame {chessboard.get_frame_number()}:")
        if not self.chessboards:
            pos_matrix = chessboard.get_pos_matrix()
            for i in range(len(pos_matrix)):
                for j in range(len(pos_matrix[0])):
                    if pos_matrix[i][j] != 0:
                        self.create_piece((i, j))
            self.chessboards.append(chessboard)
            game_state, unknown_pieces_count = self.create_game_state(self.squares)
            if(EXPERIMENT_MODE):
                self.results_analysis["game_state_unknown_pieces"].append(unknown_pieces_count)
            self.add_game_state(self.chessboards[-1].get_frame_number(), game_state)
            return
        # Otherwise we should update the model.
        self.chessboards.append(chessboard)
        move_type, moves = self.get_movements()
        # If there was an error we don't know anymore who is playing
        if move_type == -1:
            self.set_color_playing(2)
            # Remove the board if there was an error.
            self.chessboards.pop()
            print(f"\t🗑️  Gamestate discarded")
            return
        if move_type == -2:
            # Reset all logic (but keeps the boards and the states)
            print(f"\t❌ Too much movement detected ({moves}), reset of the logic 🧹🪣")
            self.reset_logic()
            pos_matrix = chessboard.get_pos_matrix()
            for i in range(len(pos_matrix)):
                for j in range(len(pos_matrix[0])):
                    if pos_matrix[i][j] != 0:
                        self.create_piece((i, j))
            self.chessboards.append(chessboard)
            game_state, unknown_pieces_count = self.create_game_state(self.squares)
            if(EXPERIMENT_MODE):
                self.results_analysis["game_state_unknown_pieces"].append(unknown_pieces_count)
            self.add_game_state(self.chessboards[-1].get_frame_number(), game_state)
            return
        if move_type == 0:
            print("\t🏜️  No movement detected")
        # Manage the consequence of the movement in the model.
        if move_type > 0:
            if len(moves) == 0:
                print("❌ Moves cant be empty")
                return
            for i, (start_pos, end_pos) in enumerate(moves):
                print(f"\t🔎 Piece moved from {Solver.pos_to_string(start_pos)} to {Solver.pos_to_string(end_pos)}")
                piece = self.squares[start_pos[1]][start_pos[0]].get_piece()
                c = 0
                for piece_curr in self.pieces:
                    c += 1
                if self.chessboards[-1].get_frame_number() == 7350:
                    print(f"Piece types before : {piece.get_types()}")
                    print(f"White : {Piece.white_types_left}")
                    print(f"Black : {Piece.black_types_left}")
                if not piece:
                    print(f"\t🚧 Could not find the piece at {Solver.pos_to_string(start_pos)}.")
                    continue
                # Enpassant victime of first piece. Piece existence is already validated in the get_movements method if it returned 2.
                if move_type == 2:
                    # From top to bot is black
                    if dy == 1:
                        color = BLACK
                    # Opposite is white
                    else:
                        color = WHITE
                    # Attacker
                    if i == 0:
                        piece.set_type(color*PAWN)
                    # Captured piece
                    if i == 1:
                        piece.set_type(-color*PAWN)
                        piece.set_captured(self.squares[end_pos[1]][end_pos[0]])
                        break
                # If this is a castling we know that it must be a king or a rook.
                if move_type == 3:
                    # Check if that piece is a king or a rook
                    if start_pos[0] == 4:
                        piece_type = KING
                    else:
                        piece_type = ROOK
                    # Check if it is white or black
                    if start_pos[1] == 0:
                        piece.set_type(-piece_type)
                    else:
                        piece.set_type(piece_type)
                # Check move if not castling
                elif not piece.is_move_legal(chessboard, start_pos, end_pos):
                    print(f"\t🚧 Illegal move for piece at {Solver.pos_to_string(start_pos)}.")
                piece.move(self.squares[end_pos[1]][end_pos[0]], self.chessboards[-1].get_frame_number())
                if move_type == 4:
                    continue
                elif piece.get_color() != 0:
                    self.set_color_playing(piece.get_color())
                elif abs(self.color_playing) == WHITE:
                    piece.set_color(self.color_playing)
                else:
                    piece.set_temp_color(self.color_playing)
            # Change the color playing
            # In case of double mouvement it comes back to the same color.
            if move_type == 4:
                pass
            elif move_type != 1:
                self.color_playing *= -1
            else:
                dx = abs(start_pos[0] - end_pos[0])
                dy = abs(start_pos[1] - end_pos[1])
                # If this may be a roque in two part, we set the color playing to unknown.
                # King roque move
                if piece and (KING in piece.get_types() or -KING in piece.get_types()) and in_list(start_pos, [[0,4], [7,4]]) and (dx == 2 and dy == 0):
                    self.set_color_playing(2)
                # Rook roque move
                elif piece and (ROOK in piece.get_types() or -ROOK in piece.get_types()) and in_list(start_pos, [[7,7], [0,7], [7,0], [0,0]]) and ((dx == 2 or dx == 3) and dy == 0):
                    self.set_color_playing(2)
                else:
                    self.color_playing *= -1
            # Try to decide every piece type/color because one move may influence others.
            for piece in self.pieces:
                piece.decide_piece()
            game_state, unknown_pieces_count = self.create_game_state(self.squares)
            if(EXPERIMENT_MODE):
                self.results_analysis["game_state_unknown_pieces"].append(unknown_pieces_count)
            self.add_game_state(self.chessboards[-1].get_frame_number(), game_state)

    def set_color_playing(self, color):
        """
        Set the color playing to the given color.
        Args:
            color: The color playing. 1 for white, -1 for black, 2 for unknown
        """
        if abs(self.color_playing) == WHITE and abs(color) == WHITE:
            if self.color_playing != color:
                print("\t🚧 Color problem.")
            return
        # Define which temporary color was the white one
        if color == WHITE:
            white = self.color_playing
        elif color == BLACK:
            white = -self.color_playing
        # Remove previous temp color if we have to reset temp color.
        else:
            for piece in self.pieces:
                piece.set_temp_color(0)
                self.color_playing = color
            return
        # Backtrack to set the color for the piece having the temp color.
        for piece in self.pieces:
            piece.replace_temp_color(white)
        self.color_playing = color

    def get_movements(self):
        """
        Get the movements done based on two position matrix.
        Returns:
            (move_type, moves): move_type is the type of movement done. -1 for error, 0 for no movement, 
            1 for simple movement, 2 for enpassant, 3 for castling, 4 for double basic mouvement. And moves is a list of movements.
        """
        if len(self.chessboards) < 2:
            print("\t🚧 This method needs at least two chessboards.")
            return (-1, [])
        # Get the difference
        matrix_dif = self.chessboards[-1].get_pos_matrix() - self.chessboards[-2].get_pos_matrix()
        # Count the cells that changed
        nb_change = np.count_nonzero(matrix_dif)
        if nb_change == 0:
            return (0, [])
        pos_list = [pos[::-1] for pos in np.argwhere(matrix_dif != 0)]
        # Init the positions
        start_pos_list = []
        end_pos_list = []
        # Find the positions
        for pos in pos_list:
            if matrix_dif[pos[1]][pos[0]] == 1:
                end_pos_list.append(pos)
            else:
                start_pos_list.append(pos)
        # Piece cannot appear out of nowhere
        if len(end_pos_list) > len(start_pos_list):
            print("\t🚧 A piece cannot appear out of nowhere")
            return (-1, [])
        # Capture
        if nb_change == 1:
            print("\t🔎 Capture detected")
            start_pos = start_pos_list[0]
            start_piece = self.squares[start_pos[1]][start_pos[0]].get_piece()
            if start_piece is None:
                print("\t🚧 No piece found in starting square.")
                return (-1, [])
            most_varying_squares = VideoProcessing.create_variation_list(self.chessboards[-2], self.chessboards[-1])
            valid_position_found = False
            for i in range(len(most_varying_squares)):
                square_pos, square_variation = most_varying_squares[i]
                if start_piece.is_move_legal(self.chessboards[-1], start_pos, square_pos):
                    # We do no take if it is the same color
                    if (self.squares[start_pos[1]][start_pos[0]] == self.squares[square_pos[1]][square_pos[0]] and \
                        start_piece.get_color() == self.squares[square_pos[1]][square_pos[0]].get_piece().get_color() and
                        (start_piece.get_color() == 1 or start_piece.get_color() == -1)):
                        print("\t🚧 Cannot capture a piece of the same color.")
                        continue
                    valid_position_found = True
                    break
            if not valid_position_found:
                print("\t🚧 No legal move found for the capture")
                return (-1, [])
            if square_variation < 0.1:
                print(f"\tVariation score:\033[31m {square_variation:.2f} (too low)\033[0m, capture abandonned")
                return (-1, [])
            elif square_variation < 0.2:
                print(f"\tVariation score:\033[33m {square_variation:.2f} (low)\033[0m")
            else:
                print(f"\tVariation score:\033[32m {square_variation:.2f} (good)\033[0m")
            return (1, [[start_pos, square_pos]])
        # Simple movement
        elif nb_change == 2:
            # Check if we have a start and end pos
            if len(start_pos_list) != 1 or len(end_pos_list) != 1:
                print("\t🚧 A move must be a start and an end position.")
                return (-1, [])
            start_piece = self.squares[start_pos_list[0][1]][start_pos_list[0][0]].get_piece()
            if start_piece:
                # Verify if the move is legal.
                if start_piece.is_move_legal(self.chessboards[-1], start_pos_list[0], end_pos_list[0]):
                    return (1,[[start_pos_list[0], end_pos_list[0]]])
                else:
                    print(f"\t🚧 A piece with types : {start_piece.get_types()} tried to move from {Solver.pos_to_string(start_pos_list[0])} to {Solver.pos_to_string(end_pos_list[0])}")
            return (-1, [])
        # Enpassant
        elif nb_change == 3:
            # Enpassant needs two start pos and one end pos
            if len(start_pos_list) == 2 or len(end_pos_list) == 1:
                piece1 = self.squares[start_pos_list[0][1]][start_pos_list[0][0]].get_piece()
                piece2 = self.squares[end_pos_list[0][1]][end_pos_list[0][0]].get_piece()
                # Both pieces must exist and pos must be possible for enpassant
                if piece1 and piece2 and (end_pos_list[0][1] in [2, 5]) and start_pos_list[0][1] == start_pos_list[1][1]:
                    # Pieces color must be unknown or opposite
                    if piece1.get_color() == 0 or piece2.get_color() == 0 or piece1.get_color() == -piece2.get_color():
                        # Both pieces must be possible pawns
                        if (PAWN in piece1.get_types() or -PAWN in piece1.get_types()) and (PAWN in piece2.get_types() or -PAWN in piece2.get_types()):
                            for index, real_start in enumerate([start_pos_list[0], start_pos_list[1]]):
                                if real_start[0] != end_pos_list[0][0]:
                                    break
                            print("\t🔎 Enpassant detected.")
                            # Return the same end_pos for color determination
                            return(2, [[start_pos_list[index], end_pos_list[0]], [start_pos_list[(index+1)%2], end_pos_list[0]]])
            return (-1, [])
        # Castling
        elif nb_change == 4:
            output = self.is_castling(start_pos_list, end_pos_list)
            if output:
                print("\t🏰 Castling")
                return (3, output)
            # Manage double simple movement
            if len(start_pos_list) == 2 and len(end_pos_list) == 2:
                piece1 = self.squares[start_pos_list[0][1]][start_pos_list[0][0]].get_piece()
                piece2 = self.squares[start_pos_list[1][1]][start_pos_list[1][0]].get_piece()
                for i in range(len(end_pos_list)):
                    if piece1.is_move_legal(self.chessboards[-1], start_pos_list[0], end_pos_list[i]):
                        if piece2.is_move_legal(self.chessboards[-1], start_pos_list[1], end_pos_list[(i+1)%2]):
                            print("\t🔎 Double simple movement detected.")
                            return (4, [[start_pos_list[0], end_pos_list[i]], [start_pos_list[1], end_pos_list[(i+1)%2]]])
            return (-1, [])
        else:
            # Fix for video fix_5_3.mp4
            return (-2, nb_change)

    def create_piece(self, pos):
        """
        Create a piece at the given position and add it to the pieces list.
        Args:
            pos: The position of the piece.
        """
        self.pieces.append(self.squares[pos[0]][pos[1]].create_piece())

    def create_game_state(self, squares):
        game_state = [[0 for _ in range(8)] for _ in range(8)]
        unknown_pieces_count = 0
        for i in range(8):
            for j in range(8):
                piece = squares[i][j].get_piece()
                if piece:
                    if piece.get_current_type() == UNKNOWN_TYPE:
                        sign = 1
                        if piece.get_color() == BLACK:
                            sign = -1
                        game_state[i][j] = sign*UNKNOWN_TYPE
                    else:
                        game_state[i][j] = piece.get_current_type()
        for piece in self.pieces:
            if piece.get_current_type() == UNKNOWN_TYPE:
                unknown_pieces_count += 1
        return game_state, unknown_pieces_count

    def add_game_state(self, frame_number, game_state_matrix):
        """
        Add a new game state to the results.
        Args:    
            Frame number must be the "real" frame number
            game_state_matrix is a 8x8 matrix with -1, -2, ..., -7, 1, ..., 2, ..., 7 values representing the pieces on the board,
            see constants.py for pieces values
        """
        self.results["game_states"].append({
            "frame": frame_number,
            "gs" : game_state_matrix,
            "time": time.time() - t0
        })

    def save_results(self):
        if(EXPERIMENT_MODE):
            self.results = self.results | self.results_analysis
        output_file = f"{TASK_FOLDER}/{VIDEO_NAME}.json"
        print(f"🎉 Results saved in {output_file}")
        json.dump(self.results, open(output_file, "w"), indent=4, separators=(',', ': '))

    def pos_to_string(pos):
        """Return the string representation of a position."""
        return chr(ord('a') + pos[0]) + str(8 - pos[1])

    def string_to_pos(string):
        """Return the position from a string."""
        return (ord(string[0]) - ord('a'), 8 - int(string[1]))

    def is_castling(self, start_pos_list, end_pos_list):
        """
        Check if the movements may be derived from a castling.
        Args:
            start_pos_list: The list of starting positions.
            end_pos_list: The list of ending positions.
        Return:
            False if it is not a castling, else the movements for the castling.
        """
        # Castling is two movements combined
        if len(start_pos_list) != 2 or len(end_pos_list) != 2:
            return False
        # Two possible starting positions for the king
        for king_start in [[4,0], [4,7]]:
            # Try with the next pos if this one doesn't match
            if not in_list(king_start, start_pos_list):
                continue
            # Check if the possible king ends at the right position
            for i, king_end in enumerate([[2, king_start[1]], [6, king_start[1]]]):
                # Try with the next pos if this one doesn't match
                if not in_list(king_end, end_pos_list):
                    continue
                # Verify the rook positions
                rook_start = [0 if i == 0 else 7, king_start[1]]
                if not in_list(rook_start, start_pos_list):
                    return False
                rook_end = [3 if i == 0 else 5, king_start[1]]
                if not in_list(rook_end, end_pos_list):
                    return False
                # If the king pos and rook pos are correct it must be castling
                return [[king_start, king_end], [rook_start, rook_end]]
        return False
