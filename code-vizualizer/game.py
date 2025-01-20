import numpy as np
from chessboard import Chessboard
from piece import Piece
from square import Square
from team import Team
from utils import in_list
from constants import *

class Game:
    """Class that manage the game."""
    def __init__(self, chessboard):
        self.chessboard = chessboard
        self.pieces = []
        self.pos_matrices = []
        self.team_matrices = []
        self.gamestates = []
        #TODO implement backtracking for color. Give a value if we don't know. When we come back to a value we can
        # swap those two colors with the corresponding one. (for example give 2 and 3 then come back to swap 2 by 1 and 3 by -1)
        self.color_playing = 2          # 1 for white, -1 for black, 2 and -2 for unknown
        self.teams = [Team("white"), Team("black")]
        # Init squares
        self.squares = [[None for _ in range(8)] for _ in range(8)]
        for j in range(8):
            for i in range(8):
                pos = (i, j)
                self.squares[j][i] = Square((pos))

    def update_game(self):
        """
        Update all the game knowledge based on the new frame.
        """
        # Get the new position matrix
        self.pos_matrices.append(self.chessboard.get_pos_matrix())
        self.team_matrices.append(self.chessboard.get_team_matrix())
        # If this is the first matrix of position we can create the pieces.
        if len(self.pos_matrices) == 1:
            for j, line in enumerate(self.pos_matrices[0]):
                for i, square in enumerate(line):
                    if square == 1:
                        self.create_piece((j, i))
        # Otherwise we should update the model.
        else:
            # Find the movement done.
            move_type, moves = self.get_movements()
            # If there was an error we don't know anymore who is playing
            if move_type == -1:
                self.set_color_playing(2)
            # Manage the consequence of the movement in the model.
            if move_type > 0:
                for start_pos, end_pos in moves:
                    print(f"Piece moved from {self.pos_to_string(start_pos)} to {self.pos_to_string(end_pos)}")
                    piece = self.squares[start_pos[1]][start_pos[0]].get_piece()
                    if piece:
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
                                # Black are playing
                                self.set_color_playing(-1)
                                self.color_playing = -1
                            else:
                                piece.set_type(piece_type)
                                # White are playing
                                self.set_color_playing(1)
                        if abs(self.color_playing) == 1:
                            piece.set_color(self.color_playing)
                        else:
                            piece.set_temp_color(self.color_playing)
                        piece.move(self.squares[end_pos[1]][end_pos[0]])
                    else:
                        print(f"Could not find the piece at {self.pos_to_string(start_pos)}.")
                # Change the color playing
                #TODO check if it could be a roque in two part before changing the color.
                self.color_playing *= -1
                # print("\n\nNew move")
                for piece in self.pieces:
                    piece.decide_piece()
                    # print(f"Piece types: {piece.get_types()}")
                #TODO check if we can assign a type to a piece based on all pieces.

    def set_color_playing(self, color):
        """
        Set the color playing to the given color.
        """
        if abs(self.color_playing) == 1:
            if self.color_playing != color:
                print("Color problem.")
            return
        # Define which temporary color was the white one
        if color == 1:
            white = self.color_playing
        elif color == -1:
            white = -self.color_playing
        # Remove previous temp color if we have to reset temp color.
        else:
            for piece in self.pieces:
                piece.set_temp_color(0)
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
            1 for simple movement, 2 for enpassant, 3 for castling. And moves is a list of movements.
        """
        # print(self.pos_matrices[-1])
        if len(self.pos_matrices) < 2:
            print("This method needs at least two position matrix.")
            return (-1, [])
        # Get the difference
        matrix_dif = self.pos_matrices[-1] - self.pos_matrices[-2]
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
        # Capture
        if nb_change == 1:
            # Search for -1 because piece disapeared from the starting position
            if len(end_pos_list) == 1:
                print("Detection of a piece appearing.")
                return (-1, [])

            # Get the matrix
            variation_matrix = self.chessboard.get_variation_matrix()
            # Sort index of the matrix tuples in descending order relative to the variation value
            flat_indices = np.argsort(variation_matrix, axis=None)[::-1]
            # Transform the flat indices in matrix indices
            positions = [np.unravel_index(index, variation_matrix.shape) for index in flat_indices]
            # Creates a list of (varation, square_position) tuples, sorted by descending order relative to the variation value 
            sorted_values = [(variation_matrix[pos], pos) for pos in positions]
            # pprint.pprint(sorted_values[:10])
            print("Capture detected :")
            return (1, [[start_pos_list[0], sorted_values[0][1][::-1]]])
        # Simple movement
        elif nb_change == 2:
            # Check if we have a start and end pos
            if len(start_pos_list) != 1 or len(end_pos_list) != 1:
                print("Error in the detection of the movement.")
                return (-1, [])
            return (1,[[start_pos_list[0], end_pos_list[0]]])
        # Enpassant
        elif nb_change == 3:
            print("Enpassant detected.")
            #TODO
            return (2, [])
        # Castling
        elif nb_change == 4:
            output = self.is_castling(start_pos_list, end_pos_list)
            if output:
                print("Castling:")
                return (3, output)
            #TODO add something to manage double simple movement if we have time. Otherwise merge knowledge of both pieces.
            else:
                print("More than one movement detected.")
                return (-1, [])
        else:
            print("Error in the detection of the movement.")
            return (-1, [])

    def create_piece(self, pos):
        """
        Create a piece at the given position.
        """
        self.pieces.append(self.squares[pos[0]][pos[1]].create_piece())

    def add_gamestate(self):
        #TODO Compute the gamestate and store it if needed.
        cur_gamestate = []
        self.gamestates.append(cur_gamestate)

    def pos_to_string(self, pos):
        return chr(ord('a') + pos[0]) + str(8 - pos[1])

    def is_castling(self, start_pos_list, end_pos_list):
        """
        Check if the movements may be derived from a castling.
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