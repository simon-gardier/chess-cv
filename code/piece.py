from constants import *
import copy
class Piece:
    """Class manage the logic and data for a specific piece."""
    # Shared variables to keep track of the number of pieces left overall and per color.
    white_types_left = {PAWN: 8, ROOK: 2, BISHOP: 2, KNIGHT: 2, KING: 1, QUEEN: 1}
    black_types_left = {PAWN: 8, ROOK: 2, BISHOP: 2, KNIGHT: 2, KING: 1, QUEEN: 1}

    def __init__(self, square, color=0):
        self.square = square            # The square object that represent the square the piece is on.
        self.color = color              # -1 for black, 1 for white and 0 for unknown.
        self.temp_color = 0             # 0 for no temp color, 2 and -2 for temp colors.
        self.current_type = UNKNOWN_TYPE            # The type of the piece, -7 for unknown.
        self.types = WHITE_PIECES + BLACK_PIECES    # The possible types for the piece.
        self.captured = False           # Whether the piece is captured or not.
    
    def get_color(self):
        """Returns the color of the piece."""
        return self.color

    def get_temp_color(self):
        """Returns the temporary color of the piece."""
        return self.temp_color

    def get_types(self):
        """Returns the possible types for this piece."""
        return self.types
    
    def get_current_type(self):
        """Returns the current type of the piece."""
        return self.current_type
    
    def get_captured(self):
        """Returns if the piece is captured."""
        return self.captured

    def set_color(self, color):
        """
        Set the color of the piece.
        Args:
            color: The color to be assigned to the piece.
        """
        if self.color == color:
            return
        elif self.color == 0:
            self.color = color
            self.temp_color = 0
            self.decide_piece()
        else:
            print(f"\t🚧 Error: This piece already has a different color assigned.")
    
    def set_temp_color(self, temp_color):
        """
        Set a temporary color to the piece.
        Args:
            temp_color: The temporary color to be assigned to the piece.
        """
        if self.color == 0 :
            self.temp_color = temp_color
        else:
            self.temp_color = 0

    def set_type(self, piece_type):
        """
        Set the type of the piece.
        Args:
            piece_type: The type to be assigned to the piece.
        """
        if piece_type == ROOK or piece_type == -ROOK:
            print("\t🏰 One less rook")
        # Already set
        if self.current_type == piece_type:
            return
        # Has to be set
        elif self.current_type == UNKNOWN_TYPE:
            self.current_type = piece_type
            self.types = [piece_type]
            if piece_type > 0:
                types_left = self.white_types_left
                self.set_color(1)
            else:
                types_left = self.black_types_left
                self.set_color(-1)
            types_left[abs(piece_type)] -= 1
        # Already set but not the same
        else:
            print(f"\t🚧 Error: This piece already has a different type assigned.")

    def set_captured(self, color_enemy):
        """
        Set the piece as captured.
        Args:
            color_enemy: The color of the enemy pieces.
        """
        self.captured = True
        if self.current_type == UNKNOWN_TYPE:
            # Pieces must be of opposite colors to capture each other.
            if self.color == 0:
                self.color = -color_enemy
            # King cannot be captured
            if KING in self.types:
                self.types.remove(KING)
            self.decide_piece()
        self.square = None

    def move(self, new_square, frame_nb):
        """
        Move the piece to another square, update the possible types and decide the piece if possible.
        Args:
            new_square: The square where the piece is moving.
        """
        old_square = self.square
        dx, dy = self.get_movement(old_square, new_square)
        # Filter the possible types based on the movement.
        decided = None
        if self.current_type == UNKNOWN_TYPE:
            self.move_filter(dx, dy)
            decided = self.decide_piece()
        old_square.remove_piece()
        new_square.add_piece(self)
        self.square = new_square
    
    def move_filter(self, dx, dy):
        """
        Update the possible types for the piece based on the movement and the remaining types available.
        Args:
            dx: The movement in the x direction.
            dy: The movement in the y direction.
        """
        if self.current_type != UNKNOWN_TYPE:
            return
        # Discard types that are not possible based on the movement.
        self.types = [piece_type for piece_type in self.check_type(dx, dy) if piece_type in self.types]
        if len(self.types) == 0:
            print("\t🚧 Error: No possible types left for this piece.")

    def decide_piece(self):
        """
        Decide the piece type if possible based on the remaining types and the color.
        """
        # TODO fix the hot fix
        if self.captured:
            return None
        if len(self.types) == 1:
            return None
        # If there is nothing to decide, return.
        if self.current_type != UNKNOWN_TYPE and self.color != 0:
            return
        # If we know the piece we know the color
        elif self.current_type != UNKNOWN_TYPE and self.color == 0:
            if self.current_type > 0:
                self.color = 1
            else:
                self.color = -1
            return None
        elif self.color != 0:
            # Keep types with the right color.
            if self.color == 2:
                print(f"Wrong color for : {self.types}, {self.square.get_pos()}")
            self.types = [piece_type for piece_type in self.types if self.color*piece_type >= 1]
        # Remove types that are already given to other pieces.
        current_types = copy.deepcopy(self.types)
        for piece_type in current_types:
            # White piece type
            if piece_type > 0:
                types_left = self.white_types_left
            # Black piece type
            else:
                types_left = self.black_types_left
            # If this type is not possible anymore, remove it from the list.
            if types_left[abs(piece_type)] == 0:
                self.types.remove(piece_type)
        # If there is only one type left attribute it.
        if len(self.types) == 1:
            self.set_type(self.types[0])
            return self.types[0]
        
        if len(self.types) == 0:
            print("\t🚧 Error: No possible types left for this piece.")
        return None

    def replace_temp_color(self, white):
        """
        If a temporary color was assigned to the piece, replace it with the right one.
        Args:
            white: The temp color that was used for white pieces.
        """
        # If there is no temp color, return.
        if self.temp_color == 0:
            return
        # If the temp color is the white equivalent, set the color to white, otherwise set it to black.
        if self.temp_color == white:
            self.set_color(1)
        else:
            self.set_color(-1)

    def get_movement(self, old_square, new_square):
        """ 
        Gives the movement vector from one square to another. 
        Args:
            old_square: The square the piece was on.
            new_square: The square the piece ends on.
        Returns:
            vector: A tuple representing the movement in both direction.
        """
        old_x, old_y = old_square.get_pos()
        new_x, new_y = new_square.get_pos()
        return ((new_x - old_x), (new_y - old_y))

    def check_type(self, dx, dy):
        """
        Checks if a move is possible for a piece type and check for potential types.
        Args:
            move: The move to be checked as a tuple of start and end positions ((x1, y1), (x2, y2))
        Returns:
            possible_types: A list of possible types for the piece.
        """
        # Init the list
        possible_types = []
        # Foreach possible type, check if the move is possible. Append the type if it is.
        if (dx==0 and (abs(dy)==1 or abs(dy)==2)) or (abs(dx)==1 and abs(dy)==1):
            if dy > 0:
                possible_types.append(-PAWN)
            else:
                possible_types.append(PAWN)
        if dx == 0 or dy == 0:
            possible_types.append(ROOK)
            possible_types.append(-ROOK)
        if (abs(dx), abs(dy)) in [(1, 2), (2, 1)]:
            possible_types.append(KNIGHT)
            possible_types.append(-KNIGHT)
        if abs(dx) == abs(dy):
            possible_types.append(BISHOP)
            possible_types.append(-BISHOP)
        if dx == 0 or dy == 0 or abs(dx) == abs(dy):
            possible_types.append(QUEEN)
            possible_types.append(-QUEEN)
        if max(abs(dx), abs(dy)) == 1 or ((abs(dx) == 2) and abs(dy) == 0):
            possible_types.append(KING)
            possible_types.append(-KING)
        # Return the list of possible types
        return possible_types

    def is_move_legal(self, chessboard, start_pos, end_pos):
        """
        Check if a move is legal for all the remaining types of the piece object
        Only the intermidiate positions are checked
        Coordinates : (x, y), with (0, 0) in a8, (7, 7) in h1
        Args:
            chessboard: The chessboard object
            start_pos: The start position of the move
            end_pos: The end position of the move
        Returns:
            is_legal: True if the move is legal, False otherwise
        """
        for piece_type in self.types:
            if Piece.is_move_legal_for(piece_type, chessboard, start_pos, end_pos):
                return True

    def is_move_legal_for(piece_type, chessboard, start_pos, end_pos):
        """
        Check if a move is legal for a specific piece type, the types are defined in contants.py
        Only the intermidiate positions are checked
        Coordinates : (x, y), with (0, 0) in a8, (7, 7) in h1
        Args:
            piece_type: The type of the piece
            chessboard: The chessboard object
            start_pos: The start position of the move
            end_pos: The end position of the move
        Returns:
            is_legal: True if the move is legal, False otherwise
        """
        if(abs(piece_type) == PAWN):
            return Piece.is_pawn_move_legal(chessboard, piece_type > 0, start_pos, end_pos)
        if(abs(piece_type) == ROOK):
            return Piece.is_rook_move_legal(chessboard, piece_type > 0, start_pos, end_pos)
        if(abs(piece_type) == KNIGHT):   
            return Piece.is_knight_move_legal(chessboard, start_pos, end_pos)
        if(abs(piece_type) == BISHOP):
            return Piece.is_bishop_move_legal(chessboard, start_pos, end_pos)
        if(abs(piece_type) == KING):
            return Piece.is_king_move_legal(chessboard, start_pos, end_pos)
        if(abs(piece_type) == QUEEN):
            return Piece.is_queen_move_legal(chessboard, start_pos, end_pos)
        return False

    def is_pawn_move_legal(chessboard, is_white, start_pos, end_pos):
        """
        Check if a move is legal for a pawn
        Coordinates : (x, y), with (0, 0) in a8, (7, 7) in h1
        Args:
            chessboard: The chessboard object
            is_white: The type of the piece
            start_pos: The start position of the move
            end_pos: The end position of the move
        Returns:
            is_legal: True if the move is legal, False otherwise
        """
        board = chessboard.get_pos_matrix()
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        dx, dy = end_x - start_x, end_y - start_y
        if is_white:
            # Pawn double step
            if start_y == 6 and end_y == 4 and start_x == end_x and board[5][start_x] == 0:
                return True
            # Pawn step
            if start_x == end_x and start_y - end_y == 1:
                return True
            if abs(dx) == 1 and dy == -1:
                return True
        else:
            if start_y == 1 and end_y == 3 and start_x == end_x and board[2][start_x] == 0:
                return True
            if start_x == end_x and end_y - start_y == 1:
                return True
            if abs(dx) == 1 and dy == 1:
                return True
        return False

    def is_rook_move_legal(chessboard, is_white, start_pos, end_pos):
        board = chessboard.get_pos_matrix()
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        if(start_x != end_x and start_y != end_y):
            return False
        dx, dy = end_x - start_x, end_y - start_y
        dx, dy = dx//abs(dx) if dx else 0, dy//abs(dy) if dy else 0
        # Deal with castling, not complete (e.g. rook could have move than come back to its starting position, piece jumped is not the king...)
        # If the white rook is on the first line at their starting position
        if is_white and dy == 0 and start_y == 7 and (start_x == 0 or start_x == 7):
            if start_x == 0 and end_x == 3 and board[7][1] == 0:
                return True
            if start_x == 7 and end_x == 5:
                return True
        # If the black rook is on the first line at their starting position
        if not is_white and dy == 0 and start_y == 0 and (start_x == 0 or start_x == 7):
            if start_x == 0 and end_x == 3 and board[0][1] == 0:
                return True
            if start_x == 7 and end_x == 5:
                return True
        return Piece.is_reachable(chessboard, start_pos, start_pos, end_pos, dx, dy)

    def is_knight_move_legal(chessboard, start_pos, end_pos):
        board = chessboard.get_pos_matrix()
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        dx, dy = end_x - start_x, end_y - start_y
        return (abs(dx), abs(dy)) in [(1, 2), (2, 1)]

    def is_bishop_move_legal(chessboard, start_pos, end_pos):
        board = chessboard.get_pos_matrix()
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        dx, dy = end_x - start_x, end_y - start_y
        if abs(dx) == abs(-dy):
            dx, dy = dx//abs(dx) if dx else 0, dy//abs(dy) if dy else 0
            return Piece.is_reachable(chessboard, start_pos, start_pos, end_pos, dx, dy)
        return False

    def is_queen_move_legal(chessboard, start_pos, end_pos):
        board = chessboard.get_pos_matrix()
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        dx, dy = end_x - start_x, end_y - start_y
        if dx == 0 or dy == 0 or abs(dx) == abs(dy):
            dx, dy = dx//abs(dx) if dx else 0, dy//abs(dy) if dy else 0
            return Piece.is_reachable(chessboard, start_pos, start_pos, end_pos, dx, dy)
        return False

    def is_king_move_legal(chessboard, start_pos, end_pos):
        board = chessboard.get_pos_matrix()
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        dx, dy = end_x - start_x, end_y - start_y
        if max(abs(dx), abs(dy)) == 1:
            return True
        return False

    def is_reachable(chessboard, start_pos, curr_pos, end_pos, dx, dy):
        """
        Check if a position is reachable from another position. Only intermidiate positions are checked
        Args:F
            chessboard: The chessboard object
            curr_pos: The current position
            end_pos: The end position
            dx: The movement in the x direction (-1, 0, 1)
            dy: The movement in the y direction (-1, 0, 1)
        """
        curr_pos_x, curr_pos_y = curr_pos
        board = chessboard.get_pos_matrix()
        if curr_pos_x < 0 or curr_pos_x > 7 or curr_pos_y < 0 or curr_pos_y > 7:
            print(f"\t🚧 Error: Position  is out of the board.")
            return False
        if (curr_pos==end_pos).all():
            return True
        if board[curr_pos_y][curr_pos_x] != 0 and not (curr_pos==start_pos).all():
            return False
        return Piece.is_reachable(chessboard, start_pos, (curr_pos_x+dx, curr_pos_y+dy), end_pos, dx, dy)
