from constants import *

class Piece:
    """Class manage the logic and data for a specific piece."""
    # Shared variables to keep track of the number of pieces left overall and per color.
    all_types_left = {PAWN: 16, ROOK: 4, BISHOP: 4, KNIGHT: 4, KING: 2, QUEEN: 2} 
    white_types_left = {PAWN: 8, ROOK: 2, BISHOP: 2, KNIGHT: 2, KING: 1, QUEEN: 1}
    black_types_left = {PAWN: 8, ROOK: 2, BISHOP: 2, KNIGHT: 2, KING: 1, QUEEN: 1}

    def __init__(self, square, color=0):
        self.color = color              # -1 for black, 1 for white and 0 for unknown
        self.temp_color = 0             # 0 for no temp color, 2 and -2 for temp colors
        self.square = square
        self.current_type = 7
        self.types = WHITE_PIECES + BLACK_PIECES
        if color == -1:
            self.current_type = -7
            self.types = BLACK_PIECES
        if color == 1:
            self.current_type = 7
            self.types = WHITE_PIECES
        self.captured = False

    def get_types(self):
        """Returns the possible types for this piece."""
        return self.types
    
    def get_color(self):
        """Returns the color of the piece."""
        return self.color

    def move(self, new_square):
        """
        Move the piece to another square, update the possible types and decide the piece if possible.
        Args:
            new_square: The square where the piece is moving.
        """
        old_square = self.square
        dx, dy = self.get_movement(old_square, new_square)
        if self.current_type == 7:
            self.move_filter(dx, dy)
            self.decide_piece()
        #TODO if this piece could be a pawn and reach the opposite side for this pawn color, 
        # add rook, bishop, knight and queen to the types.
        old_square.remove_piece()
        new_square.add_piece(self)
        self.square = new_square
        # if self.current_type != 7:
        #     print(f"Piece is {self.current_type}")
        # else:
        #     print(f"Piece could be: {self.types}")
    
    def move_filter(self, dx, dy):
        """
        Update the possible types for the piece based on the movement and the remaining types available.
        Args:
            dx: The movement in the x direction.
            dy: The movement in the y direction.
        """
        if self.current_type != 7:
            return
        # Discard types that are not possible based on the movement.
        self.types = [piece_type for piece_type in self.check_type(dx, dy) if piece_type in self.types]

    def decide_piece(self):
        """
        Decide the piece type if possible based on the remaining types and the color.
        """
        # If there is nothing to decide, return.
        if self.current_type != 7 and self.color != 0:
            return
        # If we know the piece we know the color
        elif self.current_type != 7 and self.color == 0:
            if self.current_type > 0:
                self.color = 1
            else:
                self.color = -1
            return
        elif self.color != 0:
            # Keep types with the right color.
            self.types = [piece_type for piece_type in self.types if self.color*piece_type >= 1]
        # Remove types that are already given to other pieces.
        for piece_type in self.types.copy():
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

    def set_type(self, piece_type):
        # Already set
        if self.current_type == piece_type:
            return
        # Has to be set
        elif self.current_type == 7:
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
            print(f"Error: This piece already has a different type assigned.")

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
            self.decide_piece()
        else:
            print(f"Error: This piece already has a different color assigned.")
    
    def set_temp_color(self, temp_color):
        """
        Set a temporary color to the piece.
        Args:
            temp_color: The temporary color to be assigned to the piece.
        """
        if self.color == 0 :
            self.temp_color = temp_color

    def replace_temp_color(self, white):
        """
        If a temporary color was assigned to the piece, replace it with the right one.
        Args:
            white: The temp color that was used for white pieces.
        """
        if self.temp_color == 0:
            return
        if self.temp_color == white:
            self.set_color(1)
        else:
            self.set_color(-1)

    def set_captured(self, color_enemy):
        self.captured = True
        if self.current_type == 7:
            if self.color == 0:
                self.color = -color_enemy
            # King cannot be captured
            if KING in self.types:
                self.types.remove(KING)
            self.decide_piece()
        self.square = None

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
        """
        #TODO add position to check special moves: Roque + jump
        possible_types = []
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

        return possible_types
