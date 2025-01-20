from piece import Piece

class Square:
    """Class manage the data and logic of a specific square."""
    def __init__(self, pos):
        self.pos = pos
        self.piece = None

    def get_pos(self):
        """Returns the position in the grid of the square."""
        return self.pos

    def get_piece(self):
        """Returns the piece on the square."""
        return self.piece

    def remove_piece(self):
        """Removes the piece that was present on the square."""
        self.piece = None

    def add_piece(self, piece):
        """
        Adds a piece on the square and capture the previous one if any.
        Args:
            piece: The piece that arrives on the square
        """
        # If a new piece comes when another is present, the old one is captured.
        if self.piece != None:
            # Make the piece captured and the piece moving of different color.
            self.piece.set_captured(piece.get_color())
        self.piece = piece

    def create_piece(self, color=0):
        """Create a piece on the square."""
        self.piece = Piece(self, color)
        return self.piece

