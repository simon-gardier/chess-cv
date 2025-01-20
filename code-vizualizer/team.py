class Team:
    """Class manage the logic based on a whole team."""
    def __init__(self, color):
        # Initialization
        self.pieces = []
        self.color = color

    def add_piece(self, piece):
        """
        Adds a piece to the team.
        Args:
            piece: an instance of Piece.
        """
        self.pieces.append(piece)