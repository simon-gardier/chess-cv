class Chessboard:
    """Stores the data about the chessboard seen at frame frame_number"""
    def __init__(self, frame_number, stickers=None, corners_pos=None, board_frame=None, pieces_contours=None, pos_matrix=None, team_matrix=None, found=True):
        self.frame_number = frame_number
        self.stickers = stickers
        self.corners_pos = corners_pos
        self.board_frame = board_frame
        self.pieces_contours = pieces_contours
        self.pos_matrix = pos_matrix
        self.team_matrix = team_matrix
        self.found = found

    def get_frame_number(self):
        """Return the frame number of the chessboard"""
        return self.frame_number

    def get_stickers(self):
        """Return the stickers of the chessboard"""
        return self.stickers

    def get_corners_pos(self):
        """Return the corners position of the chessboard"""
        return self.corners_pos

    def get_board_frame(self):
        """Return the board frame of the chessboard"""
        return self.board_frame

    def get_pieces_contours(self):
        """Return the pieces contours of the chessboard"""
        return self.pieces_contours

    def get_pos_matrix(self):
        """Return the position matrix of the chessboard"""
        return self.pos_matrix

    def get_team_matrix(self):
        """Return the team matrix of the chessboard"""
        return self.team_matrix

    def is_identified(self):
        """Return true if the chessboard has been found in the frame or not"""
        return self.found
