class Sticker:
    """Class manage the data for one sticker."""
    def __init__(self, pos, color):
        self.color = color
        self.pos = pos
        self.contour = []

    def is_empty(self):
        return len(self.contour) == 0

    def get_pos(self):
        """Returns a tuple representing the position in the image of the sticker."""
        return self.pos

    def get_contour(self):
        """Returns a tuple representing the contour of the sticker."""
        return self.contour

    def set_pos(self, new_pos):
        """
        Set the new position of the sticker.
        Args:
            new_pos: A tuple representing the new position in the image of the sticker.
        """
        self.pos = (int(new_pos[0]), int(new_pos[1]))

    def set_contour(self, new_contour):
        """
        Set the new contour of the sticker.
        Args:
            new_contour: tuple[Sequence[MatLike], MatLike] from cv2.findContours().
        """
        self.contour = new_contour
