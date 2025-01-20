import cv2

class Sticker:
    """Class manage the data for one sticker."""
    def __init__(self, pos, color):
        self.color = color
        self.pos = pos
        self.contour = []

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
        if new_contour is None:
            return
        self.contour = new_contour

    def update_contour(self, color_mask):
        """
        Find the contour based on the given mask.
        Args:
            color_mask: The mask to find the contour with.
        """
        # Find the matching patterns
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hulls = [cv2.convexHull(contour) for contour in contours]
        hulls = [hull for hull in hulls if cv2.contourArea(hull) < 4628] # 4628 = (1920/8)*(1080/8)/7
        # Only keep the biggest contour
        if hulls:
            self.set_contour(max(hulls, key=cv2.contourArea, default=None))
            # Compute the contour moments to find the contour centers
            moments = cv2.moments(self.get_contour())
            if moments["m00"] != 0:
                x = int(moments["m10"] / moments["m00"])
                y = int(moments["m01"] / moments["m00"])
                self.set_pos([x, y])