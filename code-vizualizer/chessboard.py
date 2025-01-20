import numpy as np
from constants import *
from utils import *
from line_profiler import profile
import pprint

from square import Square
from sticker import Sticker
from sklearn.cluster import KMeans

class Chessboard:
    """Class manage the data and logic of the chessboard."""
    def __init__(self):
        # Init stickers
        self.stickers = [Sticker((FRAME_SIZE[0], FRAME_SIZE[1]/2), "blue"),
                         Sticker((0, FRAME_SIZE[1]/2), "pink")]
        self.sticker_dist = FRAME_SIZE[0]
        self.square_size = int(GAMESTATE_WINDOW_SIZE[0] / 8)
        # Init corners
        #TODO keep count of the nb of pieces on the board. If it increase discard the frame, it must be fingers.
        self.corner_pos = []
        for i in range(4):
            self.corner_pos.append((int(i%2), i//2))
        self.prev_cropped_frame = None
        self.curr_cropped_frame = None
        self.pieces_contours = None
        self.variation_matrix = np.zeros((8, 8), dtype=np.float32)
        self.pos_matrix = np.zeros((8, 8), dtype=np.float32)
        self.team_matrix = np.zeros((8, 8), dtype=np.float32)

    
    def get_corner_pos(self):
        """Returns the list of all corners positions"""
        return self.corner_pos


    def set_corner_pos(self, index, pos):
        """Set the corner position"""
        self.corner_pos[index] = (int(pos[0]), int(pos[1]))


    def get_stickers(self):
        """Returns a list of both sticker"""
        return self.stickers


    @profile
    def update_corners(self, frame):
        """
        Finds the stickers and corners and update the data.
        Args:
            frame: The frame to look at to find the corners.
        Returns:
            corners_found: True if it could find the corners.
        """
        self.identify_stickers(frame)
        return self.find_corners(frame)

    def identify_stickers(self, frame):
        """
        Identify the stickers on the frame.
        Args:
            frame: A frame with no annotation and BGR colors.
        """
        # Mask only the blue and pink pixels in the frame
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Equalize the saturation channel
        hsv_frame[:, :, 1] = cv2.equalizeHist(hsv_frame[:, :, 1])
        lower_blue = np.array(LOWER_BLUE_HSV)
        upper_blue = np.array(UPPER_BLUE_HSV)
        blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        lower_pink = np.array(LOWER_PINK_HSV)
        upper_pink = np.array(UPPER_PINK_HSV)
        pink_mask = cv2.inRange(hsv_frame, lower_pink, upper_pink)

        # Find the matching patterns
        self.stickers[BLUE].update_contour(blue_mask)
        self.stickers[PINK].update_contour(pink_mask)
        
        # Compute the distance between the two points and the orientation of the board
        self.sticker_dist = np.linalg.norm(np.array(self.stickers[BLUE].get_pos()) - np.array(self.stickers[PINK].get_pos()))


    @profile
    def find_corners(self, frame):
        """
        Find the corners using black squares detection.
        Args:
            frame: The frame to look at to find the corners.
        Returns:
            corners_found: True if it could find the corners.
        """
        # Find the black squares
        black_squares = self.find_black_squares(frame)
        # This method needs 32 squares to work.
        if len(black_squares) != 32:
            return False
        # Compute the center of every square
        centers = []
        for i, square in enumerate(black_squares):
            moments = cv2.moments(square)
            if moments["m00"] != 0:
                x = int(moments["m10"] / moments["m00"])
                y = int(moments["m01"] / moments["m00"])
                centers.append((x, y, i))
        # Candidate for a1 and h8 corners
        #TODO optimize this or good enough ?
        top = max(centers, key=lambda p: p[1])
        right = max(centers, key=lambda p: p[0])
        bot = min(centers, key=lambda p: p[1])
        left = min(centers, key=lambda p: p[0])
        top_right = max(centers, key=lambda p: (p[0] + p[1]))
        bot_right = max(centers, key=lambda p: (p[0] - p[1]))
        bot_left = min(centers, key=lambda p: (p[0] + p[1]))
        top_left = min(centers, key=lambda p: (p[0] - p[1]))
        # Find the pair that define the best line for which 8 corners can get to the fastest.
        min_sum = float("inf")
        pair = []
        for p1, p2 in [(top, bot), (top_left, bot_right), (left, right), (bot_left, top_right)]:
            cur_sum = self.sum_dist_line(p1, p2, centers)
            if cur_sum < min_sum:
                min_sum = cur_sum
                pair = [p1, p2]
        # Define variable based on the winning pair
        centers_pair = [(pair[0][0],pair[0][1]), (pair[1][0],pair[1][1])]
        squares = [black_squares[pair[0][2]], black_squares[pair[1][2]]]
        # Get all the points from the two squares in one list.
        points = np.concatenate(squares).reshape(-1, 2)
        # Find which points are not on the axis a1-h8.
        to_determine = self.get_furthest(pair[0], pair[1], points)
        # Determine the grid position of those.
        px_points, grid_pos = self.assign_border_corner(centers_pair, to_determine)
        # If one grid pos had no point assigned
        if len(px_points) == 0:
            return False
        # Add the a1 and h8 centers for the homography
        grid_points = np.concatenate((grid_pos, [[0.5,0.5], [7.5,7.5]])).reshape(-1, 1, 2)
        px_points = np.concatenate((px_points, centers_pair)).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(grid_points, px_points)
        # Create the list of the expected center of all black squares.
        grid_points = []
        for i in range(8):
            for j in range(8):
                if (i + j)%2 != 0:
                    continue
                grid_points.append([i+0.5, j+0.5])
        grid_points = np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)
        # Find the place in the image corresponding to those points.
        estimated_px_points = cv2.perspectiveTransform(grid_points, homography)
        ordered_centers = self.adjust_points(estimated_px_points, centers)
        ordered_centers = np.array(ordered_centers, dtype=np.float32).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(grid_points, ordered_centers)
        # Create the list of the corners grid pos.
        grid_points = np.array([[0, 0], [0, 8], [8, 8], [8, 0]], dtype=np.float32).reshape(-1, 1, 2)
        # Find the place in the image corresponding to those points.
        estimated_px_points = cv2.perspectiveTransform(grid_points, homography)
        # Assign the position to the corners.
        #TODO use the pair defined above to find a1 and h8
        self.assign_corners(estimated_px_points)
        return len(estimated_px_points) != 0


    @profile
    def find_black_squares(self, frame):
        """
        Find the black squares on the given frame.
        Args:
            frame: BGR frame to look at.
        Returns:
            black_squares: The list of 32 contours with 4 vertices corresponding to black squares.
        """
        # Gray frame
        cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Binary + inverse
        _, cur_frame = cv2.threshold(cur_frame, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Isolate squares from each other with an opening.
        cur_frame = cv2.morphologyEx(cur_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        #TODO Make it more robust based on the pieces used
        # Get (external) contours
        contours, _ = cv2.findContours(cur_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Keep the contour that looks like black squares
        black_squares = []
        area_list = []
        for i, contour in enumerate(contours):
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            _, _, width, height = cv2.boundingRect(hull)
            # Extent should be at least 0.5
            if hull_area/(width*height) < 0.48:
                continue
            # Width should always be larger (perspective of square)
            if width/height < 0.95:
                continue
            # Discard small objects
            if hull_area < 200:
                continue
            # Discard large objects (value here is theoretical max size of a square)
            if hull_area > (1920/8)*(1080/8):
                continue
            # Compute an approximation, it should be composed of 4 vertices (5 in case of bad approx)
            poly_approx = cv2.approxPolyDP(hull, 0.02*cv2.arcLength(hull, True), True)
            if len(poly_approx) < 4 and len(poly_approx) > 5:
                continue
            black_squares.append(poly_approx)
            area_list.append(hull_area)
        # Compute the list of close contours.
        squares_index_list = self.group_squares(black_squares)
        squares = []
        for squares_index in squares_index_list:
            if len(squares_index) < 32:
                continue
            cur_group = [contour for i, contour in enumerate(black_squares) if i in squares_index]
            cur_group = [square for i, square in enumerate(cur_group) if self.not_inside(square, cur_group, i)]
            if len(cur_group) == 32:
                squares = cur_group
                break
        return squares

    def assign_corners(self, points):
        """
        Assign the given points to the corners based on their position and the stickers positions.
        Args:
            points: The list of points to assign to corners.
        """
        # Get the centroid of the estimated corners.
        centroid = np.mean(np.array([point.ravel() for point in points]), axis=0)
        # Get the vector defined by the sticker (pink -> blue)
        h_line_vec = np.array(self.stickers[1].get_pos()) - np.array(self.stickers[0].get_pos())
        # Vector of the perpendicular
        v_line_vec = np.array([-h_line_vec[1], h_line_vec[0]])
        for point in points:
            # Vector from the centroid to the point
            rel_vec = point - centroid
            # Compute if the vector are pointing in the same direction or not.
            h_line = np.dot(rel_vec, h_line_vec)
            v_line = np.dot(rel_vec, v_line_vec)
            # Considering left = pink and right = blue:
            # Top left (h8)
            if h_line >= 0 and v_line >= 0:
                self.set_corner_pos(2, point.ravel())
            # Bottom left (a8)
            elif h_line >= 0 and v_line < 0:
                self.set_corner_pos(1, point.ravel())
            # Bottom right (a1)
            elif h_line < 0 and v_line < 0:
                self.set_corner_pos(0, point.ravel())
            # Top right (h1)
            else:
                self.set_corner_pos(3, point.ravel())


    def sum_dist_line(self, p1, p2, points, number=8):
        """
        Return the sum of the distance of the closest points to the line defined by p1 and p2.
        Args:
            p1: Point represented by (x, y, i) used to define the line.
            p2: Point represented by (x, y, i) used to define the line.
            points: Points represented by (x, y, i).
            number: The number of points we should evaluate for the sum.
        Returns:
            sum_dist = The sum of the distances.
        """
        # Define the line.
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = p2[0] * p1[1] - p1[0] * p2[1]
        points =  np.array(points)
        # Compute the distance from each point to the line.
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c) / np.sqrt(a**2 + b**2)
        # Sort the distances.
        distances = sorted(distances)
        # Return the sum of the 8 smallest distances.
        return np.sum(distances[:number])


    def get_furthest(self, p1, p2, points, number=4):
        """
        Find the points the further away from the line.
        Args:
            p1: Point represented by (x, y, i) used to define the line.
            p2: Point represented by (x, y, i) used to define the line.
            points: Points represented by (x, y, i).
            number: The number of points we should return at the end.
        Returns:
            furthest_points: The np.array of points the further away from the line.
        """
        # Define the line.
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = p2[0] * p1[1] - p1[0] * p2[1]
        points =  np.array(points)
        # Compute the distance from each point to the line.
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c) / np.sqrt(a**2 + b**2)
        # Sort the distances and the points array at the same time based on distance.
        sorted_points = sorted(zip(distances, points), key=lambda pair: pair[0], reverse=True)
        # Create a list with the fitting points
        furthest_points = [point for _, point in sorted_points[:number]]
        # Return the list in a np.array
        return np.array(furthest_points)


    def assign_border_corner(self, line, points):
        """
        Assign the given points to four position (0,1), (7,8) and their reverse based on their 
        position to the line.
        Args:
            line: A list of the two points used to define the line.
            points: The list of points for which it should determine the position.
        Returns:
            (pos_list, grid_pos): A tuple of list that gives the position on the frame and the grid.
        """
        # Get the centroid of the points.
        centroid = np.mean(np.array([point.ravel() for point in points]), axis=0)
        # Get the vector defined by the center of corner squares
        h_line_vec = np.array(line[1]) - np.array(line[0])
        # Vector of the perpendicular
        v_line_vec = np.array([-h_line_vec[1], h_line_vec[0]])
        pos_list = 4*[0]
        grid_pos = 4*[0]
        for point in points:
            # Vector from the centroid to the point
            rel_vec = point - centroid
            # Compute if the vector are pointing in the same direction or not.
            h_line = np.dot(rel_vec, h_line_vec)
            v_line = np.dot(rel_vec, v_line_vec)
            # Considering left = pink and right = blue:
            # Top left (7, 8)
            if h_line >= 0 and v_line >= 0:
                pos_list[2] = point.ravel()
                grid_pos[2] = [7, 8]
            # Bottom left (8, 7)
            elif h_line >= 0 and v_line < 0:
                pos_list[3] = point.ravel()
                grid_pos[3] = [8, 7]
            # Bottom right (1, 0)
            elif h_line < 0 and v_line < 0:
                pos_list[1] = point.ravel()
                grid_pos[1] = [1, 0]
            # Top right (0, 1)
            else:
                pos_list[0] = point.ravel()
                grid_pos[0] = [0, 1]
        # Verify that all points got a different grid position assigned
        if not all(isinstance(pos, np.ndarray) for pos in pos_list):
            return [], []
        return pos_list, grid_pos


    def adjust_points(self, estimation, real):
        """
        For each point in the first list, find the closest point from the second list.
        Args:
            estimation: The list of point to estimate.
            real: The list of real positions.
        Returns:
            new_real: The list of the position adjusted in the same order as the first list.
        """
        # Keep only the position of each point
        real = [[target[0], target[1]] for target in real]
        real_ordered = []
        for guess in estimation:
            min_dist = float("inf")
            target_index = None
            for i, target in enumerate(real):
                cur_dist = np.linalg.norm(guess - target)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    target_index = i
            real_ordered.append(real[target_index])
        return real_ordered


    def update_board(self, base_frame):
        """
        Update the position matrix and the variation matrix
        """
        self.prev_cropped_frame = self.curr_cropped_frame
        self.curr_cropped_frame = self.crop_board(base_frame)
        self.pieces_contours = self.find_pieces(self.curr_cropped_frame)
        #TODO don't compute the rest if we only need that when there is a move (still need to save curr_cropped_frame in self.prev_cropped_frame at the end)
        self.team_matrix = self.assign_contours_team(self.curr_cropped_frame, self.pieces_contours)
        self.pos_matrix = self.create_pos_matrix()
        self.variation_matrix = self.create_variation_matrix(self.curr_cropped_frame, self.pieces_contours)


    def create_variation_matrix(self, board_frame, pieces_contours):
        """
        Update the variation matrix based on the content of the board square from the current and previous frame
        """
        variation_matrix = np.zeros((8, 8), dtype=np.float32)
        if self.prev_cropped_frame is None:
            return variation_matrix

        # Current pieces contours
        curr_frame_bgr = board_frame
        curr_team_matrix = self.team_matrix

        # Previous pieces contours
        prev_pieces_contours = self.find_pieces(self.prev_cropped_frame)
        prev_team_matrix = self.assign_contours_team(self.prev_cropped_frame, prev_pieces_contours)

        # Fill a chessboard matrix with the previous frame data. 
        # Each value is a tuple with the exact coordinates of the center of the contours if the square contains one.
        prev_contours_map = {}
        for i in range(len(prev_pieces_contours)):
            moments = cv2.moments(prev_pieces_contours[i])
            if moments["m00"] == 0:
                continue
            x = moments["m10"] / moments["m00"]
            y = moments["m01"] / moments["m00"]
            contour_pixel_position = np.array([x, y])
            x /= self.square_size
            y /= self.square_size
            prev_contours_map[(int(y), int(x))] = contour_pixel_position
        
        # Fill the variation matrix with values between 0 and 1 represeting the variation in the squares.
        # The variation is at most composed of : team variation, piece contour center position variation, piece contour mean color variation
        for i in range(len(pieces_contours)):
            moments = cv2.moments(pieces_contours[i])
            if moments["m00"] == 0:
                continue
            x = moments["m10"] / moments["m00"]
            y = moments["m01"] / moments["m00"]
            contour_pixel_position = np.array([x, y])
            x /= self.square_size
            y /= self.square_size
            mask = np.zeros(board_frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [pieces_contours[i]], contourIdx=-1, color=255, thickness=cv2.FILLED)
            curr_frame_bgr_content = curr_frame_bgr[mask == 255]
            prev_frame_bgr_content = self.prev_cropped_frame[mask == 255]
            mean_bgr_curr = np.mean(curr_frame_bgr_content, axis=0)
            mean_bgr_prev = np.mean(prev_frame_bgr_content, axis=0)
            bgr_diff = np.abs(mean_bgr_curr - mean_bgr_prev)
            nb_parameters = 0
            diff_norm = 0

            # Color variation
            diff_norm += (np.linalg.norm(bgr_diff) / (np.sqrt(3) * 255))
            nb_parameters += 1

            # Team variation
            # If the new team assignement in this square is different from the previous 
            if (prev_team_matrix[int(y)][int(x)] != curr_team_matrix[int(y)][int(x)]) and prev_team_matrix[int(y)][int(x)] != 0:
                diff_norm += 0.5
            # Outside the if, we want to decrease the variation of the squares that didnt change from one team to another, the team variation is considered as 0.
            nb_parameters += 1

            # Contour center position variation in square
            prev_contour_center = prev_contours_map.get((int(y), int(x)), None)
            if isinstance(prev_contour_center, np.ndarray):
                square_top_left = np.array([self.square_size * int(x), self.square_size * int(y)])
                contour_relative_center = contour_pixel_position - square_top_left
                prev_contour_relative_center = prev_contour_center - square_top_left
                center_variation = (np.linalg.norm(np.array(contour_relative_center) - np.array(prev_contour_relative_center)) / (np.sqrt(self.square_size ** 2 + self.square_size ** 2))) * 10
                center_variation = np.clip(center_variation, 0, 1)
                # print(f"Square : {square_top_left}, curr : {contour_relative_center} {contour_pixel_position}, prev : {prev_contour_relative_center}, variation : {center_variation}")
                diff_norm += center_variation
                nb_parameters += 1
            variation_matrix[int(y)][int(x)] = diff_norm / nb_parameters

        return variation_matrix


    def assign_contours_team(self, board_frame, pieces_contours):
        """
        Returns a list with, for each contour, the team appartenance (1 or 2)
        """
        board_frame_hsv = cv2.cvtColor(board_frame, cv2.COLOR_BGR2HSV)
        # Cluster the pieces contours
        contours_mean_hues = []
        for contour in pieces_contours:
            mask = np.zeros(board_frame_hsv.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED)
            # Calculte mean hue for the piece
            mean_hsv = cv2.mean(board_frame_hsv[:, :], mask=mask)
            mean_hue = mean_hsv[0]
            mean_sat = mean_hsv[1]
            mean_val = mean_hsv[2]
            contours_mean_hues.append([mean_sat, mean_val])

        contours_mean_hues = np.array(contours_mean_hues)
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(contours_mean_hues)
        pieces_cluster_assignments = kmeans.labels_

        cluster_means = [0, 0]
        cluster_sizes = np.bincount(pieces_cluster_assignments)
        for i in range(len(pieces_contours)):
            cluster_means[pieces_cluster_assignments[i]] += contours_mean_hues[i, 0]

        cluster_means[0] /= cluster_sizes[0]
        cluster_means[1] /= cluster_sizes[1]
        if cluster_means[0] > cluster_means[1]:
            pieces_cluster_assignments = 1 - pieces_cluster_assignments

        team_matrix = np.zeros((8, 8), dtype=int)
        for i in range(len(pieces_contours)):
            moments = cv2.moments(pieces_contours[i])
            if moments["m00"] != 0:
                # Get the center of the piece
                x = moments["m10"] / moments["m00"]
                y = moments["m01"] / moments["m00"]
                # Get the relative position of the piece in the chessboard
                x /= self.square_size
                y /= self.square_size
                # Set the value to 1 to indicate a piece
                team_matrix[int(y)][int(x)] = pieces_cluster_assignments[i] + 1
        return team_matrix

    def create_pos_matrix(self):
        """
        Update the position matrix based on the piece contours.
        0,0 is the top left corner of the board (i.e a8)
        """
        # Init the matrix
        pos_matrix = np.zeros((8, 8), dtype=int)
        for i in range(len(self.pieces_contours)):
            moments = cv2.moments(self.pieces_contours[i])
            if moments["m00"] != 0:
                # Get the center of the piece
                x = moments["m10"] / moments["m00"]
                y = moments["m01"] / moments["m00"]
                # Get the relative position of the piece in the chessboard
                x /= self.square_size
                y /= self.square_size
                # Set the value to 1 to indicate a piece
                pos_matrix[int(y)][int(x)] = 1
        return pos_matrix


    def crop_board(self, base_frame):
        """Return a square frame that contains only the chessboard area."""
        points = np.array(self.get_corner_pos(), dtype=np.float32)
        destination_points = np.array([
            [0, GAMESTATE_WINDOW_SIZE[H] - 1],
            [0, 0],
            [GAMESTATE_WINDOW_SIZE[W] - 1, 0],
            [GAMESTATE_WINDOW_SIZE[W] - 1, GAMESTATE_WINDOW_SIZE[H] - 1]
        ], dtype=np.float32)
        perspective_matrix = cv2.getPerspectiveTransform(points, destination_points)
        squared_image = cv2.warpPerspective(base_frame, perspective_matrix, GAMESTATE_WINDOW_SIZE)
        return squared_image


    def find_pieces(self, cb_frame):
        """Find the piece contours on the chessboard."""
        #TODO: Correct glitch on the border of the board
        #TODO instead of dividing the task by color, we could use the adaptative threshold and then apply the mask
        #on this threshold to remove the square borders.
        # Gray frame
        test_frame = cv2.cvtColor(cb_frame, cv2.COLOR_BGR2GRAY)
        test_frame = cv2.medianBlur(test_frame, 5)
        # Create a black frame with the dimension of cb_frame
        mask_list = []
        mask_list.append(np.zeros_like(test_frame))
        mask_list.append(np.zeros_like(test_frame))
        # Margin is used to avoid border slighly wrong
        margin = 5
        width = GAMESTATE_WINDOW_SIZE[0]//8
        # Go through all squares from top (8) to bottom (1) and left (a) to right (h)
        for i in range(9):
            for j in range(9):
                sqr_color = (i+j)%2
                # Get the square position
                sqr_top_left = (i*width + margin, j*width + margin)
                sqr_bot_right = ((i+1)*width - margin, (j+1)*width - margin)
                # Add the square to the mask
                cv2.rectangle(mask_list[sqr_color], sqr_top_left, sqr_bot_right, 255, -1)
        for i in range(2):
            # Add the equivalent of thickness/2 black pixels on the border of the chessboard.
            cv2.rectangle(mask_list[i], (0, 0), (GAMESTATE_WINDOW_SIZE), 0, 16)

        # Apply the threshold on the frame masked
        test_list = []
        thresh_list = []
        USE_OTSU = True
        for i in range(2):
            masked_frame = test_frame
            # Get rid of pieces on square borders that would be used in inpaint.
            if i == 0:
                masked_frame = cv2.dilate(masked_frame, np.ones((3, 3), np.uint8))
            else:
                masked_frame = cv2.erode(masked_frame, np.ones((3, 3), np.uint8))
            # Apply an adaptive threshold to find the pieces borders
            thresh_frame = cv2.adaptiveThreshold(masked_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 9, 4)
            thresh_frame = cv2.bitwise_and(thresh_frame, mask_list[i])
            # Apply the mask on the threshold to remove noise.
            thresh_frame = cv2.bitwise_and(thresh_frame, mask_list[i])
            # Add a border to prevent glitch with closing
            border_size = 50
            thresh_frame = cv2.copyMakeBorder(thresh_frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
            # Use opening to remove the noise.
            thresh_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            # Use closing to fill the holes in the piece borders.
            thresh_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), iterations=1)
            # Pieces on white are detected too small, and the opposite for black.
            if i == 0:
                thresh_frame = cv2.dilate(thresh_frame, np.ones((3, 3), np.uint8))
            else:
                thresh_frame = cv2.erode(thresh_frame, np.ones((7, 7), np.uint8))
            # Come back to the previous size
            thresh_frame = thresh_frame[border_size:-border_size, border_size:-border_size]
            test_list.append(masked_frame)
            thresh_list.append(thresh_frame)
        # Find contours per color to avoid merging
        contours = []
        for i in range(2):
            contours.append(cv2.findContours(thresh_list[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        contours = contours[0] + contours[1]
        # Get the convex hull for each contour
        hulls = [cv2.convexHull(contour) for contour in contours]
        return hulls

    def get_variation_matrix(self):
        """
        Returns a matrix representing the variation of color in the area of the contours from the previous frame compared to the new one, scaled between 0 and 1.
        """
        return self.variation_matrix


    def get_team_matrix(self):
        """
        Returns a matrix representing the grouping of the pieces on the board.
        """
        return self.team_matrix


    def get_pos_matrix(self):
        """
        Returns the position matrix of the pieces on the chessboard.
        """
        return self.pos_matrix


    def get_cropped_frame(self):
        """
        Returns the cropped, squared, frame of the board. The attribute is updated by chessboard.update_board.
        """
        return self.curr_cropped_frame.copy()


    def group_squares(self, contours):
        groups = []
        grouped = []
        for i in range(len(contours)):
            if i in grouped:
                continue
            else:
                cur_group = []
                cur_group.append(i)
                grouped.append(i)
                groups.append(cur_group)
                self.search_neighbor(i, contours, cur_group, grouped)
        return groups

    def search_neighbor(self, cur_index, contours, cur_group, grouped):
        for i, contour in enumerate(contours):
            if i in grouped:
                continue
            for corner in contours[cur_index]:
                for neighbor_corner in contour:
                    # Compute manahttan distance
                    man_dist = corner - neighbor_corner
                    if abs(man_dist[0][0]) + abs(man_dist[0][1]) < MAX_BORDER_DIST:
                        cur_group.append(i)
                        grouped.append(i)
                        self.search_neighbor(i, contours, cur_group, grouped)
                        break
                else:
                    continue
                break


    def not_inside(self, square, squares, index):
        for i, other_square in enumerate(squares):
            if i == index:
                continue
            for corner in square:
                if cv2.pointPolygonTest(other_square, (int(corner[0][0]), int(corner[0][1])), False) > 0:
                    return False
        return True
    
    
    def frame_black_square(self, frame):
        # Gray frame
        cur_frame = frame.copy()
        cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        # cur_frame = cv2.equalizeHist(cur_frame)
        # Binary + inverse
        _, cur_frame = cv2.threshold(cur_frame, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Isolate squares from each other with an opening.
        cur_frame = cv2.morphologyEx(cur_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        #TODO Make it more robust based on the pieces used
        # Get (external) contours
        contours, _ = cv2.findContours(cur_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cur_frame, contours, -1, (255, 0, 0), 2)
        # Keep the contour that looks like black squares
        black_squares = []
        area_list = []
        for i, contour in enumerate(contours):
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            _, _, width, height = cv2.boundingRect(hull)
            # Extent should be at least 0.5
            if hull_area/(width*height) < 0.48:
                continue
            # Width should always be larger (perspective of square)
            if width/height < 0.95:
                continue
            # Discard small objects
            if hull_area < 200:
                continue
            # Discard large objects (value here is theoretical max size of a square)
            if hull_area > (1920/8)*(1080/8):
                continue
            # Compute an approximation, it should be composed of 4 vertices (5 in case of bad approx)
            poly_approx = cv2.approxPolyDP(hull, 0.02*cv2.arcLength(hull, True), True)
            if len(poly_approx) < 4 and len(poly_approx) > 5:
                continue
            black_squares.append(poly_approx)
            area_list.append(hull_area)
        cv2.drawContours(cur_frame, black_squares, -1, (0, 0, 255), 2)
        squares_index_list = self.group_squares(black_squares)
        squares = []
        for squares_index in squares_index_list:
            if len(squares_index) < 32:
                continue
            cur_group = [contour for i, contour in enumerate(black_squares) if i in squares_index]
            #TODO Can optimize not_inside()
            cur_group = [square for i, square in enumerate(cur_group) if self.not_inside(square, cur_group, i)]
            cv2.drawContours(cur_frame, cur_group, -1, (255, 255, 0), 2)
            if len(cur_group) == 32:
                squares = cur_group
                break
        cv2.drawContours(cur_frame, squares, -1, (0, 255, 0), 2)
        return cur_frame