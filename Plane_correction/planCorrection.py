import math
import cv2
import numpy as np
import sympy


def simplify_segmentation(points, epsilon=1.0, closed=True):
    """
    Simplifies a curve composed of segmentation points using the Douglas-Peucker algorithm.
    :param points: A list or array of (x, y) coordinates representing the curve.
    :param epsilon: Maximum distance between the original curve and its approximation.
    :param closed: Whether the curve is closed (True) or open (False).
    :return: A list of simplified (x, y) points.
    """
    points_array = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(points_array)
    simplified_hull = cv2.approxPolyDP(hull, epsilon, closed)
    return simplified_hull.reshape(-1, 2)


def appx_best_fit_ngon(points, n=4):
    """
    Approximate the best fit n-gon for a given set of points.

    :param points: A list of (x, y) tuples or a NumPy array of points.
    :param n: The desired number of vertices for the n-gon.
    :return: A list of (x, y) tuples representing the vertices of the approximated n-gon.
    """
    # Calculate the convex hull of the points
    hull = cv2.convexHull(np.array(points, dtype=np.float32)).reshape(-1, 2)
    # Convert points to sympy.Point objects for further geometric processing
    hull = [sympy.Point(*pt) for pt in hull]

    # Iteratively reduce the number of vertices in the convex hull until we reach n vertices
    while len(hull) > n:
        best_candidate = None
        for edge_idx_1 in range(len(hull)):
            # Identify the adjacent vertices for the current edge
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)
            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            # Create sympy Points for all vertices involved
            edge_pt_1, edge_pt_2 = hull[edge_idx_1], hull[edge_idx_2]
            adj_pt_1, adj_pt_2 = hull[adj_idx_1], hull[adj_idx_2]

            # Form a polygon to calculate the angles at the edge points
            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1, angle2 = subpoly.angles[edge_pt_1], subpoly.angles[edge_pt_2]

            # Check if the sum of the angles is greater than 180°, otherwise skip this edge
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

                # Find the intersection point if we delete the current edge
            intersect = sympy.Line(adj_pt_1, edge_pt_1).intersection(sympy.Line(edge_pt_2, adj_pt_2))[0]
            # Calculate the area of the new triangle that would be formed
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)

            # Choose the candidate with the smallest area
            if not best_candidate or area < best_candidate[1]:
                # Create a new hull by replacing the edge with the intersection point
                better_hull = list(hull)
                better_hull[edge_idx_1] = intersect
                del better_hull[edge_idx_2]
                best_candidate = (better_hull, area)

                # Raise an error if no candidate was found (which should not happen with a convex hull)
        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

            # Update the hull with the best candidate found
        hull = best_candidate[0]

        # Convert the final hull points back to integer tuples
    hull = [(int(pt.x), int(pt.y)) for pt in hull]
    return hull


def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def correct_perspective(image, points):
    # Assuming points are ordered as [top-left, top-right, bottom-right, bottom-left]
    tl, tr, br, bl = points

    # Compute width as average of top and bottom widths
    top_width = calculate_distance(tl[0], tl[1], tr[0], tr[1])
    bottom_width = calculate_distance(bl[0], bl[1], br[0], br[1])
    width = int((top_width + bottom_width) / 2)

    # Compute height as average of left and right heights
    left_height = calculate_distance(tl[0], tl[1], bl[0], bl[1])
    right_height = calculate_distance(tr[0], tr[1], br[0], br[1])
    height = int((left_height + right_height) / 2)


    # Define the target points
    targets = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32)
    corners = np.array(points, dtype=np.float32)

    # Apply perspective transform
    M = cv2.getPerspectiveTransform(corners, targets)
    warped_image = cv2.warpPerspective(image, M, (width, height))

    return warped_image