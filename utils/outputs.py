import numpy as np
import math
import cv2
import os
import json
import svgwrite
import ezdxf

# BACKGROUND COLOR IN THE FLOOR PLAN
BACKGROUND_COLOR = (49, 40, 33)

# DISTANCE BETWEEN INTERIOR WALLS AND EXTERIOR WALLS
DISPLACEMENT = 20


def parallel_external_wall(points, displacement):
    """
    Computation of the displacement of a point from the inner wall
    to generate the same point for the outside wall. This is done
    by estimating the perpendicular vector to the wall and then
    displacing the point by a factor using the normalized perpendicular vector.

    Arguments:
        points: 3 points, connected by the point of interest
                in the middle (2 connected walls).
        displacement: factor by which a wall will be displaced.

    Returns:
        new_point: displaced point that will be part of the outside wall.
    """
    x, y = points[1]
    x1, y1 = points[0]
    x2, y2 = points[2]
    dx = x2 - x1
    dy = y2 - y1
    p = [-dy / math.dist([dx], [dy]), dx / math.dist([dx], [dy])]
    new_point = [int(x - (displacement * p[0])), int(y - (displacement * p[1]))]
    return new_point


def bisector_external_walls(inside_points, fine_region_interest, displacement=5):
    """
    Computation of the points that comprise the outside wall. To do this, 1st
    the direction vectors of the points (interior wall) are obtained by summing the perpendicular
    vectors of the 2 walls that each point is connected to. 2nd check if they the wall are almost
    parallel, if they are, use the parallel_external_wall function to compute the displace point.
    Otherwise, check if the direction vector points towards the interior of the floor or outside.
    If it points to the interior utilize the oposite direction to compute the displaced point.

    Arguments:
        inside_points: points that comprise the inside wall.
        fin_region_interest: binary map that points what is inside and what is outside of the floor.
        displacement: factor to displace the points and form the exterior wall.
    Returns:
        outside: points that comprise the exterior wall.
    """
    outside = []
    for idx, point in enumerate(inside_points):
        if idx == 0:
            left = inside_points[-1]
        else:
            left = inside_points[idx - 1]

        if idx == len(inside_points) - 1:
            right = inside_points[0]
        else:
            right = inside_points[idx + 1]

        d1 = np.subtract(point, left)
        d2 = np.subtract(point, right)

        # sum of perpendicular vectors
        db = (d1 / np.linalg.norm(d1)) + (d2 / np.linalg.norm(d2))

        # check if walls are almost parallel
        if abs(db[0]) > 0.4 or abs(db[1]) > 0.4:
            new_point = (
                np.add(np.asarray(point), displacement * db).astype(np.int32).tolist()
            )
            # check if the vector points towards the interior
            if fine_region_interest[new_point[1], new_point[0]]:
                new_point = (
                    np.add(np.asarray(point), -displacement * db)
                    .astype(np.int32)
                    .tolist()
                )
        else:
            dbp = np.asarray(-d1[1], d1[0])
            new_point = (
                np.add(np.asarray(point), displacement * dbp).astype(np.int32).tolist()
            )
            new_point = parallel_external_wall([left, point, right], -displacement)

        outside.append(new_point)

    outside.append(outside[0])

    return outside


def dotted_lines(external_points, plan):
    """
    Draw dotted lines in the floor plan (exterior walls).

    Arguments:
        external_points: points that comprise the corners of the outisde wall of the floor.
        plan: canvas where the lines will be drawn.
    """
    num_points_per_pixel = 0.2  # number of dots per pixel
    for i in range(len(external_points) - 1):
        start_point = tuple(external_points[i])
        end_point = tuple(external_points[i + 1])
        # compute line length
        line_length = int(math.dist(start_point, end_point))
        # get the number of points that will be drawn to represent the line
        num_points = int(num_points_per_pixel * line_length)
        # generate the points the will form the line
        x_coords = np.linspace(start_point[0], end_point[0], num_points)
        y_coords = np.linspace(start_point[1], end_point[1], num_points)
        for x, y in zip(x_coords, y_coords):
            cv2.circle(plan, (int(x), int(y)), 1, (255, 255, 255), -1)


def draw_ordered_corners(ordered_corners, plan_shape, name="ordered_corners.png"):
    """
    Draw points and their index in a black canvas

    Arguments:
        ordered_corners: list with points to be drawn
        plan_shape: the shape that the canvas should have
        name: file name of the output file
    """
    corners_plan = np.zeros((plan_shape[0], plan_shape[1], 3), np.uint8)
    for idx, point in enumerate(ordered_corners):
        x, y = point
        # Draw the point (circle)
        cv2.circle(corners_plan, (x, y), 5, (0, 0, 255), -1)  # Red point

        # Put the index of the point near it
        cv2.putText(
            corners_plan,
            str(idx),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    cv2.imwrite(name, corners_plan)


def straight_lines(inside_points, plan):
    """
    Draw straight lines to form the interior walls of the floor plan.

    Arguments:
        inside_points: list of corner points that comprise the interior wall of the floor.
        plan: canvas where the lines will be drawn.
    """
    for i in range(len(inside_points) - 1):
        start_point = tuple(inside_points[i])
        end_point = tuple(inside_points[i + 1])
        cv2.line(plan, start_point, end_point, (255, 255, 255), 1)


def draw_plan(ordered_corners, fine_roi, verbose=True, verbose_path="", output_path=""):
    """
    Function to draw the final simplified floor plan of the scanned room.

    Arguments:
        ordered_corners: corners of the interior wall of the floor
        fine_roi: binary map pointing what is the inside and the outside in the floor plan
        verbose: bool to show intermediate results
        verbose_path: path where the intermediate results will be saved
        output_path: path where the final plan will be saved
    """
    final_plan = np.ones((fine_roi.shape[0], fine_roi.shape[1], 3), np.uint8)
    final_plan[:] = BACKGROUND_COLOR
    if verbose:
        draw_ordered_corners(
            ordered_corners,
            fine_roi.shape,
            name=os.path.join(verbose_path, "11_ordered_corners.png"),
        )

    external_walls_corners = bisector_external_walls(
        ordered_corners, fine_roi, DISPLACEMENT
    )
    ordered_corners.append(ordered_corners[0])

    straight_lines(ordered_corners, final_plan)
    dotted_lines(external_walls_corners, final_plan)

    cv2.imwrite(os.path.join(output_path, "final_plan.png"), final_plan)


def walls2vectors(corners, shape, path=""):
    """
    Conversion of the floor plan from points to lines saved in SVG, JSON and
    DXF formats.

    Arguments:
        corners: interior corners from the floor plan.
        shape: shape of the output floor plan.
        path: path where the files will be stored.
    """
    corners.append(corners[0])

    floor_plan_svg = svgwrite.Drawing(
        os.path.join(path, "floor_map.svg"), size=(shape[1], shape[0])
    )
    floor_plan_json = {
        "floor_plan": {"width": shape[1], "height": shape[0], "walls": []}
    }
    doc = ezdxf.new()
    msp = doc.modelspace()

    for i in range(len(corners) - 1):
        x1, y1 = corners[i]
        x2, y2 = corners[i + 1]

        floor_plan_json["floor_plan"]["walls"].append(
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        )
        floor_plan_svg.add(
            floor_plan_svg.line(
                tuple(corners[i]), tuple(corners[i + 1]), stroke="white", stroke_width=3
            )
        )
        msp.add_line(tuple(corners[i]), tuple(corners[i + 1]))

    doc.saveas(os.path.join(path, "floor_map.dxf"))
    floor_plan_svg.save()
    with open(os.path.join(path, "floor_map.json"), "w") as json_file:
        json.dump(floor_plan_json, json_file)
