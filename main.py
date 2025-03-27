import argparse
import cv2
import os
from utils.transforms import *
from utils.outputs import *


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="path to the pgm file", default="scans/room2.pgm")
    parser.add_argument(
        "--verbose", help="show intermediate results", action="store_true"
    )
    parser.add_argument(
        "--output_path", help="path to place script ouput(s)", default="outputs"
    )
    return parser.parse_args()


def block_plan_alignment(image, verbose, verbose_path):
    """
    Series of functions to align the scanned floor plan with the x and y axes.

    Arguments:
        image: original floor plan.
        verbose: bool to save intermediate results.
        verbose_path: path to save intermediate results.

    Returns:
        rotated_plan: aligned scanned floor plan.
    """
    extended_plan = expanded_map(
        image,
        verbose=verbose,
        name=os.path.join(verbose_path, "1_expanded_original.png"),
    )
    plan_angle = map_orientation(image)
    rotated_plan = rotate_map(
        extended_plan,
        -plan_angle,
        verbose=verbose,
        name=os.path.join(verbose_path, "2_rotated_original.png"),
    )

    return rotated_plan


def block_plan_binarization(aligned_plan, verbose, verbose_path):
    """
    Series of functions to binarize the scanned floor plan.

    Arguments:
        aligned_map: aligned scanned floor plan.
        verbose: bool to save intermediate results.
        verbose_path: path to save intermediate results.
    Returns:
        resized_roi_plan: binary mask highlighting the ROI.
        resized_edges_plan: binary plan highlihgting the contour of the ROI.
    """

    coarse_filtered_plan = coarse_canny(
        aligned_plan, verbose=verbose, path=verbose_path
    )

    roi, roi_contour = get_region_interest(
        coarse_filtered_plan,
        verbose=verbose,
        img_name=os.path.join(verbose_path, "6_fine_roi.png"),
    )

    edges_plan = cv2.Canny(roi, 200, 225)

    resized_edges_plan = contour_based_resize(
        edges_plan,
        roi_contour,
        extra=EXTRA_SPACE,
        verbose=verbose,
        name=os.path.join(verbose_path, "7_fine_edges.png"),
    )

    resized_roi_plan = contour_based_resize(
        roi,
        roi_contour,
        extra=EXTRA_SPACE,
        verbose=verbose,
        name=os.path.join(verbose_path, "8_fine_roi_resized.png"),
    )

    return resized_roi_plan, resized_edges_plan


def block_corner_extraction(lines_plan, roi, verbose, verbose_path):
    """
    Series of functions extract the end points of the lines that comprise the
    floor plan.

    Arguments:
        lines_plan: simplified map with hough lines.
        verbose: bool to save intermediate results.
        verbose_path: path to save intermediate results.
    Returns:
        filtered_corners: ordered end points of the lines.
        mask: binary mask of the simplified ROI, with inside set as True and outisde set as False.
    """
    corners = corner_finder(
        lines_plan, verbose=verbose, name=os.path.join(verbose_path, "10_corners.png")
    )

    # ORDER THE CORNERS TO THEN CONNECT LINES
    ordered_corners = point_ordering(corners, get_contour_interest(roi))

    filtered_corners = corner_angle_filter(ordered_corners)

    mask = np.zeros_like(lines_plan[:, :, 1].squeeze())
    cv2.fillPoly(mask, pts=[np.asarray(filtered_corners, dtype=np.int32)], color=255)

    return filtered_corners, mask


if __name__ == "__main__":
    args = argument_parser()
    img_path = args.img
    verbose = args.verbose
    output_path = args.output_path
    verbose_path = os.path.join(output_path, "verbose")
    EXTRA_SPACE = 60

    # CREATE OUTPUT FOLDERS
    os.makedirs(output_path, exist_ok=True)
    if verbose:
        os.makedirs(verbose_path, exist_ok=True)

    # READ PGM FILE
    img = cv2.imread(cv2.samples.findFile(img_path), cv2.IMREAD_GRAYSCALE)

    # ROTATION OF MAP TO BE ALIGNED WITH X AND Y AXES
    rotated_plan = block_plan_alignment(img, verbose, verbose_path)

    # IMAGE PROCESSING/BINARIZATION
    resized_roi_plan, resized_edges_plan = block_plan_binarization(
        rotated_plan, verbose, verbose_path
    )

    # LINE EXTRACTION THROUGH SLIDING WINDOW (needs roi filled and roi edges)
    raw_lines_plan = sliding_window(
        resized_edges_plan,
        verbose=verbose,
        name=os.path.join(verbose_path, "9_raw_line_map.png"),
    )

    # CORNERS EXTRACTION
    corners, mask = block_corner_extraction(
        raw_lines_plan, resized_roi_plan, verbose, verbose_path
    )

    # OUTPUT GENERATION
    draw_plan(
        corners,
        mask,
        verbose=verbose,
        verbose_path=verbose_path,
        output_path=output_path,
    )

    # GENERATE VECTOR FILES
    walls2vectors(corners, mask.shape, output_path)
