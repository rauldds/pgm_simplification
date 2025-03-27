import cv2
import numpy as np
import math
import os

def expanded_map(original_scan_map, scale_factor=1.5, background_value=205, verbose=True, name='expanded_map.png'):
    """
    Extend scan plan dimensions by a specific factor (keeping the scanned area in the middle).

    Arguments:
        original_scan_map: original scanned plan.
        scale_factor: factor by which the dimensions will be extended.
        background_color: color of the pixels in the extended region, should be the same as 
                          the rest of the background (in the ros scanned plans it is 205).
        verbose: bool to save the expansion result.
        name: name of the expanded map if it is saved.
    
    Returns:
        expanded: original scanned map with extended width and height.
    """
    h, w = original_scan_map.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    expanded = background_value * np.ones((new_h, new_w), dtype=np.uint8)

    # Compute center offset
    y_offset = (new_h - h) // 2
    x_offset = (new_w - w) // 2

    expanded[y_offset:y_offset+h, x_offset:x_offset+w] = original_scan_map

    if verbose:
        cv2.imwrite(name, expanded)

    return expanded

def map_orientation(grayscale_map, thresh=200):
    """
    Get the angle of the floor plan by identify the longest lines in the map,
    computing their angles and then combining the angles to get the angle of the 
    shape of interest (largest shape in the plan). 

    Arguments:
        grayscale_map: gray scale floor plan with the shape to be rotated.
        thresh: minimum number of intersecting points (votes) to detect a line (using Hough method).
    
    Returns:
        rotation_angle: The rotation required to align the roi
    """
    # Extract edges from the grayscale plan
    edges = cv2.Canny(grayscale_map, 100, 200, apertureSize=3)
    lines = None
    # extract lines from the edges map
    while lines is None:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=thresh)
        thresh -= 50 

    # get the angles of the extracted lines
    angles = []
    for line in lines:
        _, theta = line[0]
        angle = np.degrees(theta)
        angles.append(angle)

    # Taking the median angle
    map_angle = np.median(angles)  
    
    # check how far is the angle from one of axes.
    # and then calculate difference to align with that axis.
    if 135 >= map_angle >= 45:
        rotation_angle = 90 - map_angle
    elif map_angle < 45:
        rotation_angle = -map_angle
    else:
        rotation_angle = 180 - map_angle
    return rotation_angle

def rotate_map(grayscale_map, angle, verbose=True, name="2_rotated_map.png"):
    """
    Apply a rotation to a floor plan.

    Arguments:
        grayscale_map: grayscale floor plan (should be extended, otherwise parts of the plan might get cut when rotating).
        angle: rotation to be applied to the map.
        verbose: bool to save the rotated map.
        name: name of the file of the rotated map if it is saved.
    """
    (h, w) = grayscale_map.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0) 
    rotated = cv2.warpAffine(grayscale_map, M, (int(w), int(h)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if verbose:
        cv2.imwrite(name, rotated)
    return rotated

def coarse_canny(grayscale_map, verbose=True, path=""):
    """
    Coarse feature extraction from the floor plan. The background points are set the same color as the interior points
    to only have gradients in the wall (which are black) and then by using the canny detector identify edges. Then use
    coarse closing operation to filter noise and connect edges and have a closes floor contour.

    Arguments:
        grayscale_map: scanned floor map in grayscale.
        verbose: bool to save intermediate results.
        path: folder where the intermediate results would be stored.
    
    Returns:
        morph: binarized and simplified floor plan.
    """
    canny_map = grayscale_map
    mask = canny_map == 205
    canny_map[mask] = 255
    edges = cv2.Canny(canny_map, 200, 300)  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (75, 75))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, 
                               kernel, iterations=1)
    if verbose:
        cv2.imwrite(os.path.join(path, '3_pre_canny_map.png'), 
                    canny_map)
        cv2.imwrite(os.path.join(path, '4_canny_map.png'), 
                    edges)
        cv2.imwrite(os.path.join(path, '5_after_canny_map.png'), 
                    morph)
    return morph

def get_contour_interest(grayscale_map):
    """
    Get the contours in the floor plan and keep only the biggest one,
    which should correspond to the actual floor plan.

    Arguments:
        grayscale_map: binary or grayscale map that contains the floor plan.
    Returns:
        contour_interest: contour of the floor plan.
    """
    contours, _ = cv2.findContours(grayscale_map, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    contour_interest = []
    for contour in contours:
        if len(contour) > len(contour_interest):
            contour_interest = contour
    return contour_interest

def get_region_interest(grayscale_map, verbose=True, img_name="filled_contour.png"):
    """
    Based on a grayscale map extract the region of interest (floor plan).

    Arguments:
        grayscale_map: floor map in binary or grayscale format (processed, raw).
        verbose: bool to save the region of interest as an image.
        img_name: name of the image if it is generated
    Returns:
        mask: mask representing the region of interest in the floor map.
        contour: contour of the region of interest in the floor map.
    """
    mask = np.zeros_like(grayscale_map)
    contour_interest = get_contour_interest(grayscale_map)
    cv2.fillPoly(mask, pts=[contour_interest], color=255)
    if verbose:
        cv2.imwrite(img_name, mask)
    return mask, contour_interest

def get_lines(window):
    """
    Given a small image region, identify lines (of at least 25 units of length)
    using the hough method (if any). This is done to filter small lines which 
    might be noise.

    Arguments:
        window: small region from the floor plan.
    
    Returns:
        lines: all the lines detected in the window.
    """
    lines = cv2.HoughLinesP(
                window,
                1, 
                np.pi/180, 
                threshold=25, 
                minLineLength=25, 
                maxLineGap=50 
                )
    if lines is not None:
        return lines
    return []

def sliding_window(edges_map, window_size = 100, stride = 50, verbose=True, name="line_map.png"):
    """
    Use the sliding window method to draw the contour of the floor plan more precisely
    using straight lines, since after filtering the shape is quite irregular and there
    could still exist small noise bulges in the map. 
    Using Hough Lines method in the whole image resulted in an over simplification
    of the region, so this was the best alternative I could come up with to identify lines
    without oversimplifying the map.

    Arguments:
        edges_map: binary edge map of the floor map
        window_size: size of the sliding window
        stride: sliding window step
        verbose: bool to save the resulting map of lines in an image
        name: name of the image (if it should be saved)
    
    Returns:
        line_map: map with the lines generated after sliding the window over the whole image.
    """
    h, w = edges_map.shape[:2]
    line_map = np.ones((h, w, 3), np.uint8)*255
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            window = edges_map[y:min(y+window_size, h), x:min(x+window_size, w)]
            # check if there is any edge in the window
            if np.any(window):
                lines = get_lines(window)
                if len(lines)>0:
                    for points in lines:
                        x1,y1,x2,y2=points[0]
                        cv2.line(line_map, (x1+x, y1+y), (x2+x, y2+y),(255,0,0),2)
    if verbose:
        cv2.imwrite(name,line_map)

    return line_map

def contour_based_resize(img, cnt, extra = 10, verbose = True, name="smallest_bbox.png"):
    """
    Resize the processed floor plan to reduce its size based one the smallest possible
    bounding box that would enclose the ROI.

    Arguments:
        img: floor map to be resized
        cnt: contour of the ROI in the floor map.
        extra: additional space required to leave room to draw the outside walls.
        verbose: bool to save the resized floor map.
        name: name of the file that would contain the resized floor map.
    
    Returns:
        img: resized floor map.
    """
    x, y, w, h = cv2.boundingRect(cnt) 
    
    if verbose:
        # draw the bounding rectangle 
        img = cv2.rectangle(img, (x-extra, y-extra), (x+w+extra, y+h+extra), (0, 255, 0), 2)
        cv2.imwrite(name, img[y-extra:y+h+extra, x-extra:x+w+extra])
    return img[y-extra:y+h+extra, x-extra:x+w+extra]

def corner_finder(lines_map, verbose=True, name="corners.png"):
    """
    Find the corners of the floor map based on a well segmented contour 
    of the ROI. The maximum number of corners to be found is set to 100, and the
    corners have to be at least 35 units apart.

    Arguments:
        lines_map: floor plan resulting from the Hough method and sliding window processing,
        verbose: bool to save the identified corners in an image.
        name: name of the image with the corners.

    Returns:
        corners: identified corners in lines_map.
    """
    gray = cv2.cvtColor(lines_map,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.1, minDistance=35)
    
    if verbose:
        for i, corner in enumerate(corners):
            x, y = np.round(corner.ravel()).astype(int)
            cv2.circle(lines_map, (x, y), 3, (0, 255, 0), -1)
        
        cv2.imwrite(name, lines_map)
    return corners

def point_ordering(corner_points, contour):
    """
    Since harris detector or corner detectors in general do not provide the corners in an order manner,
    in this function based on the points that comprise the contour of the ROI (and are ordered), 
    the corners are ordered by going through the contour points and checking if any of the corners is close to them.

    Arguments:
        corner_points: unordered corners from the floor plan.
        contour: contour of the ROI.
    Returns:
        matched_points: ordered corner points
    """
    # Define threshold distance
    threshold = 10
    # list to store the points in order and the index of the points that have already been added.
    matched_points = []
    idxs = []
    for point in contour:
        contour_point = point[0] 
        for idx, ref_point in enumerate(corner_points):
            if idx in idxs:
                continue
            dist = math.dist(contour_point, ref_point[0])
            if dist < threshold:
                matched_points.append([int(ref_point[0][0]), int(ref_point[0][1])])
                idxs.append(idx)
    return matched_points

def corner_angle_filter(corners):
    """
    Even after the detection of corners, there might be points that are not really corners.
    Therefore, in this function the corners are once more filtered based on their angle.
    If their angle is small (below 30 degrees) they are filtered (noise, windows or open doors).
    If their angle is almost 180 they are filtered (2 walls are not necessary, single line segment is enough)
    """
    filtered_corners = []
    for idx, point in enumerate(corners):
        if idx == 0:
            left = corners[-1]
        else:
            left = corners[idx-1]
        
        if idx == len(corners)-1:
            right = corners[0]
        else:
            right = corners[idx+1]
        
        d1 = np.subtract(point, left)
        d2 = np.subtract(point, right)
        dot = np.dot(d1,d2)
        m_d1 = np.linalg.norm(d1)
        m_d2 = np.linalg.norm(d2)
        angle = np.rad2deg(np.acos(dot/(m_d1*m_d2)))
        c_a = angle>30
        o_a = angle<160
        if o_a and c_a:
            filtered_corners.append(point)
            #print(idx)
    return filtered_corners





