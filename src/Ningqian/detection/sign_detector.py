"""
Direction sign detector — blue circle signs with white arrows.

Usage:
  Crop the raw 320x240 image: top 60px, bottom 30px -> 320x150, then pass in. 
  Original aggressive cropping will cut off signs.
    cropped = image[60:210, :]
    dets = detect_signs(cropped, road_mask=None)

Strategy:
  HSV blue filtering → morphological cleanup → contour shape validation
  → distance estimation → arrow direction detection

All thresholds are relative to image size, so if the image is resized the
detector still works. 

The signs are blue circles with a white arrow, on a black pole.
"""

import cv2
import numpy as np


# --- Relative thresholds (fractions of image dimensions) ---
# Calibrated on 320x160 crops, expressed as ratios for portability

# Area bounds as fraction of total image area (h*w)
MIN_AREA_RATIO = 0.0007     # 35 / (320*160) ≈ 0.0007
MAX_AREA_RATIO = 0.035      # 1800 / (320*160) ≈ 0.035

# Max bounding box dimension as fraction of image height
MAX_DIM_RATIO = 0.50        # 80 / 160 = 0.50

# Edge margin as fraction of image width
EDGE_MARGIN_RATIO = 0.047   # 15 / 320 ≈ 0.047

# Bottom zone: signs can't be in bottom 30% (ground in front of car)
BOTTOM_ZONE_RATIO = 0.70

# Shape validation
MIN_FILL_RATIO = 0.50       # area / enclosing circle area
MIN_ASPECT_NEAR_SQUARE = 0.70  # min(w,h)/max(w,h)

# HSV range for blue signs
BLUE_HSV_LOWER = np.array([85, 50, 20])
BLUE_HSV_UPPER = np.array([140, 255, 255])

# Distance thresholds as fraction of image width
# At 320px wide: far<15, approaching<28, close<40
DIST_FAR_RATIO = 0.047      # 15/320
DIST_MEDIUM_RATIO = 0.088   # 28/320
DIST_CLOSE_RATIO = 0.125    # 40/320


def _estimate_distance(sign_diameter, img_width):
    """
    Estimate distance category from sign apparent size (image-relative).

    Returns:
        (category, action) tuple
    """
    ratio = sign_diameter / img_width
    if ratio < DIST_FAR_RATIO:
        return 'far', 'none'
    elif ratio < DIST_MEDIUM_RATIO:
        return 'approaching', 'slow_down'
    elif ratio < DIST_CLOSE_RATIO:
        return 'close', 'prepare_turn'
    else:
        return 'imminent', 'turn_now'


def _detect_arrow_direction(roi):
    """
    Detect arrow direction (left/right) inside a sign ROI.

    Uses two complementary methods and combines their votes:
    1. White pixel mass: compare bright (white arrow) pixels in left vs right half
    2. Horizontal Sobel energy: arrowhead creates stronger edges on the pointed side

    Args:
        roi: BGR image of the sign region

    Returns:
        'left', 'right', or 'unknown'
    """
    bh, bw = roi.shape[:2]
    if min(bw, bh) < 8:
        return 'unknown'

    grey_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Circular mask covering the sign face (exclude pole/background)
    circle_mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.circle(circle_mask, (bw // 2, bh // 2), int(min(bw, bh) * 0.45), 255, -1)

    # --- Method 1: White pixel weighted centroid ---
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv_roi, np.array([0, 0, 130]), np.array([180, 100, 255]))
    white_in_circle = cv2.bitwise_and(white_mask, circle_mask)
    white_px = np.count_nonzero(white_in_circle)

    centroid_vote = 0
    if white_px >= 3:
        ys, xs = np.where(white_in_circle > 0)
        centroid_x = np.mean(xs)
        center_x = bw / 2.0
        offset_ratio = (centroid_x - center_x) / (bw / 2.0)
        if offset_ratio < -0.08:
            centroid_vote = -1
        elif offset_ratio > 0.08:
            centroid_vote = 1

    # --- Method 2: Horizontal Sobel energy asymmetry ---
    masked_grey = cv2.bitwise_and(grey_roi, circle_mask)
    sobel_x = cv2.Sobel(masked_grey, cv2.CV_64F, 1, 0, ksize=3)
    mid = bw // 2
    left_energy = np.sum(np.abs(sobel_x[:, :mid]))
    right_energy = np.sum(np.abs(sobel_x[:, mid:]))
    total_energy = left_energy + right_energy

    sobel_vote = 0
    if total_energy > 0:
        ratio = left_energy / (right_energy + 1e-6)
        if ratio > 1.25:
            sobel_vote = -1
        elif ratio < 0.75:
            sobel_vote = 1

    total_vote = centroid_vote + sobel_vote
    if total_vote <= -1:
        return 'left'
    elif total_vote >= 1:
        return 'right'
    return 'unknown'


def _build_detection(image, hsv, x, y, bw, bh, area, fill_ratio, squareness, w, h):
    """Build a detection dict with distance, arrow, and position info."""
    cx = x + bw // 2
    cy = y + bh // 2
    sign_diameter = max(bw, bh)
    distance_cat, distance_action = _estimate_distance(sign_diameter, w)

    arrow_dir = 'unknown'
    if distance_cat in ('approaching', 'close', 'imminent'):
        roi = image[y:y+bh, x:x+bw]
        arrow_dir = _detect_arrow_direction(roi)

    frame_center_x = w // 2
    if cx < frame_center_x - w * 0.06:
        sign_side = 'left'
    elif cx > frame_center_x + w * 0.06:
        sign_side = 'right'
    else:
        sign_side = 'center'

    return {
        'bbox': (x, y, bw, bh),
        'center': (cx, cy),
        'area': area,
        'fill_ratio': fill_ratio,
        'squareness': squareness,
        'sign_diameter': sign_diameter,
        'distance': distance_cat,
        'action': distance_action,
        'arrow_direction': arrow_dir,
        'sign_side': sign_side,
        'class': 'direction_sign'
    }


def detect_signs(image, road_mask=None):
    """
    Detect blue direction signs with white arrows.

    HSV blue filter → morphological cleanup → contour validation → distance/arrow.

    Args:
        image: BGR image (any resolution, pre-cropped)
        road_mask: optional road region mask (same size as image)

    Returns:
        List of detection dicts with keys:
            bbox, center, area, fill_ratio, squareness, sign_diameter,
            distance, action, arrow_direction, sign_side, class
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    min_area = max(20, int(MIN_AREA_RATIO * h * w))
    max_area = int(MAX_AREA_RATIO * h * w)
    max_dim = int(MAX_DIM_RATIO * min(h, w))
    edge_margin = max(5, int(EDGE_MARGIN_RATIO * w))

    # HSV blue filter
    raw_mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)

    # Morphological cleanup — 5x5 close to merge split blue halves
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Bright-blue pass with bigger close kernel for fragmented signs
    bright_blue_mask = cv2.inRange(hsv, np.array([85, 120, 60]), np.array([140, 255, 255]))
    kernel_close_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bright_blue_mask = cv2.morphologyEx(bright_blue_mask, cv2.MORPH_OPEN, kernel_open)
    bright_blue_mask = cv2.morphologyEx(bright_blue_mask, cv2.MORPH_CLOSE, kernel_close_big)

    combined_mask = cv2.bitwise_or(mask, bright_blue_mask)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y_pos, bw, bh = cv2.boundingRect(cnt)

        cx_temp = x + bw // 2
        cy_temp = y_pos + bh // 2
        if cx_temp < edge_margin or cx_temp > w - edge_margin:
            continue
        if cy_temp > h - edge_margin:
            continue
        if cy_temp > h * BOTTOM_ZONE_RATIO:
            continue

        squareness = min(bw, bh) / (max(bw, bh) + 1e-6)
        if squareness < MIN_ASPECT_NEAR_SQUARE:
            continue

        if max(bw, bh) > max_dim:
            continue

        # Color validation
        cnt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        mean_v = cv2.mean(hsv[:, :, 2], mask=cnt_mask)[0]
        mean_s = cv2.mean(hsv[:, :, 1], mask=cnt_mask)[0]
        if mean_v < 75:
            continue
        if mean_s < 90:
            continue

        # Fill ratio
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        enclosing_area = np.pi * radius * radius
        fill_ratio = area / (enclosing_area + 1e-6)

        fill_threshold = MIN_FILL_RATIO if mean_s < 110 else 0.25
        if fill_ratio < fill_threshold:
            continue

        # Road proximity check
        if road_mask is not None:
            pole_check_h = max(40, int(0.5 * h))
            pole_check_w = max(20, int(0.125 * w))
            cx_r = x + bw // 2
            check_y1 = min(h - 1, y_pos + bh)
            check_y2 = min(h, y_pos + bh + pole_check_h)
            check_x1 = max(0, cx_r - pole_check_w)
            check_x2 = min(w, cx_r + pole_check_w)
            if check_y2 > check_y1 and check_x2 > check_x1:
                pole_region = road_mask[check_y1:check_y2, check_x1:check_x2]
                if pole_region.size > 0 and np.mean(pole_region > 0) < 0.05:
                    continue

        det = _build_detection(image, hsv, x, y_pos, bw, bh, area, fill_ratio, squareness, w, h)
        detections.append(det)

    return detections


def draw_detections(image, detections, colour=(255, 0, 0), thickness=2):
    """Draw bounding boxes and labels on image. Returns annotated copy."""
    annotated = image.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, thickness)
        dist = det.get('distance', '?')
        arrow = det.get('arrow_direction', '?')
        action = det.get('action', '?')
        label = f"sign {dist} arrow:{arrow} [{action}]"
        cv2.putText(annotated, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
    return annotated


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sign_detector.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    from track_detector import get_road_region
    road_mask, _ = get_road_region(img)

    dets = detect_signs(img, road_mask=road_mask)
    print(f"Found {len(dets)} sign(s)")
    for d in dets:
        print(f"  bbox={d['bbox']}, fill={d['fill_ratio']:.2f}, sq={d['squareness']:.2f}")

    annotated = draw_detections(img, dets)
    cv2.imwrite("sign_detection_result.png", annotated)
    print("Saved: sign_detection_result.png")
