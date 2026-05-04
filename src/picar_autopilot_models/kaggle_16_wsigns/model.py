"""
Deploy: Dean's exp 16 (16_2_attention_block) + sign detection.

Mimics Dean's exp 16 model.py pattern (loads .h5 directly via TF).
Adds inline sign detector + state machine for sign override.

Files needed in this folder:
  model.py         — this file
  best_model.h5    — Dean exp 16 weights (copy from 16_2_attention_block/)

Usage on Pi:
  python3 run.py --model dean16_signs --mode drive --duration 60
"""

import os
import time
import numpy as np
import tensorflow as tf
import cv2


# ─── Sign-detector thresholds (calibrated on lab — confirmed working) ────────
MIN_AREA_RATIO = 0.0007
MAX_AREA_RATIO = 0.035
MAX_DIM_RATIO = 0.50
EDGE_MARGIN_RATIO = 0.047
BOTTOM_ZONE_RATIO = 0.70
MIN_FILL_RATIO = 0.50
MIN_ASPECT_NEAR_SQUARE = 0.70
BLUE_HSV_LOWER = np.array([85, 50, 20])
BLUE_HSV_UPPER = np.array([140, 255, 255])
DIST_FAR_RATIO = 0.047
DIST_MEDIUM_RATIO = 0.088
DIST_CLOSE_RATIO = 0.125

SIGN_CROP_TOP = 60
SIGN_CROP_BOTTOM = 30
SIGN_DETECTION_INTERVAL = 1       # every frame — most responsive
DUMP_FIRST_FRAME = False
DEBUG_SIGN_VERBOSE = False
SIGN_GRACE_FRAMES = 5             # for Strategy 2: keep override active for N frames after sign last seen

# Sign override behaviour — tuned from training-data analysis (median sharp turns)
SIGN_LEFT_ANGLE = 65              # was 60 — matches median training sharp-left turn
SIGN_RIGHT_ANGLE = 110            # was 120 — matches median training sharp-right turn
# (sticky duration unused now — see Strategy 2 below)


# ─── Sign detector functions (inline — no folder imports needed on Pi) ───────

def _estimate_distance(sign_diameter, img_width):
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
    bh, bw = roi.shape[:2]
    if min(bw, bh) < 8:
        return 'unknown'
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    circle_mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.circle(circle_mask, (bw // 2, bh // 2), int(min(bw, bh) * 0.45), 255, -1)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv_roi, np.array([0, 0, 130]), np.array([180, 100, 255]))
    white_in_circle = cv2.bitwise_and(white_mask, circle_mask)
    white_px = np.count_nonzero(white_in_circle)

    centroid_vote = 0
    if white_px >= 3:
        ys, xs = np.where(white_in_circle > 0)
        offset_ratio = (np.mean(xs) - bw / 2.0) / (bw / 2.0)
        if offset_ratio < -0.08: centroid_vote = -1
        elif offset_ratio > 0.08: centroid_vote = 1

    masked = cv2.bitwise_and(grey, circle_mask)
    sx = cv2.Sobel(masked, cv2.CV_64F, 1, 0, ksize=3)
    mid = bw // 2
    le = np.sum(np.abs(sx[:, :mid]))
    re = np.sum(np.abs(sx[:, mid:]))
    sobel_vote = 0
    if le + re > 0:
        r = le / (re + 1e-6)
        if r > 1.25: sobel_vote = -1
        elif r < 0.75: sobel_vote = 1

    total = centroid_vote + sobel_vote
    if total <= -1: return 'left'
    if total >= 1: return 'right'
    return 'unknown'


def detect_signs(image):
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    min_area = max(20, int(MIN_AREA_RATIO * h * w))
    max_area = int(MAX_AREA_RATIO * h * w)
    max_dim = int(MAX_DIM_RATIO * min(h, w))
    edge_margin = max(5, int(EDGE_MARGIN_RATIO * w))

    raw = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, ko)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)

    bb = cv2.inRange(hsv, np.array([85, 120, 60]), np.array([140, 255, 255]))
    kcb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bb = cv2.morphologyEx(bb, cv2.MORPH_OPEN, ko)
    bb = cv2.morphologyEx(bb, cv2.MORPH_CLOSE, kcb)

    combined = cv2.bitwise_or(mask, bb)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area: continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cx = x + bw // 2
        cy = y + bh // 2
        if cx < edge_margin or cx > w - edge_margin: continue
        if cy > h - edge_margin or cy > h * BOTTOM_ZONE_RATIO: continue
        squareness = min(bw, bh) / (max(bw, bh) + 1e-6)
        if squareness < MIN_ASPECT_NEAR_SQUARE: continue
        if max(bw, bh) > max_dim: continue

        cnt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        mv = cv2.mean(hsv[:, :, 2], mask=cnt_mask)[0]
        ms = cv2.mean(hsv[:, :, 1], mask=cnt_mask)[0]
        if mv < 75 or ms < 90: continue

        (_, _), radius = cv2.minEnclosingCircle(cnt)
        fill = area / (np.pi * radius * radius + 1e-6)
        thresh = MIN_FILL_RATIO if ms < 110 else 0.25
        if fill < thresh: continue

        diam = max(bw, bh)
        dist, action = _estimate_distance(diam, w)
        arrow = 'unknown'
        if dist in ('approaching', 'close', 'imminent'):
            roi = image[y:y + bh, x:x + bw]
            arrow = _detect_arrow_direction(roi)

        detections.append({
            'bbox': (x, y, bw, bh),
            'sign_diameter': diam,
            'distance': dist,
            'action': action,
            'arrow_direction': arrow,
            'fill_ratio': fill,
        })
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CLASS — Dean exp 16 + sign detection
# ─────────────────────────────────────────────────────────────────────────────

class Model:
    saved_model_path = 'best_model.h5'

    def __init__(self):
        # Load Dean's exp 16 .h5 directly (TF 2.15 compatible)
        model_filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.saved_model_path
        )
        self.model = tf.keras.models.load_model(model_filepath)
        print("[INFO] Dean exp 16 model loaded (1 attention block, full backbone).")
        print(f"[INFO] Input: {self.model.input_shape}")
        print(f"[INFO] Outputs: {[o.name for o in self.model.outputs]}")

        # State for sign override (Strategy 2: hold until sign out of sight)
        self.frame_count = 0
        self.last_sign_detections = []
        self.active_override_angle = None     # currently held override angle (None = not active)
        self.frames_since_seen_sign = 9999    # for grace period

    def preprocess(self, image):
        """Mirrors Dean's exp 14 model.py preprocess EXACTLY (no color conversion).
           Dean's deployed model.py uses BGR-as-is (whatever autopilot gives).
           Dean's exp 14 alone works on Pi → keep BGR pass-through.
        """
        im = tf.cast(image, tf.float32) / 255.0
        im = im[110:-30, :, :]
        im = tf.image.resize(im, [96, 160])
        im = tf.expand_dims(im, axis=0)
        return im

    def _run_signs(self, image):
        """Run sign detector on cropped top portion of frame."""
        h, w = image.shape[:2]
        bottom = h - SIGN_CROP_BOTTOM
        if bottom <= SIGN_CROP_TOP:
            return []
        sign_crop = image[SIGN_CROP_TOP:bottom, :]

        # Diagnostic: count blue pixels in the crop so we know if signs WOULD be detectable
        if DEBUG_SIGN_VERBOSE:
            hsv = cv2.cvtColor(sign_crop, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
            blue_px = np.count_nonzero(blue_mask)
            crop_h, crop_w = sign_crop.shape[:2]
            blue_pct = 100.0 * blue_px / (crop_h * crop_w)
            print(f"  [sign-debug] crop={crop_h}x{crop_w} blue_px={blue_px} ({blue_pct:.2f}%)", end="")

        try:
            return detect_signs(sign_crop)
        except Exception as e:
            print(f"[WARN] sign detection error: {e}")
            return []

    def predict(self, image):
        self.frame_count += 1

        # Dump first received frame so we can verify image format from autopilot
        if DUMP_FIRST_FRAME and self.frame_count == 1:
            try:
                cv2.imwrite('/home/pi/dean16_first_frame.png', image)
                h, w = image.shape[:2]
                print(f"[DEBUG] First frame: shape=({h}, {w}, {image.shape[2]}), "
                      f"dtype={image.dtype}, mean={image.mean():.1f}, "
                      f"saved to /home/pi/dean16_first_frame.png")
            except Exception as e:
                print(f"[DEBUG] could not dump first frame: {e}")

        # ─── 1. Steering (Dean's exp 16 — every frame) ───
        x = self.preprocess(image)
        preds = self.model.predict(x, verbose=0)
        pred_angle_raw = float(preds[0][0][0])    # angle output
        pred_speed_raw = float(preds[1][0][0])    # speed output

        # Snap speed to binary 0/1, then × 35
        speed = int(np.round(pred_speed_raw)) * 35
        # Map angle linearly: 0.0 → 50°, 0.5 → 90°, 1.0 → 130°
        angle = (pred_angle_raw * 80.0) + 50.0

        # ─── 2. Sign detection (every Nth frame) ───
        if self.frame_count % SIGN_DETECTION_INTERVAL == 0:
            self.last_sign_detections = self._run_signs(image)

        # ─── 3. State machine: sign override (Strategy 2: hold until out-of-sight) ───
        # Trigger override at IMMINENT distance with left/right arrow.
        # KEEP override active as long as ANY sign is visible (any distance) —
        # plus a small grace period after sign disappears.
        # Once sign passes (e.g. car has turned past the junction), model takes over.
        sign_overriding = False

        # Did we see any sign this frame?
        any_sign_visible = False
        for sign in self.last_sign_detections:
            arr = sign.get('arrow_direction', 'unknown')
            dist = sign.get('distance', 'far')
            any_sign_visible = True
            # Trigger override only at imminent distance with valid arrow
            if dist == 'imminent' and self.active_override_angle is None:
                if arr == 'left':
                    self.active_override_angle = SIGN_LEFT_ANGLE
                    break
                elif arr == 'right':
                    self.active_override_angle = SIGN_RIGHT_ANGLE
                    break

        # Update grace counter
        if any_sign_visible:
            self.frames_since_seen_sign = 0
        else:
            self.frames_since_seen_sign += 1

        # Apply override if active AND sign was seen recently
        if self.active_override_angle is not None:
            if self.frames_since_seen_sign < SIGN_GRACE_FRAMES:
                # Sign still visible (or recently was) → hold the override
                angle = self.active_override_angle
                sign_overriding = True
            else:
                # Sign has been gone for grace period → end override, model resumes
                self.active_override_angle = None

        # Clamp
        angle = float(np.clip(angle, 50, 130))
        speed = int(np.clip(speed, 0, 35))

        # ─── 4. Print per frame ───
        n_signs = len(self.last_sign_detections)
        sign_str = ""
        if n_signs > 0:
            parts = []
            for d in self.last_sign_detections:
                parts.append(f"[{d['arrow_direction']}|{d['distance']}|d={d['sign_diameter']}]")
            sign_str = " SIGNS: " + " ".join(parts)
        ovr_str = f" OVERRIDE→{int(angle)}° (since_seen={self.frames_since_seen_sign})" if sign_overriding else ""

        print(f'Raw angle: {pred_angle_raw:.3f} → {angle:.0f}° | '
              f'Raw speed: {pred_speed_raw:.3f} → {speed} | '
              f'signs={n_signs}{ovr_str}{sign_str}')

        return angle, speed
