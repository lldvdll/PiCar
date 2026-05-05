"""
TPU/INT8 inference + sign-detection sticky override + frame recording.

Combines three behaviours in one model.py:
  1. TPU/.tflite inference (with INT8 CPU fallback) reading crop from
     experiment_details.json. StatefulPartitionedCall:N output ordering
     auto-detected (TFLite often reverses Keras order).
  2. HSV-based blue-circle sign detection + arrow direction; sticky angle
     override (hold the override until the sign is out of sight + grace
     frames). Tuned values copied from kaggle_16_wsigns.
  3. Per-frame raw camera frames saved to disk (for re-training data):
       ~/recordings/<model_folder_name>/<SCENARIO>/<ts>_<angle>_<speed>.png
     The recorded angle is the FINAL angle (after override), so a future
     training run learns "when sign visible, steer to 65°/110°".

Files needed in this folder:
  model.py                 — this file
  experiment_details.json  — read for crop sizes
  model_edgetpu.tflite     — Edge TPU compiled (preferred)
  model_int8.tflite        — INT8 CPU fallback if libedgetpu unavailable
"""

import os
import json
import time
import numpy as np
import cv2

try:
    import tflite_runtime.interpreter as tflite
    _USE_TFLITE_RT = True
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    _USE_TFLITE_RT = False


# ─── EDIT PER RECORDING SESSION ────────────────────────────────────────────
SCENARIO = 'fig_8_junction'
# SCENARIO = 'fig_8_junction_obs'
# SCENARIO = 'fig_8_bend_inside'
# SCENARIO = 'fig_8_bend_outside'
# SCENARIO = 'oval_outside'
# SCENARIO = 'oval_inside'
# SCENARIO = 't_left_turn'
# SCENARIO = 't_right_turn'
OUTPUT_EXT = 'png'   # or 'jpg' if disk is too slow
# ───────────────────────────────────────────────────────────────────────────


# ─── Sign-detector thresholds (copied from kaggle_16_wsigns — calibrated at lab) ───
MIN_AREA_RATIO = 0.0007
MAX_AREA_RATIO = 0.035
MAX_DIM_RATIO = 0.50
EDGE_MARGIN_RATIO = 0.047
BOTTOM_ZONE_RATIO = 0.70
MIN_FILL_RATIO = 0.50
MIN_ASPECT_NEAR_SQUARE = 0.70
BLUE_HSV_LOWER = np.array([85, 50, 20])
BLUE_HSV_UPPER = np.array([140, 255, 255])

# ─── Sign distance thresholds (TUNABLE, per direction) ────────────────────
# Two distance classes:
#   approaching  — sign diameter < threshold  (no override)
#   imminent     — sign diameter ≥ threshold  (triggers override)
# Different thresholds for left vs right since right turns at T-junctions
# (unprotected, crossing opposite lane) typically need more lead time.
# Lower value = override fires earlier (more lead time, more false positives).
# Higher value = override fires later (closer to sign, more confident).
SIGN_IMMINENT_PX_LEFT  = 25   # fires for left-arrow signs at this diameter
SIGN_IMMINENT_PX_RIGHT = 30   # fires for right-arrow signs at this diameter
# ───────────────────────────────────────────────────────────────────────────

# ─── Pre-override stabilization band ──────────────────────────────────────
# When a sign is visible but not yet imminent, the model's predicted angle
# can drift toward an early turn. To stop premature turning, force the angle
# to SIGN_STABILIZE_ANGLE (or clamp into ±SIGN_STABILIZE_BAND of it) for any
# sign with diameter in [SIGN_STABILIZE_MIN_PX, imminent_threshold).
SIGN_STABILIZE_MIN_PX = 18    # below this the sign is too far to bother
SIGN_STABILIZE_ANGLE  = 90    # straight-ahead angle (centre of 50–130 range)
SIGN_STABILIZE_BAND   = 0     # 0 = hard-set to 90; e.g. 2 → clamp into [88, 92]
# ───────────────────────────────────────────────────────────────────────────

SIGN_CROP_TOP = 60
SIGN_CROP_BOTTOM = 30
SIGN_DETECTION_INTERVAL = 1       # every frame
SIGN_GRACE_FRAMES = 5             # hold override N frames after sign out of sight

# Sign override angles — tuned from training-data median sharp turns
SIGN_LEFT_ANGLE = 65
SIGN_RIGHT_ANGLE = 110

DUMP_FIRST_FRAME = False
DEBUG_SIGN_VERBOSE = False


# ─── Sign detector functions (inline — no folder imports needed on Pi) ───────

def _estimate_distance(sign_diameter, arrow_direction):
    """Two-class distance using direction-specific threshold:
       'imminent' if diameter ≥ threshold for that direction, else 'approaching'.
       Unknown-direction signs use the LEFT threshold as a conservative default.
    """
    if arrow_direction == 'right':
        threshold = SIGN_IMMINENT_PX_RIGHT
    else:
        threshold = SIGN_IMMINENT_PX_LEFT
    return 'imminent' if sign_diameter >= threshold else 'approaching'


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
        # Detect arrow FIRST so distance can use direction-specific threshold
        roi = image[y:y + bh, x:x + bw]
        arrow = _detect_arrow_direction(roi)
        dist = _estimate_distance(diam, arrow)   # 'approaching' or 'imminent'

        detections.append({
            'bbox': (x, y, bw, bh),
            'sign_diameter': diam,
            'distance': dist,
            'arrow_direction': arrow,
            'fill_ratio': fill,
        })
    return detections


# ─── Frame recording ────────────────────────────────────────────────────────

def record_state(img, speed, angle):
    """Save raw camera frame with filename = <ts>_<angle>_<speed>.<ext>."""
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.basename(model_dir)
    save_dir = os.path.join(os.path.expanduser("~"), "recordings", model_name, SCENARIO)
    os.makedirs(save_dir, exist_ok=True)
    timestamp_ms = int(time.time() * 1000)
    if OUTPUT_EXT == 'png':
        filename = f"{timestamp_ms}_{int(angle)}_{int(speed)}.png"
        cv2.imwrite(os.path.join(save_dir, filename), img)
    else:
        filename = f"{timestamp_ms}_{int(angle)}_{int(speed)}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ─── Model class ────────────────────────────────────────────────────────────

class Model:
    def __init__(self):
        here = os.path.dirname(os.path.abspath(__file__))

        # Read crop / resize from experiment_details.json
        with open(os.path.join(here, 'experiment_details.json')) as f:
            cfg = json.load(f)
        self.crop_top = int(cfg['CROP_TOP_PIXELS'])
        self.crop_bot = int(cfg['CROP_BOTTOM_PIXELS'])
        self.target_h = int(cfg['IMG_HEIGHT_TARGET'])
        self.target_w = int(cfg['IMG_WIDTH_TARGET'])
        exp = cfg.get('EXPERIMENT_NAME', os.path.basename(here))

        # Pick TPU model first; fall back to CPU INT8
        tpu_path = os.path.join(here, 'model_edgetpu.tflite')
        cpu_path = os.path.join(here, 'model_int8.tflite')

        if os.path.exists(tpu_path):
            try:
                if _USE_TFLITE_RT:
                    delegate = [tflite.load_delegate('libedgetpu.so.1')]
                else:
                    delegate = [tflite.experimental.load_delegate('libedgetpu.so.1')]
                self.interpreter = tflite.Interpreter(model_path=tpu_path,
                                                       experimental_delegates=delegate)
                print(f"[INFO] {exp}: using Coral Edge TPU ({tpu_path}).")
            except Exception as e:
                print(f"[WARN] {exp}: Coral TPU unavailable ({e}); falling back to CPU INT8.")
                self.interpreter = tflite.Interpreter(model_path=cpu_path)
        else:
            print(f"[INFO] {exp}: no TPU file, using CPU INT8 ({cpu_path}).")
            self.interpreter = tflite.Interpreter(model_path=cpu_path)

        self.interpreter.allocate_tensors()
        self.in_detail = self.interpreter.get_input_details()[0]
        self.out_details = self.interpreter.get_output_details()

        # Output-index detection (StatefulPartitionedCall:N parsing)
        self.angle_idx, self.speed_idx = 0, 1
        for i, od in enumerate(self.out_details):
            n = od.get('name', '')
            nlower = n.lower()
            if 'angle' in nlower:
                self.angle_idx = i
            elif 'speed' in nlower:
                self.speed_idx = i
            elif 'StatefulPartitionedCall' in n:
                suffix = n.rsplit(':', 1)[-1]
                if suffix == '0':
                    self.angle_idx = i
                elif suffix == '1':
                    self.speed_idx = i

        self.in_scale, self.in_zp = self.in_detail['quantization']
        self.out_scales = [od['quantization'][0] for od in self.out_details]
        self.out_zps    = [od['quantization'][1] for od in self.out_details]
        self.in_dtype  = self.in_detail['dtype']

        # Sign override state (Strategy 2: hold until out of sight + grace period)
        self.frame_count = 0
        self.last_sign_detections = []
        self.active_override_angle = None
        self.frames_since_seen_sign = 9999

        print(f"[INFO] crop=[{self.crop_top}:-{self.crop_bot}], "
              f"resize={self.target_h}x{self.target_w}, "
              f"in_dtype={self.in_dtype}, in_q=({self.in_scale:.4f}, {self.in_zp}), "
              f"angle_idx={self.angle_idx}, speed_idx={self.speed_idx}")
        print(f"[INFO] SIGNS enabled — interval={SIGN_DETECTION_INTERVAL}, "
              f"imminent_px=L{SIGN_IMMINENT_PX_LEFT}/R{SIGN_IMMINENT_PX_RIGHT}, "
              f"stabilize≥{SIGN_STABILIZE_MIN_PX}px→{SIGN_STABILIZE_ANGLE}°"
              f"{f'±{SIGN_STABILIZE_BAND}' if SIGN_STABILIZE_BAND > 0 else ''}, "
              f"grace_frames={SIGN_GRACE_FRAMES}, "
              f"left={SIGN_LEFT_ANGLE}°, right={SIGN_RIGHT_ANGLE}°")
        print(f"[INFO] RECORDING enabled — scenario='{SCENARIO}'")

    def preprocess(self, image):
        """Crop, resize, quantize for the model input."""
        img = image[self.crop_top:-self.crop_bot, :, :]
        img = cv2.resize(img, (self.target_w, self.target_h),
                         interpolation=cv2.INTER_LINEAR)
        if self.in_dtype == np.uint8:
            x = img.astype(np.uint8)
        else:
            x = (img.astype(np.float32) / 255.0)
            if self.in_dtype == np.int8:
                x = (x / self.in_scale + self.in_zp).round().astype(np.int8)
            else:
                x = x.astype(self.in_dtype)
        return np.expand_dims(x, axis=0)

    def _dequant(self, idx):
        raw = self.interpreter.get_tensor(self.out_details[idx]['index'])
        scale, zp = self.out_scales[idx], self.out_zps[idx]
        if scale == 0:
            return float(raw.flatten()[0])
        return float((raw.flatten()[0].astype(np.int32) - zp) * scale)

    def _run_signs(self, image):
        """Sign detector on [SIGN_CROP_TOP:-SIGN_CROP_BOTTOM] of the raw frame.
           This is INDEPENDENT of the model's crop — sign detection works on
           a looser crop so signs above the model's input area are still seen.
        """
        h, w = image.shape[:2]
        bottom = h - SIGN_CROP_BOTTOM
        if bottom <= SIGN_CROP_TOP:
            return []
        sign_crop = image[SIGN_CROP_TOP:bottom, :]

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

        if DUMP_FIRST_FRAME and self.frame_count == 1:
            try:
                cv2.imwrite('/home/pi/57_tpu_rec_first_frame.png', image)
                h, w = image.shape[:2]
                print(f"[DEBUG] First frame: shape=({h}, {w}, {image.shape[2]}), "
                      f"dtype={image.dtype}, mean={image.mean():.1f}")
            except Exception as e:
                print(f"[DEBUG] could not dump first frame: {e}")

        raw_image = image.copy()                   # save before preprocess

        # ─── 1. Steering inference (TPU/INT8) ───
        x = self.preprocess(image)
        self.interpreter.set_tensor(self.in_detail['index'], x)
        self.interpreter.invoke()

        pred_angle_raw = self._dequant(self.angle_idx)
        pred_speed_raw = self._dequant(self.speed_idx)

        speed = int(np.round(pred_speed_raw)) * 35
        angle = (pred_angle_raw * 80.0) + 50.0

        # ─── 2. Sign detection (every Nth frame) ───
        if self.frame_count % SIGN_DETECTION_INTERVAL == 0:
            self.last_sign_detections = self._run_signs(image)

        # ─── 3. Sign override state machine (Strategy 2: sticky) ───
        sign_overriding = False
        sign_stabilizing = False
        any_sign_visible = False
        any_sign_in_stabilize_band = False
        # Override fires when distance crosses SIGN_IMMINENT_PX (defined at top).
        # Tune that constant to fire earlier/later.
        for sign in self.last_sign_detections:
            arr = sign.get('arrow_direction', 'unknown')
            dist = sign.get('distance', 'approaching')
            diam = sign.get('sign_diameter', 0)
            any_sign_visible = True
            if dist == 'imminent' and self.active_override_angle is None:
                if arr == 'left':
                    self.active_override_angle = SIGN_LEFT_ANGLE
                    break
                elif arr == 'right':
                    self.active_override_angle = SIGN_RIGHT_ANGLE
                    break
            # Pre-imminent: sign visible & "approaching" but big enough that
            # the model would be tempted to start turning early — clamp angle.
            if dist == 'approaching' and diam >= SIGN_STABILIZE_MIN_PX:
                any_sign_in_stabilize_band = True

        # Update grace counter
        if any_sign_visible:
            self.frames_since_seen_sign = 0
        else:
            self.frames_since_seen_sign += 1

        # Apply override if active AND sign was seen recently
        if self.active_override_angle is not None:
            if self.frames_since_seen_sign < SIGN_GRACE_FRAMES:
                angle = self.active_override_angle
                sign_overriding = True
            else:
                # Sign has been gone for grace period → release override
                self.active_override_angle = None

        # Pre-override stabilization — only when no override is active yet.
        # Holds the car straight while a sign approaches, so the model can't
        # commit to a turn before the imminent threshold fires.
        if not sign_overriding and any_sign_in_stabilize_band:
            if SIGN_STABILIZE_BAND <= 0:
                angle = SIGN_STABILIZE_ANGLE
            else:
                lo = SIGN_STABILIZE_ANGLE - SIGN_STABILIZE_BAND
                hi = SIGN_STABILIZE_ANGLE + SIGN_STABILIZE_BAND
                angle = float(np.clip(angle, lo, hi))
            sign_stabilizing = True

        # ─── 4. Clamp ───
        angle = float(np.clip(angle, 50, 130))
        speed = int(np.clip(speed, 0, 35))

        # ─── 5. Print per frame ───
        n_signs = len(self.last_sign_detections)
        sign_str = ""
        if n_signs > 0:
            parts = []
            for d in self.last_sign_detections:
                parts.append(f"[{d['arrow_direction']}|{d['distance']}|d={d['sign_diameter']}]")
            sign_str = " SIGNS: " + " ".join(parts)
        if sign_overriding:
            ovr_str = f" OVERRIDE→{int(angle)}° (since_seen={self.frames_since_seen_sign})"
        elif sign_stabilizing:
            ovr_str = f" STABILIZE→{int(angle)}°"
        else:
            ovr_str = ""

        print(f'Raw angle: {pred_angle_raw:.3f} → {angle:.0f}° | '
              f'Raw speed: {pred_speed_raw:.3f} → {speed} | '
              f'signs={n_signs}{ovr_str}{sign_str}')

        # ─── 6. Record final (post-override) angle + speed ───
        # Recording the FINAL angle means future training data shows
        # "when this sign was visible at this distance, the right answer was 65°/110°"
        record_state(raw_image, speed, angle)

        return angle, speed
