import sys
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
MODEL_PATH = "pose_landmarker.task"
DEFAULT_IMAGE_PATH = "test-image.png"

WHITE = (255, 255, 255)

# BlazePose landmark indices we care about for the head:
#   0  = nose (top of head proxy)
#   7  = left ear
#   8  = right ear
#   9  = mouth left  (lower head proxy)
#  10  = mouth right
# We'll use: nose (0), the visible ear (7 or 8), and mouth midpoint (9/10)

# Body skeleton connections — face landmarks stripped out, just the 3 head points
POSE_CONNECTIONS = [
    # head: nose → ear, nose → mouth
    (0, 7), (0, 8),
    (0, 9), (0, 10),
    # shoulders
    (11, 12),
    # left arm (wrist=15, index tip=19)
    (11, 13), (13, 15),
    # right arm (wrist=16, index tip=20)
    (12, 14), (14, 16),
    # torso
    (23, 24),
    # left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Only draw dots for these indices
HEAD_LANDMARKS  = {0, 7, 8, 9, 10}
ARM_LANDMARKS   = {11, 12, 13, 14, 15, 16, 19, 20}  # shoulders, elbows, wrists, index tips
BODY_LANDMARKS  = set(range(23, 33))
DRAWN_LANDMARKS = HEAD_LANDMARKS | ARM_LANDMARKS | BODY_LANDMARKS


# ── Helpers ────────────────────────────────────────────────────────────────────
def download_file(url: str, dest: str) -> None:
    import os
    if os.path.exists(dest):
        print(f"[INFO] '{dest}' already exists – skipping download.")
        return
    print(f"[INFO] Downloading {url} → {dest} …")
    urllib.request.urlretrieve(url, dest)
    print(f"[INFO] Saved to '{dest}'.")


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    annotated_image = np.copy(rgb_image)
    h, w = annotated_image.shape[:2]

    # ── Blue segmentation mask ─────────────────────────────────────────────────
    if detection_result.segmentation_masks:
        mask = np.squeeze(detection_result.segmentation_masks[0].numpy_view())
        blue_layer = np.zeros_like(annotated_image)
        blue_layer[:] = (0, 0, 255)
        mask_smooth = cv2.GaussianBlur(mask, (15, 15), 0)
        alpha = (mask_smooth[..., np.newaxis] * 0.6).astype(np.float32)
        annotated_image = (
            annotated_image.astype(np.float32) * (1 - alpha)
            + blue_layer.astype(np.float32) * alpha
        ).astype(np.uint8)

    # ── Skeleton ───────────────────────────────────────────────────────────────
    for pose_landmarks in detection_result.pose_landmarks:
        pts = [
            (int(lm.x * w), int(lm.y * h))
            for lm in pose_landmarks
        ]

        # Connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(pts) and end_idx < len(pts):
                cv2.line(annotated_image, pts[start_idx], pts[end_idx],
                         WHITE, thickness=3, lineType=cv2.LINE_AA)

        # Dots — only for the landmarks we care about
        for idx in DRAWN_LANDMARKS:
            if idx < len(pts):
                cv2.circle(annotated_image, pts[idx], radius=6,
                           color=WHITE, thickness=2, lineType=cv2.LINE_AA)

        # ── Head tilt angle ───────────────────────────────────────────────────
        # Use nose (top) and mouth midpoint (bottom) as the head axis.
        # Pick whichever ear has higher visibility for the side reference.
        lms = pose_landmarks
        nose  = pts[0]
        mouth = (
            (pts[9][0] + pts[10][0]) // 2,
            (pts[9][1] + pts[10][1]) // 2,
        )
        ear_idx = 7 if lms[7].visibility >= lms[8].visibility else 8
        ear = pts[ear_idx]

        # Angle of the nose→mouth vector from vertical (positive = tilted right)
        dx = mouth[0] - nose[0]
        dy = mouth[1] - nose[1]
        tilt_deg = np.degrees(np.arctan2(dx, dy))  # arctan2(x,y) gives angle from vertical

        print(f"[INFO] Head tilt: {tilt_deg:+.1f}°  (ear used: {'left' if ear_idx == 7 else 'right'})")

        # Head axis line (cyan)
        cv2.line(annotated_image, ear, mouth, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # ── Torso spine ────────────────────────────────────────────────────────
        l_shoulder, r_shoulder = pts[11], pts[12]
        l_hip,      r_hip      = pts[23], pts[24]
        torso_top    = ((l_shoulder[0] + r_shoulder[0]) // 2,
                        (l_shoulder[1] + r_shoulder[1]) // 2)
        torso_bottom = ((l_hip[0] + r_hip[0]) // 2,
                        (l_hip[1] + r_hip[1]) // 2)

        cv2.circle(annotated_image, torso_top,    6, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(annotated_image, torso_bottom, 6, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.line(annotated_image, torso_bottom, torso_top, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # ── Neck / head chain ─────────────────────────────────────────────────
        neck_head_mid = ((torso_top[0] + nose[0]) // 2,
                         (torso_top[1] + nose[1]) // 2)
        ear_mid = ((pts[7][0] + pts[8][0]) // 2,
                   (pts[7][1] + pts[8][1]) // 2)
        cv2.circle(annotated_image, neck_head_mid, 6, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(annotated_image, ear_mid,        6, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.line(annotated_image, torso_top, neck_head_mid, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.line(annotated_image, neck_head_mid, ear_mid,   (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # ── Hand angle lines (magenta: wrist → index tip) ─────────────────────
        MAGENTA = (255, 0, 255)
        for wrist_idx, tip_idx, label in [(15, 19, "left"), (16, 20, "right")]:
            if wrist_idx < len(pts) and tip_idx < len(pts):
                cv2.line(annotated_image, pts[wrist_idx], pts[tip_idx],
                         MAGENTA, thickness=2, lineType=cv2.LINE_AA)

    return annotated_image


# ── Main ───────────────────────────────────────────────────────────────────────
def main(image_path: str = DEFAULT_IMAGE_PATH) -> None:
    download_file(MODEL_URL, MODEL_PATH)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    elif raw.shape[2] == 4:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
    else:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw)
    detection_result = detector.detect(image)

    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    output_path = "output_landmarks.jpg"
    cv2.imwrite(output_path, annotated_bgr)
    print(f"[INFO] Saved to '{output_path}'.")

    cv2.imshow("Pose Landmarks", annotated_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    main(img)