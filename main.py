from fastapi import FastAPI, Request, UploadFile, File, Form, Header, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import io
import cv2
import numpy as np
import uvicorn
import sqlite3
import hashlib
import os
import json
import uuid
import re
import math
import requests
from datetime import datetime, timezone
from datetime import timedelta
from passlib.context import CryptContext
from jose import jwt, JWTError
from PIL import Image, ImageOps


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(title="FiberHash / Metalens Authentication API")

SECRET_KEY = "challengeproof_secret_key_change_later"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STORAGE SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "fiberhash.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id TEXT PRIMARY KEY,
            product_name TEXT,
            brand TEXT,
            batch_code TEXT,
            master_image_path TEXT,
            master_image_hash TEXT,
            created_at TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS verification_events (
            id TEXT PRIMARY KEY,
            product_id TEXT,
            status TEXT,
            decision TEXT,
            is_match INTEGER,
            trust_score REAL,
            quality_score REAL,
            blur_variance REAL,
            brightness REAL,
            glare_score REAL,
            resolution_width INTEGER,
            resolution_height INTEGER,
            inlier_count INTEGER,
            good_match_count INTEGER,
            total_keypoints_master INTEGER,
            total_keypoints_scan INTEGER,
            scan_image_hash TEXT,
            replay_warning INTEGER,
            message TEXT,
            created_at TEXT,
            raw_result_json TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS unit_fingerprints (
            unit_id TEXT PRIMARY KEY,
            order_id TEXT,
            seller_id TEXT,
            buyer_id TEXT,
            marketplace_name TEXT,
            product_id TEXT,
            product_name TEXT,
            brand TEXT,
            batch_code TEXT,
            package_image_path TEXT,
            package_image_hash TEXT,
            seal_image_path TEXT,
            seal_image_hash TEXT,
            created_at TEXT
        )
        """
     )
    cursor.execute("""
       CREATE TABLE IF NOT EXISTS unit_verification_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_id TEXT,
            decision TEXT,
            package_match INTEGER,
            seal_match INTEGER,
            trust_score REAL,
            ai_risk_level TEXT,
            ai_risk_score REAL,
            created_at TEXT
     )
""")
    cursor.execute("""
       CREATE TABLE IF NOT EXISTS challenge_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT UNIQUE,
            order_id TEXT,
            marketplace_name TEXT,
            seller_id TEXT,
            buyer_id TEXT,
            unit_id TEXT,
            case_type TEXT,
            case_status TEXT,
            trigger_reason TEXT,
            verification_decision TEXT,
            package_match INTEGER,
            seal_match INTEGER,
            trust_score REAL,
            risk_level TEXT,
            recommended_action TEXT,
            created_at TEXT
        )
        """
    )
    
    cursor.execute( """
       CREATE TABLE IF NOT EXISTS challenge_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            challenge_id TEXT UNIQUE,
            order_id TEXT,
            marketplace_name TEXT,
            seller_id TEXT,
            buyer_id TEXT,
            unit_id TEXT,
            product_id TEXT,
            challenge_source TEXT,
            challenge_reason TEXT,
            challenge_status TEXT,
            customer_notes TEXT,
            seller_response TEXT,
            response_at TEXT,
            created_at TEXT
        )
 """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS challenge_timeline (
            event_id TEXT PRIMARY KEY,
            challenge_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_title TEXT NOT NULL,
            event_description TEXT,
            actor_type TEXT,
            actor_id TEXT,
            old_status TEXT,
            new_status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
         )
 """)
    cursor.execute("""
       CREATE TABLE IF NOT EXISTS seller_trust_metrics (
            seller_id TEXT PRIMARY KEY,
            total_challenges INTEGER DEFAULT 0,
            accepted_challenges INTEGER DEFAULT 0,
            rejected_challenges INTEGER DEFAULT 0,
            passed_verifications INTEGER DEFAULT 0,
            failed_verifications INTEGER DEFAULT 0,
            last_updated TEXT
         )
""") 
    
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS sellers (
            seller_id TEXT PRIMARY KEY,
            seller_name TEXT NOT NULL,
            seller_slug TEXT UNIQUE NOT NULL,
            public_url TEXT NOT NULL,
            created_at TEXT
         )
 """)
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            seller_id TEXT,
            reset_token TEXT,
            reset_token_expires TEXT,
            created_at TEXT
            
        )
""")
    try:
        cursor.execute(
            "ALTER TABLE users ADD COLUMN email_verified INTEGER DEFAULT 0"
    )
    except: 
        pass
    try:
        cursor.execute(
            "ALTER TABLE users ADD COLUMN verification_token TEXT"
    )
    except: 
        pass
    try:
        cursor.execute(
            "ALTER TABLE users ADD COLUMN verification_token_expires TEXT"
    )
    except: 
        pass
    try:
        cursor.execute(
            "ALTER TABLE users ADD COLUMN reset_token TEXT"
        )
    except:
        pass
        
    try:
        cursor.execute(
        "ALTER TABLE users ADD COLUMN reset_token_expires TEXT"
        )
    except:
        pass
        
    conn.commit()
    conn.close()


init_db()


# ============================================================
# BASIC UTILITIES
# ============================================================

def sha256_bytes(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def save_bytes_to_file(file_bytes: bytes, filename_prefix: str) -> str:
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{filename_prefix}_{file_id}.jpg")

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    return file_path


def read_file_bytes(file_path: str) -> bytes:
    with open(file_path, "rb") as f:
        return f.read()

def decode_image(image_bytes: bytes):
    if not image_bytes:
        return None

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")

        rgb = np.array(pil_img)

        # Convert RGB to BGR because the rest of your OpenCV code expects BGR
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        return bgr

    except Exception as e:
        print("IMAGE DECODE FAILED:", str(e))
        return None
# ============================================================
# QUIET ZONE DETECTION
# ============================================================

QUIET_ZONE_CANONICAL_SIZE = 512

# Percentage of the detected boundary excluded from the
# fingerprint region so the physical boundary itself does
# not contaminate the surface texture.
QUIET_ZONE_BORDER_INSET = 0.06

# Conservative acceptance threshold.
QUIET_ZONE_MIN_CONFIDENCE = 0.78

# If two candidates score too similarly, we do not guess.
QUIET_ZONE_MIN_SCORE_MARGIN = 0.08 
def _order_quiet_zone_points(points):
    """
    Return four points in this order:

        top-left
        top-right
        bottom-right
        bottom-left

    This is required before perspective correction.
    """
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(
        points,
        axis=1,
    ).reshape(-1)

    return np.array(
        [
            points[np.argmin(coordinate_sum)],
            points[np.argmin(coordinate_difference)],
            points[np.argmax(coordinate_sum)],
            points[np.argmax(coordinate_difference)],
        ],
        dtype=np.float32,
    )


def _quiet_zone_geometry(points):
    """
    Validate the basic geometry of a four-corner candidate.

    Returns:
        ordered_points,
        side_lengths,
        aspect_score,
        right_angle_score,
        area

    Returns None when geometry is invalid.
    """
    ordered = _order_quiet_zone_points(points)

    side_lengths = np.linalg.norm(
        np.roll(ordered, -1, axis=0) - ordered,
        axis=1,
    )

    if np.any(side_lengths < 1.0):
        return None

    width = float(
        (side_lengths[0] + side_lengths[2]) / 2.0
    )

    height = float(
        (side_lengths[1] + side_lengths[3]) / 2.0
    )

    if width <= 0.0 or height <= 0.0:
        return None

    aspect_ratio = min(width, height) / max(width, height)

    angle_errors = []

    for index in range(4):
        previous = (
            ordered[(index - 1) % 4]
            - ordered[index]
        )

        following = (
            ordered[(index + 1) % 4]
            - ordered[index]
        )

        denominator = (
            np.linalg.norm(previous)
            * np.linalg.norm(following)
        )

        if denominator <= 1e-6:
            return None

        cosine = abs(
            float(
                np.dot(previous, following)
                / denominator
            )
        )

        angle_errors.append(cosine)

    right_angle_score = 1.0 - min(
        1.0,
        float(np.mean(angle_errors)) / 0.35,
    )

    area = abs(
        float(
            cv2.contourArea(ordered)
        )
    )

    return (
        ordered,
        side_lengths,
        aspect_ratio,
        right_angle_score,
        area,
    )
    
def _warp_quiet_zone(
    image,
    points,
    size=QUIET_ZONE_CANONICAL_SIZE,
):
    """
    Perspective-correct the detected physical Quiet Zone
    into a square canonical image.
    """
    ordered = _order_quiet_zone_points(points)

    destination = np.array(
        [
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        ordered,
        destination,
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (size, size),
        flags=cv2.INTER_CUBIC,
    )
    
def _quiet_zone_surface_metrics(warped):
    """
    Measure whether the interior resembles a relatively
    coherent unprinted physical surface.

    These are rejection signals, not proof of authenticity.
    """
    height, width = warped.shape[:2]

    margin = max(
        8,
        int(min(height, width) * 0.08),
    )

    inner = warped[
        margin:height - margin,
        margin:width - margin,
    ]

    inner_gray = cv2.cvtColor(
        inner,
        cv2.COLOR_BGR2GRAY,
    )

    inner_hsv = cv2.cvtColor(
        inner,
        cv2.COLOR_BGR2HSV,
    )

    edges = cv2.Canny(
        cv2.GaussianBlur(
            inner_gray,
            (3, 3),
            0,
        ),
        40,
        120,
    )

    edge_density = float(
        np.mean(edges > 0)
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(
            25,
            int(width * 0.08),
        ),
        minLineLength=max(
            35,
            int(width * 0.15),
        ),
        maxLineGap=8,
    )

    long_line_count = 0

    if lines is not None:
        for line in lines[:, 0]:
            line_length = float(
                np.hypot(
                    line[2] - line[0],
                    line[3] - line[1],
                )
            )

            if line_length >= width * 0.15:
                long_line_count += 1

    saturation_std = float(
        np.std(inner_hsv[:, :, 1])
    )

    value_std = float(
        np.std(inner_hsv[:, :, 2])
    )

    return {
        "edge_density": edge_density,
        "long_line_count": long_line_count,
        "saturation_std": saturation_std,
        "value_std": value_std,
    }
 def extract_quiet_zone(
    image,
    capture_context="factory_registration",
):
    """
    Detect, validate and canonicalise the physical
    Quiet Zone.

    The Quiet Zone is the intentionally unprinted physical
    square reserved for FiberHash surface fingerprinting.

    This function performs:
        1. Candidate discovery
        2. Geometric validation
        3. Candidate de-duplication
        4. Candidate scoring
        5. Ambiguity rejection
        6. Perspective correction
        7. Canonical extraction

    It fails closed.

    It NEVER substitutes an arbitrary image when the
    Quiet Zone cannot be confidently identified.
    """

    # --------------------------------------------------------
    # 1. BASIC INPUT VALIDATION
    # --------------------------------------------------------

    if image is None:
        return {
            "success": False,
            "reason": "INVALID_IMAGE",
            "confidence": 0.0,
            "corners": None,
            "image": None,
            "capture_context": capture_context,
        }

    if not isinstance(image, np.ndarray):
        return {
            "success": False,
            "reason": "INVALID_IMAGE_TYPE",
            "confidence": 0.0,
            "corners": None,
            "image": None,
            "capture_context": capture_context,
        }

    if image.size == 0:
        return {
            "success": False,
            "reason": "EMPTY_IMAGE",
            "confidence": 0.0,
            "corners": None,
            "image": None,
            "capture_context": capture_context,
        }

    if image.ndim != 3 or image.shape[2] != 3:
        return {
            "success": False,
            "reason": "INVALID_IMAGE_CHANNELS",
            "confidence": 0.0,
            "corners": None,
            "image": None,
            "capture_context": capture_context,
        }

    height, width = image.shape[:2]

    # The detector needs enough resolution to identify the
    # physical boundary reliably.
    if width < 600 or height < 600:
        return {
            "success": False,
            "reason": "IMAGE_TOO_SMALL_FOR_QUIET_ZONE_DETECTION",
            "confidence": 0.0,
            "corners": None,
            "image": None,
            "capture_context": capture_context,
        }

    # --------------------------------------------------------
    # 2. PREPARE EDGE IMAGE
    # --------------------------------------------------------

    try:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    except cv2.error:
        return {
            "success": False,
            "reason": "GRAYSCALE_CONVERSION_FAILED",
            "confidence": 0.0,
            "corners": None,
            "image": None,
            "capture_context": capture_context,
        }

    # Several edge configurations make the detector less
    # dependent on one particular lighting/contrast condition.
    configurations = [
        (20, 80, 3),
        (30, 100, 3),
        (40, 120, 5),
        (50, 150, 7),
        (60, 180, 9),
    ]

    candidates = []

    # --------------------------------------------------------
    # 3. CANDIDATE DISCOVERY
    # --------------------------------------------------------

    for lower_threshold, upper_threshold, kernel_size in configurations:

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        edges = cv2.Canny(
            blurred,
            lower_threshold,
            upper_threshold,
        )

        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )

        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:

            if contour is None or len(contour) < 4:
                continue

            contour_area = float(
                cv2.contourArea(contour)
            )

            if not np.isfinite(contour_area):
                continue

            area_ratio = (
                contour_area
                / float(width * height)
            )

            # The Quiet Zone must not be required to occupy a
            # fixed 1% of the entire camera photograph.
            # Full-resolution phone captures contain substantial
            # surrounding image area, so that fixed ratio can
            # discard a genuine physical Quiet Zone before the
            # four-corner geometry checks are reached.
            #
            # Retain a conservative lower bound for very small
            # candidates, while allowing the minimum ratio to fall
            # for large camera frames. The existing geometric and
            # confidence checks remain unchanged.
            min_area_ratio = min(
                0.01,
                max(
                    0.0025,
                    10000.0 / float(width * height),
                ),
            )

            if area_ratio < min_area_ratio:
                continue

            if area_ratio > 0.45:
                continue

            perimeter = float(
                cv2.arcLength(
                    contour,
                    True,
                )
            )

            if not np.isfinite(perimeter):
                continue

            if perimeter <= 0.0:
                continue

            approximation = cv2.approxPolyDP(
                contour,
                0.04 * perimeter,
                True,
            )

            if approximation is None:
                continue

            if len(approximation) != 4:
                continue

            if not cv2.isContourConvex(
                approximation
            ):
                continue

            candidate_points = (
                approximation.reshape(4, 2)
            )

            # ------------------------------------------------
            # 4. GEOMETRIC VALIDATION
            # ------------------------------------------------

            if not np.all(
                np.isfinite(candidate_points)
            ):
                continue

            geometry = _quiet_zone_geometry(
                candidate_points
            )

            if geometry is None:
                continue

            (
                ordered,
                side_lengths,
                aspect_ratio,
                right_angle_score,
                area,
            ) = geometry

            if not np.all(
                np.isfinite(ordered)
            ):
                continue

            if not np.all(
                np.isfinite(side_lengths)
            ):
                continue

            if not np.isfinite(
                aspect_ratio
            ):
                continue

            if not np.isfinite(
                right_angle_score
            ):
                continue

            if aspect_ratio < 0.72:
                continue

            if right_angle_score < 0.70:
                continue

            if area <= 0.0:
                continue

            # ------------------------------------------------
            # 5. INTEGER GEOMETRY FOR OPENCV
            # ------------------------------------------------

            ordered_int = np.round(
                ordered
            ).astype(np.int32)

            if ordered_int.shape != (4, 2):
                continue

            x, y, box_width, box_height = (
                cv2.boundingRect(
                    ordered_int
                )
            )

            if box_width <= 0 or box_height <= 0:
                continue

            # ------------------------------------------------
            # 6. PERSPECTIVE CORRECTION
            # ------------------------------------------------

            try:
                warped = _warp_quiet_zone(
                    image,
                    ordered,
                    QUIET_ZONE_CANONICAL_SIZE,
                )
            except (
                cv2.error,
                ValueError,
                TypeError,
            ):
                continue

            if warped is None:
                continue

            if warped.size == 0:
                continue

            expected_size = (
                QUIET_ZONE_CANONICAL_SIZE,
                QUIET_ZONE_CANONICAL_SIZE,
            )

            if warped.shape[:2] != expected_size:
                continue

            # ------------------------------------------------
            # 7. SURFACE METRICS
            # ------------------------------------------------

            try:
                metrics = _quiet_zone_surface_metrics(
                    warped
                )
            except (
                cv2.error,
                ValueError,
                TypeError,
            ):
                continue

            edge_density = float(
                metrics.get(
                    "edge_density",
                    1.0,
                )
            )

            long_line_count = int(
                metrics.get(
                    "long_line_count",
                    999,
                )
            )

            if not np.isfinite(
                edge_density
            ):
                continue

            # ------------------------------------------------
            # 8. CENTRE POSITION SCORE
            # ------------------------------------------------
            #
            # The 45 × 45 mm guide provides a weak positional
            # prior only.
            #
            # It NEVER defines the extraction region.

            center = np.mean(
                ordered,
                axis=0,
            )

            if not np.all(
                np.isfinite(center)
            ):
                continue

            normalized_center_distance = float(
                np.hypot(
                    (
                        center[0]
                        - width / 2.0
                    ) / (width / 2.0),
                    (
                        center[1]
                        - height / 2.0
                    ) / (height / 2.0),
                )
            )

            center_score = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - (
                        normalized_center_distance
                        / 0.90
                    ),
                ),
            )

            # ------------------------------------------------
            # 9. PHYSICAL SIZE PLAUSIBILITY
            # ------------------------------------------------
            #
            # This is deliberately broad.
            #
            # We do NOT hard-code a pixel size because the
            # physical Quiet Zone can occupy different pixel
            # dimensions depending on camera distance.

            size_score = 1.0

            if area_ratio < 0.015:
                size_score = max(
                    0.0,
                    min(
                        1.0,
                        area_ratio / 0.015,
                    ),
                )

            elif area_ratio > 0.25:
                size_score = max(
                    0.0,
                    min(
                        1.0,
                        1.0
                        - (
                            (area_ratio - 0.25)
                            / 0.20
                        ),
                    ),
                )

            # ------------------------------------------------
            # 10. SURFACE SIGNALS
            # ------------------------------------------------
            #
            # IMPORTANT:
            #
            # These are deliberately WEAK signals.
            #
            # Natural Quiet Zone material contains texture.
            # We therefore must NOT reject it simply because
            # it contains edges or microstructure.
            #
            # A later quality stage can make the stronger
            # "usable for fingerprinting" decision.

            surface_edge_score = max(
                0.0,
                min(
                    1.0,
                    (
                        0.25
                        - edge_density
                    ) / 0.25,
                ),
            )

            straight_structure_score = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - (
                        long_line_count
                        / 20.0
                    ),
                ),
            )

            # ------------------------------------------------
            # 11. FINAL CANDIDATE SCORE
            # ------------------------------------------------
            #
            # Geometry dominates.
            #
            # This is intentional: finding the physical
            # boundary is the primary detection problem.

            score = (
                45.0
                * right_angle_score
                +
                35.0
                * aspect_ratio
                +
                10.0
                * size_score
                +
                5.0
                * center_score
                +
                3.0
                * surface_edge_score
                +
                2.0
                * straight_structure_score
            )

            if not np.isfinite(score):
                continue

            candidates.append(
                {
                    "score": float(score),
                    "corners": ordered.copy(),
                    "warped": warped,
                    "metrics": metrics,
                    "area_ratio": float(area_ratio),
                }
            )

    # --------------------------------------------------------
    # 12. NO CANDIDATES
    # --------------------------------------------------------

    if not candidates:
        return {
            "success": False,
            "reason": "QUIET_ZONE_NOT_DETECTED",
            "confidence": 0.0,
            "corners": None,
            "image": None,
            "capture_context": capture_context,
        }

    # --------------------------------------------------------
    # 13. DE-DUPLICATE SAME-PHYSICAL-ZONE CANDIDATES
    # --------------------------------------------------------
    #
    # Different edge thresholds can produce slightly different
    # contours around the SAME physical Quiet Zone.
    #
    # Those must not be treated as separate competing zones.

    candidates.sort(
        key=lambda candidate:
        candidate["score"],
        reverse=True,
    )

    distinct_candidates = []

    for candidate in candidates:

        candidate_corners = candidate["corners"]

        is_duplicate = False

        candidate_area = max(
            candidate["area_ratio"],
            1e-6,
        )

        candidate_scale = math.sqrt(
            candidate_area
        ) * max(
            width,
            height,
        )

        duplicate_distance_threshold = max(
            8.0,
            candidate_scale * 0.08,
        )

        for existing in distinct_candidates:

            existing_corners = existing["corners"]

            if existing_corners.shape != (4, 2):
                continue

            mean_corner_distance = float(
                np.mean(
                    np.linalg.norm(
                        candidate_corners
                        - existing_corners,
                        axis=1,
                    )
                )
            )

            if (
                np.isfinite(
                    mean_corner_distance
                )
                and mean_corner_distance
                <= duplicate_distance_threshold
            ):
                is_duplicate = True
                break

        if not is_duplicate:
            distinct_candidates.append(
                candidate
            )

    if not distinct_candidates:
        return {
            "success": False,
            "reason": "QUIET_ZONE_CANDIDATE_CLUSTERING_FAILED",
            "confidence": 0.0,
            "corners": None,
            "image": None,
            "capture_context": capture_context,
        }

    # Highest-scoring distinct candidate.
    best = distinct_candidates[0]

    best_score = float(
        best["score"]
    )

    # --------------------------------------------------------
    # 14. CONFIDENCE
    # --------------------------------------------------------

    base_confidence = max(
        0.0,
        min(
            1.0,
            best_score / 100.0,
        ),
    )

    # Only compare against a genuinely DISTINCT candidate.
    if len(distinct_candidates) > 1:

        second_score = float(
            distinct_candidates[1]["score"]
        )

        score_margin = (
            best_score
            - second_score
        )

        margin_confidence = max(
            0.0,
            min(
                1.0,
                score_margin / 20.0,
            ),
        )

        combined_confidence = (
            0.80
            * base_confidence
            +
            0.20
            * margin_confidence
        )

        # Fail closed when the best and second-best physical candidates
        # are too close to call. The configured margin is expressed as
        # confidence units (0.08 = 8 score points out of 100).
        if (
            score_margin / 100.0
            < QUIET_ZONE_MIN_SCORE_MARGIN
        ):
            return {
                "success": False,
                "reason": "QUIET_ZONE_DETECTION_AMBIGUOUS",
                "confidence": round(combined_confidence, 4),
                "corners": None,
                "image": None,
                "capture_context": capture_context,
            }

    else:
        # With only one distinct geometric candidate,
        # there is no ambiguity penalty.
        combined_confidence = (
            base_confidence
        )

    # --------------------------------------------------------
    # 15. FAIL CLOSED ON LOW CONFIDENCE
    # --------------------------------------------------------

    if (
        combined_confidence
        < QUIET_ZONE_MIN_CONFIDENCE
    ):
        return {
            "success": False,
            "reason": "QUIET_ZONE_DETECTION_CONFIDENCE_TOO_LOW",
            "confidence": round(
                combined_confidence,
                4,
            ),
            "corners": None,
            "image": None,
            "capture_context": capture_context,
        }

        # --------------------------------------------------------
    # 16. CANONICAL EXTRACTION
    # --------------------------------------------------------

    canonical_size = (
        QUIET_ZONE_CANONICAL_SIZE
    )

    try:
        canonical = best["warped"]

        if (
            canonical is None
            or canonical.size == 0
        ):
            return {
                "success": False,
                "reason": "CANONICAL_EXTRACTION_FAILED",
                "confidence": 0.0,
                "corners": None,
                "image": None,
                "capture_context": capture_context,
            }

        if canonical.shape[:2] != (
            canonical_size,
            canonical_size,
        ):
            canonical = cv2.resize(
                canonical,
                (
                    canonical_size,
                    canonical_size,
                ),
                interpolation=cv2.INTER_CUBIC,
            )

    except (
        cv2.error,
        ValueError,
        TypeError,
    ):
        return {
            "success": False,
            "reason": "CANONICAL_RESIZE_FAILED",
            "confidence": 0.0,
            "corners": None,
            "image": None,
            "capture_context": capture_context,
        }
    
            

        canonical = cv2.resize(
            canonical,
            (
                canonical_size,
                canonical_size,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

    except (
        cv2.error,
        ValueError,
        TypeError,
    ):
        return {
            "success": False,
            "reason": "CANONICAL_RESIZE_FAILED",
            "confidence": 0.0,
            "corners": None,
            "image": None,
            "capture_context": capture_context,
        }

    # --------------------------------------------------------
    # 17. FINAL RESULT
    # --------------------------------------------------------

    return {
        "success": True,
        "reason": "QUIET_ZONE_DETECTED",
        "confidence": round(
            combined_confidence,
            4,
        ),
        "corners": [
            [
                round(float(point[0]), 2),
                round(float(point[1]), 2),
            ]
            for point in best["corners"]
        ],
        "image": canonical,
        "capture_context": capture_context,
        "metrics": best["metrics"],
    }   

def normalize_image(image, target_size=1024):
    if image is None:
        return None

    h, w = image.shape[:2]

    # Centre square crop
    side = min(h, w)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    square = image[y1:y1 + side, x1:x1 + side]

    # Resize every image to the same size before SIFT
    interpolation = cv2.INTER_AREA if side > target_size else cv2.INTER_CUBIC
    resized = cv2.resize(
        square,
        (target_size, target_size),
        interpolation=interpolation
    )

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for lighting/shadow normalisation
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)
    return enhanced
def crop_largest_contour_region(image, min_area_ratio=0.02):
    if image is None:
        return None

    original = image.copy()
    height, width = image.shape[:2]
    image_area = height * width

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return original

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < image_area * min_area_ratio:
        return original

    x, y, w, h = cv2.boundingRect(largest)

    padding = 10
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, width)
    y2 = min(y + h + padding, height)

    cropped = original[y1:y2, x1:x2]

    return cropped


def isolate_package_patch(image):
    return crop_largest_contour_region(image, min_area_ratio=0.01)


def isolate_seal_area(image):
    return crop_largest_contour_region(image, min_area_ratio=0.02)

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


# ============================================================
# IMAGE QUALITY CHECKS
# ============================================================

def check_blur(image):
    if image is None:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return round(float(variance), 2)


def check_brightness(image):
    if image is None:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return round(brightness, 2)


def check_glare(image):
    """
    Simple glare estimate:
    counts very bright pixels as a percentage of the image.
    Higher score means more glare/reflection.
    """
    if image is None:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(gray > 245)
    total_pixels = gray.size

    glare_score = (bright_pixels / total_pixels) * 100
    return round(float(glare_score), 2)


def get_resolution(image):
    if image is None:
        return 0, 0

    height, width = image.shape[:2]
    return width, height


def quality_assessment(image):
    """
    Produces quality metrics and a practical scan quality score.
    Score is not forensic truth; it is a usability/scan-quality indicator.
    """
    if image is None:
        return {
            "quality_score": 0.0,
            "blur_variance": 0.0,
            "brightness": 0.0,
            "glare_score": 0.0,
            "width": 0,
            "height": 0,
            "quality_flags": ["INVALID_IMAGE"],
        }

    blur_variance = check_blur(image)
    brightness = check_brightness(image)
    glare_score = check_glare(image)
    width, height = get_resolution(image)

    flags = []
    score = 100.0

    if width < 300 or height < 300:
        flags.append("LOW_RESOLUTION")
        score -= 25

    if blur_variance < 80:
        flags.append("IMAGE_TOO_BLURRY")
        score -= 30

    if brightness < 45:
        flags.append("IMAGE_TOO_DARK")
        score -= 20

    if brightness > 220:
        flags.append("IMAGE_TOO_BRIGHT")
        score -= 20

    if glare_score > 8:
        flags.append("GLARE_DETECTED")
        score -= 20

    score = max(0.0, min(100.0, score))

    return {
        "quality_score": round(score, 2),
        "blur_variance": blur_variance,
        "brightness": brightness,
        "glare_score": glare_score,
        "width": width,
        "height": height,
        "quality_flags": flags,
    }


# ============================================================
# FEATURE MATCHING + HOMOGRAPHY
# ============================================================

def sift_match(master_gray, scan_gray):
    """
    SIFT + FLANN matching + Lowe ratio test + homography inlier check.

    This is still MVP/R&D-level, but stronger than a plain feature-count comparison.
    """
    result = {
        "trust_score": 0.0,
        "good_match_count": 0,
        "inlier_count": 0,
        "total_keypoints_master": 0,
        "total_keypoints_scan": 0,
        "homography_found": False,
        "match_quality": "insufficient",
    }

    if master_gray is None or scan_gray is None:
        return result

    try:
        sift = cv2.SIFT_create()
    except Exception:
        return result

    kp1, des1 = sift.detectAndCompute(master_gray, None)
    kp2, des2 = sift.detectAndCompute(scan_gray, None)

    result["total_keypoints_master"] = len(kp1) if kp1 is not None else 0
    result["total_keypoints_scan"] = len(kp2) if kp2 is not None else 0

    if des1 is None or des2 is None:
        return result

    if len(kp1) < 8 or len(kp2) < 8:
        return result

    FLANN_INDEX_KDTREE = 1

    index_params = dict(
        algorithm=FLANN_INDEX_KDTREE,
        trees=5
    )

    search_params = dict(
        checks=50
    )

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        raw_matches = flann.knnMatch(des1, des2, k=2)
    except Exception:
        return result

    good_matches = []

    for pair in raw_matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    good_match_count = len(good_matches)
    result["good_match_count"] = good_match_count

    inlier_count = 0
    homography_found = False

    if good_match_count >= 8:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        try:
            matrix, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                5.0
            )

            if matrix is not None and mask is not None:
                homography_found = True
                inlier_count = int(mask.sum())

        except Exception:
            homography_found = False
            inlier_count = 0

    result["inlier_count"] = inlier_count
    result["homography_found"] = homography_found

    if good_match_count > 0:
        inlier_ratio = inlier_count / good_match_count
    else:
        inlier_ratio = 0.0
    # Phone-friendly R&D scoring:
    # Good matches and geometric inliers matter more than raw keypoint percentage.
    match_score = min(good_match_count / 25, 1.0) * 35
    inlier_score = min(inlier_count / 20, 1.0) * 45
    geometry_score = inlier_ratio * 20
    trust_score = match_score + inlier_score + geometry_score
    trust_score = max(0.0, min(100.0, trust_score))

    result["trust_score"] = round(trust_score, 2)
    if result["trust_score"] >= 60 and inlier_count >= 10:
        result["match_quality"] = "strong"
    elif result["trust_score"] >= 35 and inlier_count >= 6:
        result["match_quality"] = "moderate"
    elif result["trust_score"] >= 20:
        result["match_quality"] = "weak"
    else:
        result["match_quality"] = "poor"
    return result


# ============================================================
# DECISION LOGIC
# ============================================================

def make_decision(trust_score, quality_score, inlier_count, quality_flags):
    """
    Pass / Review / Fail logic.
    This is intentionally conservative for an R&D MVP.
    """

    if "INVALID_IMAGE" in quality_flags:
        return {
            "decision": "fail",
            "is_match": False,
            "message": "Invalid image file. Please upload a clear image.",
        }
        
    if trust_score >= 80 and inlier_count >= 20:
        return {
               "decision": "pass",
               "is_match": True,
               "message": "MATCH: verified (image quality warning)."
       }
        
    if "IMAGE_TOO_BLURRY" in quality_flags:
        return {
            "decision": "review",
            "is_match": False,
            "message": "Scan quality is too blurry. Please rescan under better conditions.",
        }

    if quality_score < 50:
        return {
            "decision": "review",
            "is_match": False,
            "message": "Scan quality is weak. Please rescan before making a final decision.",
        }

    if trust_score >= 60 and inlier_count >= 10:
        return {
            "decision": "pass",
            "is_match": True,
            "message": "VERIFIED GENUINE",
        }

    if trust_score >= 35 and inlier_count >= 6:
        return {
            "decision": "review",
            "is_match": False,
            "message": "POSSIBLE MATCH. Manual review recommended.",
        }

    return {
        "decision": "fail",
        "is_match": False,
        "message": "MISMATCH: POSSIBLE COUNTERFEIT OR WRONG PRODUCT",
    }
    
def calculate_ai_risk(package_match, seal_match, package_result, seal_result):
        package_result = package_result or {} 
        seal_result = seal_result or {}
        package_quality_flags = package_result.get("quality", {}).get("quality_flags", [])
        seal_quality_flags = seal_result.get("quality", {}).get("quality_flags", [])

        package_trust = package_result.get("trust_score", 0)
        seal_trust = seal_result.get("trust_score", 0)

        reasons = []

        if package_match and seal_match:
           risk_level = "low"
           recommended_action = "Accept verification result."
           reasons.append("Package and seal both matched the registered unit.")

        elif package_match and not seal_match:
           risk_level = "high"
           recommended_action = "Flag as possible tampering, resealing, or seal replacement."
           reasons.append("Package matched but seal did not match.")

        elif not package_match and seal_match:
           risk_level = "high"
           recommended_action = "Flag as possible component mismatch or suspicious seal transfer."
           reasons.append("Seal matched but package did not match.")

        else:
           risk_level = "high"
           recommended_action = "Reject or escalate as possible counterfeit or unknown product."
           reasons.append("Both package and seal failed verification.")

        if "IMAGE_TOO_BLURRY" in package_quality_flags or "IMAGE_TOO_BLURRY" in seal_quality_flags:
           reasons.append("One or more scans were blurry.")

        if "GLARE_DETECTED" in package_quality_flags or "GLARE_DETECTED" in seal_quality_flags:
           reasons.append("Glare was detected in one or more scans.")
        if package_trust < 35 or seal_trust < 35:
           reasons.append("One or more trust scores were below review threshold.")

        return {
          "risk_level": risk_level,
          "risk_score": 10 if risk_level == "low" else 85,
          "risk_reasons": reasons,
          "recommended_action": recommended_action,
    }


# ============================================================
# DATABASE HELPERS
# ============================================================

def add_timeline_event(
    challenge_id,
    event_type,
    event_title,
    event_description="",
    actor_type="",
    actor_id="",
    old_status="",
    new_status=""
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO challenge_timeline (
            event_id,
            challenge_id,
            event_type,
            event_title,
            event_description,
            actor_type,
            actor_id,
            old_status,
            new_status,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            challenge_id,
            event_type,
            event_title,
            event_description,
            actor_type,
            actor_id,
            old_status,
            new_status,
            now_iso()
        )
    )

    conn.commit()
    conn.close()
    
def create_product_record(product_name, brand, batch_code, master_image_path, master_image_hash):
    product_id = str(uuid.uuid4())

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO products (
            id,
            product_name,
            brand,
            batch_code,
            master_image_path,
            master_image_hash,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            product_id,
            product_name,
            brand,
            batch_code,
            master_image_path,
            master_image_hash,
            now_iso(),
        ),
    )

    conn.commit()
    conn.close()

    return product_id


def create_unit_record(unit_id, order_id, seller_id, buyer_id, marketplace_name, product_id, product_name, brand, batch_code, package_image_path, package_image_hash, seal_image_path, seal_image_hash):
    unit_id = unit_id.strip()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO unit_fingerprints (
            unit_id,
            order_id,
            seller_id,
            buyer_id,
            marketplace_name,
            product_id,
            product_name,
            brand,
            batch_code,
            package_image_path,
            package_image_hash,
            seal_image_path,
            seal_image_hash,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            unit_id, 
            order_id,
            seller_id,
            buyer_id,
            marketplace_name,
            product_id,
            product_name,
            brand,
            batch_code,
            package_image_path,
            package_image_hash,
            seal_image_path,
            seal_image_hash,
            now_iso(),
        ),
    )

    conn.commit()
    conn.close()

    return unit_id

def get_unit_record(unit_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            unit_id,
            order_id,
            seller_id,
            buyer_id,
            marketplace_name,
            product_id,
            product_name,
            brand,
            batch_code,
            package_image_path,
            package_image_hash,
            seal_image_path,
            seal_image_hash,
            created_at
        FROM unit_fingerprints
        WHERE unit_id = ?
        """,
        (unit_id,),
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "unit_id": row[0],
        "order_id": row[1],
        "seller_id": row[2],
        "buyer_id": row[3],
        "marketplace_name": row[4],
        "product_id": row[5],
        "product_name": row[6],
        "brand": row[7],
        "batch_code": row[8],
        "package_image_path": row[9],
        "package_image_hash": row[10],
        "seal_image_path": row[11],
        "seal_image_hash": row[12],
        "created_at": row[13],
    }
def log_unit_verification_event(unit_id, decision, package_match, seal_match, trust_score, ai_risk):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO unit_verification_events (
            unit_id,
            decision,
            package_match,
            seal_match,
            trust_score,
            ai_risk_level,
            ai_risk_score,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            unit_id,
            decision,
            int(package_match),
            int(seal_match),
            trust_score,
            ai_risk.get("risk_level"),
            ai_risk.get("risk_score"),
            now_iso(),
        ),
    )

    conn.commit()
    conn.close()
    
def create_challenge_case(
    order_id,
    marketplace_name,
    seller_id,
    buyer_id,
    unit_id,
    case_type,
    trigger_reason,
    verification_decision,
    package_match,
    seal_match,
    trust_score,
    risk_level,
    recommended_action,
):
    case_id = str(uuid.uuid4())

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO challenge_cases (
            case_id,
            order_id,
            marketplace_name,
            seller_id,
            buyer_id,
            unit_id,
            case_type,
            case_status,
            trigger_reason,
            verification_decision,
            package_match,
            seal_match,
            trust_score,
            risk_level,
            recommended_action,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case_id,
            order_id,
            marketplace_name,
            seller_id,
            buyer_id,
            unit_id,
            case_type,
            "open",
            trigger_reason,
            verification_decision,
            int(package_match),
            int(seal_match),
            trust_score,
            risk_level,
            recommended_action,
            now_iso(),
        ),
    )

    conn.commit()
    conn.close()

    return case_id
def get_product(product_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            product_name,
            brand,
            batch_code,
            master_image_path,
            master_image_hash,
            created_at
        FROM products
        WHERE id = ?
        """,
        (product_id,),
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "product_name": row[1],
        "brand": row[2],
        "batch_code": row[3],
        "master_image_path": row[4],
        "master_image_hash": row[5],
        "created_at": row[6],
    }


def scan_hash_seen_before(scan_hash):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM verification_events
        WHERE scan_image_hash = ?
        """,
        (scan_hash,),
    )

    count = cursor.fetchone()[0]
    conn.close()

    return count > 0


def log_verification_event(product_id, result):
    event_id = str(uuid.uuid4())

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO verification_events (
            id,
            product_id,
            status,
            decision,
            is_match,
            trust_score,
            quality_score,
            blur_variance,
            brightness,
            glare_score,
            resolution_width,
            resolution_height,
            inlier_count,
            good_match_count,
            total_keypoints_master,
            total_keypoints_scan,
            scan_image_hash,
            replay_warning,
            message,
            created_at,
            raw_result_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            product_id,
            result.get("status"),
            result.get("decision"),
            1 if result.get("is_match") else 0,
            safe_float(result.get("trust_score")),
            safe_float(result.get("quality", {}).get("quality_score")),
            safe_float(result.get("quality", {}).get("blur_variance")),
            safe_float(result.get("quality", {}).get("brightness")),
            safe_float(result.get("quality", {}).get("glare_score")),
            int(result.get("quality", {}).get("width", 0)),
            int(result.get("quality", {}).get("height", 0)),
            int(result.get("matching", {}).get("inlier_count", 0)),
            int(result.get("matching", {}).get("good_match_count", 0)),
            int(result.get("matching", {}).get("total_keypoints_master", 0)),
            int(result.get("matching", {}).get("total_keypoints_scan", 0)),
            result.get("scan_image_hash"),
            1 if result.get("replay_warning") else 0,
            result.get("message"),
            now_iso(),
            json.dumps(result),
        ),
    )

    conn.commit()
    conn.close()

    return event_id


# ============================================================
# CORE VERIFICATION FUNCTION
# ============================================================

def run_verification(master_bytes, scan_bytes, product_id=None):
    master_hash = sha256_bytes(master_bytes)
    scan_hash = sha256_bytes(scan_bytes)

    master_raw = decode_image(master_bytes)
    scan_raw = decode_image(scan_bytes)

    quality = quality_assessment(scan_raw)

    master_gray = normalize_image(master_raw)
    scan_gray = normalize_image(scan_raw)

    matching = sift_match(master_gray, scan_gray)

    decision = make_decision(
        trust_score=matching["trust_score"],
        quality_score=quality["quality_score"],
        inlier_count=matching["inlier_count"],
        quality_flags=quality["quality_flags"],
    )

    replay_warning = scan_hash_seen_before(scan_hash)

    result = {
        "status": "success",
        "product_id": product_id,
        "decision": decision["decision"],
        "is_match": decision["is_match"],
        "trust_score": matching["trust_score"],
        "threshold_policy": {
            "pass": "trust_score >= 60 and inlier_count >= 10",
            "review": "trust_score >= 35 and inlier_count >= 6, or weak image quality",
            "fail": "below review threshold or invalid image",
        },
        "message": decision["message"],
        "quality": quality,
        "matching": matching,
        "master_image_hash": master_hash,
        "scan_image_hash": scan_hash,
        "replay_warning": replay_warning,
        "replay_message": "This exact scan image has been submitted before." if replay_warning else "No duplicate scan detected.",
        "created_at": now_iso(),
    }

    event_id = log_verification_event(product_id, result)

    result["event_id"] = event_id

    return result


# ============================================================
# FLEXIBLE MULTIPART FILE READER
# ============================================================

async def get_uploaded_file_bytes(form, possible_names):
    for name in possible_names:
        if name in form:
            file_obj = form[name]

            if hasattr(file_obj, "read"):
                file_bytes = await file_obj.read()
                return file_bytes, name

    return None, None


# ============================================================
# API ENDPOINTS
# ============================================================
# ============================================================
# AUTH HELPERS
# ============================================================

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
def require_admin_user(request: Request):
    authorization = (
        request.headers.get("authorization")
        or request.headers.get("Authorization")
    )

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization token")

    authorization = authorization.strip()

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization token")

    token = authorization.split(" ", 1)[1].strip()

    if not token:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")

        return payload

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")   
        
@app.post("/api/v1/auth/register")
async def register_user(
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    seller_name: str = Form(None)
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT email FROM users WHERE email = ?",
        (email,)
    )

    existing = cursor.fetchone()

    if existing:
        conn.close()
        raise HTTPException(
            status_code=409,
            detail= "Email already exists"
        )
    verification_token = str(uuid.uuid4())

    verification_token_expires = (
    datetime.utcnow() + timedelta(hours=24)
    ).isoformat()
    user_id = str(uuid.uuid4())
    seller_id = None
    
    if role == "seller" and seller_name:
        seller_id = str(uuid.uuid4())

        seller_slug = (
            seller_name.lower()
            .replace(" ", "-")
            .replace(".", "")
        )

        public_url = f"/seller/{seller_slug}"

        cursor.execute(
            """
            INSERT INTO sellers
            (seller_id, seller_name, seller_slug, public_url, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                seller_id,
                seller_name,
                seller_slug,
                public_url,
                datetime.utcnow().isoformat()
            )
        )

    cursor.execute(
        """
        INSERT INTO users
        (user_id, email, password_hash, role, seller_id,email_verified,verification_token,verification_token_expires, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            email,
            hash_password(password),
            role,
            seller_id,
            0,
            verification_token,
            verification_token_expires,
            datetime.utcnow().isoformat()
        )
    )

    conn.commit()
    conn.close()
    
    verification_link = (
        f"https://fiber-hash-seal-lock-03xeqt.flutterflow.app/verify-email"
        f"?token={verification_token}"
    )
    send_verification_email(email, verification_link)
    return {
        "success": True,
        "user_id": user_id,
        "seller_id": seller_id,
        "verification_link": verification_link
    } 
    
@app.get("/api/v1/auth/verify-email")
async def verify_email(token: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT user_id, verification_token_expires
        FROM users
        WHERE verification_token = ?
          AND email_verified = 0
        """,
        (token,)
    )

    user = cursor.fetchone()

    if not user:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="Invalid verification token"
        )

    user_id, expires = user

    if datetime.utcnow() > datetime.fromisoformat(expires):
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="Verification token has expired"
        )

    cursor.execute(
        """
        UPDATE users
        SET email_verified = 1,
            verification_token = NULL,
            verification_token_expires = NULL
        WHERE user_id = ?
        """,
        (user_id,)
    )

    conn.commit()
    conn.close()

    return {
        "success": True,
        "message": "Email verified successfully"
    }   
@app.post("/api/v1/auth/login")
async def login_user(
    email: str = Form(...),
    password: str = Form(...)
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT user_id, email, password_hash, role, seller_id,email_verified
        FROM users
        WHERE email = ?
        """,
        (email,)
    )

    user = cursor.fetchone()

    conn.close()
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail= "Invalid email or password"
        )

    user_id, email, password_hash, role, seller_id, email_verified = user

    if not verify_password(password, password_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
    if not email_verified:
        raise HTTPException(
            status_code=403,
            detail="Please verify your email before logging in."
        )

    access_token = create_access_token({
        "sub": user_id,
        "role": role
    })
    
    seller_name = None
    if seller_id:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT seller_name
            FROM sellers
            WHERE seller_id = ?
        """, (seller_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            seller_name = row[0]

    return {
        "success": True,
        "access_token": access_token,
        "user_id": user_id,
        "role": role,
        "seller_id": seller_id,
        "seller_name": seller_name
    }
    
def send_reset_email(recipient_email: str, reset_link: str):
    api_key = os.getenv("RESEND_API_KEY")
    sender = os.getenv("EMAIL_FROM")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "from": sender,
        "to": [recipient_email],
        "subject": "ChallengeProof Password Reset",
        "html": f"""
        <h2>ChallengeProof Password Reset</h2>
        <p>You requested to reset your password.</p>
        <p>Click the link below to choose a new password:</p>
        <p><a href="{reset_link}">{reset_link}</a></p>
        <p>This link expires in 1 hour.</p>
        <p>If you did not request this, you can safely ignore this email.</p>
        """,
    }

    response = requests.post(
        "https://api.resend.com/emails",
        headers=headers,
        json=payload,
    )

    return response   
    
def send_verification_email(recipient_email: str, verification_link: str):
    api_key = os.getenv("RESEND_API_KEY")
    sender = os.getenv("EMAIL_FROM")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "from": sender,
        "to": [recipient_email],
        "subject": "ChallengeProof Email Verification",
        "html": f"""
        <h2>Welcome to ChallengeProof</h2>
        <p>Please verify your email address to activate your account.</p>
        <p>
            <a href="{verification_link}">
                Verify My Email
            </a>
        </p>
        <p>This verification link expires in 24 hours.</p>
        <p>If you did not create this account, you can safely ignore this email.</p>
        """,
    }

    response = requests.post(
        "https://api.resend.com/emails",
        headers=headers,
        json=payload,
    )

    return response   
    
@app.post("/api/v1/auth/forgot-password")
async def forgot_password(
    email: str = Form(...)
):
    email = email.strip().lower()
    email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    if not re.match(email_pattern, email):
        return {
            "success": False,
            "message": "Please enter a valid email address."
        }
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT user_id
        FROM users
        WHERE email = ?
        """,
        (email,)
    )

    user = cursor.fetchone()

    if not user:
        conn.close()
        return {
            "success": True,
            "message": "If the email exists, a reset link has been generated."
        }

    reset_token = str(uuid.uuid4())
    reset_token_expires = (
        datetime.utcnow() + timedelta(hours=1)
    ).isoformat()

    cursor.execute(
        """
        UPDATE users
        SET reset_token = ?,
            reset_token_expires = ?
        WHERE email = ?
        """,
        (
            reset_token,
            reset_token_expires,
            email
        )
    )

    conn.commit()
    conn.close()

    reset_link = (
        f"https://fiber-hash-seal-lock-03xeqt.flutterflow.app/resetPasswordPage"
        f"?token={reset_token}"
    )
    try:
        send_reset_email(email, reset_link) 
    except Exception as e:
        print(f"Email send failed: {e}")

    return {
        "success": True,
        "message": "Password reset link generated.",
        "reset_link": reset_link
    }    

    return {
        "success": True,
        "message": "Password reset link generated.",
        "reset_link": reset_link
    }  
    
@app.post("/api/v1/auth/reset-password")
async def reset_password(
    token: str = Form(...),
    new_password: str = Form(...)
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT user_id, reset_token_expires
        FROM users
        WHERE reset_token = ?
        """,
        (token,)
    )

    user = cursor.fetchone()

    if not user:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="Invalid reset token"
        )

    user_id, reset_token_expires = user

    if datetime.utcnow() > datetime.fromisoformat(reset_token_expires):
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="Reset token has expired"
        )

    cursor.execute(
        """
        UPDATE users
        SET password_hash = ?,
            email_verified = 1,
            reset_token = NULL,
            reset_token_expires = NULL
        WHERE user_id = ?
        """,
        (
            hash_password(new_password),
            user_id
        )
    )

    conn.commit()
    conn.close()

    return {
        "success": True,
        "message": "Password reset successfully."
    }    
@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "FiberHash / Metalens Authentication API is running.",
        "docs": "/docs",
        "verify_direct": "/api/v1/verify",
        "register_product": "/api/v1/products/register",
        "verify_product": "/api/v1/products/verify",
        "debug_upload": "/api/v1/debug-upload",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": now_iso(),
    }


@app.post("/api/v1/verify")
async def verify_direct(
    master_image: UploadFile = File(...),
    field_scan: UploadFile = File(...)
):
    """
    Direct comparison endpoint.
    FlutterFlow sends two uploaded files:
    - master_image
    - field_scan

    This endpoint does not require a product database record.
    """

    try:
        master_bytes = await master_image.read()
        field_bytes = await field_scan.read()

        result = run_verification(master_bytes, field_bytes)

        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "decision": "fail",
                "is_match": False,
                "trust_score": 0.0,
                "message": str(e),
            },
        )
# The old centre-crop / largest-square ROI path has intentionally been removed.
# Quiet Zone extraction must always be performed by extract_quiet_zone() on the
# ORIGINAL camera frame. That detector owns candidate discovery, geometry,
# perspective correction and canonical extraction.
    
    
@app.post("/api/v1/units/verify")
async def verify_unit(
    unit_id: str = Form(...),
    package_scan: UploadFile = File(...),
    seal_scan: UploadFile = File(...),
    package_capture_context: str = Form("consumer_scan"),
    seal_capture_context: str = Form("consumer_scan"),
):
    try:
        unit = get_unit_record(unit_id)
        if not unit:
            print("===== UNIT NOT FOUND DEBUG =====")
            print("received unit_id:", unit_id)
            print("unit_id length:", len(unit_id) if unit_id else 0)
            print("===============================")
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "decision": "fail",
                    "package_match": False,
                    "seal_match": False,
                    "trust_score": 0,
                    "message": "Unit not found",
                    "received_unit_id": unit_id,
                    "ai_risk": {
                        "risk_level": "high",
                        "risk_score": 85,
                        "recommended_action": "Unit ID was not found in the registered unit database.",
                        "risk_reasons": [
                            "The unit_id sent during verification does not match a registered public unit ID."
                        ],
                    },
                },
            )

        package_scan_bytes = await package_scan.read()
        seal_scan_bytes = await seal_scan.read()

        package_scan_img_raw = decode_image(package_scan_bytes)
        seal_scan_img_raw = decode_image(seal_scan_bytes)

        if package_scan_img_raw is None or seal_scan_img_raw is None:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "decision": "fail",
                    "package_match": False,
                    "seal_match": False,
                    "trust_score": 0,
                    "message": "Package or seal scan could not be decoded.",
                },
            )

        package_qz_result = extract_quiet_zone(
            package_scan_img_raw,
            capture_context=package_capture_context,
        )
        seal_qz_result = extract_quiet_zone(
            seal_scan_img_raw,
            capture_context=seal_capture_context,
        )

        if not package_qz_result.get("success") or not seal_qz_result.get("success"):
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "decision": "review",
                    "package_match": False,
                    "seal_match": False,
                    "trust_score": 0,
                    "message": "One or both scans failed Quiet Zone detection.",
                    "package_quiet_zone": {
                        "success": package_qz_result.get("success", False),
                        "reason": package_qz_result.get("reason"),
                        "confidence": package_qz_result.get("confidence", 0.0),
                    },
                    "seal_quiet_zone": {
                        "success": seal_qz_result.get("success", False),
                        "reason": seal_qz_result.get("reason"),
                        "confidence": seal_qz_result.get("confidence", 0.0),
                    },
                },
            )

        package_scan_img = package_qz_result.get("image")
        seal_scan_img = seal_qz_result.get("image")

        if package_scan_img is None or seal_scan_img is None:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "decision": "review",
                    "package_match": False,
                    "seal_match": False,
                    "trust_score": 0,
                    "message": "Quiet Zone detection succeeded but canonical extraction returned no image.",
                },
            )

        os.makedirs("debug_rois", exist_ok=True)
        cv2.imwrite("debug_rois/verify_package_quiet_zone.jpg", package_scan_img)
        cv2.imwrite("debug_rois/verify_seal_quiet_zone.jpg", seal_scan_img)

        package_ok, package_encoded = cv2.imencode(".jpg", package_scan_img)
        seal_ok, seal_encoded = cv2.imencode(".jpg", seal_scan_img)

        if not package_ok or not seal_ok:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "decision": "fail",
                    "package_match": False,
                    "seal_match": False,
                    "trust_score": 0,
                    "message": "Canonical Quiet Zone extraction could not be encoded.",
                },
            )

        package_scan_bytes = package_encoded.tobytes()
        seal_scan_bytes = seal_encoded.tobytes()
        print("===== VERIFY DEBUG =====")
        print("unit_id:", unit_id)
        print("unit record:", unit)
        print("package_image_path:", unit.get("package_image_path"))
        print("seal_image_path:", unit.get("seal_image_path"))
        print("package_scan received:", package_scan is not None)
        print("seal_scan received:", seal_scan is not None)
        print("========================")
        
        if not unit.get("package_image_path") or not unit.get("seal_image_path"):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "decision": "fail",
                    "package_match": False,
                    "seal_match": False,
                    "trust_score": 0,
                    "message": "Package and seal baselines are not registered for this unit yet."
                },
            )

        package_result = run_verification(
        master_bytes=read_file_bytes(unit["package_image_path"]),
        scan_bytes=package_scan_bytes,
        product_id=unit_id,
   )

        seal_result = run_verification(
            master_bytes=read_file_bytes(unit["seal_image_path"]),
            scan_bytes=seal_scan_bytes,
            product_id=unit_id,
        )
        print("===== UNIT VERIFY MATCH DEBUG =====")
        print("unit_id:", unit_id)
        print("package decision:", package_result.get("decision"))
        print("package trust:", package_result.get("trust_score"))
        print("package inliers:", package_result.get("matching", {}).get("inlier_count"))
        print("package good matches:", package_result.get("matching", {}).get("good_match_count"))
        print("package master keypoints:", package_result.get("matching", {}).get("total_keypoints_master"))
        print("package scan keypoints:", package_result.get("matching", {}).get("total_keypoints_scan"))
        print("seal decision:", seal_result.get("decision"))
        print("seal trust:", seal_result.get("trust_score"))
        print("seal inliers:", seal_result.get("matching", {}).get("inlier_count"))
        print("seal good matches:", seal_result.get("matching", {}).get("good_match_count"))
        print("seal master keypoints:", seal_result.get("matching", {}).get("total_keypoints_master"))
        print("seal scan keypoints:", seal_result.get("matching", {}).get("total_keypoints_scan"))
        print("===================================")
        package_match = package_result["decision"] == "pass"
        seal_match = seal_result["decision"] == "pass"

        if package_match and seal_match:
            decision = "pass"
            trust_score = 100.0

        elif package_match or seal_match:
            decision = "review"
            trust_score = 50.0

        else:
            decision = "fail"
            trust_score = 0.0
        ai_risk = calculate_ai_risk(
            package_match,
            seal_match,
            package_result,
            seal_result,
        ) 
        log_unit_verification_event(
            unit_id,
            decision,
            package_match,
            seal_match,
            trust_score,
            ai_risk,
        )
        case_id = create_challenge_case(
            order_id=unit.get("order_id") or f"ORDER-{unit_id}",
            marketplace_name=unit.get("marketplace_name") or "UNKNOWN_MARKETPLACE",
            seller_id=unit.get("seller_id") or "UNKNOWN_SELLER",
            buyer_id=unit.get("buyer_id") or "UNKNOWN_BUYER",
            unit_id=unit_id,
            case_type="verification_challenge",
            trigger_reason="verification_result",
            verification_decision=decision,
            package_match=package_match,
            seal_match=seal_match,
            trust_score=trust_score,
            risk_level=ai_risk.get("risk_level"),
            recommended_action=ai_risk.get("recommended_action"),
        )
        
        print("challenge case_id:", case_id)

        seller_id = unit.get("seller_id")
        order_id = unit.get("order_id")

        if seller_id and order_id:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT challenge_id
                FROM challenge_requests
                WHERE seller_id = ?
                  AND order_id = ?
                  AND challenge_status = 'open_accepted_by_seller'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (seller_id, order_id),
            )

            challenge_row = cursor.fetchone()

            if challenge_row:
                if decision == "pass":
                    cursor.execute(
                        """
                        UPDATE seller_trust_metrics
                        SET
                            passed_verifications = passed_verifications + 1,
                            last_updated = ?
                        WHERE seller_id = ?
                        """,
                        (now_iso(), seller_id),
                    )

                    new_status = "verified_passed"
                else:
                    cursor.execute(
                        """
                        UPDATE seller_trust_metrics
                        SET
                            failed_verifications = failed_verifications + 1,
                            last_updated = ?
                        WHERE seller_id = ?
                        """,
                        (now_iso(), seller_id),
                    )

                    new_status = "verified_failed"

                cursor.execute(
                    """
                    UPDATE challenge_requests
                    SET challenge_status = ?
                    WHERE challenge_id = ?
                    """,
                    (new_status, challenge_row[0]),
                )

            conn.commit()
            conn.close()
       
        return {
            "status": "verified",
            "case_id": case_id,
            "decision": decision,
            "package_match": package_match,
            "seal_match": seal_match,
            "trust_score": trust_score,
            "package_trust_score": package_result.get("trust_score", 0),
            "seal_trust_score": seal_result.get("trust_score", 0),
            "package_inlier_count": package_result.get("matching", {}).get("inlier_count", 0),
            "seal_inlier_count": seal_result.get("matching", {}).get("inlier_count", 0),
            "package_good_match_count": package_result.get("matching", {}).get("good_match_count", 0),
            "seal_good_match_count": seal_result.get("matching", {}).get("good_match_count", 0),
            "package_keypoints_master": package_result.get("matching", {}).get("total_keypoints_master", 0),
            "package_keypoints_scan": package_result.get("matching", {}).get("total_keypoints_scan", 0),
            "seal_keypoints_master": seal_result.get("matching", {}).get("total_keypoints_master", 0),
            "seal_keypoints_scan": seal_result.get("matching", {}).get("total_keypoints_scan", 0),
            "ai_risk": ai_risk,
            "package_result": package_result,
            "seal_result": seal_result,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "decision": "fail",
                "message": str(e),
            },
        )
@app.post("/api/v1/products/register")
async def register_product(
    product_name: str = Form("Unnamed Product"),
    brand: str = Form("Unknown Brand"),
    batch_code: str = Form("Unknown Batch"),
    master_image: UploadFile = File(...),
):
    """
    Registers/mints a product reference.
    Stores the master image and product metadata.
    """

    try:
        master_bytes = await master_image.read()

        if not master_bytes:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No master image received.",
                },
            )

        master_raw = decode_image(master_bytes)

        if master_raw is None:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Master image could not be decoded.",
                },
            )

        master_hash = sha256_bytes(master_bytes)
        master_path = save_bytes_to_file(master_bytes, "master")

        product_id = create_product_record(
            product_name=product_name,
            brand=brand,
            batch_code=batch_code,
            master_image_path=master_path,
            master_image_hash=master_hash,
        )

        return {
            "status": "success",
            "message": "Product master reference registered.",
            "product_id": product_id,
            "product_name": product_name,
            "brand": brand,
            "batch_code": batch_code,
            "master_image_hash": master_hash,
            "created_at": now_iso(),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
            },
        )

@app.post("/api/v1/units/register")
async def register_unit(
    unit_id: str = Form(...),
    order_id: str = Form(...),
    seller_id: str = Form(...),
    buyer_id: str = Form(...),
    marketplace_name: str = Form(...),
    product_id: str = Form(...),
    product_name: str = Form(...),
    brand: str = Form(...),
    batch_code: str = Form(...),
    package_image: UploadFile | None = File(None),
    seal_image: UploadFile | None = File(None),
    package_capture_context: str = Form("factory_registration"),
    seal_capture_context: str = Form("factory_registration"),
):    

    package_img = None
    seal_img = None
    package_qz_result = None
    seal_qz_result = None

    if package_image is not None:
        package_bytes = await package_image.read()
        package_raw = decode_image(package_bytes)

        if package_raw is None:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Package baseline image could not be decoded.",
                },
            )

        package_qz_result = extract_quiet_zone(
            package_raw,
            capture_context=package_capture_context,
        )

        if not package_qz_result.get("success"):
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "message": "Package baseline Quiet Zone could not be confidently detected.",
                    "reason": package_qz_result.get("reason"),
                    "confidence": package_qz_result.get("confidence", 0.0),
                    "capture_context": package_capture_context,
                },
            )

        package_img = package_qz_result.get("image")

    if seal_image is not None:
        seal_bytes = await seal_image.read()
        seal_raw = decode_image(seal_bytes)

        if seal_raw is None:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Seal baseline image could not be decoded.",
                },
            )

        seal_qz_result = extract_quiet_zone(
            seal_raw,
            capture_context=seal_capture_context,
        )

        if not seal_qz_result.get("success"):
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "message": "Seal baseline Quiet Zone could not be confidently detected.",
                    "reason": seal_qz_result.get("reason"),
                    "confidence": seal_qz_result.get("confidence", 0.0),
                    "capture_context": seal_capture_context,
                },
            )

        seal_img = seal_qz_result.get("image")


# CASE 1: RAW UNIT REGISTRATION ONLY
# This only runs when no package_image and no seal_image file was sent.
    if package_image is None and seal_image is None:
        existing_unit = get_unit_record(unit_id)
        if existing_unit is None:
            unit_id = create_unit_record(
                unit_id,
                order_id,
                seller_id,
                buyer_id,
                marketplace_name,
                product_id,
                product_name,
                brand,
                batch_code,
                None,
                None,
                None,
                None,
            )    
        else:
            unit_id = unit_id
        return {
           "status": "raw_registered",
           "unit_id": unit_id,
           "order_id": order_id,
           "seller_id": seller_id,
           "buyer_id": buyer_id,
           "marketplace_name": marketplace_name,
           "product_id": product_id,
           "product_name": product_name,
           "brand": brand,
           "batch_code": batch_code,
           "message": "Raw Unit ID registered. Package and seal baselines can be captured next."
    }

# CASE 2: PARTIAL FILE UPLOAD IS NOT ALLOWED
    if package_image is None or seal_image is None:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Both package_image and seal_image are required for baseline registration."
        }
    )

# CASE 3: FILES WERE SENT BUT PROCESSING FAILED
    if package_img is None or seal_img is None:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Package or seal image could not be processed. Please retake both baseline images."
        }
    )
    
    if  package_img is not None and seal_img is not None:
        _, package_encoded = cv2.imencode(".jpg", package_img)
        _, seal_encoded = cv2.imencode(".jpg", seal_img)
        os.makedirs("debug_rois", exist_ok=True)
        cv2.imwrite("debug_rois/register_package_roi.jpg", package_img)
        cv2.imwrite("debug_rois/register_seal_roi.jpg", seal_img)
        package_bytes = package_encoded.tobytes()
        seal_bytes = seal_encoded.tobytes()
        package_hash = hashlib.sha256(package_bytes).hexdigest()
        seal_hash = hashlib.sha256(seal_bytes).hexdigest()
        package_file_path = f"uploads/{package_hash}.jpg"
        seal_file_path = f"uploads/{seal_hash}.jpg"

        with open(package_file_path, "wb") as f:
           f.write(package_bytes)

        with open(seal_file_path, "wb") as f:
           f.write(seal_bytes)
        with open(seal_file_path, "wb") as f:
            f.write(seal_bytes)
            print("===== REGISTER BASELINE DEBUG =====")
            print("unit_id:", unit_id)
            print("package_image received:", package_image is not None)
            print("seal_image received:", seal_image is not None)
            print("package_file_path:", package_file_path)
            print("seal_file_path:", seal_file_path)
            print("package_hash:", package_hash)
            print("seal_hash:", seal_hash)
            print("===================================")
     
        unit_id = create_unit_record(
            unit_id,
            order_id,
            seller_id,
            buyer_id,
            marketplace_name,
            product_id,
            product_name,
            brand,
            batch_code,
            package_file_path,
            package_hash,
            seal_file_path,
            seal_hash
    )   
        return {
            "status": "registered",
            "unit_id": unit_id,
            "order_id": order_id,
            "seller_id": seller_id,
            "buyer_id": buyer_id,
            "marketplace_name": marketplace_name,
            "product_id": product_id,
            "product_name": product_name,
            "brand": brand,
            "batch_code": batch_code,
            "package_hash": package_hash,
            "seal_hash": seal_hash,
            "message": "Package and seal baselines registered successfully."
        }
        
@app.post("/api/v1/units/brand-baseline-register")
async def brand_baseline_register_unit(
    unit_id: str = Form(...),
    seller_id: str = Form(...),
    product_id: str = Form(...),
    product_name: str = Form(...),
    brand: str = Form(...),
    batch_code: str = Form(...),
):
    order_id = "BRAND-BASELINE"
    buyer_id = "N/A"
    marketplace_name = "BRAND_REGISTRY"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()

    cursor.execute(
        """
        INSERT OR REPLACE INTO unit_fingerprints (
            unit_id,
            order_id,
            seller_id,
            buyer_id,
            marketplace_name,
            product_id,
            product_name,
            brand,
            batch_code,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            unit_id,
            order_id,
            seller_id,
            buyer_id,
            marketplace_name,
            product_id,
            product_name,
            brand,
            batch_code,
            now,
        ),
    )

    conn.commit()
    conn.close()

    return {
        "status": "brand_baseline_registered",
        "unit_id": unit_id,
        "seller_id": seller_id,
        "product_id": product_id,
        "product_name": product_name,
        "brand": brand,
        "batch_code": batch_code,
        "order_id": order_id,
        "buyer_id": buyer_id,
        "marketplace_name": marketplace_name,
        "message": "Brand baseline unit registered. Package and seal baselines can be captured next.",
    } 
    
@app.post("/api/v1/units/brand-baseline-images")
async def register_brand_baseline_images(
    unit_id: str = Form(...),
    package_image: UploadFile = File(...),
    seal_image: UploadFile = File(...),
    package_capture_context: str = Form("brand_baseline"),
    seal_capture_context: str = Form("brand_baseline"),
):   
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM unit_fingerprints WHERE unit_id = ?",
            (unit_id,),
        )

        existing = cursor.fetchone()
        conn.close()

        if existing is None:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "Brand baseline unit not found. Register the brand baseline unit first.",
                    "unit_id": unit_id,
                },
            )

        package_bytes = await package_image.read()
        seal_bytes = await seal_image.read()

        package_raw = decode_image(package_bytes)
        seal_raw = decode_image(seal_bytes)

        if package_raw is None or seal_raw is None:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Brand baseline package or seal image could not be decoded.",
                },
            )

        package_qz_result = extract_quiet_zone(
            package_raw,
            capture_context=package_capture_context,
        )
        seal_qz_result = extract_quiet_zone(
            seal_raw,
            capture_context=seal_capture_context,
        )

        if not package_qz_result.get("success") or not seal_qz_result.get("success"):
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "message": "One or both brand baseline images failed Quiet Zone detection.",
                    "package_quiet_zone": {
                        "success": package_qz_result.get("success", False),
                        "reason": package_qz_result.get("reason"),
                        "confidence": package_qz_result.get("confidence", 0.0),
                    },
                    "seal_quiet_zone": {
                        "success": seal_qz_result.get("success", False),
                        "reason": seal_qz_result.get("reason"),
                        "confidence": seal_qz_result.get("confidence", 0.0),
                    },
                },
            )

        package_img = package_qz_result.get("image")
        seal_img = seal_qz_result.get("image")

        package_ok, package_encoded = cv2.imencode(".jpg", package_img)
        seal_ok, seal_encoded = cv2.imencode(".jpg", seal_img)

        if not package_ok or not seal_ok:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Canonical Quiet Zone images could not be encoded.",
                },
            )

        package_bytes = package_encoded.tobytes()
        seal_bytes = seal_encoded.tobytes()

        package_hash = hashlib.sha256(package_bytes).hexdigest()
        seal_hash = hashlib.sha256(seal_bytes).hexdigest()
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)

        package_file_path = os.path.join(
            uploads_dir,
            f"{package_hash}.jpg",
        )
        seal_file_path = os.path.join(
            uploads_dir,
            f"{seal_hash}.jpg",
        )
        with open(package_file_path, "wb") as package_file:
            package_file.write(package_bytes)
        with open(seal_file_path, "wb") as seal_file:
            seal_file.write(seal_bytes)
 
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE unit_fingerprints
            SET
                package_image_path = ?,
                seal_image_path = ?,
                package_image_hash = ?,
                seal_image_hash = ?,
                created_at = ?
            WHERE unit_id = ?
            """,
            (
                package_file_path,
                seal_file_path,
                package_hash,
                seal_hash,
                now,
                unit_id,
            ),
        )

        conn.commit()
        conn.close()

        return {
            "status": "brand_baseline_images_registered",
            "unit_id": unit_id,
            "package_hash": package_hash,
            "seal_hash": seal_hash,
            "package_capture_context": package_capture_context,
            "seal_capture_context": seal_capture_context,
            "message": "Brand package and seal baseline images registered successfully.",
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
            },
        )
    
@app.post("/api/v1/products/verify")
async def verify_registered_product(
    product_id: str = Form(...),
    field_scan: UploadFile = File(...),
):
    """
    Verifies a scan against a stored product master image.
    """

    try:
        product = get_product(product_id)

        if product is None:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "decision": "fail",
                    "is_match": False,
                    "trust_score": 0.0,
                    "message": "Product ID not found.",
                    "product_id": product_id,
                },
            )

        scan_bytes = await field_scan.read()

        if not scan_bytes:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "decision": "fail",
                    "is_match": False,
                    "trust_score": 0.0,
                    "message": "No field scan image received.",
                },
            )

        master_bytes = read_file_bytes(product["master_image_path"])

        result = run_verification(
            master_bytes=master_bytes,
            scan_bytes=scan_bytes,
            product_id=product_id,
        )

        result["product"] = {
            "product_id": product["id"],
            "product_name": product["product_name"],
            "brand": product["brand"],
            "batch_code": product["batch_code"],
            "created_at": product["created_at"],
        }

        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "decision": "fail",
                "is_match": False,
                "trust_score": 0.0,
                "message": str(e),
            },
        )

@app.get("/api/v1/products")
async def list_products():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            u.product_id,
            u.product_name,
            u.brand,
            u.seller_id,
            s.seller_name,
            COUNT(u.unit_id) AS units_registered,
            MAX(u.created_at) AS latest_registered_at
        FROM unit_fingerprints u
        LEFT JOIN sellers s
            ON u.seller_id = s.seller_id
        GROUP BY
            u.product_id,
            u.product_name,
            u.brand,
            u.seller_id,
            s.seller_name
        ORDER BY latest_registered_at DESC
        """
    )

    rows = cursor.fetchall()
    conn.close()

    products = []

    for row in rows:
        products.append(
            {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "brand": row["brand"],
                "seller_id": row["seller_id"],
                "seller_name": row["seller_name"],
                "units_registered": row["units_registered"],
                "latest_registered_at": row["latest_registered_at"],
            }
        )

    return {
        "status": "success",
        "count": len(products),
        "products": products,
    }
  
@app.get("/api/v1/products/{product_id}")
async def get_product_by_id(product_id: str):
    product = get_product(product_id)

    if product is None:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": "Product not found.",
            },
        )

    return {
        "status": "success",
        "product": {
            "id": product["id"],
            "product_name": product["product_name"],
            "brand": product["brand"],
            "batch_code": product["batch_code"],
            "master_image_hash": product["master_image_hash"],
            "created_at": product["created_at"],
        },
    }


@app.get("/api/v1/events")
async def list_events(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            product_id,
            status,
            decision,
            is_match,
            trust_score,
            quality_score,
            replay_warning,
            message,
            created_at
        FROM verification_events
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    conn.close()

    events = []

    for row in rows:
        events.append(
            {
                "event_id": row[0],
                "product_id": row[1],
                "status": row[2],
                "decision": row[3],
                "is_match": bool(row[4]),
                "trust_score": row[5],
                "quality_score": row[6],
                "replay_warning": bool(row[7]),
                "message": row[8],
                "created_at": row[9],
            }
        )

    return {
        "status": "success",
        "events": events,
    }
@app.post("/api/v1/challenges/request")
async def request_challenge(payload: dict):
    challenge_id = str(uuid.uuid4())

    order_id = payload.get("order_id")
    marketplace_name = payload.get("marketplace_name")
    seller_id = payload.get("seller_id")
    buyer_id = payload.get("buyer_id")
    unit_id = payload.get("unit_id")
    product_id = payload.get("product_id")
    challenge_source = payload.get("challenge_source", "buyer")
    challenge_reason = payload.get("challenge_reason")
    customer_notes = payload.get("customer_notes", "")

    if not buyer_id:
        buyer_id = f"BUYER-PUBLIC-{str(uuid.uuid4())[:8].upper()}"

    if not order_id:
        order_id = f"PUBLIC-CHALLENGE-{str(uuid.uuid4())[:8].upper()}"

    challenge_status = "requested"
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO challenge_requests (
            challenge_id,
            order_id,
            marketplace_name,
            seller_id,
            buyer_id,
            unit_id,
            product_id,
            challenge_source,
            challenge_reason,
            challenge_status,
            customer_notes,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            challenge_id,
            order_id,
            marketplace_name,
            seller_id,
            buyer_id,
            unit_id,
            product_id,
            challenge_source,
            challenge_reason,
            challenge_status,
            customer_notes,
            now_iso(),
        ),
    )

    conn.commit()
    conn.close()
    
    add_timeline_event(
         challenge_id=challenge_id,
         event_type="CHALLENGE_REQUESTED",
         event_title="Challenge requested",
         event_description=f"Challenge requested for order {order_id}",
         actor_type="buyer",
         actor_id=buyer_id,
         old_status="",
         new_status=challenge_status
    )

    return {
        "status": "success",
        "challenge_id": challenge_id,
        "challenge_status": challenge_status,
        "message": "Challenge request created successfully.",
    } 
    
@app.get("/api/v1/challenges/requests")
async def list_challenge_requests(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT 
            cr.*,
            s.seller_name
        FROM challenge_requests cr
        LEFT JOIN sellers s
            ON cr.seller_id = s.seller_id
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()

    return {
        "status": "success",
        "count": len(rows),
        "requests": [dict(row) for row in rows]
    }
    
@app.get("/api/v1/challenges/buyer/{buyer_id}")
async def get_buyer_challenges(buyer_id: str):

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT 
            cr.*,
            s.seller_name
        FROM challenge_requests cr
        LEFT JOIN sellers s
            ON cr.seller_id = s.seller_id
        WHERE cr.buyer_id = ?
        ORDER BY cr.created_at DESC
        """,
        (buyer_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    return {
        "status": "success",
        "count": len(rows),
        "challenges": [dict(row) for row in rows]
    }  
    
@app.get("/api/v1/challenges/detail/{challenge_id}")
async def get_challenge_detail(challenge_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT 
            cr.*,
            s.seller_name
        FROM challenge_requests cr
        LEFT JOIN sellers s
            ON cr.seller_id = s.seller_id
        WHERE cr.challenge_id = ?
        """,
        (challenge_id,)
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return {
            "status": "error",
            "message": "challenge not found"
        }

    return {
        "status": "success",
        "challenge": dict(row)
    }   
@app.patch("/api/v1/challenges/{challenge_id}/seller-response")
async def seller_response_to_challenge(challenge_id: str, payload: dict):
    seller_response = payload.get("seller_response")

    if seller_response not in ["accepted", "rejected"]:
        return {
            "status": "error",
            "message": "seller_response must be accepted or rejected",
        }

    if seller_response == "accepted":
        challenge_status = "open_accepted_by_seller"
    else:
        challenge_status = "open_rejected_by_seller"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE challenge_requests
        SET
            seller_response = ?,
            response_at = ?,
            challenge_status = ?
        WHERE challenge_id = ?
        """,
        (
            seller_response,
            now_iso(),
            challenge_status,
            challenge_id,
        ),
    )

    updated_count = cursor.rowcount

    cursor.execute(
        "SELECT seller_id FROM challenge_requests WHERE challenge_id = ?",
        (challenge_id,),
    )
    seller_row = cursor.fetchone()

    if seller_row:
        seller_id = seller_row[0]

        cursor.execute(
            """
            INSERT OR IGNORE INTO seller_trust_metrics (
                seller_id,
                total_challenges,
                accepted_challenges,
                rejected_challenges,
                passed_verifications,
                failed_verifications,
                last_updated
            )
            VALUES (?, 0, 0, 0, 0, 0, ?)
            """,
            (seller_id, now_iso()),
        )

        if seller_response == "accepted":
            cursor.execute(
                """
                UPDATE seller_trust_metrics
                SET
                    total_challenges = total_challenges + 1,
                    accepted_challenges = accepted_challenges + 1,
                    last_updated = ?
                WHERE seller_id = ?
                """,
                (now_iso(), seller_id),
            )
        else:
            cursor.execute(
                """
                UPDATE seller_trust_metrics
                SET
                    total_challenges = total_challenges + 1,
                    rejected_challenges = rejected_challenges + 1,
                    last_updated = ?
                WHERE seller_id = ?
                """,
                (now_iso(), seller_id),
            )

    conn.commit()
    conn.close()

    if updated_count == 0:
        return {
            "status": "error",
            "message": "Challenge request not found.",
        }

    add_timeline_event(
         challenge_id=challenge_id,
         event_type="SELLER_RESPONSE",
         event_title="Seller responded to challenge",
         event_description=f"Seller response: {seller_response}",
         actor_type="seller",
         actor_id=seller_id,
         old_status="requested",
         new_status=challenge_status
    )
    return {
        "status": "success",
        "challenge_id": challenge_id,
        "seller_response": seller_response,
        "challenge_status": challenge_status,
        "message": "Seller response recorded successfully.",
    }

def make_seller_slug(seller_name: str) -> str:
    slug = seller_name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "seller"

@app.get("/api/v1/challenges/{challenge_id}/timeline")
async def get_challenge_timeline(challenge_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            event_id,
            challenge_id,
            event_type,
            event_title,
            event_description,
            actor_type,
            actor_id,
            old_status,
            new_status,
            created_at
        FROM challenge_timeline
        WHERE challenge_id = ?
        ORDER BY created_at ASC
        """,
        (challenge_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    return {
        "status": "success",
        "challenge_id": challenge_id,
        "count": len(rows),
        "timeline": [dict(row) for row in rows]
    }

@app.post("/api/v1/sellers/onboard")
async def onboard_seller(
    seller_id: str = Form(...),
):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT
             seller_name,
             seller_slug,
             public_url,
             created_at
         FROM sellers
         WHERE seller_id = ?
         """,
         (seller_id,)
    )
    
    row = cursor.fetchone()
    
    print("===== ONBOARD SELLER =====")
    print("seller_id:", seller_id)
    print("row:", row)
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Seller not found")
    
    seller_name = row["seller_name"]
    seller_slug = row["seller_slug"]
    public_url = row["public_url"]
    created_at = row["created_at"]
        
    print("seller_name:", seller_name)
    print("seller_slug:", seller_slug)
    print("public_url:", public_url)
    print("created_at:", created_at)
       
    conn.close()
        
        
    print({
        "status": "success",
        "seller_id": seller_id,
        "seller_name": seller_name,
        "seller_slug": seller_slug,
        "public_url": public_url,
        "created_at": created_at,
    })
    print("Returning onboard response")
    
    return {
        "status": "success",
        "seller_id": seller_id,
        "seller_name": seller_name,
        "seller_slug": seller_slug,
        "public_url": public_url,
        "created_at": created_at
    }
    
@app.get("/api/v1/sellers")
async def list_sellers():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            seller_id,
            seller_name,
            seller_slug,
            public_url,
            created_at
        FROM sellers
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    sellers = []
    for row in rows:
        sellers.append(dict(row))

    return {
        "status": "success",
        "count": len(sellers),
        "sellers": sellers
    }    
@app.get("/api/v1/sellers/profile/{seller_slug}")
async def get_seller_public_profile(seller_slug: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
           s.seller_id,
           s.seller_name,
           s.seller_slug,
           s.public_url,
           s.created_at,
           COALESCE(m.total_challenges, 0) AS total_challenges,
           COALESCE(m.accepted_challenges, 0) AS accepted_challenges,
           COALESCE(m.rejected_challenges, 0) AS rejected_challenges,
           COALESCE(m.passed_verifications, 0) AS passed_verifications,
           COALESCE(m.failed_verifications, 0) AS failed_verifications
    FROM sellers s
    LEFT JOIN seller_trust_metrics m
        ON s.seller_id = m.seller_id
    WHERE s.seller_slug = ?
    """,
    (seller_slug,),
)

    seller = cursor.fetchone()
    conn.close()

    if not seller:
        return {
            "status": "not_found",
            "message": "Seller profile not found",
            "seller_slug": seller_slug,
        }
    total = seller["passed_verifications"] + seller["failed_verifications"]

    if total > 0:
        trust_score = round((seller["passed_verifications"] / total) * 100)
    else:
        trust_score = 100

    return {
        "status": "success",
        "seller_id": seller["seller_id"],
        "seller_name": seller["seller_name"],
        "seller_slug": seller["seller_slug"],
        "public_url": seller["public_url"],
        "created_at": seller["created_at"],
        "total_challenges": seller["total_challenges"],
        "accepted_challenges": seller["accepted_challenges"],
        "rejected_challenges": seller["rejected_challenges"],
        "passed_verifications": seller["passed_verifications"],
        "failed_verifications": seller["failed_verifications"],
        "trust_score": trust_score,
    }   


@app.get("/api/v1/sellers/{seller_id}/trust-metrics")
async def get_seller_trust_metrics(seller_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM seller_trust_metrics
        WHERE seller_id = ?
        """,
        (seller_id,),
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return {
            "status": "success",
            "seller_id": seller_id,
            "total_challenges": 0,
            "accepted_challenges": 0,
            "rejected_challenges": 0,
            "passed_verifications": 0,
            "failed_verifications": 0,
            "acceptance_rate": 0,
            "pass_rate": 0,
            "last_updated": None,
        }

    data = dict(row)

    total = data.get("total_challenges", 0) or 0
    accepted = data.get("accepted_challenges", 0) or 0
    passed = data.get("passed_verifications", 0) or 0
    failed = data.get("failed_verifications", 0) or 0
    total_verifications = passed + failed

    acceptance_rate = round((accepted / total) * 100, 2) if total > 0 else 0
    pass_rate = round((passed / total_verifications) * 100, 2) if total_verifications > 0 else 0

    data["status"] = "success"
    data["acceptance_rate"] = acceptance_rate
    data["pass_rate"] = pass_rate

    return data
    
@app.get("/api/v1/challenge-cases")
async def list_challenge_cases(
    limit: int = 20, 
    admin_user: dict = 
    Depends(require_admin_user)
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
    SELECT
        c.case_id,
        cr.challenge_id,
        s.seller_name,
        c.order_id,
        c.marketplace_name,
        c.seller_id,
        c.buyer_id,
        c.unit_id,
        u.product_id,
        u.product_name,
        u.brand,
        u.batch_code,
        c.case_type,
        c.case_status,
        c.trigger_reason,
        c.verification_decision,
        c.package_match,
        c.seal_match,
        c.trust_score,
        c.risk_level,
        c.recommended_action,
        c.created_at
    FROM challenge_cases c
    LEFT JOIN unit_fingerprints u
    ON c.unit_id = u.unit_id
    LEFT JOIN challenge_requests cr
        ON c.order_id = cr.order_id
    LEFT JOIN sellers s
        ON c.seller_id = s.seller_id
    ORDER BY c.created_at DESC
    LIMIT ?
    """,
        (limit,),
    )

    rows = cursor.fetchall()
    conn.close()

    cases = []

    for row in rows:
        cases.append(
            {
                "case_id": row[0],
                "challenge_id": row[1],
                "seller_name": row[2],
                "order_id": row[3],
                "marketplace_name": row[4],
                "seller_id": row[5],
                "buyer_id": row[6],
                "unit_id": row[7],
                "product_id": row[8],
                "product_name": row[9],
                "brand": row[10],
                "batch_code": row[11],
                "case_type": row[12],
                "case_status": row[13],
                "trigger_reason": row[14],
                "verification_decision": row[15],
                "package_match": bool(row[16]),
                "seal_match": bool(row[17]),
                "trust_score": row[18],
                "risk_level": row[19],
                "recommended_action": row[20],
                "created_at": row[21],
            }
        )
        
    return {
        "status": "success",
        "count": len(cases),
        "cases": cases,
    }
    
@app.patch("/api/v1/challenge-cases/{case_id}/status")
async def update_challenge_case_status(
    case_id: str, 
    payload: dict,
    admin_user: dict =
    Depends(require_admin_user)
    ):
    allowed_statuses = {
        "in_progress",
        "reviewed",
        "closed",
        "flagged_high_risk"
    }

    new_status = payload.get("status")

    if not new_status:
        return {
            "status": "error",
            "message": "Missing status field"
        }

    if new_status not in allowed_statuses:
        return {
            "status": "error",
            "message": f"Invalid status. Allowed values: {list(allowed_statuses)}"
        }

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if new_status == "flagged_high_risk":
        cursor.execute(
            """
            UPDATE challenge_cases
            SET case_status = ?,
                risk_level = ?,
                recommended_action = ?
            WHERE case_id = ?
            """,
            (
                "flagged_high_risk",
                "high",
                "Flagged by admin for high-risk review.",
                case_id
            )
        )
    else:
        cursor.execute(
            """
            UPDATE challenge_cases
            SET case_status = ?
            WHERE case_id = ?
            """,
            (
                new_status,
                case_id
            )
        )

    conn.commit()

    if cursor.rowcount == 0:
        conn.close()
        return {
            "status": "error",
            "message": "Challenge case not found",
            "case_id": case_id
        }

    conn.close()

    return {
        "status": "success",
        "case_id": case_id,
        "new_status": new_status
    }
    
@app.post("/api/v1/debug-upload")
async def debug_upload(request: Request):
    form = await request.form()

    return {
        "status": "debug",
        "received_keys": list(form.keys()),
        "types": {
            key: str(type(value))
            for key, value in form.items()
        },
    }
@app.get("/phone-register", response_class=HTMLResponse)
async def phone_register_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FiberHash Phone Registration</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                padding: 20px;
                background: #111;
                color: #fff;
            }
            input, button {
                width: 100%;
                margin: 10px 0;
                padding: 12px;
                font-size: 16px;
            }
            button {
                background: #16a34a;
                color: white;
                border: none;
                border-radius: 6px;
            }
            pre {
                background: #222;
                padding: 12px;
                overflow-x: auto;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
        <h2>FiberHash SealLock Phone Registration</h2>

        <label>Public Unit ID</label>
        <input id="unit_id" type="text" placeholder="Example: TEST200">

        <label>Product Name</label>
        <input id="product_name" type="text" value="Test Product">

        <label>Brand</label>
        <input id="brand" type="text" value="Test Brand">

        <label>Batch Code</label>
        <input id="batch_code" type="text" value="TEST-BATCH">

        <label>Package baseline</label>
        <input id="package_image" type="file" accept="image/*" capture="environment">

        <label>Seal baseline</label>
        <input id="seal_image" type="file" accept="image/*" capture="environment">

        <button onclick="submitRegister()">Register Unit</button>

        <h3>Result</h3>
        <pre id="result">Waiting...</pre>

        <script>
            async function submitRegister() {
                const unitId = document.getElementById("unit_id").value;
                const productName = document.getElementById("product_name").value;
                const brand = document.getElementById("brand").value;
                const batchCode = document.getElementById("batch_code").value;
                const packageFile = document.getElementById("package_image").files[0];
                const sealFile = document.getElementById("seal_image").files[0];

                if (!unitId || !packageFile || !sealFile) {
                    document.getElementById("result").textContent =
                        "Please enter Unit ID and select both baseline images.";
                    return;
                }

                const formData = new FormData();

                // Your backend uses product_id as the public physical unit ID
                formData.append("product_id", unitId);
                formData.append("product_name", productName);
                formData.append("brand", brand);
                formData.append("batch_code", batchCode);
                formData.append("package_image", packageFile);
                formData.append("seal_image", sealFile);

                document.getElementById("result").textContent = "Registering...";

                try {
                    const response = await fetch("/api/v1/units/register", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();
                    document.getElementById("result").textContent =
                        JSON.stringify(data, null, 2);

                } catch (err) {
                    document.getElementById("result").textContent =
                        "Error: " + err.message;
                }
            }
        </script>
    </body>
    </html>
    """    
@app.get("/phone-test", response_class=HTMLResponse)
async def phone_test_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FiberHash Phone Camera Test</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                padding: 20px;
                background: #111;
                color: #fff;
            }
            input, button {
                width: 100%;
                margin: 10px 0;
                padding: 12px;
                font-size: 16px;
            }
            button {
                background: #4f46e5;
                color: white;
                border: none;
                border-radius: 6px;
            }
            pre {
                background: #222;
                padding: 12px;
                overflow-x: auto;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
        <h2>FiberHash SealLock Phone Test</h2>

        <label>Unit ID</label>
        <input id="unit_id" type="text" placeholder="Enter public unit ID">

        <label>Package scan</label>
        <input id="package_scan" type="file" accept="image/*" capture="environment">

        <label>Seal scan</label>
        <input id="seal_scan" type="file" accept="image/*" capture="environment">

        <button onclick="submitVerify()">Verify</button>

        <h3>Result</h3>
        <pre id="result">Waiting...</pre>

        <script>
            async function submitVerify() {
                const unitId = document.getElementById("unit_id").value;
                const packageFile = document.getElementById("package_scan").files[0];
                const sealFile = document.getElementById("seal_scan").files[0];

                if (!unitId || !packageFile || !sealFile) {
                    document.getElementById("result").textContent =
                        "Please enter unit ID and select both images.";
                    return;
                }

                const formData = new FormData();
                formData.append("unit_id", unitId);
                formData.append("package_scan", packageFile);
                formData.append("seal_scan", sealFile);

                document.getElementById("result").textContent = "Submitting...";

                try {
                    const response = await fetch("/api/v1/units/verify", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();
                    document.getElementById("result").textContent =
                        JSON.stringify(data, null, 2);

                } catch (err) {
                    document.getElementById("result").textContent =
                        "Error: " + err.message;
                }
            }
        </script>
    </body>
    </html>
    """    
@app.get("/debug/{filename}")
async def get_debug_roi(filename: str):
    filepath = os.path.join("debug_rois", filename)

    if not os.path.exists(filepath):
        return JSONResponse(
            status_code=404,
            content={"detail": "Not Found"}
        )

    return FileResponse(filepath)

@app.get("/api/v1/admin/seller-trust-dashboard")
async def seller_trust_dashboard():

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        s.seller_id,
        s.seller_name,
        s.seller_slug,
        COALESCE(m.total_challenges, 0) AS total_challenges,
        COALESCE(m.accepted_challenges, 0) AS accepted_challenges,
        COALESCE(m.rejected_challenges, 0) AS rejected_challenges,
        COALESCE(m.passed_verifications, 0) AS passed_verifications,
        COALESCE(m.failed_verifications, 0) AS failed_verifications,
        m.last_updated
    FROM sellers s
    LEFT JOIN seller_trust_metrics m
        ON s.seller_id = m.seller_id
    ORDER BY
        COALESCE(m.passed_verifications, 0) DESC,
        COALESCE(m.accepted_challenges, 0) DESC
""")

    rows = cursor.fetchall()
    conn.close()

    sellers = []

    for row in rows:

        total_challenges = row["total_challenges"] or 0
        accepted_challenges = row["accepted_challenges"] or 0
        passed_verifications = row["passed_verifications"] or 0

        acceptance_rate = (
            accepted_challenges / total_challenges * 100
            if total_challenges > 0 else 0
        )

        pass_rate = (
            passed_verifications / accepted_challenges * 100
            if accepted_challenges > 0 else 0
        )

        sellers.append({
            "seller_name": row["seller_name"],
            "seller_slug": row["seller_slug"],
            "seller_id": row["seller_id"],
            "total_challenges": total_challenges,
            "accepted_challenges": accepted_challenges,
            "rejected_challenges": row["rejected_challenges"],
            "passed_verifications": row["passed_verifications"],
            "failed_verifications": row["failed_verifications"],
            "acceptance_rate": round(acceptance_rate, 2),
            "pass_rate": round(pass_rate, 2),
            "last_updated": row["last_updated"]
        })

    return {
        "status": "success",
        "seller_count": len(sellers),
        "sellers": sellers
    }
@app.post("/debug/quiet-zone")
async def debug_quiet_zone(
    file: UploadFile = File(...)
):
    """
    Temporary diagnostic endpoint.

    Receives the original full camera photograph,
    runs Quiet Zone detection, and returns the
    detected canonical Quiet Zone as a PNG.

    This endpoint does NOT register anything.
    """

    try:
        # ----------------------------------------------------
        # 1. READ ORIGINAL IMAGE
        # ----------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="No image data received.",
            )

        # ----------------------------------------------------
        # 2. DECODE ORIGINAL IMAGE
        # ----------------------------------------------------

        image = decode_image(image_bytes)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Unable to decode uploaded image.",
            )

        # ----------------------------------------------------
        # 3. RUN QUIET ZONE DETECTION
        # ----------------------------------------------------

        result = extract_quiet_zone(
            image,
            capture_context="debug",
        )

        if not result.get("success"):
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "reason": result.get(
                        "reason",
                        "QUIET_ZONE_DETECTION_FAILED",
                    ),
                    "confidence": result.get(
                        "confidence",
                        0.0,
                    ),
                },
            )

        # ----------------------------------------------------
        # 4. GET CANONICAL QUIET ZONE
        # ----------------------------------------------------

        canonical = result.get("image")

        if canonical is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Quiet Zone detection succeeded "
                    "but returned no image."
                ),
            )
        # ----------------------------------------------------
        # DEBUG: DRAW THE SELECTED QUIET ZONE CORNERS
        # ON THE ORIGINAL IMAGE
        # ----------------------------------------------------

        debug_image = image.copy()

        corners = result.get("corners")

        if not corners or len(corners) != 4:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Quiet Zone detection succeeded "
                    "but did not return four corners."
                ),
            )

        debug_points = np.array(
            corners,
            dtype=np.int32,
        ).reshape(4, 2)

        # Draw the selected physical Quiet Zone boundary.
        cv2.polylines(
            debug_image,
            [debug_points],
            True,
            (0, 255, 0),
            4,
        )

        # Draw each selected corner.
        for index, point in enumerate(debug_points):
            x = int(point[0])
            y = int(point[1])

            cv2.circle(
                debug_image,
                (x, y),
                12,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                debug_image,
                str(index + 1),
                (x + 15, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        canonical = debug_image

        # ----------------------------------------------------
        # 5. ENCODE AS PNG
        # ----------------------------------------------------

        success, encoded = cv2.imencode(
            ".png",
            canonical,
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Unable to encode extracted "
                    "Quiet Zone."
                ),
            )

        # ----------------------------------------------------
        # 6. RETURN EXTRACTED QUIET ZONE
        # ----------------------------------------------------

        response = Response(
            content=encoded.tobytes(),
            media_type="image/png",
        )

        response.headers[
            "X-Quiet-Zone-Confidence"
        ] = str(
            result.get(
                "confidence",
                0.0,
            )
        )

        return response

    except HTTPException:
        raise

    except Exception as e:

        print(
            "QUIET ZONE DEBUG FAILED:",
            str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Quiet Zone diagnostic failed."
            ),
        )
