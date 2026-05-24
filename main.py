from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
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
from datetime import datetime, timezone
from PIL import Image, ImageOps


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(title="FiberHash / Metalens Authentication API")

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

    except Exception:
        return None

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
        
    if trust_score >= 80 and inlier_count >= 1000:
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


def create_unit_record(unit_id,product_name, brand, batch_code, package_image_path, package_image_hash, seal_image_path, seal_image_hash):
    unit_id = unit_id.strip()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO unit_fingerprints (
            unit_id,
            product_name,
            brand,
            batch_code,
            package_image_path,
            package_image_hash,
            seal_image_path,
            seal_image_hash,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            unit_id,
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
        "product_name": row[1],
        "brand": row[2],
        "batch_code": row[3],
        "package_image_path": row[4],
        "package_image_hash": row[5],
        "seal_image_path": row[6],
        "seal_image_hash": row[7],
        "created_at": row[8],
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
def crop_center_square(img):
    h, w = img.shape[:2]
    crop_size = int(min(h, w) * 0.70)

    cx = w // 2
    cy = h // 2

    x1 = max(0, cx - crop_size // 2)
    y1 = max(0, cy - crop_size // 2)
    x2 = min(w, cx + crop_size // 2)
    y2 = min(h, cy + crop_size // 2)

    return img[y1:y2, x1:x2]


def isolate_square_roi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 2000:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)

            if 0.75 <= ratio <= 1.25:
                candidates.append((area, x, y, w, h))

    if not candidates:
        return crop_center_square(img)

    _, x, y, w, h = max(candidates, key=lambda item: item[0])

    pad = int(min(w, h) * 0.0)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)

    return img[y:y+h, x:x+w]


def isolate_unprinted_package_surface(img):
    return isolate_square_roi(img)


def isolate_seal_surface(img):
    return isolate_square_roi(img)
    
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
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "decision": "fail",
                    "message": "Unit not found",
                },
            )

        package_scan_bytes = await package_scan.read()
        seal_scan_bytes = await seal_scan.read()

        package_scan_img = decode_image(package_scan_bytes)
        seal_scan_img = decode_image(seal_scan_bytes)

        package_scan_img = isolate_unprinted_package_surface(package_scan_img)
        seal_scan_img = isolate_seal_surface(seal_scan_img)
        os.makedirs("debug_rois", exist_ok=True)
        cv2.imwrite("debug_rois/verify_package_roi.jpg", package_scan_img)
        cv2.imwrite("debug_rois/verify_seal_roi.jpg", seal_scan_img)
        _, package_encoded = cv2.imencode(".jpg", package_scan_img)
        _, seal_encoded = cv2.imencode(".jpg", seal_scan_img)

        package_scan_bytes = package_encoded.tobytes()
        seal_scan_bytes = seal_encoded.tobytes()

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
        return {
            "status": "verified",
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
    if  package_image is not None:
        package_bytes = await package_image.read()
        package_img = decode_image(package_bytes)
        package_img = isolate_unprinted_package_surface(package_img)

    if  seal_image is not None:
        seal_bytes = await seal_image.read()
        seal_img = decode_image(seal_bytes)
        seal_img = isolate_seal_surface(seal_img) 


# CASE 1: RAW UNIT REGISTRATION ONLY
# This only runs when no package_image and no seal_image file was sent.
    if package_image is None and seal_image is None:
        existing_unit = get_unit_record(product_id)

        if existing_unit is None:
            unit_id = create_unit_record(
            product_id,
            product_name,
            brand,
            batch_code,
            None,
            None,
            None,
            None
        )
        else:
            unit_id = product_id
        return {
           "status": "raw_registered",
           "unit_id": unit_id,
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

        unit_id = create_unit_record(
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
            "product_id": product_id,
            "product_name": product_name,
            "brand": brand,
            "batch_code": batch_code,
            "package_hash": package_hash,
            "seal_hash": seal_hash,
            "message": "Package and seal baselines registered successfully."
        }
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

