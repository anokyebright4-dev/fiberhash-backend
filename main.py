from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uvicorn
import sqlite3
import hashlib
import os
import json
import uuid
from datetime import datetime, timezone


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

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def normalize_image(image):
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)
    return enhanced


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

    max_keypoints = max(result["total_keypoints_master"], result["total_keypoints_scan"])

    if max_keypoints <= 0:
        return result

    feature_score = (good_match_count / max_keypoints) * 100

    if good_match_count > 0:
        inlier_ratio = inlier_count / good_match_count
    else:
        inlier_ratio = 0.0

    homography_bonus = inlier_ratio * 40

    trust_score = (feature_score * 0.6) + homography_bonus
    trust_score = max(0.0, min(100.0, trust_score))

    result["trust_score"] = round(trust_score, 2)

    if result["trust_score"] >= 65 and inlier_count >= 12:
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

    if trust_score >= 65 and inlier_count >= 12:
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


def create_unit_record(product_name, brand, batch_code, package_image_path, package_image_hash, seal_image_path, seal_image_hash):
    unit_id = str(uuid.uuid4())

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO unit_fingerprints (
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
            "pass": "trust_score >= 65 and inlier_count >= 12",
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

@app.post("/api/v1/units/verify")
async def verify_unit(
    unit_id: str = Form(...),
    package_scan: UploadFile = File(...),
    seal_scan: UploadFile = File(...),
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
        return {
           "status": "verified",
           "decision": decision,
           "package_match": package_match,
           "seal_match": seal_match,
           "trust_score": trust_score,
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
    product_name: str = Form(...),
    brand: str = Form(...),
    batch_code: str = Form(...),
    package_image: UploadFile = File(...),
    seal_image: UploadFile = File(...)
):
    package_bytes = await package_image.read()
    seal_bytes = await seal_image.read()

    package_hash = hashlib.sha256(package_bytes).hexdigest()
    seal_hash = hashlib.sha256(seal_bytes).hexdigest()
    package_file_path = f"uploads/{package_hash}.jpg"
    seal_file_path = f"uploads/{seal_hash}.jpg"

    with open(package_file_path, "wb") as f:
        f.write(package_bytes)

    with open(seal_file_path, "wb") as f:
        f.write(seal_bytes) 

    unit_id = create_unit_record(
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
        "package_hash": package_hash,
        "seal_hash": seal_hash
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


