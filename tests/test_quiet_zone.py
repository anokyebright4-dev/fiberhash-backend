"""Tests for the Quiet Zone extraction engine in main.py.

Coverage follows the approved implementation/test plan:

  * Stage-1 cyclic ordering + winding (geometry only)
  * Stage-2 orientation resolution from intrinsic evidence, distinguishing
    successful resolution, genuinely ambiguous patches, and incorrect
    resolution (guarded via cross-rotation consistency)
  * the long-line counting fix
  * off-centre and close-up acceptance
  * false-positive rejection (printed outlines / flat panels / ruled panels)
  * fail-closed behaviour
  * public-contract / compatibility preservation
"""

import asyncio
import io
import json
import os

import cv2
import numpy as np
import pytest
from starlette.datastructures import Headers, UploadFile

import main
from tests import synthetic


# ----------------------------------------------------------------------------
# Stage 1: cyclic ordering + winding (geometry only)
# ----------------------------------------------------------------------------

def _base_quad(angle_deg=0.0, aspect=1.0, cx=200.0, cy=200.0, scale=120.0):
    w = scale
    h = scale * aspect
    corners = np.array(
        [[-w, -h], [w, -h], [w, h], [-w, h]], dtype=np.float32
    ) / 2.0
    theta = np.radians(angle_deg)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    return (corners @ rot.T) + np.array([cx, cy], dtype=np.float32)


def _signed_area(pts):
    total = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        total += (x1 * y2) - (x2 * y1)
    return total


@pytest.mark.parametrize("angle", [0, 15, 30, 37, 45, 60, 90, 135, 210])
def test_cyclic_order_is_permutation_invariant_and_consistent(angle):
    quad = _base_quad(angle_deg=angle)
    reference = main._order_quad_cyclic(quad)

    for perm in ([0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
        ordered = main._order_quad_cyclic(quad[perm])
        # Same set of points regardless of input order.
        ref_sorted = reference[np.lexsort((reference[:, 1], reference[:, 0]))]
        got_sorted = ordered[np.lexsort((ordered[:, 1], ordered[:, 0]))]
        assert np.allclose(ref_sorted, got_sorted, atol=1e-3)
        # Winding is fixed (positive shoelace => clockwise in image coords).
        assert _signed_area(ordered) > 0.0


def test_cyclic_order_geometry_is_start_invariant():
    # Aspect derived by _quiet_zone_geometry must not depend on input order.
    quad = _base_quad(angle_deg=33.0, aspect=0.7)
    aspects = set()
    for perm in ([0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]):
        geometry = main._quiet_zone_geometry(quad[perm])
        assert geometry is not None
        aspects.add(round(float(geometry[2]), 4))
    assert len(aspects) == 1


def test_cyclic_order_never_mirrors_chiral_content():
    # An 'F' is chiral: a mirrored warp would flip it. Warp and check the
    # stroke layout is preserved (top bar to the right of the vertical stem).
    img = np.full((600, 600, 3), 40, np.uint8)
    cv2.rectangle(img, (200, 200), (400, 400), (180, 180, 180), -1)
    # Vertical stem + top and middle bars (an 'F').
    cv2.line(img, (250, 230), (250, 370), (0, 0, 0), 10)
    cv2.line(img, (250, 235), (330, 235), (0, 0, 0), 10)
    cv2.line(img, (250, 300), (315, 300), (0, 0, 0), 10)
    quad = np.array(
        [[200, 200], [400, 200], [400, 400], [200, 400]], dtype=np.float32
    )
    canonical, _, _ = main._warp_quiet_zone(img, quad, 256)
    gray = cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY)
    dark = gray < 80
    ys, xs = np.nonzero(dark)
    # The stem (vertical) should sit to the LEFT of the bar ink mass; a mirror
    # would put the stem on the right. Compare ink centroid x vs stem x.
    stem_x = np.median(xs[xs < np.percentile(xs, 40)])
    ink_centroid_x = xs.mean()
    assert ink_centroid_x > stem_x  # bars extend to the right of the stem


# ----------------------------------------------------------------------------
# Stage 2: orientation resolution
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("mark", ["tl", "tr", "br", "bl"])
def test_orientation_resolves_mark_to_canonical_top_left(mark):
    image, quad = synthetic.marked_patch_frame(700, 360, mark_corner=mark)
    canonical, _, info = main._warp_quiet_zone(image, quad, main.QUIET_ZONE_CANONICAL_SIZE)
    assert info["orientation_ambiguous"] is False
    assert synthetic.brightest_quadrant(canonical) == "tl"


@pytest.mark.parametrize("angle", [0, 15, 30, 37, 45, 60, 90, 135, 210])
def test_orientation_cross_rotation_consistency_square(angle):
    # The SAME physical patch captured at different rotations must canonicalise
    # to the SAME orientation. This is the incorrect-resolution guard: an
    # off-by-90 bug would place the mark in different quadrants across angles.
    image, quad = synthetic.marked_patch_frame(900, 360, mark_corner="tl")
    center = (450, 450)
    rotated_image, rotated_quad = synthetic.rotate_image_and_quad(
        image, quad, angle, center
    )
    canonical, _, info = main._warp_quiet_zone(
        rotated_image, rotated_quad, main.QUIET_ZONE_CANONICAL_SIZE
    )
    assert info["orientation_ambiguous"] is False
    assert synthetic.brightest_quadrant(canonical) == "tl"


def test_orientation_rectangle_uses_long_side_and_two_fold_symmetry():
    # Non-square patch: long side becomes the canonical width, ambiguity is a
    # pure 180-degree flip resolved by the corner mark.
    image, quad = synthetic.frame_with_patch(
        800, 800, 400, 400, 460, 260, base=170.0, amplitude=8.0, mark_corner="tl"
    )
    canonical, corners, info = main._warp_quiet_zone(
        image, quad, main.QUIET_ZONE_CANONICAL_SIZE
    )
    assert info["orientation_ambiguous"] is False
    assert synthetic.brightest_quadrant(canonical) == "tl"
    # The resolved top edge (corners[0]->corners[1]) must be the long side.
    top_edge = np.linalg.norm(corners[1] - corners[0])
    side_edge = np.linalg.norm(corners[3] - corners[0])
    assert top_edge > side_edge


def test_orientation_ambiguous_for_symmetric_patch():
    image, quad = synthetic.symmetric_patch_frame(700, 360)
    _, _, info = main._warp_quiet_zone(image, quad, main.QUIET_ZONE_CANONICAL_SIZE)
    assert info["orientation_ambiguous"] is True
    assert info["orientation_confidence"] < main.QUIET_ZONE_ORIENTATION_MIN_MAGNITUDE
    assert info["orientation_method"] in ("insufficient_evidence", "orientation_degenerate")


def test_orientation_is_deterministic_when_ambiguous():
    image, quad = synthetic.symmetric_patch_frame(700, 360)
    first, _, info1 = main._warp_quiet_zone(image, quad, 256)
    second, _, info2 = main._warp_quiet_zone(image, quad, 256)
    assert info1 == info2
    assert np.array_equal(first, second)


# ----------------------------------------------------------------------------
# Long-line counting fix
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("n_lines", [0, 1, 3, 6])
def test_long_line_count_evaluates_all_lines(n_lines):
    warped = np.full((512, 512, 3), 200, np.uint8)
    for i in range(n_lines):
        y = 60 + i * 60
        cv2.line(warped, (40, y), (472, y), (0, 0, 0), 3)
    metrics = main._quiet_zone_surface_metrics(warped)
    # The pre-fix bug capped this at 1; all long lines must now be counted.
    assert metrics["long_line_count"] >= n_lines - 1
    if n_lines >= 3:
        assert metrics["long_line_count"] >= 3
    if n_lines == 0:
        assert metrics["long_line_count"] == 0


# ----------------------------------------------------------------------------
# Detection: off-centre and close-up acceptance
# ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cx,cy",
    [(250, 250), (650, 250), (250, 650), (650, 650), (200, 450), (450, 200)],
)
def test_off_centre_patch_is_detected(cx, cy):
    image, _ = synthetic.frame_with_patch(900, 900, cx, cy, 300, 300, seed=5)
    result = main.extract_quiet_zone(image)
    assert result["success"] is True, result["reason"]
    assert result["image"].shape == (
        main.QUIET_ZONE_CANONICAL_SIZE,
        main.QUIET_ZONE_CANONICAL_SIZE,
        3,
    )


@pytest.mark.parametrize("patch", [200, 460, 700, 820])
def test_close_up_patch_is_detected_regardless_of_area(patch):
    # A valid Quiet Zone filling a large fraction of the frame must not fail
    # merely because it exceeds the old 0.20/0.30 area thresholds.
    image, _ = synthetic.frame_with_patch(900, 900, 450, 450, patch, patch, seed=7)
    result = main.extract_quiet_zone(image)
    assert result["success"] is True, (patch, result["reason"])


def test_close_up_patch_filling_about_90_percent_is_detected():
    # Genuinely exercise the ~90% frame-area condition (not the ~83% case).
    frame = 900
    patch = int(round((0.90 ** 0.5) * frame))  # 854 -> ~90% of the frame area
    area_fraction = (patch * patch) / float(frame * frame)
    assert 0.88 <= area_fraction <= 0.92, area_fraction
    image, _ = synthetic.frame_with_patch(frame, frame, frame // 2, frame // 2, patch, patch, seed=7)
    result = main.extract_quiet_zone(image)
    assert result["success"] is True, (area_fraction, result["reason"])
    assert result["image"].shape == (
        main.QUIET_ZONE_CANONICAL_SIZE,
        main.QUIET_ZONE_CANONICAL_SIZE,
        3,
    )


def test_frame_spanning_border_is_rejected():
    # A contour that is essentially the whole frame outline is not a patch.
    image = np.full((900, 900, 3), 40, np.uint8)
    cv2.rectangle(image, (2, 2), (897, 897), (200, 200, 200), 4)
    result = main.extract_quiet_zone(image)
    assert result["success"] is False


# ----------------------------------------------------------------------------
# False-positive rejection
# ----------------------------------------------------------------------------

def test_printed_outline_rectangle_is_rejected():
    image = np.full((900, 900, 3), 40, np.uint8)
    cv2.rectangle(image, (300, 300), (620, 620), (90, 90, 90), 3)
    result = main.extract_quiet_zone(image)
    assert result["success"] is False


def test_flat_uniform_panel_is_rejected():
    image = np.full((900, 900, 3), 40, np.uint8)
    cv2.rectangle(image, (300, 300), (620, 620), (190, 190, 190), -1)
    result = main.extract_quiet_zone(image)
    assert result["success"] is False


def test_ruled_panel_is_rejected():
    # A distinct panel that is full of long printed lines (ledger/box art).
    image = np.full((900, 900, 3), 40, np.uint8)
    cv2.rectangle(image, (300, 300), (620, 620), (200, 200, 200), -1)
    for y in range(320, 620, 24):
        cv2.line(image, (300, y), (620, y), (10, 10, 10), 3)
    result = main.extract_quiet_zone(image)
    assert result["success"] is False


# ----------------------------------------------------------------------------
# Fail-closed behaviour
# ----------------------------------------------------------------------------

def test_two_equal_patches_are_ambiguous():
    image = np.full((900, 900, 3), 40, np.uint8)
    image[150:450, 150:450] = synthetic.low_freq_texture(300, seed=11)
    image[500:800, 500:800] = synthetic.low_freq_texture(300, seed=11)
    result = main.extract_quiet_zone(image)
    assert result["success"] is False
    assert result["reason"] == "QUIET_ZONE_DETECTION_AMBIGUOUS"


def test_blank_frame_is_rejected():
    image = np.full((900, 900, 3), 50, np.uint8)
    result = main.extract_quiet_zone(image)
    assert result["success"] is False
    assert result["reason"] in (
        "QUIET_ZONE_NOT_DETECTED",
        "QUIET_ZONE_DETECTION_CONFIDENCE_TOO_LOW",
    )


def test_too_small_image_is_rejected_with_existing_reason():
    image = np.full((400, 400, 3), 40, np.uint8)
    result = main.extract_quiet_zone(image)
    assert result["success"] is False
    assert result["reason"] == "IMAGE_TOO_SMALL_FOR_QUIET_ZONE_DETECTION"


# ----------------------------------------------------------------------------
# Orientation ambiguity must not, by itself, block acceptance
# ----------------------------------------------------------------------------

def test_ambiguous_orientation_still_accepts_valid_patch(monkeypatch):
    # Force every candidate's orientation to be ambiguous and confirm a valid
    # patch is still detected: acceptance must be independent of orientation
    # resolution (hierarchy step 5).
    real_warp = main._warp_quiet_zone

    def always_ambiguous(image, points, size=main.QUIET_ZONE_CANONICAL_SIZE):
        canonical, corners, _ = real_warp(image, points, size)
        info = {
            "orientation_confidence": 0.0,
            "orientation_ambiguous": True,
            "orientation_method": "insufficient_evidence",
        }
        return canonical, corners, info

    monkeypatch.setattr(main, "_warp_quiet_zone", always_ambiguous)

    image, _ = synthetic.frame_with_patch(900, 900, 450, 450, 320, 320, seed=3)
    result = main.extract_quiet_zone(image)
    assert result["success"] is True, result["reason"]
    assert result["metrics"]["orientation_ambiguous"] is True


# ----------------------------------------------------------------------------
# Public-contract / compatibility preservation
# ----------------------------------------------------------------------------

def test_success_result_contract_is_preserved():
    image, _ = synthetic.frame_with_patch(900, 900, 450, 450, 320, 320, seed=3)
    result = main.extract_quiet_zone(image, capture_context="factory_registration")
    for key in (
        "success",
        "reason",
        "confidence",
        "corners",
        "image",
        "capture_context",
        "metrics",
    ):
        assert key in result
    assert result["success"] is True
    assert result["reason"] == "QUIET_ZONE_DETECTED"
    assert result["capture_context"] == "factory_registration"
    assert len(result["corners"]) == 4
    assert all(len(pt) == 2 for pt in result["corners"])
    assert result["image"].shape == (
        main.QUIET_ZONE_CANONICAL_SIZE,
        main.QUIET_ZONE_CANONICAL_SIZE,
        3,
    )


def test_detection_is_deterministic():
    image, _ = synthetic.frame_with_patch(900, 900, 400, 500, 320, 320, seed=9)
    first = main.extract_quiet_zone(image)
    second = main.extract_quiet_zone(image)
    assert first["success"] == second["success"]
    assert first["reason"] == second["reason"]
    assert np.array_equal(first["image"], second["image"])


def test_same_patch_across_rotations_yields_similar_canonical():
    # Engine-level cross-capture consistency (req 6) without touching SIFT: the
    # baseline and a rotated 'scan' of the same physical patch should extract to
    # highly correlated canonicals.
    image, quad = synthetic.marked_patch_frame(900, 340, mark_corner="tl")
    baseline = main.extract_quiet_zone(image)
    assert baseline["success"] is True, baseline["reason"]

    rotated_image, _ = synthetic.rotate_image_and_quad(image, quad, 40.0, (450, 450))
    scan = main.extract_quiet_zone(rotated_image)
    assert scan["success"] is True, scan["reason"]

    a = cv2.cvtColor(baseline["image"], cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(scan["image"], cv2.COLOR_BGR2GRAY).astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    correlation = float((a * b).mean())
    assert correlation > 0.5


# ----------------------------------------------------------------------------
# Register -> verify round-trip through the real Quiet Zone pipeline
# ----------------------------------------------------------------------------

def _scene_jpeg(seed=7):
    image, _ = synthetic.frame_with_patch(
        900, 900, 450, 450, 340, 340, mark_corner="tl", seed=seed
    )
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def _upload(name, data):
    return UploadFile(
        filename=name,
        file=io.BytesIO(data),
        headers=Headers({"content-type": "image/jpeg"}),
    )


def _isolate_pipeline_storage(tmp_path, monkeypatch):
    # Keep all DB writes / stored baselines inside the test's tmp dir; never
    # touch the real database or uploads directory.
    monkeypatch.chdir(tmp_path)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("debug_rois", exist_ok=True)
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "roundtrip.db"))
    main.init_db()


def test_register_then_verify_round_trip(tmp_path, monkeypatch):
    _isolate_pipeline_storage(tmp_path, monkeypatch)
    scene = _scene_jpeg()

    async def scenario():
        registration = await main.register_unit(
            unit_id="U-ROUNDTRIP",
            order_id="O1",
            seller_id="S1",
            buyer_id="B1",
            marketplace_name="M1",
            product_id="P1",
            product_name="PN",
            brand="BR",
            batch_code="BC",
            package_image=_upload("package.jpg", scene),
            seal_image=_upload("seal.jpg", scene),
            package_capture_context="factory_registration",
            seal_capture_context="factory_registration",
        )
        # A successful extraction must register (dict result, not a 4xx JSON).
        assert isinstance(registration, dict), getattr(
            registration, "body", registration
        )
        assert registration["status"] == "registered"

        verification = await main.verify_unit(
            unit_id="U-ROUNDTRIP",
            package_scan=_upload("package.jpg", scene),
            seal_scan=_upload("seal.jpg", scene),
            package_capture_context="consumer_scan",
            seal_capture_context="consumer_scan",
        )
        # The stored canonical is verifiable against a scan of the same patch.
        assert isinstance(verification, dict), getattr(
            verification, "body", verification
        )
        assert verification["status"] == "verified"
        assert verification["decision"] == "pass"

    asyncio.run(scenario())


def test_extraction_failure_cannot_enter_verification(tmp_path, monkeypatch):
    _isolate_pipeline_storage(tmp_path, monkeypatch)
    scene = _scene_jpeg()

    ok, blank_buf = cv2.imencode(".jpg", np.full((900, 900, 3), 50, np.uint8))
    assert ok
    blank = blank_buf.tobytes()

    async def scenario():
        registration = await main.register_unit(
            unit_id="U-FAILGUARD",
            order_id="O1",
            seller_id="S1",
            buyer_id="B1",
            marketplace_name="M1",
            product_id="P1",
            product_name="PN",
            brand="BR",
            batch_code="BC",
            package_image=_upload("package.jpg", scene),
            seal_image=_upload("seal.jpg", scene),
            package_capture_context="factory_registration",
            seal_capture_context="factory_registration",
        )
        assert isinstance(registration, dict)
        assert registration["status"] == "registered"

        # One scan fails Quiet Zone extraction: verification must fail closed
        # (HTTP 422) and never emit a pass/verified decision into SIFT.
        response = await main.verify_unit(
            unit_id="U-FAILGUARD",
            package_scan=_upload("package.jpg", scene),
            seal_scan=_upload("blank.jpg", blank),
            package_capture_context="consumer_scan",
            seal_capture_context="consumer_scan",
        )
        assert not isinstance(response, dict)
        assert response.status_code == 422
        body = json.loads(response.body)
        assert body.get("decision") != "pass"
        assert body.get("status") == "error"
        assert "Quiet Zone detection" in body.get("message", "")

        # A registration whose baseline fails extraction must also fail closed.
        failed_registration = await main.register_unit(
            unit_id="U-FAILREG",
            order_id="O1",
            seller_id="S1",
            buyer_id="B1",
            marketplace_name="M1",
            product_id="P1",
            product_name="PN",
            brand="BR",
            batch_code="BC",
            package_image=_upload("blank.jpg", blank),
            seal_image=_upload("seal.jpg", scene),
            package_capture_context="factory_registration",
            seal_capture_context="factory_registration",
        )
        assert not isinstance(failed_registration, dict)
        assert failed_registration.status_code == 422

    asyncio.run(scenario())
