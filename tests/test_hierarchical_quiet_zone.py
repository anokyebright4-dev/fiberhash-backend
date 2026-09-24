"""Deterministic localisation tests for the hierarchical Quiet Zone scanner.

These fixtures deliberately use no physical-size or capture-guide information.
They assert localisation, not merely a successful response.
"""

import asyncio
import io
import inspect
import json

import cv2
import numpy as np
from starlette.datastructures import UploadFile

import main


def _texture(width, height, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (height, width)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=6)
    noise = cv2.normalize(noise, None, 135, 220, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)


def _nested_scene():
    image = np.full((1000, 1200, 3), 35, dtype=np.uint8)
    # A deliberately salient outer/package surface.
    outer = np.array([[120, 140], [1070, 110], [1100, 875], [150, 900]], np.int32)
    cv2.fillConvexPoly(image, outer, (120, 120, 120))
    cv2.polylines(image, [outer], True, (240, 240, 240), 8)
    # The intended inner physical Quiet Zone is fully nested and textured.
    quiet = np.array([[390, 300], [760, 270], [785, 650], [410, 680]], np.float32)
    source = np.array([[0, 0], [359, 0], [359, 359], [0, 359]], np.float32)
    transform = cv2.getPerspectiveTransform(source, quiet)
    patch = _texture(360, 360, seed=44)
    projected = cv2.warpPerspective(patch, transform, (1200, 1000))
    mask = cv2.warpPerspective(np.full((360, 360), 255, np.uint8), transform, (1200, 1000))
    image[mask > 0] = projected[mask > 0]
    cv2.polylines(image, [quiet.astype(np.int32)], True, (245, 245, 245), 5)
    return image, quiet


def _mean_corner_error(actual, expected):
    actual = np.asarray(actual, dtype=np.float32)
    expected = main._qz_order_quad(expected)
    return float(np.mean(np.linalg.norm(main._qz_order_quad(actual) - expected, axis=1)))


def _terminal_candidate(**overrides):
    """One fully evidenced candidate for decision-predicate unit tests."""
    candidate = {
        "sides": [
            {
                "coverage": main.QUIET_ZONE_MIN_SIDE_COVERAGE + 0.1,
                "relative_contrast": main.QUIET_ZONE_MIN_SIDE_RELATIVE_CONTRAST + 0.1,
            }
            for _ in range(4)
        ],
        "post": {
            "core_std": 12.0,
            "directional_coherence": 0.25,
            "edge_band_ratio_max": 1.1,
        },
        "surface": {"eligible": True},
        "warped": np.zeros((512, 512, 3), dtype=np.uint8),
        "aspect": 0.9,
        "right_angle": 0.9,
        "score": max(0.9, main.QUIET_ZONE_MIN_CONFIDENCE),
    }
    candidate.update(overrides)
    return candidate


def _directional_motion_blur(image, length, angle_degrees):
    """Deterministic linear motion kernel; unlike Gaussian softness it has direction."""
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    transform = cv2.getRotationMatrix2D(
        ((length - 1) / 2.0, (length - 1) / 2.0),
        angle_degrees,
        1.0,
    )
    kernel = cv2.warpAffine(kernel, transform, (length, length))
    kernel /= float(kernel.sum())
    return cv2.filter2D(image, -1, kernel)


def _surface_with_printed_graphic_scene():
    """One material patch plus a separate high-contrast foreground glyph."""
    image = np.full((1000, 1200, 3), (45, 125, 45), dtype=np.uint8)
    expected = np.array(
        [[690, 420], [1049, 420], [1049, 779], [690, 779]], dtype=np.float32,
    )
    image[420:780, 690:1050] = _texture(360, 360, seed=95)
    cv2.rectangle(image, (690, 420), (1049, 779), (20, 20, 20), 6)
    # A white printed foreground graphic, intentionally not a filled surface.
    cv2.putText(image, "&", (380, 350), cv2.FONT_HERSHEY_SIMPLEX, 5.2,
                (245, 245, 245), 18, cv2.LINE_AA)
    return image, expected


def test_nested_inner_boundary_beats_larger_outer_package_boundary():
    image, expected = _nested_scene()
    result = main.extract_quiet_zone(image)
    assert result["success"], result
    assert _mean_corner_error(result["corners"], expected) < 18.0
    assert result["image"].shape == (512, 512, 3)
    assert result["metrics"]["detection_source"] == "hierarchical_boundary_scanner"
    assert result["metrics"]["terminal_acceptance"] is True
    assert result["metrics"]["ambiguity_competition_reached"] is False


def test_false_outer_rectangle_is_not_returned_as_a_successful_crop():
    image, expected = _nested_scene()
    result = main.extract_quiet_zone(image)
    assert result["success"], result
    detected_area = abs(cv2.contourArea(np.asarray(result["corners"], dtype=np.float32)))
    outer_area = abs(cv2.contourArea(np.array([[120, 140], [1070, 110], [1100, 875], [150, 900]], np.float32)))
    expected_area = abs(cv2.contourArea(expected))
    assert abs(detected_area - expected_area) < abs(detected_area - outer_area)


def test_printed_chroma_neutral_panel_is_not_a_semantic_quiet_zone_candidate():
    """A printed panel must not create ambiguity with a physical patch."""
    rng = np.random.default_rng(55)
    image = np.full((1000, 1200, 3), (40, 125, 40), dtype=np.uint8)
    # This is deliberately a high-contrast, chroma-neutral information panel,
    # rather than a material surface.  Its printed lines make it a realistic
    # false region proposal if filled low-chroma masks are treated as patches.
    cv2.rectangle(image, (35, 580), (560, 930), (230, 230, 230), -1)
    for y in range(630, 900, 45):
        cv2.rectangle(image, (65, y), (520, y + 12), (25, 25, 25), -1)

    expected = np.array(
        [[760, 180], [1059, 180], [1059, 479], [760, 479]], dtype=np.float32,
    )
    patch = rng.integers(80, 190, (300, 300), dtype=np.uint8)
    patch = cv2.GaussianBlur(patch, (0, 0), 1.5)
    image[180:480, 760:1060] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image, (760, 180), (1059, 479), (20, 20, 20), 7)

    result = main.extract_quiet_zone(image)
    assert result["success"], result
    assert _mean_corner_error(result["corners"], expected) < 25.0


def test_printed_foreground_graphic_does_not_compete_with_filled_surface():
    image, expected = _surface_with_printed_graphic_scene()
    result = main.extract_quiet_zone(image)
    assert result["success"], result
    assert _mean_corner_error(result["corners"], expected) < 25.0
    assert result["metrics"]["surface_eligibility"]["eligible"] is True
    assert result["metrics"]["terminal_acceptance"] is True
    assert result["metrics"]["ambiguity_competition_reached"] is False


def test_handwritten_marks_do_not_make_a_physical_surface_ineligible():
    image, expected = _nested_scene()
    # Sparse internal strokes model handwriting without changing the physical
    # boundary or relying on text/OCR semantics.
    cv2.line(image, (500, 410), (610, 490), (30, 30, 30), 3)
    cv2.line(image, (560, 540), (650, 470), (30, 30, 30), 3)
    result = main.extract_quiet_zone(image)
    assert result["success"], result
    assert _mean_corner_error(result["corners"], expected) < 22.0
    assert result["metrics"]["surface_eligibility"]["eligible"] is True
    assert result["metrics"]["terminal_acceptance"] is True


def test_material_eligibility_rejects_sparse_graphic_warp():
    image, _ = _surface_with_printed_graphic_scene()
    graphic = image[80:430, 340:710]
    graphic = cv2.resize(graphic, (512, 512), interpolation=cv2.INTER_CUBIC)
    post = main._qz_post_warp_validation(graphic)
    assert post is not None
    eligibility = main._qz_surface_material_eligibility(graphic, post)
    assert eligibility["eligible"] is False


def test_high_confidence_alone_cannot_establish_terminal_acceptance():
    candidate = _terminal_candidate(score=0.99, sides=[])
    validation = main._qz_terminal_validation(candidate)
    assert validation["accepted"] is False
    assert validation["checks"]["confidence_sufficient"] is True
    assert validation["checks"]["four_side_support"] is False


def test_high_scoring_graphic_that_fails_material_eligibility_cannot_terminally_win():
    candidate = _terminal_candidate(score=0.99, surface={"eligible": False})
    validation = main._qz_terminal_validation(candidate)
    assert validation["accepted"] is False
    assert validation["checks"]["confidence_sufficient"] is True
    assert validation["checks"]["surface_eligible"] is False


def test_plain_unoutlined_material_surface_remains_detectable():
    """A surface transition can establish a Quiet Zone without a drawn border."""
    image = np.full((1000, 1200, 3), 35, dtype=np.uint8)
    expected = np.array([[390, 300], [760, 270], [785, 650], [410, 680]], dtype=np.float32)
    source = np.array([[0, 0], [359, 0], [359, 359], [0, 359]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, expected)
    patch = _texture(360, 360, seed=44)
    projected = cv2.warpPerspective(patch, transform, (1200, 1000))
    mask = cv2.warpPerspective(np.full((360, 360), 255, np.uint8), transform, (1200, 1000))
    image[mask > 0] = projected[mask > 0]

    result = main.extract_quiet_zone(image)
    assert result["success"], result
    assert _mean_corner_error(result["corners"], expected) < 20.0
    assert result["metrics"]["surface_eligibility"]["eligible"] is True
    assert result["metrics"]["terminal_acceptance"] is True


def test_off_centre_quiet_zone_does_not_require_a_centre_prior():
    image = np.full((1000, 1200, 3), 30, dtype=np.uint8)
    expected = np.array([[60, 610], [365, 590], [390, 910], [75, 925]], np.float32)
    source = np.array([[0, 0], [319, 0], [319, 319], [0, 319]], np.float32)
    transform = cv2.getPerspectiveTransform(source, expected)
    patch = _texture(320, 320, seed=7)
    projection = cv2.warpPerspective(patch, transform, (1200, 1000))
    mask = cv2.warpPerspective(np.full((320, 320), 255, np.uint8), transform, (1200, 1000))
    image[mask > 0] = projection[mask > 0]
    cv2.polylines(image, [expected.astype(np.int32)], True, (245, 245, 245), 5)
    result = main.extract_quiet_zone(image)
    assert result["success"], result
    assert _mean_corner_error(result["corners"], expected) < 20.0


def test_modest_directional_motion_blur_retains_recoverable_geometry():
    image, expected = _nested_scene()
    blurred = _directional_motion_blur(image, length=9, angle_degrees=35)
    result = main.extract_quiet_zone(blurred)
    assert result["success"], result
    assert _mean_corner_error(result["corners"], expected) < 24.0
    assert result["image"].shape == (512, 512, 3)
    assert "canonical_quality" in result["metrics"]


def test_severe_directional_motion_blur_fails_closed_when_geometry_is_unusable():
    image, _ = _nested_scene()
    blurred = _directional_motion_blur(image, length=401, angle_degrees=35)
    result = main.extract_quiet_zone(blurred)
    assert not result["success"]
    assert result["image"] is None


def test_canonical_quality_separates_blur_from_localisation():
    blurred_canonical = np.full((512, 512, 3), 140, dtype=np.uint8)
    quality = main.canonical_quiet_zone_quality(blurred_canonical)
    assert quality["quality_flags"] == ["IMAGE_TOO_BLURRY"]

    rng = np.random.default_rng(123)
    sharp_canonical = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    sharp_quality = main.canonical_quiet_zone_quality(sharp_canonical)
    assert sharp_quality["width"] == 512
    assert sharp_quality["height"] == 512


def test_existing_verification_policy_receives_quality_warning_after_localisation():
    decision = main.make_decision(
        trust_score=85,
        quality_score=70,
        inlier_count=22,
        quality_flags=["IMAGE_TOO_BLURRY"],
    )
    # This intentionally preserves the current downstream policy: a strong
    # fingerprint remains a pass with a quality warning, rather than becoming
    # a false Quiet Zone localisation failure.
    assert decision["decision"] == "pass"
    assert decision["is_match"] is True


def test_downstream_verification_retains_blur_quality_assessment():
    canonical = np.full((512, 512, 3), 140, dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", canonical)
    assert encoded_ok

    result = main.run_verification(encoded.tobytes(), encoded.tobytes())

    assert "IMAGE_TOO_BLURRY" in result["quality"]["quality_flags"]
    assert result["decision"] == "review"
    assert "matching" in result


def test_brand_baseline_accepts_localised_canonical_with_quality_warning(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "evidence.db"))
    monkeypatch.setattr(main, "QUIET_ZONE_EVIDENCE_DIR", str(tmp_path / "evidence-files"))
    monkeypatch.chdir(tmp_path)
    main.init_db()
    main.create_unit_record(
        "UNIT-QUALITY", "ORDER", "SELLER", "BUYER", "MARKET", "PRODUCT",
        "Product", "Brand", "Batch", None, None, None, None,
    )
    canonical = np.full((512, 512, 3), 140, dtype=np.uint8)

    monkeypatch.setattr(main, "decode_image", lambda _: np.zeros((900, 900, 3), dtype=np.uint8))
    monkeypatch.setattr(
        main,
        "extract_quiet_zone",
        lambda image, capture_context: {
            "success": True,
            "reason": "QUIET_ZONE_DETECTED",
            "confidence": 0.9,
            "image": canonical,
            "metrics": {"canonical_quality": main.canonical_quiet_zone_quality(canonical)},
            "capture_context": capture_context,
        },
    )
    package = UploadFile(filename="package.jpg", file=io.BytesIO(b"package"))
    seal = UploadFile(filename="seal.jpg", file=io.BytesIO(b"seal"))

    response = asyncio.run(
        main.register_brand_baseline_images(
            "UNIT-QUALITY",
            package,
            seal,
            package_capture_context="brand_baseline",
            seal_capture_context="brand_baseline",
        )
    )
    assert response["status"] == "brand_baseline_images_registered"
    assert response["unit_id"] == "UNIT-QUALITY"
    assert (tmp_path / "uploads" / f"{response['package_hash']}.jpg").is_file()
    assert (tmp_path / "uploads" / f"{response['seal_hash']}.jpg").is_file()

    conn = main.sqlite3.connect(main.DB_PATH)
    try:
        assert conn.execute("SELECT COUNT(*) FROM quiet_zone_evidence").fetchone()[0] == 2
    finally:
        conn.close()


def test_seller_registration_accepts_localised_canonical_with_quality_warning(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "seller-quality.db"))
    monkeypatch.setattr(main, "QUIET_ZONE_EVIDENCE_DIR", str(tmp_path / "evidence-files"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uploads").mkdir()
    main.init_db()
    canonical = np.full((512, 512, 3), 140, dtype=np.uint8)
    quality = main.canonical_quiet_zone_quality(canonical)
    assert quality["quality_flags"] == ["IMAGE_TOO_BLURRY"]

    monkeypatch.setattr(main, "decode_image", lambda _: np.zeros((900, 900, 3), dtype=np.uint8))
    monkeypatch.setattr(
        main,
        "extract_quiet_zone",
        lambda image, capture_context: {
            "success": True,
            "reason": "QUIET_ZONE_DETECTED",
            "confidence": 0.9,
            "image": canonical,
            "metrics": {"canonical_quality": quality},
            "capture_context": capture_context,
        },
    )

    response = asyncio.run(
        main.register_unit(
            "UNIT-SELLER-QUALITY", "ORDER", "SELLER", "BUYER", "MARKET", "PRODUCT",
            "Product", "Brand", "Batch",
            UploadFile(filename="package.jpg", file=io.BytesIO(b"package")),
            UploadFile(filename="seal.jpg", file=io.BytesIO(b"seal")),
            package_capture_context="seller_registration",
            seal_capture_context="seller_registration",
        )
    )

    assert response["status"] == "registered"
    assert response["unit_id"] == "UNIT-SELLER-QUALITY"
    assert (tmp_path / "uploads" / f"{response['package_hash']}.jpg").is_file()
    assert (tmp_path / "uploads" / f"{response['seal_hash']}.jpg").is_file()

    conn = main.sqlite3.connect(main.DB_PATH)
    try:
        assert conn.execute("SELECT COUNT(*) FROM quiet_zone_evidence").fetchone()[0] == 2
    finally:
        conn.close()


def test_brand_baseline_ambiguity_exposes_candidates_and_preserves_raw_upload(
    tmp_path,
    monkeypatch,
):
    """Ambiguity diagnostics retain raw bytes without becoming QZ evidence."""
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "diagnostics.db"))
    monkeypatch.setattr(main, "QUIET_ZONE_DIAGNOSTIC_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setattr(main, "QUIET_ZONE_DIAGNOSTIC_TTL_SECONDS", 3600)
    monkeypatch.chdir(tmp_path)
    main.init_db()
    main.create_unit_record(
        "UNIT-AMBIGUITY", "ORDER", "SELLER", "BUYER", "MARKET", "PRODUCT",
        "Product", "Brand", "Batch", None, None, None, None,
    )
    package_bytes = b"exact package multipart payload"
    seal_bytes = b"exact seal multipart payload"
    decoded_images = [
        np.zeros((900, 900, 3), dtype=np.uint8),
        np.ones((900, 900, 3), dtype=np.uint8),
    ]
    monkeypatch.setattr(main, "decode_image", lambda _: decoded_images.pop(0))
    canonical = np.full((512, 512, 3), 127, dtype=np.uint8)
    ambiguity = {
        "winner": {
            "final_score": 0.93,
            "candidate_confidence": 0.93,
            "corners": [[10, 10], [500, 10], [500, 500], [10, 500]],
            "candidate_source": "material_region",
            "candidate_area": 240100.0,
            "geometry_score": 0.96,
            "surface_eligibility": {"eligible": True, "tile_gradient_p30": 9.0},
            "score_components": {"geometry_contribution": 0.35},
        },
        "competitor": {
            "final_score": 0.89,
            "candidate_confidence": 0.89,
            "corners": [[100, 100], [450, 100], [450, 450], [100, 450]],
            "candidate_source": "edge_contour",
            "candidate_area": 122500.0,
            "geometry_score": 0.88,
            "surface_eligibility": {"eligible": True, "tile_gradient_p30": 8.0},
            "score_components": {"geometry_contribution": 0.30},
        },
        "score_difference": 0.04,
        "score_ratio": 0.957,
    }

    def extractor(image, capture_context):
        if image[0, 0, 0] == 0:
            return {
                "success": False,
                "reason": "QUIET_ZONE_DETECTION_AMBIGUOUS",
                "confidence": 0.93,
                "image": None,
                "metrics": {"ambiguity": ambiguity},
            }
        return {
            "success": True,
            "reason": "QUIET_ZONE_DETECTED",
            "confidence": 0.9,
            "image": canonical,
            "metrics": {"canonical_quality": main.canonical_quiet_zone_quality(canonical)},
        }

    monkeypatch.setattr(main, "extract_quiet_zone", extractor)
    response = asyncio.run(
        main.register_brand_baseline_images(
            "UNIT-AMBIGUITY",
            UploadFile(filename="package.jpg", file=io.BytesIO(package_bytes)),
            UploadFile(filename="seal.jpg", file=io.BytesIO(seal_bytes)),
        )
    )
    payload = json.loads(response.body)
    package = payload["package_quiet_zone"]
    assert response.status_code == 422
    assert package["reason"] == "QUIET_ZONE_DETECTION_AMBIGUOUS"
    assert package["ambiguity_diagnostics"] == ambiguity
    assert "diagnostic_upload" in package
    assert package["diagnostic_upload"]["debug_resubmission"] == {
        "endpoint": "/debug/quiet-zone",
        "multipart_field": "file",
    }
    assert "diagnostic_upload" not in payload["seal_quiet_zone"]

    diagnostic_id = package["diagnostic_upload"]["diagnostic_id"]
    stored = main.get_failed_quiet_zone_diagnostic_upload(diagnostic_id)
    assert stored["original_bytes"] == package_bytes
    retrieved = asyncio.run(main.get_failed_quiet_zone_upload_original(diagnostic_id))
    assert retrieved.body == package_bytes
    assert retrieved.headers["cache-control"] == "no-store"

    conn = main.sqlite3.connect(main.DB_PATH)
    try:
        assert conn.execute("SELECT COUNT(*) FROM quiet_zone_diagnostic_uploads").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM quiet_zone_evidence").fetchone()[0] == 0
    finally:
        conn.close()


def test_flat_panel_fails_closed():
    image = np.full((900, 900, 3), 40, dtype=np.uint8)
    cv2.rectangle(image, (250, 250), (650, 650), (180, 180, 180), -1)
    cv2.rectangle(image, (250, 250), (650, 650), (245, 245, 245), 5)
    result = main.extract_quiet_zone(image)
    assert not result["success"]
    assert result["image"] is None


def test_two_equally_supported_zones_are_ambiguous():
    image = np.full((1000, 1000, 3), 35, dtype=np.uint8)
    for x, y, seed in ((100, 100, 1), (560, 560, 1)):
        image[y:y + 300, x:x + 300] = _texture(300, 300, seed)
        cv2.rectangle(image, (x, y), (x + 299, y + 299), (245, 245, 245), 5)
    result = main.extract_quiet_zone(image)
    assert not result["success"]
    assert result["reason"] == "QUIET_ZONE_DETECTION_AMBIGUOUS"
    assert result["corners"] is None
    assert result["image"] is None
    ambiguity = result["metrics"]["ambiguity"]
    assert result["confidence"] == round(ambiguity["winner"]["final_score"], 4)
    assert ambiguity["score_difference"] >= 0.0
    assert 0.0 < ambiguity["score_ratio"] <= 1.0
    for candidate_name in ("winner", "competitor"):
        candidate = ambiguity[candidate_name]
        assert len(candidate["corners"]) == 4
        assert candidate["candidate_area"] > 0.0
        assert candidate["candidate_source"] in {"edge_contour", "material_region"}
        assert candidate["surface_eligibility"]["eligible"] is True
        assert candidate["terminal_validation"]["accepted"] is True
        assert "geometry_contribution" in candidate["score_components"]


def test_debug_quiet_zone_returns_the_extractor_canonical_png(monkeypatch):
    """The debug endpoint must not substitute an annotated source frame."""
    source = np.full((900, 900, 3), (20, 80, 150), dtype=np.uint8)
    canonical = np.zeros((512, 512, 3), dtype=np.uint8)
    canonical[:, :, 0] = 17
    canonical[:, :, 1] = 93
    canonical[:, :, 2] = 201
    calls = []

    def extractor(image, capture_context):
        calls.append((image.shape, capture_context))
        return {
            "success": True,
            "reason": "QUIET_ZONE_DETECTED",
            "confidence": 0.8765,
            "corners": [[100, 100], [700, 100], [700, 700], [100, 700]],
            "image": canonical,
            "capture_context": capture_context,
        }

    monkeypatch.setattr(main, "extract_quiet_zone", extractor)
    encoded_ok, encoded_source = cv2.imencode(".jpg", source)
    assert encoded_ok
    upload = UploadFile(filename="camera.jpg", file=io.BytesIO(encoded_source.tobytes()))
    response = asyncio.run(main.debug_quiet_zone(upload))

    assert len(calls) == 1
    assert calls[0][0] == source.shape
    assert calls[0][1] == "debug"
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["X-Quiet-Zone-Confidence"] == "0.8765"
    decoded = cv2.imdecode(np.frombuffer(response.body, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (512, 512, 3)
    assert np.array_equal(decoded, canonical)


def test_quiet_zone_evidence_is_append_only_and_served_from_controlled_id(tmp_path, monkeypatch):
    temp_dir = tmp_path
    monkeypatch.setattr(main, "DB_PATH", str(temp_dir / "evidence.db"))
    monkeypatch.setattr(main, "QUIET_ZONE_EVIDENCE_DIR", str(temp_dir / "evidence-files"))
    main.init_db()
    canonical = np.full((512, 512, 3), 123, dtype=np.uint8)
    result = {
        "success": True,
        "confidence": 0.91,
        "reason": "QUIET_ZONE_DETECTED",
        "capture_context": "test",
        "metrics": {"localisation_confidence": 0.91, "canonical_quality": {"quality_score": 100}},
    }
    first = main.save_quiet_zone_evidence(
        "UNIT-1", "package", "brand_baseline", canonical, result, related_event_id="UNIT-1",
    )
    second = main.save_quiet_zone_evidence("UNIT-1", "seal", "verification", canonical, result)
    assert first["evidence_id"] != second["evidence_id"]
    history = asyncio.run(main.get_quiet_zone_evidence("UNIT-1"))
    assert history["count"] == 2
    assert [item["capture_type"] for item in history["evidence"]] == ["package", "seal"]
    assert history["evidence"][0]["related_event_id"] == "UNIT-1"
    conn = main.sqlite3.connect(main.DB_PATH)
    try:
        stored_diagnostics = conn.execute(
            "SELECT diagnostics_json FROM quiet_zone_evidence WHERE evidence_id = ?",
            (first["evidence_id"],),
        ).fetchone()[0]
        assert json.loads(stored_diagnostics)["localisation_confidence"] == 0.91
    finally:
        conn.close()
    image = asyncio.run(main.get_quiet_zone_evidence_image(first["evidence_id"]))
    decoded = cv2.imread(image.path)
    assert decoded.shape == (512, 512, 3)
    assert np.array_equal(decoded, canonical)


def test_quiet_zone_evidence_rejects_failed_or_noncanonical_results(tmp_path, monkeypatch):
    temp_dir = tmp_path
    monkeypatch.setattr(main, "DB_PATH", str(temp_dir / "evidence.db"))
    monkeypatch.setattr(main, "QUIET_ZONE_EVIDENCE_DIR", str(temp_dir / "evidence-files"))
    main.init_db()
    canonical = np.zeros((512, 512, 3), dtype=np.uint8)
    failed = {"success": False, "reason": "QUIET_ZONE_NOT_DETECTED"}

    try:
        main.save_quiet_zone_evidence("UNIT-1", "package", "verification", canonical, failed)
        assert False, "failed extraction must not create evidence"
    except ValueError:
        pass
    try:
        main.save_quiet_zone_evidence(
            "UNIT-1", "package", "verification", np.zeros((500, 500, 3), dtype=np.uint8),
            {"success": True},
        )
        assert False, "non-canonical image must not create evidence"
    except ValueError:
        pass

    conn = main.sqlite3.connect(main.DB_PATH)
    try:
        assert conn.execute("SELECT COUNT(*) FROM quiet_zone_evidence").fetchone()[0] == 0
    finally:
        conn.close()


def test_quiet_zone_evidence_root_is_configurable_and_not_exposed(tmp_path, monkeypatch):
    temp_dir = tmp_path
    configured_root = temp_dir / "custom-evidence-root"
    monkeypatch.setattr(main, "DB_PATH", str(temp_dir / "evidence.db"))
    monkeypatch.setattr(main, "QUIET_ZONE_EVIDENCE_DIR", str(configured_root))
    main.init_db()
    canonical = np.full((512, 512, 3), 47, dtype=np.uint8)
    evidence = main.save_quiet_zone_evidence(
        "UNIT-2", "package", "verification", canonical,
        {"success": True, "confidence": 0.5, "reason": "QUIET_ZONE_DETECTED"},
    )

    assert main._quiet_zone_evidence_root() == str(configured_root)
    assert (configured_root / f"{evidence['evidence_id']}.png").is_file()
    history = asyncio.run(main.get_quiet_zone_evidence("UNIT-2"))
    assert history["evidence"][0]["image_url"] == (
        f"/api/v1/quiet-zone-evidence/{evidence['evidence_id']}/image"
    )
    assert str(configured_root) not in str(history)
    try:
        asyncio.run(main.get_quiet_zone_evidence_image("..%2Ffiberhash.db"))
        assert False, "an arbitrary filename must not be retrievable"
    except main.HTTPException as error:
        assert error.status_code == 404


def test_all_production_quiet_zone_workflows_record_both_capture_types():
    expected = {
        main.verify_unit: "verification",
        main.register_unit: "seller_registration",
        main.register_brand_baseline_images: "brand_baseline",
    }
    for endpoint, workflow in expected.items():
        source = inspect.getsource(endpoint)
        assert f'"package", "{workflow}"' in source
        assert f'"seal", "{workflow}"' in source
    assert "save_quiet_zone_evidence" not in inspect.getsource(main.debug_quiet_zone)
