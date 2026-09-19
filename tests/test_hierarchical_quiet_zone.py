"""Deterministic localisation tests for the hierarchical Quiet Zone scanner.

These fixtures deliberately use no physical-size or capture-guide information.
They assert localisation, not merely a successful response.
"""

import asyncio
import io
import inspect

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


def test_nested_inner_boundary_beats_larger_outer_package_boundary():
    image, expected = _nested_scene()
    result = main.extract_quiet_zone(image)
    assert result["success"], result
    assert _mean_corner_error(result["corners"], expected) < 18.0
    assert result["image"].shape == (512, 512, 3)
    assert result["metrics"]["detection_source"] == "hierarchical_boundary_scanner"


def test_false_outer_rectangle_is_not_returned_as_a_successful_crop():
    image, expected = _nested_scene()
    result = main.extract_quiet_zone(image)
    assert result["success"], result
    detected_area = abs(cv2.contourArea(np.asarray(result["corners"], dtype=np.float32)))
    outer_area = abs(cv2.contourArea(np.array([[120, 140], [1070, 110], [1100, 875], [150, 900]], np.float32)))
    expected_area = abs(cv2.contourArea(expected))
    assert abs(detected_area - expected_area) < abs(detected_area - outer_area)


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
    assert result["reason"] in (
        "QUIET_ZONE_DETECTION_AMBIGUOUS",
        "QUIET_ZONE_DETECTION_CONFIDENCE_TOO_LOW",
        "QUIET_ZONE_NOT_DETECTED",
    )


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
