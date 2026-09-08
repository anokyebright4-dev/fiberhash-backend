"""Deterministic synthetic image helpers for Quiet Zone engine tests.

These helpers build fully synthetic scenes only. They never encode any
physical size, millimetre/pixel calibration, or the 45x45 mm capture guide:
the backend must detect the physical patch purely from image content.
"""

import cv2
import numpy as np


def low_freq_texture(size, base=180.0, amplitude=40.0, seed=0):
    """A smooth, deterministic surface texture (no long straight lines)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(1.0, size / 24.0))
    spread = float(np.ptp(noise)) + 1e-6
    noise = (noise - noise.min()) / spread
    gray = (base + (noise - 0.5) * 2.0 * amplitude).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _place_mark(image, x0, y0, patch_w, patch_h, mark_corner, radius=16):
    if mark_corner is None:
        return
    mx = x0 + radius + 4 if mark_corner in ("tl", "bl") else x0 + patch_w - radius - 4
    my = y0 + radius + 4 if mark_corner in ("tl", "tr") else y0 + patch_h - radius - 4
    cv2.circle(image, (int(mx), int(my)), radius, (255, 255, 255), -1)


def frame_with_patch(
    frame_w,
    frame_h,
    center_x,
    center_y,
    patch_w,
    patch_h,
    background=40,
    base=180.0,
    amplitude=40.0,
    mark_corner=None,
    seed=0,
):
    """A background frame containing one axis-aligned textured patch."""
    image = np.full((frame_h, frame_w, 3), background, np.uint8)
    x0 = int(center_x - patch_w // 2)
    y0 = int(center_y - patch_h // 2)
    patch = low_freq_texture(max(patch_w, patch_h), base, amplitude, seed)
    image[y0:y0 + patch_h, x0:x0 + patch_w] = patch[:patch_h, :patch_w]
    _place_mark(image, x0, y0, patch_w, patch_h, mark_corner)
    quad = np.array(
        [
            [x0, y0],
            [x0 + patch_w, y0],
            [x0 + patch_w, y0 + patch_h],
            [x0, y0 + patch_h],
        ],
        dtype=np.float32,
    )
    return image, quad


def marked_patch_frame(
    frame_size,
    patch_size,
    mark_corner,
    background=40,
    base=170.0,
    amplitude=8.0,
    seed=1,
):
    """A centred, low-texture patch dominated by a single corner mark.

    Used to isolate orientation resolution: the mark is by far the strongest
    intrinsic asymmetry, so a correct resolver always maps it to the canonical
    top-left regardless of capture rotation.
    """
    center = frame_size // 2
    return frame_with_patch(
        frame_size,
        frame_size,
        center,
        center,
        patch_size,
        patch_size,
        background=background,
        base=base,
        amplitude=amplitude,
        mark_corner=mark_corner,
        seed=seed,
    )


def symmetric_patch_frame(frame_size, patch_size, background=40):
    """A 4-fold symmetric (separable cosine) textured patch.

    Its |high-pass| energy is invariant under 90 deg rotation about the
    centre, so the energy centroid sits at the centre and canonical
    orientation is genuinely ambiguous, while the patch still has real
    surface variance and a single clean rectangular boundary.
    """
    image = np.full((frame_size, frame_size, 3), background, np.uint8)
    center = frame_size // 2
    x0 = center - patch_size // 2
    y0 = center - patch_size // 2
    yy, xx = np.mgrid[0:patch_size, 0:patch_size].astype(np.float32)
    cx = cy = (patch_size - 1) / 2.0
    pattern = (
        150.0
        + 45.0 * np.cos((xx - cx) * 0.18)
        + 45.0 * np.cos((yy - cy) * 0.18)
    )
    patch = cv2.cvtColor(pattern.clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    image[y0:y0 + patch_size, x0:x0 + patch_size] = patch
    quad = np.array(
        [
            [x0, y0],
            [x0 + patch_size, y0],
            [x0 + patch_size, y0 + patch_size],
            [x0, y0 + patch_size],
        ],
        dtype=np.float32,
    )
    return image, quad


def rotate_image_and_quad(image, quad, angle_deg, center):
    """Rotate the whole scene and the quad together (simulates capture rotation)."""
    matrix = cv2.getRotationMatrix2D(
        (float(center[0]), float(center[1])), float(angle_deg), 1.0
    )
    height, width = image.shape[:2]
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderValue=(40, 40, 40),
    )
    rotated_quad = cv2.transform(
        quad.reshape(-1, 1, 2).astype(np.float32), matrix
    ).reshape(-1, 2)
    return rotated, rotated_quad.astype(np.float32)


def brightest_quadrant(canonical):
    """Return 'tl'/'tr'/'br'/'bl' for the quadrant holding the brightest mark."""
    gray = cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    _, _, _, max_loc = cv2.minMaxLoc(gray)
    x, y = max_loc
    height, width = gray.shape[:2]
    top = y < height / 2.0
    left = x < width / 2.0
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"
