"""Run manual real-capture Quiet Zone validation from caller-supplied paths."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


def _review_board(source, result, filename):
    """Build one persistent full-frame/localisation/canonical review image."""
    max_height = 900
    scale = min(1.0, max_height / source.shape[0])
    full = cv2.resize(source, (round(source.shape[1] * scale), round(source.shape[0] * scale)))
    if result["success"]:
        corners = np.asarray(result["corners"], dtype=np.float32) * scale
        cv2.polylines(full, [corners.astype(np.int32)], True, (0, 0, 255), max(2, round(5 * scale)))
        canonical = result["image"]
        status = "PASS — visually review boundary and canonical before accepting"
        detail = f"confidence={result['confidence']:.4f}  corners={result['corners']}"
    else:
        canonical = np.full((512, 512, 3), 32, dtype=np.uint8)
        status = "FAIL — no canonical Quiet Zone emitted"
        detail = f"reason={result['reason']}  confidence={result['confidence']:.4f}"
        cv2.putText(canonical, "REJECTED", (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 255), 3, cv2.LINE_AA)
    canonical = cv2.resize(canonical, (512, 512))
    header, footer = 90, 105
    board = np.full((max(full.shape[0], 512) + header + footer, full.shape[1] + 512 + 30, 3), 245, dtype=np.uint8)
    board[header:header + full.shape[0], :full.shape[1]] = full
    board[header:header + 512, full.shape[1] + 30:] = canonical
    cv2.putText(board, filename, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(board, status, (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 110, 0) if result["success"] else (0, 0, 180), 2, cv2.LINE_AA)
    cv2.putText(board, "Original + detected quadrilateral", (18, header - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(board, "512 x 512 canonical", (full.shape[1] + 30, header - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    for index, line in enumerate((detail[i:i + 115] for i in range(0, len(detail), 115))):
        cv2.putText(board, line, (18, board.shape[0] - footer + 28 + index * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    return board


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for source in map(Path, args.images):
        image = cv2.imread(str(source))
        result = main.extract_quiet_zone(image)
        item = {
            "file": source.name,
            "shape": list(image.shape) if image is not None else None,
            "success": result["success"],
            "reason": result["reason"],
            "confidence": result["confidence"],
            "corners": result["corners"],
        }
        if image is not None:
            overlay = image.copy()
            if result["success"]:
                cv2.polylines(overlay, [np.asarray(result["corners"], np.int32)], True, (0, 0, 255), 8)
                cv2.imwrite(str(output / f"{source.name}.canonical.png"), result["image"])
            cv2.imwrite(str(output / f"{source.name}.overlay.jpg"), overlay)
            cv2.imwrite(str(output / f"{source.name}.validation.png"), _review_board(image, result, source.name))
        report.append(item)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main_cli()
