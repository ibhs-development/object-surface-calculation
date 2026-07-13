#!/usr/bin/env python3
"""Measure leaf surface area in a photograph using an ArUco scale marker.

The leaf and marker must lie on the same flat plane.  The marker's four corners
define an image-to-centimetre homography, so the result is corrected for camera
tilt as well as scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


ARUCO_DICTIONARIES = {
    name.removeprefix("DICT_"): value
    for name, value in vars(cv2.aruco).items()
    if name.startswith("DICT_") and isinstance(value, int)
}
SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class MeasurementError(RuntimeError):
    """Raised when a reliable marker or leaf measurement cannot be made."""


@dataclass(frozen=True)
class LeafMeasurement:
    image: str
    area_cm2: float
    marker_id: int
    marker_size_cm: float
    marker_edge_pixels: list[float]


@dataclass
class Analysis:
    measurement: LeafMeasurement
    annotated_image: np.ndarray
    leaf_mask: np.ndarray


def _aruco_dictionary(name: str):
    normalized = name.upper().removeprefix("DICT_")
    if normalized not in ARUCO_DICTIONARIES:
        choices = ", ".join(sorted(ARUCO_DICTIONARIES))
        raise MeasurementError(
            f"Unknown ArUco dictionary {name!r}. Available dictionaries: {choices}"
        )
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[normalized])


def _detect_marker(
    image: np.ndarray, dictionary_name: str, requested_id: int | None
) -> tuple[np.ndarray, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        _aruco_dictionary(dictionary_name), parameters
    )
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or not corners:
        raise MeasurementError(
            "No ArUco marker was detected. Check that the entire marker is visible, "
            "in focus, and matches --dictionary."
        )

    flat_ids = ids.flatten().astype(int)
    if requested_id is None:
        if len(flat_ids) != 1:
            found = ", ".join(map(str, flat_ids.tolist()))
            raise MeasurementError(
                f"Detected multiple markers ({found}); select one with --marker-id."
            )
        index = 0
    else:
        matches = np.flatnonzero(flat_ids == requested_id)
        if len(matches) == 0:
            found = ", ".join(map(str, flat_ids.tolist()))
            raise MeasurementError(
                f"Marker ID {requested_id} was not found. Detected marker IDs: {found}"
            )
        index = int(matches[0])

    # OpenCV returns top-left, top-right, bottom-right, bottom-left.
    return corners[index].reshape(4, 2).astype(np.float32), int(flat_ids[index])


def _make_leaf_mask(
    image: np.ndarray,
    marker_corners: np.ndarray,
    min_saturation: int,
    max_dark_value: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a filled mask and the selected outer leaf contour."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # A leaf is normally either chromatic or distinctly darker than white paper.
    foreground = np.where(
        (saturation >= min_saturation) | (value <= max_dark_value), 255, 0
    ).astype(np.uint8)

    # Remove the marker and a safety margin so its printed/antialiased edge can
    # never be selected as the leaf.
    edge_lengths = np.linalg.norm(
        marker_corners - np.roll(marker_corners, -1, axis=0), axis=1
    )
    marker_padding = max(3, int(round(float(np.mean(edge_lengths)) * 0.12)))
    marker_mask = np.zeros(foreground.shape, dtype=np.uint8)
    cv2.fillConvexPoly(marker_mask, np.rint(marker_corners).astype(np.int32), 255)
    marker_mask = cv2.dilate(
        marker_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * marker_padding + 1, 2 * marker_padding + 1)
        ),
    )
    foreground[marker_mask != 0] = 0

    short_side = min(image.shape[:2])
    close_size = max(3, int(round(short_side * 0.003)))
    close_size += 1 - close_size % 2
    open_size = max(3, int(round(short_side * 0.001)))
    open_size += 1 - open_size % 2
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
    )

    contours, _ = cv2.findContours(
        foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    image_area = image.shape[0] * image.shape[1]
    candidates = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= image_area * 0.0001
    ]
    if not candidates:
        raise MeasurementError(
            "No leaf-sized object was segmented. Try lowering --min-saturation or "
            "raising --max-dark-value."
        )

    leaf_contour = max(candidates, key=cv2.contourArea)
    leaf_mask = np.zeros(foreground.shape, dtype=np.uint8)
    cv2.drawContours(leaf_mask, [leaf_contour], -1, 255, cv2.FILLED)
    return leaf_mask, leaf_contour


def analyze_image(
    image_path: str | Path,
    *,
    marker_size_cm: float = 5.0,
    marker_id: int | None = 23,
    dictionary: str = "6X6_250",
    min_saturation: int = 35,
    max_dark_value: int = 180,
) -> Analysis:
    """Analyze one image and return its metric area and diagnostic images."""
    path = Path(image_path)
    if marker_size_cm <= 0:
        raise MeasurementError("--marker-size-cm must be greater than zero.")
    if not 0 <= min_saturation <= 255 or not 0 <= max_dark_value <= 255:
        raise MeasurementError("Segmentation thresholds must be between 0 and 255.")

    if not path.is_file():
        raise MeasurementError(f"Could not read image: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise MeasurementError(f"Could not read image: {path}")

    marker_corners, detected_id = _detect_marker(image, dictionary, marker_id)
    leaf_mask, leaf_contour = _make_leaf_mask(
        image, marker_corners, min_saturation, max_dark_value
    )

    metric_corners = np.array(
        [
            [0.0, 0.0],
            [marker_size_cm, 0.0],
            [marker_size_cm, marker_size_cm],
            [0.0, marker_size_cm],
        ],
        dtype=np.float32,
    )
    image_to_cm = cv2.getPerspectiveTransform(marker_corners, metric_corners)
    metric_contour = cv2.perspectiveTransform(
        leaf_contour.astype(np.float32), image_to_cm
    )
    area_cm2 = abs(float(cv2.contourArea(metric_contour)))
    if not np.isfinite(area_cm2) or area_cm2 <= 0:
        raise MeasurementError("The computed leaf area is invalid.")

    edge_lengths = np.linalg.norm(
        marker_corners - np.roll(marker_corners, -1, axis=0), axis=1
    )
    measurement = LeafMeasurement(
        image=str(path),
        area_cm2=round(area_cm2, 3),
        marker_id=detected_id,
        marker_size_cm=marker_size_cm,
        marker_edge_pixels=[round(float(length), 1) for length in edge_lengths],
    )

    annotated = image.copy()
    cv2.drawContours(annotated, [leaf_contour], -1, (0, 0, 255), 8)
    cv2.polylines(
        annotated,
        [np.rint(marker_corners).astype(np.int32)],
        True,
        (255, 0, 255),
        8,
        cv2.LINE_AA,
    )
    anchor = tuple(np.rint(leaf_contour[:, 0, :].min(axis=0)).astype(int))
    text_y = max(45, anchor[1] - 24)
    cv2.putText(
        annotated,
        f"Leaf area: {area_cm2:.2f} cm^2",
        (max(10, anchor[0]), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        (255, 255, 255),
        7,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Leaf area: {area_cm2:.2f} cm^2",
        (max(10, anchor[0]), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    return Analysis(measurement, annotated, leaf_mask)


def _write_image(path: Path, image: np.ndarray, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise MeasurementError(f"Could not write {description}: {path}")


def _write_results_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_name", "leaf_area_cm2", "status", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _find_images(directory: Path, recursive: bool) -> list[Path]:
    entries = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        (
            path
            for path in entries
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            and not path.stem.lower().endswith(("_measured", "_mask"))
        ),
        key=lambda path: str(path.relative_to(directory)).lower(),
    )


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    csv_output: str | Path | None = None,
    recursive: bool = False,
    marker_size_cm: float = 5.0,
    marker_id: int | None = 23,
    dictionary: str = "6X6_250",
    min_saturation: int = 35,
    max_dark_value: int = 180,
) -> tuple[Path, int, int]:
    """Process images in a folder and return CSV path, successes, and failures."""
    source = Path(input_dir)
    destination = Path(output_dir)
    if not source.is_dir():
        raise MeasurementError(f"Input folder does not exist: {source}")

    images = _find_images(source, recursive)
    if not images:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise MeasurementError(
            f"No supported images found in {source}. Supported extensions: {supported}"
        )

    csv_path = Path(csv_output) if csv_output else destination / "leaf_areas.csv"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    succeeded = 0
    failed = 0

    print(f"Found {len(images)} image(s) in {source}", flush=True)
    for index, image_path in enumerate(images, start=1):
        relative_path = image_path.relative_to(source)
        print(
            f"[{index}/{len(images)}] Processing {relative_path} ...",
            flush=True,
        )
        output_subdir = destination / relative_path.parent
        measured_path = output_subdir / f"{image_path.stem}_measured.jpg"
        mask_path = output_subdir / f"{image_path.stem}_mask.png"
        try:
            analysis = analyze_image(
                image_path,
                marker_size_cm=marker_size_cm,
                marker_id=marker_id,
                dictionary=dictionary,
                min_saturation=min_saturation,
                max_dark_value=max_dark_value,
            )
            _write_image(measured_path, analysis.annotated_image, "annotated image")
            _write_image(mask_path, analysis.leaf_mask, "mask image")
        except MeasurementError as exc:
            failed += 1
            rows.append(
                {
                    "image_name": str(relative_path),
                    "leaf_area_cm2": "",
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"    ERROR: {exc}", flush=True)
            continue

        succeeded += 1
        rows.append(
            {
                "image_name": str(relative_path),
                "leaf_area_cm2": f"{analysis.measurement.area_cm2:.3f}",
                "status": "ok",
                "error": "",
            }
        )
        print(
            f"    Done: {analysis.measurement.area_cm2:.3f} cm^2",
            flush=True,
        )

    _write_results_csv(csv_path, rows)

    print(
        f"Finished: {succeeded} succeeded, {failed} failed. CSV: {csv_path}",
        flush=True,
    )
    return csv_path, succeeded, failed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate leaf area in cm^2 from one photo or every photo in a folder."
        )
    )
    parser.add_argument(
        "input_path", type=Path, help="photo or folder containing photos"
    )
    parser.add_argument(
        "--marker-size-cm",
        type=float,
        default=5.0,
        help="physical side length of the marker's black square (default: 5.0)",
    )
    parser.add_argument("--marker-id", type=int, default=23, help="marker ID (default: 23)")
    parser.add_argument(
        "--dictionary", default="6X6_250", help="ArUco dictionary (default: 6X6_250)"
    )
    parser.add_argument(
        "--min-saturation",
        type=int,
        default=35,
        help="minimum HSV saturation treated as leaf (default: 35)",
    )
    parser.add_argument(
        "--max-dark-value",
        type=int,
        default=180,
        help="maximum HSV value treated as non-white/leaf (default: 180)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="single-image annotated output (default: leaf_area_annotated.jpg)",
    )
    parser.add_argument("--mask-output", type=Path, help="single-image leaf mask path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="batch measured/mask output folder (default: output)",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        help=(
            "CSV path (single default: beside measured image; "
            "batch default: OUTPUT_DIR/leaf_areas.csv)"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="also process images in subfolders during batch mode",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.input_path.is_dir():
        if args.output or args.mask_output:
            print(
                "error: --output and --mask-output are for a single image; "
                "use --output-dir for folder input.",
                file=sys.stderr,
            )
            return 2
        try:
            csv_path, succeeded, failed = process_folder(
                args.input_path,
                args.output_dir,
                csv_output=args.csv_output,
                recursive=args.recursive,
                marker_size_cm=args.marker_size_cm,
                marker_id=args.marker_id,
                dictionary=args.dictionary,
                min_saturation=args.min_saturation,
                max_dark_value=args.max_dark_value,
            )
        except MeasurementError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(
                json.dumps(
                    {
                        "input_folder": str(args.input_path),
                        "output_folder": str(args.output_dir),
                        "csv": str(csv_path),
                        "succeeded": succeeded,
                        "failed": failed,
                    },
                    indent=2,
                )
            )
        return 1 if failed else 0

    output_path = args.output or Path("leaf_area_annotated.jpg")
    csv_path = args.csv_output or output_path.with_suffix(".csv")
    try:
        analysis = analyze_image(
            args.input_path,
            marker_size_cm=args.marker_size_cm,
            marker_id=args.marker_id,
            dictionary=args.dictionary,
            min_saturation=args.min_saturation,
            max_dark_value=args.max_dark_value,
        )
        _write_image(output_path, analysis.annotated_image, "annotated image")
        if args.mask_output:
            _write_image(args.mask_output, analysis.leaf_mask, "mask image")
        _write_results_csv(
            csv_path,
            [
                {
                    "image_name": args.input_path.name,
                    "leaf_area_cm2": f"{analysis.measurement.area_cm2:.3f}",
                    "status": "ok",
                    "error": "",
                }
            ],
        )
    except MeasurementError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = asdict(analysis.measurement)
    result["annotated_image"] = str(output_path)
    result["csv"] = str(csv_path)
    if args.mask_output:
        result["mask_image"] = str(args.mask_output)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Leaf area: {analysis.measurement.area_cm2:.3f} cm^2")
        print(f"Annotated image: {output_path}")
        if args.mask_output:
            print(f"Leaf mask: {args.mask_output}")
        print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
