#!/usr/bin/env python3

from pathlib import Path
import argparse
import cv2
import numpy as np


INPUT_DIR = Path("/home/ubuntu/Homography/dataset/input_images")
OUTPUT_DIR = Path("/home/ubuntu/Homography/dataset/output_images")
MIN_CONTOUR_AREA = 5_000  # absolute floor; adaptive thresholding below can raise this


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as tl, tr, br, bl."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_document_corners(image: np.ndarray) -> np.ndarray:
    """Detect four document corners using scored contours with quad/hull preference and rect fallback."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thr_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9)
    thr_adapt_inv = 255 - thr_adapt
    _, thr_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def to_edges(src: np.ndarray) -> np.ndarray:
        closed = cv2.morphologyEx(src, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        edges_local = cv2.Canny(closed, 50, 150)
        edges_local = cv2.dilate(edges_local, None, iterations=2)
        edges_local = cv2.erode(edges_local, None, iterations=1)
        return edges_local

    edge_maps = [to_edges(thr_adapt), to_edges(thr_adapt_inv), to_edges(thr_otsu)]

    h, w = image.shape[:2]
    min_area = max(MIN_CONTOUR_AREA, 0.02 * h * w)  # allow smaller docs but avoid noise
    max_area = 0.90 * h * w  # avoid full-frame grabs

    best_pts = None
    best_score = 0.0
    best_rect_box = None
    best_rect_score = 0.0

    for edges in edge_maps:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            hull = cv2.convexHull(contour)
            peri_hull = cv2.arcLength(hull, True)
            approx_hull = cv2.approxPolyDP(hull, 0.02 * peri_hull, True)

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="float32")
            rect_area = max(rect[1][0] * rect[1][1], 1.0)

            rectangularity = area / rect_area
            area_ratio = area / (h * w)
            aspect = (
                min(rect[1][0], rect[1][1]) / max(rect[1][0], rect[1][1])
                if rect[1][0] and rect[1][1]
                else 0
            )

            corner_bonus = 0.0
            candidate = box
            if len(approx) == 4:
                candidate = approx.reshape(4, 2)
                corner_bonus = 0.3
            elif len(approx_hull) == 4:
                candidate = approx_hull.reshape(4, 2)
                corner_bonus = 0.2

            score = (
                rectangularity * 0.65
                + area_ratio * 0.20
                + aspect * 0.10
                + corner_bonus
            )

            rect_score = rectangularity * 0.75 + area_ratio * 0.25

            if score > best_score:
                best_score = score
                best_pts = candidate

            if rect_score > best_rect_score:
                best_rect_score = rect_score
                best_rect_box = box

    if best_pts is not None:
        return order_points(np.array(best_pts, dtype="float32"))
    if best_rect_box is not None:
        return order_points(np.array(best_rect_box, dtype="float32"))

    # Final fallback: image corners
    fallback = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    return order_points(fallback)


def warp_image(image: np.ndarray, src_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp image to a rectangle defined by the detected corners and return homography."""
    rect = order_points(src_pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(round(max(width_a, width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(round(max(height_a, height_b)))

    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    h_matrix, _ = cv2.findHomography(rect, dst)
    warped = cv2.warpPerspective(image, h_matrix, (max_width, max_height))
    return warped, h_matrix


def process_image(image_path: Path, output_dir: Path) -> Path:
    """Load, rectify, and save a single image plus its homography matrix."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    corners = detect_document_corners(image)
    warped, h_matrix = warp_image(image, corners)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), warped)
    np.save(str(output_dir / f"{image_path.stem}_H.npy"), h_matrix)
    return output_path


def run_batch(input_dir: Path, output_dir: Path) -> None:
    """Process all images in the input directory."""
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    )
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images. Writing outputs to {output_dir}")
    for img_path in image_paths:
        try:
            out_path = process_image(img_path, output_dir)
            print(f"Saved: {out_path}")
        except Exception as exc:
            print(f"Failed: {img_path} -> {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic document rectification using homography.")
    parser.add_argument("--input_dir", type=Path, default=INPUT_DIR, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="Directory to save rectified images.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch(args.input_dir, args.output_dir)

