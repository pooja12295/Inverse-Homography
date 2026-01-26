## Homography Image Rectification

Batch image rectification via homography. The provided script automatically detects planar quadrilateral regions, warps each image to a fronto-parallel view, and saves both the rectified image and the homography matrix.

### Layout
- `script/inverse_homography_rectification.py` — main batch rectification script.
- `dataset/input_images/` — place input images here (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`).
- `dataset/output_images/` — rectified outputs are written here along with homography matrices.

### Quick Start
1) Install dependencies (Python 3.8+):
```bash
pip install opencv-python numpy
```

2) Run batch rectification (from repo root):
```bash
python3 script/inverse_homography_rectification.py
```
Options:
```bash
python3 script/inverse_homography_rectification.py \
  --input_dir /path/to/inputs \
  --output_dir /path/to/outputs
```

### Outputs
- Rectified images: `dataset/output_images/<name>.<ext>`
- Homography matrices: `dataset/output_images/<name>_H.npy` (NumPy array mapping source → rectified)

