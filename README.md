# Real-Time People Counting System (Bidirectional)

A computer-vision project that detects, tracks, and counts people in a video stream using **Ultralytics YOLO (YOLOv10)** and **OpenCV**.  
It performs **person-only tracking** with persistent IDs and supports **bidirectional line-crossing counting** (IN/OUT) using centroid history.

> Main notebook: **`people_counting_system.ipynb`** (Google Colab GPU-friendly)

---

## Highlights

- Person detection with **YOLOv10** (e.g., `yolov10n.pt`)
- Robust multi-object tracking using **ByteTrack** (`tracker="bytetrack.yaml"`)
- **Configurable counting line**
  - Vertical line (left ↔ right)
  - Horizontal line (up ↔ down)
- **Bidirectional counting (IN/OUT)** using track centroid history
- Exports an annotated output video: **`output_count.mp4`**

---

## How It Works (Notebook-Aligned)

1. Load YOLOv10 model
2. Read frames from a video using OpenCV
3. Track only persons:
   ```python
   results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml")
   ```
4. For each tracked ID:
   - compute centroid `(cx, cy)`
   - compare previous centroid position vs current position
   - detect a **crossing event** at the counting line
5. Update counters (bidirectional):
   - **IN** when crossing happens in one direction
   - **OUT** when crossing happens in the opposite direction
6. Draw overlays (IDs, centroids, line, counts) and write output video

---

## Counting Modes

The notebook uses a `user_choice` variable to decide which line to use:

- `user_choice = 0` → **Vertical** line at `line_x = width // 2`
  - Cross **Right → Left** → increments one counter (e.g., `IN`)
  - Cross **Left → Right** → increments the other counter (e.g., `OUT`)

- `user_choice = 1` → **Horizontal** line at `line_y = height // 2`
  - Cross **Top → Bottom** → increments one counter (e.g., `IN`)
  - Cross **Bottom → Top** → increments the other counter (e.g., `OUT`)

> If you prefer different semantics (e.g., “IN=left→right”), just swap the increment logic in the notebook.

---

## Tech Stack

- Python
- Ultralytics (YOLO + tracking)
- OpenCV
- Google Colab (optional)

---

## Repository Contents

- `people_counting_system.ipynb` — end-to-end implementation (install → demo predict → video tracking → bidirectional counting → export)

---

## Setup

### Run on Google Colab (Recommended)
Open the notebook in Colab and run cells top-to-bottom.

Installs:
```bash
pip install ultralytics
```

### Run Locally
```bash
pip install ultralytics opencv-python
```

> Tracking dependencies (e.g., `lap`) may be auto-installed by Ultralytics if missing.

---

## Usage

### 1) Load model
```python
from ultralytics import YOLO
model = YOLO("yolov10n.pt")
```

### 2) Select video
```python
video_path = "/content/IMG_0628.mp4"
cap = cv2.VideoCapture(video_path)
```

### 3) Configure counting line
```python
line_x = width // 2
line_y = height // 2
user_choice = 0  # 0=vertical, 1=horizontal
```

### 4) Output
- Output video: `output_count.mp4`
- Overlays typically include:
  - centroid marker
  - track ID label
  - counting line (vertical/horizontal)
  - total counts (IN/OUT and/or total)

---

## Notes / Limitations

- Counting quality depends on tracking stability; heavy occlusion may cause **ID switches**.
- Consider adding an **offset / tolerance** region around the line to avoid jitter-based double triggers.
- For long videos, you may want to periodically clean old IDs from history if they disappear.

---

## Future Improvements (Optional)

- ROI masking for better accuracy
- Export event logs to CSV (timestamp, ID, direction)
- Live stream support (webcam / RTSP)
- UI toggles for line position and direction mapping

---

## Acknowledgements

- Ultralytics YOLO: https://github.com/ultralytics/ultralytics  
- ByteTrack (via Ultralytics tracker configs)  
- OpenCV

---

## Author

**Samriddhacoderho**  
GitHub: https://github.com/Samriddhacoderho
