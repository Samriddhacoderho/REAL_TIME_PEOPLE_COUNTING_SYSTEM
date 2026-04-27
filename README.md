# Real-Time People Counting System

A computer-vision project that detects, tracks, and counts people in a video using **Ultralytics YOLO (YOLOv10)** + **OpenCV**.  
It performs **person-only tracking**, assigns **persistent IDs**, and counts a person when they **cross a configurable virtual counting line** (vertical or horizontal).

> Main notebook: **`people_counting_system.ipynb`** (Google Colab GPU-friendly)

---

## What’s New / Current Logic (Notebook-Aligned)

- Uses `model.track(..., tracker="bytetrack.yaml")` for more stable tracking
- Supports **two counting modes** via `user_choice`:
  - `user_choice = 0` → **vertical line** counting (x-direction crossing)
  - `user_choice = 1` → **horizontal line** counting (y-direction crossing)
- Uses a simple **track history** dictionary to detect an actual **crossing event**:
  - stores previous `cx` or `cy` for each `track_id`
  - increments count only when crossing happens (and only once per ID using `counted_ids`)

---

## Key Features

- **Real-time person detection** with pretrained YOLOv10 weights (`yolov10n.pt`)
- **Multi-object tracking** with persistent IDs
- **Crossing-based counting** using previous vs current centroid position
- **Annotated output video export** to `output_count.mp4`
- Filters only the `person` class (`classes=[0]`)

---

## How It Works (High Level)

1. Load YOLOv10 model
2. Read frames from a video with OpenCV
3. Track persons with ByteTrack:
   - `results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml")`
4. For each tracked person:
   - compute centroid `(cx, cy)`
   - compare current centroid with previous centroid stored in `track_history`
5. If the centroid **crosses the chosen line**, increase the counter once per ID
6. Draw overlays (IDs, centroids, counting line, total count)
7. Write frames to `output_count.mp4`

---

## Tech Stack

- **Python**
- **Ultralytics** (YOLO + tracking)
- **OpenCV**
- **Google Colab** (optional, recommended for GPU)

---

## Repository Structure

- `people_counting_system.ipynb` — complete pipeline (install → predict demo → video tracking → counting → export)

---

## Setup & Installation

### Option A — Run on Google Colab (Recommended)
Open the notebook and run all cells in order.

The notebook installs:
```bash
pip install ultralytics
```

### Option B — Run Locally
```bash
pip install ultralytics opencv-python
```

> Note: Ultralytics tracking may require extra packages (e.g., `lap`). If missing, Ultralytics may auto-install them depending on your environment.

---

## Usage

### 1) Load the model
```python
from ultralytics import YOLO
model = YOLO("yolov10n.pt")
```

### 2) Set your input video path
Example in the notebook:
```python
video_path = "/content/IMG_0628.mp4"
cap = cv2.VideoCapture(video_path)
```

### 3) Choose counting mode
In the notebook:
```python
line_x = width // 2      # vertical line
line_y = height // 2     # horizontal line
user_choice = 0          # 0=vertical, 1=horizontal
```

### 4) Run tracking + counting
- `track_history` stores last centroid position per ID
- `counted_ids` ensures each ID is counted only once

### 5) Output
- Output file: **`output_count.mp4`**
- The rendered video includes:
  - track IDs
  - centroid markers
  - counting line (vertical or horizontal)
  - **Total People** overlay

---

## Notes / Limitations

- Current counting counts **one crossing per unique track ID**. If someone turns back and crosses again, they will not be counted again unless you reset logic.
- In crowded scenes, occlusions can still cause **ID switches**, affecting counts.
- Counting direction differs by mode:
  - Vertical mode currently triggers on a **right → left** crossing of the vertical line.
  - Horizontal mode currently triggers on a **top → bottom** crossing of the horizontal line.

(These can be extended to support bidirectional IN/OUT counting.)

---

## Future Improvements (Optional Ideas)

- Bidirectional counting (IN/OUT)
- Use an `offset` tolerance near the line (already defined in notebook; can be applied to reduce jitter)
- ROI masking to ignore irrelevant areas
- Save events to CSV (time, ID, direction)

---

## Acknowledgements

- Ultralytics YOLO: https://github.com/ultralytics/ultralytics  
- ByteTrack (via Ultralytics tracker configs)
- OpenCV

---

## Author

**Samriddhacoderho**  
GitHub: https://github.com/Samriddhacoderho
