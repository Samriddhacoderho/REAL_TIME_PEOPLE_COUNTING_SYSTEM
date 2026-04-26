# Real-Time People Counting System

A computer-vision project that detects, tracks, and counts people in a video stream using **Ultralytics YOLO (YOLOv10)** and **OpenCV**.  
The pipeline performs **person-only detection**, assigns **persistent track IDs**, and increments a counter when a person **crosses a virtual counting line**.

> Main notebook: **`people_counting_system.ipynb`** (Google Colab-ready)

---

## Key Features

- **Real-time person detection** using Ultralytics YOLO (pretrained weights)
- **Multi-object tracking** with persistent IDs (`model.track(..., persist=True)`)
- **Line-crossing based counting** (simple and effective baseline)
- **Annotated output video** generation (`output_count.mp4`)
- Person filtering via YOLO class selection (**class 0 = person**)

---

## How It Works (High Level)

1. **Load YOLOv10 model** (e.g., `yolov10n.pt`)
2. **Read frames** from a video using OpenCV
3. **Track people** in each frame:
   - detect only class `person` (`classes=[0]`)
   - retrieve bounding boxes + track IDs
4. **Compute the centroid** of each tracked box
5. **Count** a person when the centroid crosses the horizontal line (default: middle of the frame)
6. **Render overlays**:
   - centroids
   - track IDs
   - counting line
   - total count
7. **Write** annotated frames to `output_count.mp4`

---

## Tech Stack

- **Python**
- **Ultralytics** (YOLO + tracking)
- **OpenCV**
- (Optional) **Google Colab** for GPU acceleration

---

## Repository Structure

- `people_counting_system.ipynb` — main implementation notebook (install, run detection, tracking, counting, and video export)

---

## Setup & Installation

### Option A — Run on Google Colab (Recommended)
Open the notebook in Colab and run the cells in order.

The notebook installs dependencies like this:
```bash
pip install ultralytics
```

### Option B — Run Locally
1. Create a virtual environment (recommended)
2. Install dependencies:
```bash
pip install ultralytics opencv-python
```

> Note: Tracking may require extra packages (e.g., `lap`). Ultralytics can auto-install missing requirements depending on your environment.

---

## Usage

### 1) Load the YOLO model
In the notebook, the model is initialized like:
```python
from ultralytics import YOLO
model = YOLO("yolov10n.pt")
```

### 2) Provide your input video path
Example used in the notebook:
```python
video_path = "/content/walk.mp4"
```

### 3) Run tracking + counting
Core logic (summary):
- `model.track(frame, persist=True, classes=[0])`
- keep a list of `counted_ids`
- increment `count` when a track crosses the line

### 4) Output
The notebook writes an annotated video:
- **Output file:** `output_count.mp4`
- Includes the counting line and `Total People: N` overlay

---

## Configuration Tips

You can easily tweak these parameters in the notebook:

- **Counting line position**
  - Currently: `line_y = height // 2` (middle of frame)
  - Move up/down to match your scene layout

- **Counting logic**
  - Current baseline counts when `cy < line_y` and ID not counted yet  
  - You can upgrade to “crossing direction” logic (top→bottom vs bottom→top) for better accuracy.

- **Resolution / speed**
  - Use smaller YOLO models (e.g., `yolov10n`) for faster inference
  - Consider resizing frames for performance if needed

---

## Limitations (Current Baseline)

- The current counting rule is a **simple line condition**; depending on camera angle and movement direction, you may want a true **line-crossing event** (track history-based).
- **Occlusion** (people overlapping) can cause ID switches and affect accuracy in crowded scenes.
- Performance depends on hardware (GPU strongly recommended for real-time).

---

## Future Improvements (Ideas)

- Direction-aware counting (IN/OUT)
- Region-of-interest masking (ignore irrelevant areas)
- Tracking history trails for robust crossing detection
- Export analytics (CSV logs: timestamp, ID, event)
- Stream input support (webcam / RTSP)

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV community

---

## License

Add a license file if you plan to distribute this project publicly (e.g., MIT, Apache-2.0).

---

## Author

**Samriddhacoderho**  
GitHub: https://github.com/Samriddhacoderho
