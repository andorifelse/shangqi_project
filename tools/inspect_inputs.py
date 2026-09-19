"""Extract reference video contact sheets; original inputs remain untouched."""
from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root / "outputs" / "reference"
out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(root / "demo_example.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = count / fps
print(f"Video: {count} frames, {fps:.3f} fps, {duration:.2f}s")
for page in range(3):
    tiles = []
    for sec in np.linspace(page * duration / 3, (page + 1) * duration / 3 - 0.1, 8):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(sec * 1000))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode video at {sec}")
        frame = cv2.resize(frame, (640, 360))
        cv2.putText(frame, f"{sec:.1f}s", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        tiles.append(frame)
    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 8, 2)])
    cv2.imwrite(str(out / f"video_{page + 1}.jpg"), sheet)
cap.release()

