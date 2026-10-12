from pathlib import Path

import cv2
import numpy as np


def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        img[y, :, 0] = int(50 + 100 * y / height) 
        img[y, :, 1] = int(30 + 80 * y / height) 
        img[y, :, 2] = int(20 + 60 * y / height) 

    cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), -1)
    cv2.rectangle(img, (300, 100), (500, 350), (255, 0, 0), -1)

    cv2.circle(img, (500, 100), 60, (0, 0, 255), -1)
    cv2.circle(img, (150, 350), 80, (255, 255, 0), -1)

    cv2.putText(img, "TEST IMAGE", (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return img


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "sample_data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "test_synthetic.jpg"

    img = create_test_image()
    cv2.imwrite(str(out_path), img)
    print(f"Saved test image: {out_path} ({img.shape[1]}x{img.shape[0]})")
