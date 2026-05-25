import cv2
import os

files = ["img4.jpg", "ecg_warped.jpg", "ecg_warped_test.jpg", "test.jpg"]
for f in files:
    if os.path.exists(f):
        img = cv2.imread(f)
        print(f"{f}: shape={img.shape}")
    else:
        print(f"{f} does not exist")
