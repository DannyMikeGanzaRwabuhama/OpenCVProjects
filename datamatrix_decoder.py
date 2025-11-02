import cv2
import numpy as np
import sys

IMAGE_PATH = "datamatrix_food.jpg"

img_color = cv2.imread(IMAGE_PATH)
if img_color is None:
    sys.exit(f"Error: Cannot open image '{IMAGE_PATH}'")

gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# Reduce noise, preserve edges
gray = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)

# Optional: Sharpen slightly
kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])
gray = cv2.filter2D(gray, -1, kernel)

detector = cv2.barcode_BarcodeDetector()

retval = detector.detectAndDecode(gray)

if len(retval) == 4:
    ok, decoded_info, decoded_type, corners = retval
elif len(retval) == 3:
    ok, decoded_info, corners = retval
    decoded_type = None
else:
    sys.exit("Error: Unexpected return from detectAndDecode")

if not ok or not decoded_info or not any(decoded_info):
    sys.exit("Error: No Data Matrix detected. Try better lighting or closer photo.")

print("\nDecoded Data Matrix:")
print("-" * 60)
for i, text in enumerate(decoded_info):
    if text:
        print(f"[{i+1}] {text}")
print("-" * 60)

for pts in corners:
    pts = np.int32(pts).reshape((-1, 1, 2))
    cv2.polylines(img_color, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

# Add label
cv2.putText(
    img_color,
    f"DataMatrix: {len(decoded_info[0]) if decoded_info[0] else 0} chars",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2,
)

cv2.imshow("Detected Data Matrix", img_color)
print("\nPress any key to close the window...")
cv2.waitKey(0)
cv2.destroyAllWindows()