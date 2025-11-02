import cv2
import numpy as np
import sys

IMAGE_PATH = "qr_soc51.jpg"

img_color = cv2.imread(IMAGE_PATH)
if img_color is None:
    sys.exit(f"Error: Cannot open image '{IMAGE_PATH}'")

gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])
gray = cv2.filter2D(gray, -1, kernel)

qr_decoder = cv2.QRCodeDetector()

retval, decoded_info, points, _ = qr_decoder.detectAndDecode(gray)

if not retval or not decoded_info:
    sys.exit("Error: No QR code detected. Try better lighting or closer photo.")

print("\nDecoded QR Code:")
print("-" * 60)
print(decoded_info)
print("-" * 60)

if points is not None:
    points = np.int32(points).reshape((-1, 1, 2))
    cv2.polylines(img_color, [points], isClosed=True, color=(0, 255, 0), thickness=3)

cv2.putText(
    img_color,
    f"{decoded_info[:30]}{'...' if len(decoded_info) > 30 else ''}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2,
)

cv2.imshow("Detected QR Code", img_color)
print("\nPress any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()