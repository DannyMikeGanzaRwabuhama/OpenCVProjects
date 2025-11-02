import cv2
import numpy as np
import sys

IMAGE_PATH = "maxicode.png"  

img_color = cv2.imread(IMAGE_PATH)
if img_color is None:
    sys.exit(f"Error: Cannot open image '{IMAGE_PATH}'")

gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

gray = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 21, 5
)

scale = 2.0
h, w = gray.shape
if min(h, w) < 200:  
    gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

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
    sys.exit("Error: No MaxiCode detected. Try clearer image or better lighting.")

print("\nDecoded MaxiCode:")
print("-" * 70)
for i, text in enumerate(decoded_info):
    if text:
        barcode_type = decoded_type[i] if decoded_type else "Unknown"
        print(f"[{i+1}] {text}")
        print(f"    → Type: {barcode_type}")
print("-" * 70)

for pts in corners:
    pts = np.int32(pts).reshape((-1, 1, 2))
    cv2.polylines(img_color, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

first_text = next((t for t in decoded_info if t), "")
label = f"MaxiCode: {len(first_text)} chars"
cv2.putText(
    img_color,
    label,
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2,
)

cv2.imshow("Detected MaxiCode", img_color)
print("\nPress any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()