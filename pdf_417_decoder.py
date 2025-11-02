#!/usr/bin/env python3
"""
Decode a PDF-417 barcode from an image (e.g. a driver-license photo).

Requirements
------------
pip install opencv-python pylibdmtx pyzbar numpy
"""

import cv2
import numpy as np
from pyzbar import pyzbar

# ----------------------------------------------------------------------
# 1. Load the image
# ----------------------------------------------------------------------
image_path = "pdf417_license.jpg"          # <-- replace with your file name
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Cannot open {image_path}")

# ----------------------------------------------------------------------
# 2. Optional preprocessing (helps with low-contrast / blurry images)
# ----------------------------------------------------------------------
# Increase contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = clahe.apply(img)

# Slight Gaussian blur to reduce noise (keep barcode edges)
img = cv2.GaussianBlur(img, (3, 3), 0)

# ----------------------------------------------------------------------
# 3. Decode with pyzbar (PDF-417 is natively supported)
# ----------------------------------------------------------------------
barcodes = pyzbar.decode(img, symbols=[pyzbar.ZBarSymbol.PDF417])

if not barcodes:
    raise RuntimeError("No PDF-417 barcode found. Try adjusting the image or preprocessing.")

for barcode in barcodes:
    # barcode.data is bytes → decode to UTF-8 string
    decoded_text = barcode.data.decode("utf-8", errors="ignore")
    print("Decoded PDF-417 data:")
    print(decoded_text)
    # If you want the raw bytes:
    # print("Raw bytes:", barcode.data)

# ----------------------------------------------------------------------
# 4. (Optional) Show the image with the detected region highlighted
# ----------------------------------------------------------------------
# Uncomment the block below if you want a quick visual check

"""
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for b in barcodes:
    (x, y, w, h) = b.rect
    cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("Detected PDF-417", color_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""