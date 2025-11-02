import cv2
import numpy as np
from pyzbar import pyzbar

image_path = "pdf417_license.jpg"         
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Cannot open {image_path}")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = clahe.apply(img)

img = cv2.GaussianBlur(img, (3, 3), 0)

barcodes = pyzbar.decode(img, symbols=[pyzbar.ZBarSymbol.PDF417])

if not barcodes:
    raise RuntimeError("No PDF-417 barcode found. Try adjusting the image or preprocessing.")

for barcode in barcodes:
    decoded_text = barcode.data.decode("utf-8", errors="ignore")
    print("Decoded PDF-417 data:")
    print(decoded_text)
