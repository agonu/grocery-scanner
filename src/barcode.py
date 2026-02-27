"""Barcode and QR-code detection using pyzbar.

When a barcode is readable, the UPC/EAN can be used to look up the product
directly — bypassing the visual retrieval pipeline entirely.

Usage:
    from src.barcode import detect_barcodes
    hits = detect_barcodes(pil_image)
    # hits: [{"data": "0123456789012", "type": "EAN13", "box": (x1,y1,x2,y2)}]
"""

import numpy as np
from PIL import Image


def detect_barcodes(image: Image.Image) -> list[dict]:
    """Detect and decode barcodes and QR codes in a PIL image.

    Returns a list of dicts, one per detected code:
        data  — decoded string (e.g. UPC/EAN number)
        type  — symbology ("EAN13", "UPCA", "QRCODE", …)
        box   — (x1, y1, x2, y2) pixel bounding box

    Returns an empty list if pyzbar is not installed or no codes found.
    """
    try:
        from pyzbar import pyzbar
    except ImportError:
        return []

    img_array = np.array(image.convert("RGB"))
    try:
        decoded = pyzbar.decode(img_array)
    except Exception:
        return []

    results = []
    for bc in decoded:
        r = bc.rect
        results.append({
            "data": bc.data.decode("utf-8", errors="replace"),
            "type": bc.type,
            "box": (r.left, r.top, r.left + r.width, r.top + r.height),
        })
    return results
