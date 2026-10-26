import cv2
import numpy as np

def erosi(mask, k=3):
    r = k // 2
    pad = np.pad(mask, r, constant_values=0)
    out = np.zeros_like(mask, dtype=bool)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            out[y, x] = pad[y:y+k, x:x+k].all()
    return out

def dilasi(mask, k=3):
    r = k // 2
    pad = np.pad(mask, r, constant_values=0)
    out = np.zeros_like(mask, dtype=bool)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            out[y, x] = pad[y:y+k, x:x+k].any()
    return out

def opening(m, k=3):
    return dilasi(erosi(m, k), k)

def closing(m, k=3):
    return erosi(dilasi(m, k), k)

img = cv2.imread('canon.png', 0)

if img is not None:
    mask = img > 127
    
    hasil_erosi = erosi(mask, k=3)
    hasil_dilasi = dilasi(mask, k=3)
    hasil_opening = opening(mask, k=3)
    hasil_closing = closing(mask, k=3)

    cv2.imshow('Image Source', img)
    cv2.imshow('Hasil Erosi', (hasil_erosi * 255).astype(np.uint8))
    cv2.imshow('Hasil Dilasi', (hasil_dilasi * 255).astype(np.uint8))
    cv2.imshow('Hasil Opening', (hasil_opening * 255).astype(np.uint8))
    cv2.imshow('Hasil Closing', (hasil_closing * 255).astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Gambar tidak ditemukan.")