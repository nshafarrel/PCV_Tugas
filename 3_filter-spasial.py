import cv2
import numpy as np

img = cv2.imread('canon.png', 0)
tinggi, lebar = img.shape
    
# Blank image as output placeholder
img_hasil = np.zeros((tinggi, lebar), dtype=np.uint8)
    
# Kernel Spasial 7x7 (Mean filter)
kernel = np.ones((7, 7), dtype=np.float64) / 49.0
                       
# Konvolusi (Filter spasial) dengan tipe float
img_float = img.astype(np.float64)
for i in range(3, tinggi - 3):
    for j in range(3, lebar - 3):
        area = img_float[i-3 : i+4, j-3 : j+4]
        
        hasil_perhitungan = np.sum(area * kernel)
        
        # Kliping dan konversi aman ke uint8
        img_hasil[i, j] = int(np.clip(np.floor(hasil_perhitungan + 0.5), 0, 255))

cv2.imshow('Grayscale', img)
cv2.imshow('Hasil Filter Spasial 7x7 (Manual Blur)', img_hasil)  
cv2.waitKey(0)
cv2.destroyAllWindows()