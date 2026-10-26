import cv2
import numpy as np

def bgr_to_hsv_manual(img):
    img_float = img.astype(np.float64) / 255.0
    b, g, r = img_float[..., 0], img_float[..., 1], img_float[..., 2]
    
    cmax = np.max(img_float, axis=-1)
    cmin = np.min(img_float, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    s = np.zeros_like(cmax)
    v = cmax

    mask = delta != 0
    s[mask] = delta[mask] / cmax[mask]

    mask_r = mask & (cmax == r)
    mask_g = mask & (cmax == g)
    mask_b = mask & (cmax == b)

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

    hsv = np.stack([h / 2.0, s * 255.0, v * 255.0], axis=-1)
    return np.clip(np.floor(hsv + 0.5), 0, 255).astype(np.uint8) 

def in_range_manual(hsv_img, lower, upper):
    mask = (hsv_img[..., 0] >= lower[0]) & (hsv_img[..., 0] <= upper[0]) & \
           (hsv_img[..., 1] >= lower[1]) & (hsv_img[..., 1] <= upper[1]) & \
           (hsv_img[..., 2] >= lower[2]) & (hsv_img[..., 2] <= upper[2])
    return mask

def bitwise_and_manual(img, mask):
    out = np.zeros_like(img)
    out[mask] = img[mask]
    return out

# 1. Read image
img = cv2.imread("canon.png")

if img is not None:
    # 2. Show Image
    cv2.imshow("Original Image", img)

    # 3. Filter Color Image
    hsv_img = bgr_to_hsv_manual(img) 

    # Filter Range Warna Putih, Silver, dan Abu-abu
    lower_end = np.array([10, 50, 50])
    upper_end = np.array([25, 255, 255])

    # Mask manual
    mask_img = in_range_manual(hsv_img, lower_end, upper_end)
    filtered_img = bitwise_and_manual(img, mask_img)
        
    cv2.imshow('Filtered: White, Silver, Grey', filtered_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 4. Filter Color Video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break   
    
    hsv_video = bgr_to_hsv_manual(frame)
    
    # Mask manual
    mask_video = in_range_manual(hsv_video, lower_end, upper_end)
    filtered_video = bitwise_and_manual(frame, mask_video)
    
    # Show Video
    cv2.imshow("Video Original", frame)
    cv2.imshow("Video Filtered: White/Silver/Grey", filtered_video)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()