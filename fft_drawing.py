import numpy as np
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def spectrum(F):
    Fshift = fftshift(F)
    mag = 20 * np.log(np.abs(Fshift) + 1)
    return np.uint8(mag)


def inverse(img_freq):
    return np.uint8(np.abs(ifft2(img_freq)))


# 讀取圖片
img = cv2.imread('fruits.jpg', cv2.IMREAD_GRAYSCALE)
#img = img[70:150, 600:720]

# 取得影像大小
h, w = img.shape[:2]

# 初始化 mask
mask = np.ones((h, w), np.uint8)

# 滑鼠回調函數
def draw_mask(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE and flags:
        # 繪製 mask
        cv2.circle(mask, (x, y), 10, 0, -1)

def erase_freq(img_freq, mask):
    return ifftshift(fftshift(img_freq) * mask)

# 建立視窗
cv2.namedWindow('image')
cv2.namedWindow('spectrum')

# 綁定滑鼠回調函數到 'spectrum' 視窗
cv2.setMouseCallback('spectrum', draw_mask)

cv2.imshow('image', img)
img_freq = fft2(img)
spect = spectrum(img_freq)

while(1):
    display = np.minimum(spect, mask * 255)
    cv2.imshow('spectrum', display)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # Esc
        break
    elif key == 13: # Enter
        img_freq = erase_freq(img_freq, mask)
        spect = spectrum(img_freq)
        cv2.imshow('image', inverse(img_freq))

# 關閉視窗
cv2.destroyAllWindows()

cv2.waitKey(0)
cv2.destroyAllWindows()
