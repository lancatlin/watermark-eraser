import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


def display(*img, bgr=False, cmap="gray", axis=False, colorbar=False, title='', **kwargs):
    for i in img:
        if bgr:
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        plt.imshow(i, cmap=cmap, **kwargs)
        if colorbar:
            plt.colorbar()
        if title:
            plt.title(title)
        plt.axis(axis)
        if title:
            plt.savefig(f"./HW4_img/{title}.jpg")
        plt.show()

def modify(img, contrast, brightness):
    output = img * (contrast/127 + 1) - contrast + brightness
    output = np.clip(output, 0, 255)
    output = np.uint8(output)
    return output


def in_range(num1, num2, diff = 10):
    return abs(num1 - num2) <= diff


def kuwahara(image, window_size):
    # 確保窗口大小是奇數
    if window_size % 2 == 0:
        window_size += 1
    
    # 複製原始圖像
    result = image.copy()
    
    # 圖像轉換為灰度
    gray = image
    
    # 計算圖像的平均值和方差
    mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))
    mean_sq = cv2.boxFilter(gray * gray, cv2.CV_32F, (window_size, window_size))
    variance = mean_sq - mean * mean
    
    # 遍歷圖像的每個像素
    rows, cols = gray.shape
    half_size = window_size // 2
    for i in range(half_size, rows - half_size):
        for j in range(half_size, cols - half_size):
            # 對每個窗口計算局部統計信息
            mean_w = mean[i, j]
            variance_w = variance[i, j]
            
            min_variance = float('inf')
            min_intensity = 0
            
            # 遍歷窗口的每個子區域
            for x in range(i - half_size, i + half_size + 1):
                for y in range(j - half_size, j + half_size + 1):
                    # 計算子區域的方差
                    variance_s = variance[x, y]
                    
                    # 如果方差更小，更新最小方差和對應的強度值
                    if variance_s < min_variance:
                        min_variance = variance_s
                        min_intensity = gray[x, y]
            
            # 將最小方差對應的強度值分配給該像素
            result[i, j] = min_intensity
    
    return result


def find_white(image):
    # 將圖像轉換為灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 對灰度圖像進行閾值化
    _, thresholded = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

    return thresholded


def canny(image, threshold1, threshold2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, threshold1, threshold2)
    
    return edges


def band_pass(image, high, low):
    h, w = image.shape
    center = [h//2, w//2]
    high = np.sqrt(h**2 + w**2) * high / 2
    low = np.sqrt(h**2 + w**2) * low / 2

    fft_img = np.fft.fftshift(np.fft.fft2(image))

    mask = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            if low < distance < high:
                mask[i, j] = 1

    fft_img = fft_img * mask
    display(np.log(np.abs(fft_img)))
    fft_img = np.fft.ifftshift(fft_img)


    ifft_img = np.uint8(np.abs(np.fft.ifft2(fft_img)))

    return ifft_img

def find_same(pixel, threshold=10):
    # count the difference between 3 channels

    diff = 0
    for i in range(3):
        diff += abs(pixel[i] - pixel[i-1])
    return diff < threshold



def find_watermark(img):
    blured = cv2.blur(image, (5,5))
    diff_img = img - blured

    nr, nc  = img.shape[:2]

    watermark = np.zeros(img.shape[:2])
    for y in range(nr):
        for x in range(nc):
            if find_same(diff_img[y][x], 20):
                watermark[y][x] = 1
    
    return watermark

if __name__ == "__main__":
    image = cv2.imread('fruits.jpg', 0)

    test = find_watermark(image) * 255

    cv2.imshow("original", image)
    cv2.imshow("test", test)
    cv2.waitKey()