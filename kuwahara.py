import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from utils import timeit, display, cache_file


@timeit
@cache_file
def kuwahara_color(image, window_size):
    # 確保窗口大小是奇數
    if window_size % 2 == 0:
        window_size += 1
    
    # 複製原始圖像
    result = image.copy()

    fimage = image.astype(np.float32)
    
    # 計算圖像的平均值和方差
    mean = cv2.boxFilter(fimage, cv2.CV_32F, (window_size, window_size))
    mean_sq = cv2.boxFilter(fimage * fimage, cv2.CV_32F, (window_size, window_size))
    variance = np.mean(np.abs(mean_sq - mean * mean), axis=2)

    display(variance, colorbar=True)
    
    # 遍歷圖像的每個像素
    rows, cols, channels = image.shape
    half_size = window_size // 2
    for i in range(half_size, rows - half_size):
        for j in range(half_size, cols - half_size):
            idx = np.argmin(variance[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1])
            x = idx // window_size
            y = idx % window_size
            # 將最小方差對應的強度值分配給該像素
            result[i, j] = image[i - half_size + x, j - half_size + y]
    
    return result


def kuwahara_color_each(img, window_size):
    channels = cv2.split(img)
    return cv2.merge([kuwahara(c, window_size) for c in channels])


@timeit
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


def canny(image, threshold1, threshold2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, threshold1, threshold2)
    
    return edges



if __name__ == "__main__":
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = 'watermark2.jpg'
    image = cv2.imread(filename)
    result = kuwahara_color(image, 5)
    # Save the file in output folder with base name
    basename = filename.split("/")[-1].split(".")[0]
    cv2.imwrite(f"images/kuwahara_{basename}.jpg", result)
    display(result)
    plt.show()
