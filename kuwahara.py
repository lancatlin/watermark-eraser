import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from utils import timeit, display, canny
from functools import wraps


def cache_file(func):
    @wraps(func)
    def cache_file_wrapper(*args, **kwargs):
        if "basename" in kwargs:
            basename = kwargs["basename"]
            del kwargs["basename"]
            filename = f"{basename}-{func.__name__}.npy"
            try:
                return np.load(filename)
            except FileNotFoundError:
                result = func(*args, **kwargs)
                np.save(filename, result)
                return result
        return func(*args, **kwargs)

    return cache_file_wrapper


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
            idx = np.argmin(
                variance[
                    i - half_size : i + half_size + 1, j - half_size : j + half_size + 1
                ]
            )
            x = idx // window_size
            y = idx % window_size
            # 將最小方差對應的強度值分配給該像素
            result[i, j] = image[i - half_size + x, j - half_size + y]

    return result


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "watermark2.jpg"
    image = cv2.imread(filename)
    result = kuwahara_color(image, 5)
    # Save the file in output folder with base name
    basename = filename.split("/")[-1].split(".")[0]
    cv2.imwrite(f"images/kuwahara_{basename}.jpg", result)
    display(result)
    plt.show()
