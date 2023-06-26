import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import time
from functools import wraps


figure_count = 1


def display(*imgs, cmap="gray", axis=False, colorbar=False, title="", **kwargs):
    global figure_count
    for i, img in enumerate(imgs):
        # Check if the image is in BGR format, convert it to RGB
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(img, cmap=cmap, **kwargs)
        if colorbar:
            plt.colorbar()
        if title:
            plt.title(title)
        plt.axis(axis)
        try:
            plt.savefig(
                f"doc/fruits/fruits-{figure_count}.jpg",
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
            )
        except Exception as e:
            print(e)
        figure_count += 1


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


@timeit
def conv(img, kernel):
    return cv2.filter2D(img, -1, kernel)


def canny(image, threshold1, threshold2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, threshold1, threshold2)

    return edges


def key(a, b):
    return str(tuple(quantize(a)) + tuple(quantize(b)))


def quantize(x, bits=3):
    return x + (1 << bits - 1) - 1 >> bits << bits


def parse_array(arr_str):
    # 移除首尾的方括號，然後使用空格拆分字符串，再轉換為整數陣列
    arr = np.array(arr_str.strip("[]").split(), dtype=np.float32)
    return arr
