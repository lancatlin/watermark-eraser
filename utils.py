import cv2
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import time
from hashlib import md5 as hash
from functools import wraps


figure_count = 1


def display(*imgs, cmap="gray", axis=False, colorbar=False, title='', **kwargs):
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
            plt.savefig(f"doc/fruits/fruits-{figure_count}.jpg",
                        bbox_inches='tight', pad_inches=0, dpi=300)
        except Exception as e:
            print(e)
        figure_count += 1


def spectrum(F):
    Fshift = fftshift(F)
    mag = 20 * np.log(np.abs(Fshift) + 1)
    return np.uint8(mag)


def show_spectrum(*F, **kwargs):
    display(*[spectrum(f) for f in F], colorbar=True, **kwargs)


def inverse(img_freq):
    return np.uint8(np.abs(ifft2(img_freq)))


def contrast(img, brightness, c):
    output = img * (c/127 + 1) - c + brightness
    output = np.clip(output, 0, 255)
    return np.uint8(output)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


@timeit
def conv(img, kernel):
    return cv2.filter2D(img, -1, kernel)


def hist_color(img, ranges=[0, 256]):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist(img, i, col, ranges, show=False)
    plt.show()


def hist(img, i=0, color='b', ranges=[0, 256], show=True):
    histr = cv2.calcHist([img], [i], None, [256], ranges)
    plt.plot(histr, color=color)
    plt.xlim([0, 256])
    if show:
        plt.show()


def kernel_fft(kernel, img):
    # Pad kernel to match image size
    return fft2(kernel, img.shape[:2])


kernels = {
    'sobel-x': np.array([[-1, 0, 1], ]),
    'sobel-y': np.array([[1], [0], [-1], ]),
    'laplace': np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ]),
    'low_pass': np.ones((9, 9)) / 81,
}


def cache_file(func):
    @wraps(func)
    def cache_file_wrapper(*args, **kwargs):
        filename = f'{func.__name__}.npy'
        try:
            return np.load(filename)
        except FileNotFoundError:
            result = func(*args, **kwargs)
            np.save(filename, result)
            return result
    return cache_file_wrapper


def canny(image, threshold1, threshold2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, threshold1, threshold2)

    return edges


def key(a, b):
    return str(tuple(quantize(a)) + tuple(quantize(b)))


def quantize(x, bits=3):
    return x + (1 << bits-1) - 1 >> bits << bits


def parse_array(arr_str):
    # 移除首尾的方括號，然後使用空格拆分字符串，再轉換為整數陣列
    arr = np.array(arr_str.strip('[]').split(), dtype=np.float32)
    return arr
