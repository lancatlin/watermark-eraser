import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from utils import display, canny, timeit, key, quantize, parse_array
from plot import plot_3d
from cluster import ColorCluster
from kuwahara import kuwahara_color


ALPHA = 0.5
BLUR_EDGE_SIZE = 5
VARIANCE_THRESHOLD = 100


def mask_color(mixed, original):
    # return (mixed - (1 - ALPHA) * original) / ALPHA
    return (mixed - original) / ALPHA + original


def is_watermark(mask):
    return np.var(mask) < VARIANCE_THRESHOLD


def mark_direction(edge):
    kernel = np.array([
        [0, 0, -1, 0, 0],
        [0, 0, -1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, -1, 0, 0],
        [0, 0, -1, 0, 0],
    ])
    # Convolute image as float32
    signed = cv2.filter2D(edge, -1, kernel)
    return edge * signed


class Watermark:
    def __init__(self, img):
        self.img = img
        self.df = pd.DataFrame(columns=['mixed', 'background', 'foreground', 'count'],)

        print(self.df)


        self.blurred = kuwahara_color(self.img, 5)
        display(self.blurred)

        self.edge = canny(self.img, 75, 100) / 255
        display(self.edge, colorbar=True)

        self.signed_edge = mark_direction(self.edge)
        display(self.signed_edge, colorbar=True)

        self.watermark_edge = None


    def is_watermark(self, i, j, bs):
        a = self.blurred[i+bs[0], j+bs[1]]
        b = self.blurred[i-bs[0], j-bs[1]]
        k = key(a, b)
        if k in self.df.index:
            self.df.loc[k, 'count'] += 1
            return True
        else:
            return self.is_watermark_impl(a, b)


    def is_watermark_impl(self, a, b):
        mask_1 = mask_color(a, b)
        mask_2 = mask_color(b, a)
        if is_watermark(mask_1):
            self.add_data(a, b, mask_1)
            return True
        if is_watermark(mask_2):
            self.add_data(b, a, mask_2)
            return True

        return False

    
    def add_data(self, mixed, background, foreground):
        k = key(mixed, background)
        self.df.loc[k] = {
            'mixed': mixed,
            'background': background,
            'foreground': foreground,
            'count': 1,
        }
            

    @timeit
    def find_watermark(self, alpha=0.5, bs=5):
        # The additional vector to the center of the block
        horizontal = np.array([0, 1])
        vertical = np.array([1, 0])

        output = np.zeros(self.img.shape[:2], np.bool_)

        nr, nc = self.img.shape[:2]

        for i in range(bs, nr - bs):
            for j in range(bs, nc - bs):
                if self.signed_edge[i, j] > 0:      # The edge is horizontal
                    output[i, j] = self.is_watermark(i, j, bs * vertical)
                elif self.signed_edge[i, j] < 0:    # The edge is vertical
                    output[i, j] = self.is_watermark(i, j, bs * horizontal)

        self.watermark_edge = output
        return output

    
    def remove_watermark(self):
        cluster = ColorCluster(self.df)
        cluster.cluster('mixed')
        plot_3d(self.df, cluster.clustering.labels_)
        mixed_colors, mixed_var = cluster.mean_color('mixed')
        bg_colors, bg_var = cluster.mean_color('background')
        print(bg_colors)
        result = self.img.copy()

        for i in range(len(mixed_colors)):
            # 計算範圍上下界
            bias = np.sqrt(mixed_var[i]) * 3
            mixed = mixed_colors[i]
            print(mixed_colors, bias)
            lower_bound = np.clip(mixed - bias, 0, 255)
            upper_bound = np.clip(mixed + bias, 0, 255)

            # 使用布林索引過濾圖片
            filtered_img = cv2.inRange(self.img, lower_bound, upper_bound)

            # Dialate the image to fill the holes
            kernel = np.ones((3, 3), np.uint8)
            filtered_img = cv2.dilate(filtered_img, kernel, iterations=1)


            mask = self.img.copy()
            mask[filtered_img == 0] = 0
            display(mask)

            G = bg_colors[i]

            # result = result * (1 - alpha_mask) + G * alpha_mask
            result[filtered_img > 0] = G
        return result.astype(np.uint8)


    def load(self, filename):
        self.df = pd.read_csv(filename, converters={'mixed': parse_array, 'backgrond': parse_array, 'foreground': parse_array})

    def save(self, filename):
        self.df.to_csv(filename)

if __name__ == "__main__":
    # Read filename from command arguments
    filename = sys.argv[1]
    skip = len(sys.argv) > 2
    if not filename or not os.path.isfile(filename):
        filename = "watermark.jpg"

    img = cv2.imread(filename, -1)
    display(img)

    watermark = Watermark(img)
    if not skip:
        edge = watermark.find_watermark()
        watermark.save('watermark.csv')
        result = img.copy()
        result[edge] = 0
        display(result)
    else:
        watermark.load('watermark.csv')

    # print(watermark.df)
    result = watermark.remove_watermark()
    display(result)
    # Mask result on the original image
    plt.show()
