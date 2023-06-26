import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from utils import display, canny, timeit, key, quantize, parse_array
from plot import plot_3d
from cluster import ColorCluster
from kuwahara import kuwahara_color, BASENAME


BLUR_EDGE_SIZE = 5
VARIANCE_THRESHOLD = 100


def is_white(mask):
    return np.var(mask) < VARIANCE_THRESHOLD


def mark_direction(edge):
    """Mark the direction of the edge."""
    kernel = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, -1, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, -1, 0, 0],
            [0, 0, -1, 0, 0],
        ]
    )
    # Convolute image as float32
    signed = cv2.filter2D(edge, -1, kernel)
    return edge * signed


class Watermark:
    def __init__(self, filepath, alpha=0.5):
        self.alpha = alpha
        self.filepath = filepath
        self.basename = os.path.basename(filepath)
        self.img = cv2.imread(filepath)
        if self.img is None:
            raise Exception("Cannot read image file.")

        self.df = pd.DataFrame(
            columns=["mixed", "background", "foreground", "count"],
        )

        self.blurred = kuwahara_color(self.img, 5, basename=self.basename)
        display(self.blurred)

        self.edge = canny(self.img, 75, 100) / 255
        display(self.edge, colorbar=True)

        self.signed_edge = mark_direction(self.edge)
        display(self.signed_edge, colorbar=True)

        self.watermark_edge = None

    @timeit
    def find_watermark(self, bs=5):
        # The additional vector to the center of the block
        horizontal = np.array([0, 1])
        vertical = np.array([1, 0])

        output = np.zeros(self.img.shape[:2], np.bool_)

        nr, nc = self.img.shape[:2]

        for i in range(bs, nr - bs):
            for j in range(bs, nc - bs):
                if self.signed_edge[i, j] > 0:  # The edge is horizontal
                    output[i, j] = self.is_watermark(i, j, bs * vertical)
                elif self.signed_edge[i, j] < 0:  # The edge is vertical
                    output[i, j] = self.is_watermark(i, j, bs * horizontal)

        self.watermark_edge = output
        return output

    def remove_watermark(self):
        cluster = ColorCluster(self.df)
        cluster.cluster("mixed")
        plot_3d(self.df, cluster.clustering.labels_)

        mixed_colors, mixed_var = cluster.mean("mixed")
        bg_colors, bg_var = cluster.mean("background")
        result = self.img.copy()

        mask_sum = np.zeros(self.img.shape, np.uint8)

        for i in range(len(mixed_colors)):
            # 計算範圍上下界
            bias = np.sqrt(mixed_var[i]) * 3
            mixed = mixed_colors[i]

            lower_bound = np.clip(mixed - bias, 0, 255)
            upper_bound = np.clip(mixed + bias, 0, 255)

            filtered_img = cv2.inRange(self.img, lower_bound, upper_bound)

            # Dialate the filtered image
            kernel = np.ones((3, 3), np.uint8)
            filtered_img = cv2.dilate(filtered_img, kernel, iterations=1)

            mask = self.img.copy()
            mask[filtered_img == 0] = 0
            mask_sum = mask_sum | mask

            G = bg_colors[i]

            result[filtered_img > 0] = G

        display(mask_sum)

        return result.astype(np.uint8)

    def is_watermark(self, i, j, bs):
        """Check if the block is a watermark block.
        If the colors already exist in the dataframe, return True and count + 1.
        If not, call is_watermark_impl to check if it is a watermark block.
        """
        a = self.blurred[i + bs[0], j + bs[1]]
        b = self.blurred[i - bs[0], j - bs[1]]
        k = key(a, b)
        if k in self.df.index:
            self.df.loc[k, "count"] += 1
            return True
        else:
            return self.is_watermark_impl(a, b)

    def is_watermark_impl(self, a, b):
        mask_1 = self.mask_color(a, b)
        mask_2 = self.mask_color(b, a)
        if is_white(mask_1):
            self.add_data(a, b, mask_1)
            return True
        if is_white(mask_2):
            self.add_data(b, a, mask_2)
            return True

        return False

    def mask_color(self, mixed, original):
        """Calculate the color of the mask."""
        return (mixed - original) / self.alpha + original

    def add_data(self, mixed, background, foreground):
        k = key(mixed, background)
        self.df.loc[k] = {
            "mixed": mixed,
            "background": background,
            "foreground": foreground,
            "count": 1,
        }

    def mark_edge(self):
        """Mark the watermark edge on the image."""
        result = self.img.copy()
        result[self.watermark_edge] = 0
        return result

    def load(self):
        self.df = pd.read_csv(
            f"{self.basename}.csv",
            converters={
                "mixed": parse_array,
                "backgrond": parse_array,
                "foreground": parse_array,
            },
        )

    def save(self):
        self.df.to_csv(f"{self.basename}.csv")
