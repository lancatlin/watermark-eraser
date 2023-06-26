import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import display, canny
from watermark import mask_color, mark_direction, quantize


def test_points(img, points):
    for name, p in points.items():
        mixed = img[p[0, 1], p[0, 0]]
        original = img[p[1, 1], p[1, 0]]
        print(f"{name}: mixed: {mixed}, original: {original}")

        # Iterate alpha from 0.2 to 0.8, step by 0.2
        for alpha in np.arange(0.2, 0.8, 0.1):
            mask = mask_color(mixed, original, alpha=alpha).astype(np.uint8)
            print(
                f'alpha: {format(alpha, ".2f")}, mask: {mask}, std: {format(np.std(mask), ".2f")}'
            )

        print()


def prepare_test_image(filename):
    img = cv2.imread(filename, -1)
    blurred = cv2.blur(img, (5, 5))
    return blurred


def test_fruits():
    img = prepare_test_image("doc/fruits.jpg")
    # Calculate specific points of mask of observation
    points = {
        "pear": np.array([[110, 690], [100, 670]]),
        "apple": np.array([[880, 298], [880, 275]]),
        "grape": np.array([[778, 700], [760, 700]]),
        "peach": np.array([[513, 700], [514, 710]]),
        "banana": np.array([[395, 495], [407, 489]]),
        "background": np.array([[395, 88], [395, 60]]),
    }
    test_points(img, points)
    display(img)


def test_watermark():
    img = prepare_test_image("doc/watermark.jpg")
    points = {"a": np.array([[70, 58], [86, 38]])}
    test_points(img, points)
    display(img)


def test_watermark2():
    img = prepare_test_image("doc/watermark2.jpg")
    points = {
        "a": np.array([[70, 58], [86, 38]]),
        "should not vertical": np.array([[105, 34], [105, 17]]),
        "should not horizontal": np.array([[56, 57], [22, 90]]),
    }
    test_points(img, points)
    display(img)


def test_mark_directions():
    img = cv2.imread("doc/watermark2.jpg", -1)
    edge = canny(img, 75, 100) / 255
    display(edge, colorbar=True)

    signed_edge = mark_direction(edge)
    display(signed_edge, colorbar=True)


def test_quantize():
    nums = np.arange(0, 255)
    print(quantize(nums, 3))


if __name__ == "__main__":
    test_fruits()
    test_watermark2()
    test_mark_directions()
    test_quantize()
    plt.show()
