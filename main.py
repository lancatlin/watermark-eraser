import sys
import os
from matplotlib import pyplot as plt

from watermark import Watermark
from utils import display


if __name__ == "__main__":
    # Read filename from command arguments
    filename = sys.argv[1]
    skip = len(sys.argv) > 2
    if not filename or not os.path.isfile(filename):
        filename = "doc/watermark-text.jpg"

    watermark = Watermark(filename)
    if not skip:
        edge = watermark.find_watermark()
        watermark.save()
        display(watermark.mark_edge())
    else:
        watermark.load()

    result = watermark.remove_watermark()
    display(result)
    # Mask result on the original image
    plt.show()
