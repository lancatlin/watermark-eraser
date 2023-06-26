import sys
import os
from matplotlib import pyplot as plt
import argparse

from watermark import Watermark
from utils import display


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="Image file to process")
parser.add_argument(
    "-a", "--alpha", help="Alpha value for watermark", type=float, default=0.5
)
parser.add_argument(
    "-s",
    "--skip",
    help="Skip watermark detection",
    default=False,
    action="store_true",
)

args = parser.parse_args()
# Read filename from command arguments

watermark = Watermark(args.filename, args.alpha)
if not args.skip:
    edge = watermark.find_watermark()
    watermark.save()
    display(watermark.mark_edge())
else:
    watermark.load()

result = watermark.remove_watermark()
display(result)
# Mask result on the original image
plt.show()
