import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(file, new_size=None, trim=None):
    img = Image.open(file).convert('L')

    if trim:
        width, height = img.size
        top_trim = int(trim.get('top', 0) * height)
        bottom_trim = int(trim.get('bottom', 0) * height)
        left_trim = int(trim.get('left', 0) * width)
        right_trim = int(trim.get('right', 0) * width)

        img = img.crop((left_trim, top_trim, width - right_trim, height - bottom_trim))

    if new_size:
        new_size = tuple(int(el) for el in new_size)
        img = img.resize(new_size)

    return np.array(img)


def show_img(arr):
    plt.figure()
    plt.imshow(arr)
    plt.show()
