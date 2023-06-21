import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(file, new_size=None):

    img = Image.open(file).convert('L')
    if new_size:
        new_size = tuple(int(el) for el in new_size)
        img = img.resize(new_size)

    return np.array(img)


def show_img(arr):
    plt.figure()
    plt.imshow(arr)
    plt.show()
