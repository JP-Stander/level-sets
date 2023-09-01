# %%
from level_sets.utils import get_level_sets
from images.utils import load_image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def calculate_entropy(image):
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    
    # Normalize the histogram to get probabilities
    hist = hist.ravel() / hist.sum()
    
    # Compute the entropy
    entropy = -np.sum(hist[np.nonzero(hist)] * np.log2(hist[np.nonzero(hist)]))
    return entropy
# %%
img_size = 200
img1 = load_image(
    "../dtd/images/dotted/dotted_0161.jpg",
    [img_size, img_size]
)

img2 = load_image(
    "../dtd/images/dotted/dotted_0184.jpg",
    [img_size, img_size]
)

img3 = load_image(
    "../dtd/images/fibrous/fibrous_0165.jpg",
    [img_size, img_size]
)

img4 = load_image(
    "../dtd/images/fibrous/fibrous_0116.jpg",
    [img_size, img_size]
)

# %%

plt.figure()
plt.imshow(img1, 'gray')
plt.xticks([])
plt.yticks([])
plt.savefig("../paper_results/ls_v_entropy/dotted1.png")

ls1 = get_level_sets(img1)
print(f"Entropy: {calculate_entropy(img1)}")
print(f"# Level-sets: {ls1.max()}")

plt.figure()
plt.imshow(img2, 'gray')
plt.xticks([])
plt.yticks([])
plt.savefig("../paper_results/ls_v_entropy/dotted2.png")

ls2 = get_level_sets(img2)
print(f"Entropy: {calculate_entropy(img2)}")
print(f"# Level-sets: {ls2.max()}")

plt.figure()
plt.imshow(img3, 'gray')
plt.xticks([])
plt.yticks([])
plt.savefig("../paper_results/ls_v_entropy/fibrous1.png")

ls3 = get_level_sets(img3)
print(f"Entropy: {calculate_entropy(img3)}")
print(f"# Level-sets: {ls3.max()}")

plt.figure()
plt.imshow(img4, 'gray')
plt.xticks([])
plt.yticks([])
plt.savefig("../paper_results/ls_v_entropy/fibrous2.png")

ls4 = get_level_sets(img4)
print(f"Entropy: {calculate_entropy(img4)}")
print(f"# Level-sets: {ls4.max()}")
# %%
