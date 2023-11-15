# %%
import keras
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from scipy import spatial
current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(2, "/".join(current_script_directory.split("/")[:-2]))
sys.path.insert(1, "/".join(current_script_directory.split("/")[:-1]))
from images.utils import load_image
from ad_rise.utils import generate_masks
from scipy.stats import entropy
import pysal
from pysal.lib import weights
from pysal.explore import esda
from tqdm import tqdm

def similarity(true: list, pred: list):

    ## COSINE SIMILARITY
    cos_sim = 1 - spatial.distance.cosine(true, pred)
    return cos_sim

#%%
model = keras.models.load_model("results/med_deep_learning/cnn.keras")
classes = ['asthma', 'control']
images_for_inference = {
    "asthma": [
        "A5_PRP+T_40X_05.tif",
        "A4_PRP+T_40X_03.tif",
        "A15_PRP+T_10X_18.tif",
        "A29_PRP+T_40X_08.tif"
    ],
    "control": [
        "C1509_PPP_T_30K_05.tif",
        "Conradie_PPP_T_20K_02.tif",
        "C1509_PPP_T_45K_16.tif",
        "van Zyl_PPP_T_20K_03.tif",
        "resia_PPP_T_20K_05.tif"
    ]
}
class_name = "asthma"
image_name = images_for_inference[class_name][1]
image_path = f"../../colab_alisa/{class_name}/{image_name}"
image = load_image(image_path, [224, 224], trim={"bottom": 0.08}).reshape(224,224,1)

#%%
masks = generate_masks( #s=30,p=0.05
    N=5000, 
    s=30,
    p=0.05, 
    img=np.squeeze(image), 
    # sampling_method='random'
    sampling_method='proportional level-sets'
)

y_true = np.array([0,0])
y_true[classes.index(class_name)] = 1

expanded_image = np.expand_dims(image, axis=0)  # Shape: (1, 224, 224, 1)
expanded_masks = np.expand_dims(masks, axis=3)  # Shape: (1000, 224, 224, 1)

# Element-wise multiplication between masks and image
masked_images = expanded_image * expanded_masks

# Predict on all masked images in one go
all_predictions = model.predict(masked_images)

# Calculate similarities for all predictions
sim_scores = [similarity(np.array([0,1]), pred_probs.reshape(2,)) for pred_probs in all_predictions]

# Calculate the saliency map using similarity scores and masks
saliency = np.tensordot(sim_scores, masks, axes=1).reshape(image.shape[0], image.shape[1])
# saliency = np.zeros((224,224))
# spatial_variantion = []
# for mask, sim_score in tqdm(zip(masks, sim_scores)):
#     saliency += sim_score * mask
#     # w = weights.util.lat2W(224, 224)
#     # mi = esda.Moran(saliency.flatten(), w)
#     # spatial_variantion.append(mi.I)


#%%
plt.figure()
plt.imshow(image, "gray")
plt.imshow(saliency, cmap='jet', alpha=0.5)
plt.show()
# %%
plt.figure()
plt.plot(spatial_variantion)
plt.show()
# %%
