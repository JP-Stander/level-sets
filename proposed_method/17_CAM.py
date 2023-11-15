#%%
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(2, "/".join(current_script_directory.split("/")[:-2]))
sys.path.insert(1, "/".join(current_script_directory.split("/")[:-1]))
from images.utils import load_image

# Load your trained Keras model
model = load_model("results/med_deep_learning/cnn.keras")

# Choose the last convolutional layer and output layer
last_conv_layer = model.get_layer("conv2d_1")
output_layer = model.get_layer("dense_3")

# Create a new model that outputs the feature maps of the last conv layer
cam_model = Model(inputs=model.input, outputs=[last_conv_layer.output, output_layer.output])

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
image_name = images_for_inference[class_name][0]
image_path = f"../../colab_alisa/{class_name}/{image_name}"
x = load_image(image_path, [224, 224], trim={"bottom": 0.08}).reshape(224,224,1)
img = x.copy()
# %%
classes = ['asthma', 'control']
# Generate CAM
last_conv_outputs, predictions = cam_model(np.expand_dims(x, axis=0))
class_idx = np.argmax(predictions[0])
cam = last_conv_outputs[0, :, :, class_idx]

# Apply visualization and overlay CAM on the input image
heatmap = cv2.resize(cam.numpy(), (img.shape[1], img.shape[0]))
heatmap = np.uint8(heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# Display the original image, CAM, and superimposed image

plt.imshow(img)
plt.title("Original Image")
plt.show()

plt.imshow(heatmap)
plt.title("Class Activation Map (CAM)")
plt.show()

# plt.imshow(superimposed_img)
# plt.title("Superimposed Image with CAM")
# plt.show()

# %%
