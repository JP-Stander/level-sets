#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
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
#%%
# Load your trained Keras model
model = load_model("results/med_deep_learning/cnn.keras")

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
#%%
# Create a gradient tape to compute gradients
# Convert the input image to a TensorFlow tensor
inputs = tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float32)

# Create a gradient tape to compute gradients
with tf.GradientTape(persistent=True) as tape:
    tape.watch(inputs)  # Ensure that the inputs are watched by the tape
    predictions = model(inputs)
    top_prediction = tf.argmax(predictions[0])

# Calculate gradients with respect to the input image
grads = tape.gradient(predictions[:, top_prediction], inputs)

# Calculate the Saliency Map
saliency_map = tf.reduce_max(tf.abs(grads), axis=-1)[0]

# Normalize the Saliency Map
saliency_map /= tf.reduce_max(saliency_map)
#%%
# Display the original image and Saliency Map
plt.subplot(1, 2, 1)
plt.imshow(img[0])
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(saliency_map, cmap="viridis")
plt.title("Saliency Map")
plt.show()
