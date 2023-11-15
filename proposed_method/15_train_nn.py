# %%
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Sequential
current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, "/".join(current_script_directory.split("/")[:-1]))
from images.utils import load_image
from scipy import spatial
sys.path.insert(2, "/".join(current_script_directory.split("/")[:-2]))
from ad_rise.utils import generate_masks


def similarity(true: list, pred: list):

    ## COSINE SIMILARITY
    cos_sim = 1 - spatial.distance.cosine(true, pred)
    return cos_sim

# Define your data location
data_loc = "../../colab_alisa"
# %%
# Load and preprocess your data
X = []
y = []

# Iterate through folders
classes = ['asthma', 'control']
for class_name in classes:
    class_dir = os.path.join(data_loc, class_name)
    if os.path.isdir(class_dir):
        class_label = int(classes.index(class_name))  # Assuming folder names are class labels
        for image_file in os.listdir(class_dir):
            if image_file.endswith(".tif"):
                image_path = os.path.join(class_dir, image_file)
                image = load_image(image_path, [224, 224], trim={"bottom": 0.08}).reshape(224,224,1)
                X.append(image)
                y.append(class_label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and build your neural network model using TensorFlow/Keras
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile your model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#%%
# Train your model
model.fit(
    X_train,
    tf.keras.utils.to_categorical(y_train, num_classes=2),
    epochs=12,
    validation_data=(X_val, tf.keras.utils.to_categorical(y_val, num_classes=2))
)
model.save("results/med_deep_learning/cnn.keras")

# %%
idx = 10

masks = generate_masks(
    N=1000, 
    s=20,
    p=0.1, 
    img=np.squeeze(image), 
    sampling_method='proportional level-sets'
)

expanded_image = np.expand_dims(X_val[idx], axis=0)  # Shape: (1, 224, 224, 1)
expanded_masks = np.expand_dims(masks, axis=3)  # Shape: (1000, 224, 224, 1)

# Element-wise multiplication between masks and image
masked_images = expanded_image * expanded_masks

# Predict on all masked images in one go
all_predictions = model.predict(masked_images)

# Calculate similarities for all predictions
sim_scores = [similarity(np.array([0,1]), pred_probs.reshape(2,)) for pred_probs in all_predictions]

# Calculate the saliency map using similarity scores and masks
saliency2 = np.tensordot(sim_scores, masks, axes=1).reshape(X_val[idx].shape[0], X_val[idx].shape[1])

plt.figure()
plt.imshow(X_val[idx], "gray")
plt.imshow(saliency2, cmap='jet', alpha=0.5)
plt.show()
# %%
