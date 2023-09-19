# %%
import os
import cv2
import numpy as np
from images.utils import load_image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
# %%

def image_to_histogram(descriptors, kmeans):
    hist = np.zeros(num_clusters)
    labels = kmeans.predict(descriptors)
    for label in labels:
        hist[label] += 1
    return hist

def process_sublist(sublist_descriptors, kmeans):
    # Convert each descriptor to histogram
    histograms = [image_to_histogram(desc, kmeans) for desc in sublist_descriptors]

    return histograms

# %%

folders = ["dotted", "fibrous"]#os.listdir("../dtd/images")[:10]
graph_files = [f"../dtd/images/{folder}/{dir}" for folder in folders for dir in os.listdir(f"../dtd/images/{folder}")]

images = [[] for _ in folders]

for i, folder in enumerate(folders):
    images[i] = [
        load_image(os.path.join(f"../dtd/images/{folder}", image), [50,50])
        for image in os.listdir(f"../dtd/images/{folder}")
    ]

# %%
b_acc = 0
num_clusters = 100
sift = cv2.xfeatures2d.SIFT_create(n_points)

kps = [[] for _ in folders]
des = [[] for _ in folders]

for i, sublist in enumerate(images):
    for img in sublist:
        kp, de = sift.detectAndCompute(img, None)
        kps[i] += [np.array([key_point.pt for key_point in kp]).reshape(-1, 2)]
        if de is not None:
            des[i] += [de]



flattened_des = [arr for sublist in des for arr in sublist]
all_descriptors = np.vstack(flattened_des)
# num_clusters = 100

# Step 1: Create a visual vocabulary
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)

# Step 2: Represent each image as a histogram of visual words
lists_of_full = [
    process_sublist(
        sublist_descriptors,
        kmeans
    ) for sublist_descriptors in des
]

X = []
y = []

for class_index, sublist in enumerate(lists_of_full):
    X.extend(sublist)
    y.extend([class_index] * len(sublist))

# Standardize features
# scaler = StandardScaler().fit(X)
# X = scaler.transform(X)

# Step 3: Train a classifier
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accs ={
    "Logistic Regression": [],
    "Random Forest": [],
    "XGBoost": [],
    "SVM": [],
    "KNN": []
}
for train_index, test_index in kf.split(range(len(X))):
    X_train = [X[i] for i in train_index]
    X_test = [X[i] for i in test_index]
    y_train = [y[i] for i in train_index]
    y_test = [y[i] for i in test_index]
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": xgb.XGBClassifier(objective="binary:logistic", random_state=42),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = clf.score(X_test, y_test)
        accs[name] += [accuracy]
acc = np.max([np.mean(accs[name]) for name in accs.keys()])
    # if acc > b_acc:
    #     b_acc=acc
    #     b_n = num_clusters
    #     print(f"Best accuracy of {b_acc} with n {b_n}")

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load and resize the image
img = cv2.imread('../dtd/images/dotted/dotted_0038.jpg')
original_shape = img.shape[:2]
img = cv2.resize(img, (50, 50))
resize_shape = img.shape[:2]

# Load the original image for plotting
img2 = cv2.imread('../dtd/images/dotted/dotted_0038.jpg')

# Detect keypoints
sift = cv2.xfeatures2d.SIFT_create(35)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp = sift.detect(gray, None)

# Calculate the resizing factor
resize_factor_x = original_shape[1] / resize_shape[1]  # width
resize_factor_y = original_shape[0] / resize_shape[0]  # height

fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 'gray')  # Display image in RGB format

# Use a color palette for bright colors
colors = sns.color_palette("husl", len(kp))

# Plot adjusted keypoints on top of the original image
for i, k in enumerate(kp):
    adjusted_x = k.pt[0] * resize_factor_x
    adjusted_y = k.pt[1] * resize_factor_y
    adjusted_size = k.size * (resize_factor_x + resize_factor_y) / 2  # average resizing factor
    circle = plt.Circle((adjusted_x, adjusted_y), adjusted_size/2, color=colors[i], fill=False, linewidth=3)
    ax.add_patch(circle)

plt.axis('off')
plt.savefig("../paper3_results/dotted_0038_sift_keypoints.png")


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load and resize the image
img = cv2.imread('../dtd/images/fibrous/fibrous_0121.jpg')
original_shape = img.shape[:2]
img = cv2.resize(img, (50, 50))
resize_shape = img.shape[:2]

# Load the original image for plotting
img2 = cv2.imread('../dtd/images/fibrous/fibrous_0121.jpg')

# Detect keypoints
sift = cv2.xfeatures2d.SIFT_create(35)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp = sift.detect(gray, None)

# Calculate the resizing factor
resize_factor_x = original_shape[1] / resize_shape[1]  # width
resize_factor_y = original_shape[0] / resize_shape[0]  # height

fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 'gray')  # Display image in RGB format

# Use a color palette for bright colors
colors = sns.color_palette("husl", len(kp))

# Plot adjusted keypoints on top of the original image
for i, k in enumerate(kp):
    adjusted_x = k.pt[0] * resize_factor_x
    adjusted_y = k.pt[1] * resize_factor_y
    adjusted_size = k.size * (resize_factor_x + resize_factor_y) / 2  # average resizing factor
    circle = plt.Circle((adjusted_x, adjusted_y), adjusted_size/2, color=colors[i], fill=False, linewidth=3)
    ax.add_patch(circle)

plt.axis('off')
plt.savefig("../paper3_results/fibrous_0121_sift_keypoints.png")

# %%
