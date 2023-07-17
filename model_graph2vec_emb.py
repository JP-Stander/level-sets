#%%

import numpy as np
import pandas as pd
# %%

embeddings = pd.read_csv("../my_graphical_dataset/features/features.csv")
trackings = pd.read_csv("../my_graphical_dataset/tracking.csv")

embeddings = pd.merge(embeddings, trackings[["V1","V2"]].rename(columns={"V1": "type", "V2": "label"}), on="type")
embeddings["label"] = embeddings["label"].apply(lambda x: x.split("/")[0])
# %%
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X = embeddings.drop(columns=['type', 'label'])
y = embeddings['label']

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a classifier and fit it to our data
clf = SVC(kernel='linear') 
clf.fit(X_train, y_train)

# predict the labels for the test set
y_pred = clf.predict(X_test)

# print the accuracy of the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))



# %%
