# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 12:59:05 2025

@author: user
@studentID: 5885221 & 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_recall_curve, PrecisionRecallDisplay, roc_auc_score, RocCurveDisplay, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from math import isclose
np.random.seed(5885221)


""" Variables """
Test_ratio = 0.2
Random_state = 5885221
""" Functions """
def create_image(flatImage,imageSize=28):
    image = np.zeros((imageSize,imageSize))
    for i in range(imageSize):
        image[i] = flatImage[(i*imageSize):((i+1)*imageSize)]
    return image

""" Main Code """

""" Initialize Dataset """
# Read dataset and create pandas dataFrame
data = pd.read_csv('Project_Data_EE4C12_CE_IMG.csv', header=None)
dataSet = pd.DataFrame(data)

# Use pandas to seperate into images (X) and labels (Y):
Y_pd = dataSet[dataSet.columns[0]]
X_pd = dataSet.drop(dataSet.columns[0], axis=1)

# Create numpy arrays instead of using pandas
Y = Y_pd.to_numpy()
X = X_pd.to_numpy()

# Create test image of the dataset:
testImage = create_image(X[0])

# Normalize data:
X_scaled = StandardScaler().fit_transform(X)

# Show normalization on first 10 features:
print("mean of X", np.nanmean(X[:10,:],axis=1))
print("mean of X after z-score normalization", np.nanmean(X_scaled[:10,:],axis=1))

# Make a train-test split:
X_train, X_test, y_train, y_test = train_test_split(X_scaled,Y,test_size=Test_ratio,random_state=Random_state)

""" Model 1 """
""" Model 2 """
""" Model 3 """
""" Model 4 """

""" Plotting """
# Plot the image
plt.imshow(testImage, cmap='gray')  # 'gray' ensures it's displayed in grayscale
plt.title("Test Image")
plt.axis('off')  # Hide axes for clarity
plt.show()

# Plot the amount of samples for each label:
unique, counts = np.unique(Y, return_counts=True)
bar_width = 0.7  # smaller value = more space between bars
plt.bar(unique, counts, width=bar_width, color='skyblue', edgecolor='black')
plt.title("Distribution of Y")
plt.xlabel("Value")
plt.ylabel("Count")
plt.xticks(unique)  # Show each integer on the x-axis
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig("data_histogram.png")
plt.show()

