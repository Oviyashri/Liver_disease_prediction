# Importing Libraries:
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("data/indian_liver_patient.csv")

# Filling NaN Values of "Albumin_and_Globulin_Ratio" feature with Median:
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median())


# Label Encoding:
df['Gender'] = np.where(df['Gender']=='Male', 1,0)

# Droping 'Direct_Bilirubin' feature:
df = df.drop('Direct_Bilirubin', axis=1)

# Independent and Dependent Feature:
X = df.drop('Dataset',axis=1)
y = df["Dataset"]

# SMOTE Technique:
from imblearn.combine import SMOTETomek
smote = SMOTETomek()
X_smote, y_smote = smote.fit_resample(X,y)

# Train Test Split:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_smote,y_smote, test_size=0.3, random_state=33)


# RandomForestClassifier:
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train,y_train)

# Creating a pickle file for the classifier
filename = 'Liver2.pkl'
pickle.dump(RandomForest, open(filename, 'wb'))