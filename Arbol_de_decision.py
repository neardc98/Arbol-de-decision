from cProfile import label
from json import encoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#----------------------------------------------------------------------------
import os
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from itertools import cycle

plt.style.use("ggplot")
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
#-----------------------------------------------------------------------------

sns.set()
train_df = pd.read_csv("titanic-train.csv")
test_df  = pd.read_csv("titanic-test.csv")
# print(train_df.info())
# print(train_df.head)

train_df.Sex.value_counts().plot(kind="bar", color=['b', 'r'])
plt.title("Distribucion de sobrevivientes")
# plt.show()

label_encoder = preprocessing.LabelEncoder()
encoder_sex = label_encoder.fit_transform(train_df['Sex'])
#print(encoder_sex)

train_df['Age']= train_df['Age'].fillna(train_df['Age'].median)

train_df['Embarked'] = train_df['Embarked'].fillna('S')



#limpiar / dropear 
useful_features = ["Sex", "Age", "Pclass"]
train_df[useful_features].head()

y = train_df.Survived  # Target 1=Survived, 0=Did Not Survive
test = test_df.drop(columns=["PassengerId"], axis=1).copy()
X = train_df.drop(columns=["PassengerId", "Survived"], axis=1).copy()
X = train_df[useful_features]
X.info()

print(
    "Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):"
)
print(train_df["Embarked"].value_counts())
sns.countplot(x="Embarked", data=train_df, palette="Set2")
plt.show()

sns.barplot(
    x="Embarked",
    y="Survived",
    data=train_df,
)
plt.show()

ax = sns.scatterplot(
    x="Sex",
    y="Survived",
    hue="Pclass",
    data=train_df,
)
ax.set_title("Coaster Speed vs. Height")
plt.show()


X_train = X[useful_features]
X_test = test[useful_features].copy()

y_train = y
print(X_train.head())
