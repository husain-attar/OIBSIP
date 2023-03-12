import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('Iris.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df.info)
print(df.isnull().sum())
print(df.Species.value_counts)
X= df[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y= df['Species']
print(X,y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X,y)
lr.fit(X_train,y_train)
predictions = lr.predict(X)
Scores = pd.DataFrame({'Actual':y,'Predictions':predictions})
print(Scores.head())
y_test_hat=lr.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_test_hat)*100,'%')