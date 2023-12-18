# atificial neural network
import numpy as npy
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
x[:, 1] = labelencoder_X_1.fit_transform(x[:, 1]).astype(float)

labelencoder_X_2 = LabelEncoder()
x[:, 2] = labelencoder_X_2.fit_transform(x[:, 2]).astype(float)

categorical_features = [1]

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'  
)
x = preprocessor.fit_transform(x).astype(float)

x=x[:,1:] 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test=sc.transform(x_test)
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=10,epochs=100)
y_pred=classifier.predict(x_test)
y_pred =(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

