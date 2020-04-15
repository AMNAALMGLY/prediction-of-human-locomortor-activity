# -*- coding: utf-8 -*-
"""
Created on Fri May  4 12:21:33 2018

@author: Rawan
"""

#-----------------------------------

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#------------------------------------

path_1 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Wavelet_sym5_Features_Window_1500_Slide_1.pkl'
path_2 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Time_Features_Window_1500_Slide_1.pkl'
path_3 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Frequency_Features_Window_1500_Slide_1.pkl'
path_4 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\IMU_Features_Window_1500_Slide_1.pkl'

scaler = StandardScaler()
#
features = pd.read_pickle(path_3)
x=features.iloc[:,:features.shape[1]-1]
y=pd.get_dummies(features.Mode)


#df = pd.read_csv('C:\\Users\\D\\Downloads\\titanic.csv',usecols=('pclass','age','sibsp','parch','fare','survived'))
#df =df.dropna()
#x=df[['pclass','age','sibsp','parch','fare']]
#y=pd.get_dummies(df.survived.astype('category'))


#x = pd.DataFrame(load_iris().data)
#y = pd.get_dummies(pd.DataFrame(load_iris().target,dtype='category'))

x = pd.DataFrame(scaler.fit_transform(x))


x , x_test, y, y_test = train_test_split(x, y, test_size=0.2 )#, random_state = 123)
x_val , x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)#, random_state = 123)

NN_model = Sequential()
NN_model.add(Dense(100, activation='relu' , input_dim=x.shape[1]))
NN_model.add(Dense(y.shape[1], activation='softmax'))
np.random.seed(87)
NN_model.compile(optimizer='adam',metrics=['accuracy'], loss= 'categorical_crossentropy')
history = NN_model.fit(x=x , y=y , epochs=1500, validation_data=[x_val,y_val], shuffle=True ,callbacks= [EarlyStopping(monitor='val_loss', min_delta=0.001,patience=5)])

y_pred = NN_model.predict_classes(x_test)
print(metrics.accuracy_score(y_test,pd.get_dummies(y_pred)))

plt.figure(figsize=(8, 7))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylim(0,1)
plt.title('Model Accuracy with Early Stopping')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

plt.figure(figsize=(8, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0,2)
plt.title('Model Loss with Early Stopping')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')
plt.show()