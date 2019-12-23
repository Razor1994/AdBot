from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy
import os
import numpy as np

emotion_model_path = 'models/_mini_XCEPTION.89-0.65.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)

truey=[]
predy=[]
x = np.load('./modXtest.npy')
y = np.load('./modytest.npy')

yhat= emotion_classifier.predict(x)
yh = yhat.tolist()
yt = y.tolist()
count = 0

for i in range(len(y)):
    yy = max(yh[i])
    yyt = max(yt[i])
    predy.append(yh[i].index(yy))
    truey.append(yt[i].index(yyt))
    if(yh[i].index(yy)== yt[i].index(yyt)):
        count+=1

acc = (count/len(y))*100

#saving values for confusion matrix and analysis
np.save('truey', truey)
np.save('predy', predy)
print("Predicted and true label values saved")
print("Accuracy on test set :"+str(acc)+"%")
