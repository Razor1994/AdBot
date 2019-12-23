import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
 
dataset_path = 'fer2013/fer2013.csv'
image_size=(48,48)
 
def load_fer2013():
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions
 
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def get_train_valid_data():
    faces, emotions = load_fer2013()
    faces = preprocess_input(faces)
    x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=41)

    #saving the test samples to be used later
    np.save('modXtest', x_test)
    np.save('modytest', y_test)

    return x_train, y_train, x_valid, y_valid

if __name__ == "__main__":
   get_train_valid_data()