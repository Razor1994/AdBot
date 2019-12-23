import pandas as pd, cv2, csv, os, numpy as np
from sklearn.model_selection import train_test_split
dataset_1_path = 'testImages_abstract'
dataset_2_path = 'testImages_artphoto'
directory = os.path.join('c:\\', 'Users\\PranayDev\\Documents\\ComputerVision\\Project\\ImageEmotion\\agg')
image_root_directory = os.path.join('c:\\', 'Users\\PranayDev\\Documents\\ComputerVision\\Project\\ImageEmotion\\agg\\images')
directories = {'amusement':image_root_directory + '\\' + 'amusement', 
 'anger':image_root_directory + '\\' + 'anger', 
 'awe':image_root_directory + '\\' + 'awe', 
 'contentment':image_root_directory + '\\' + 'contentment', 
 'disgust':image_root_directory + '\\' + 'disgust', 
 'excitement':image_root_directory + '\\' + 'excitement', 
 'fear':image_root_directory + '\\' + 'fear', 
 'sadness':image_root_directory + '\\' + 'sadness'}
image_size = (112, 112)

def load_abstract_images():
    img_paths = []
    emotions = []
    images = []
    with open(('ImageEmotion/' + dataset_1_path + '/' + 'ABSTRACT_groundTruth.csv'), newline='') as (csvfile):
        dataset = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(dataset)
        for row in dataset:
            img_paths.append(row[0])
            emotions.append(list(map(int, row[1:])))

    for path in img_paths:
        path = path[1:len(path) - 1]
        img = cv2.imread('ImageEmotion/' + dataset_1_path + '/' + path)
        img = cv2.resize(img, image_size)
        images.append(img)

    return (
     np.array(images), np.array(emotions))


def load_rochester_data():
    images = []
    emotions = []
    for emotion_folder in directories:
        folder_path = directories[emotion_folder]
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                img = cv2.imread(folder_path + '\\' + file)
                img = cv2.resize(img, image_size)
                img = img / 255.0
                img = img - 0.5
                img = img * 2.0
                images.append(img)
                emotion = [0] * 8
                emotion[list(directories.keys()).index(emotion_folder)] = 1
                emotions.append(emotion)

    return (
     images, emotions)


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def get_train_valid_data():
    images, emotions = load_rochester_data()
    x_train, x_test, y_train, y_test = train_test_split(images, emotions, test_size=0.1, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=41)
    np.save('modXtest', x_test)
    np.save('modytest', y_test)
    return (
     x_train, y_train, x_valid, y_valid)


if __name__ == '__main__':
    get_train_valid_data()