import cv2
import dlib
from keras.preprocessing.image import img_to_array
import imutils
from keras.models import load_model
import numpy as np
import os
import random
import time

# ADs data path
image_root_directory = os.path.join('c:\\', 'Users\\PranayDev\\Documents\\ComputerVision\\Project\\ImageEmotion\\agg\\images')
directories = {
#  'amusement':image_root_directory + '\\' + 'amusement', 
 'anger':image_root_directory + '\\' + 'anger', 
 'fear':image_root_directory + '\\' + 'fear',
#  'awe':image_root_directory + '\\' + 'awe', 
 'contentment':image_root_directory + '\\' + 'contentment', 
 'sadness':image_root_directory + "\\" + '',
 'excitement':image_root_directory + '\\' + 'excitement' 
 }

image_emotion_map = {
     'anger': 0,
     'fear': 2,
     'contentment': 3,
     'sadness': 4,
     'excitement': 3
 }

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml.txt'
emotion_model_path = 'models/_mini_XCEPTION.89-0.65.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

ad_image = None

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_ad_image():
    image_size = (112, 112)
    target_emotion, folder_path = random.choice(list(directories.items()))
    print(folder_path)
    for root, dirs, files in os.walk(folder_path):
        image_index = random.randint(0, len(files))
        image_file = files[image_index]
        image = cv2.imread(folder_path + '\\' + image_file)
        img = cv2.resize(image, image_size)
        img = img / 255.0
        img = img - 0.5
        img = img * 2.0
        emotion = [0] * 6
        emotion[list(image_emotion_map.keys()).index(target_emotion)] = 1
    return image, emotion, target_emotion

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

def is_looking_in_frame(gray):
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

        if gaze_ratio <= 0.7 or gaze_ratio >= 1.7:
            return False
        else:
            return True

def get_facial_emotions(gray, frame):
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
 
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
            (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 2)
            cv2.putText(frameClone, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                        (0, 0, 255), 2)

        return label, preds, canvas

def should_change_ad(facial_emotions, ad_emotions):
    # facial_emotion_vector = np.zeros(8)
    # ad_emotions_vector = np.array(ad_emotions)
    # facial_emotion_vector[0] = facial_emotions[3]
    # facial_emotion_vector[1] = facial_emotions[0]
    # facial_emotion_vector[2] = facial_emotions[3]
    # facial_emotion_vector[3] = facial_emotions[3]
    # facial_emotion_vector[4] = facial_emotions[1]
    # facial_emotion_vector[5] = facial_emotions[5]
    # facial_emotion_vector[6] = facial_emotions[2]
    # facial_emotion_vector[7] = facial_emotions[4]
    # cos_sim = np.dot(facial_emotion_vector, ad_emotions_vector)/(np.linalg.norm(facial_emotion_vector)*np.linalg.norm(ad_emotions_vector))

    facial_emotion_vector = np.array(facial_emotions)[0:6]
    # facial_emotion_vector = facial_emotion_vector/np.linalg.norm(facial_emotion_vector)
    ad_emotions_vector = np.array(ad_emotions)
    # ad_emotions_vector[0] = ad_emotions[1]
    # ad_emotions_vector[1] = ad_emotions[4]
    # ad_emotions_vector[2] = ad_emotions[6]
    # ad_emotions_vector[3] = ad_emotions[0] + ad_emotions[2] + ad_emotions[3]
    # ad_emotions_vector[4] = ad_emotions[7]
    # ad_emotions_vector[5] = ad_emotions[5]
    cos_sim = np.dot(facial_emotion_vector, ad_emotions_vector)/(np.linalg.norm(facial_emotion_vector)*np.linalg.norm(ad_emotions_vector))

    print(cos_sim)
    if cos_sim<0.4:
        return True
    else:
        return False

while True:
    _, frame = cap.read()
    frame = imutils.resize(frame,width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = np.zeros((250, 300, 3), dtype="uint8")

    if ad_image is None:
        ad_image, ad_emotions, target_emotion = get_ad_image()
        cv2.putText(ad_image, target_emotion, (50, 100), cv2.FONT_ITALIC, 2, (0, 0, 255), 3)
        cv2.imshow("Ad", ad_image)

    isLooking = is_looking_in_frame(gray)
    should_sleep = False
    if isLooking:
        print(isLooking)
        try:
            label, facial_emotions, facial_emotions_canvas = get_facial_emotions(gray, frame)
            cv2.imshow("Facial emotions probabilities", facial_emotions_canvas)
            if label != "neutral":
                change_ad = should_change_ad(facial_emotions, ad_emotions)
                if change_ad:
                    cv2.putText(ad_image, target_emotion, (50, 100), cv2.FONT_ITALIC, 2, (0, 0, 255), 3)
                    ad_image, ad_emotions, target_emotion = get_ad_image()
                    cv2.putText(ad_image, target_emotion, (50, 100), cv2.FONT_ITALIC, 2, (0, 0, 255), 3)
                    cv2.imshow("Ad", ad_image)
                    should_sleep = True
        except Exception as e:
            print(e)
            continue

    cv2.imshow("Frame", frame)
    if should_sleep:
        time.sleep(3)
        should_sleep = False
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()