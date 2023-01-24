import os
import cv2 as cv
import numpy as np

people = []
for i in os.listdir(r'images'):
    people.append(i)

harr_cascade = cv.CascadeClassifier(r'haarcascades\haarcascade_frontalface_alt.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(r'images', person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)

            faces_rect = harr_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

            for(x, y, w, h) in faces_rect:
                # gray=cv.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),thickness=5)
                face_roi = gray[y:y+h, x:x+w]
                features.append(face_roi)
                labels.append(label)
create_train()
# print(f'length of the features = {len(features)}')
# print(f'length of the labels = {len(labels)}')
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')

# cv.face.LBPHFaceRecognizer_create().train(features, labels)
# cv.face.LBPHFaceRecognizer_create().save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)

print('training done----------------------------')