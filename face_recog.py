import numpy as np
import cv2 as cv
import os
harr_cascade = cv.CascadeClassifier(r'haarcascades\haarcascade_frontalface_alt.xml')

people = []
for i in os.listdir(r'images'):
    people.append(i)

# features = np.load(r'C:\Users\sinha\PycharmProjects\pythonProject\features.npy')
# labels = np.load(r'C:\Users\sinha\PycharmProjects\pythonProject\labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'face_trained.yml')

path = r'images\ben_affleck\2.jpg'
img_color = cv.imread(path)
img = cv.cvtColor(img_color, cv.COLOR_RGB2GRAY)

# cv.imshow("person",img)

faces_rect = harr_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1)
for(x, y, w, h) in faces_rect:
    face_roi = img[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(face_roi)
    print(f'label = {people[label]} with a confidence of {confidence}')

    cv.putText(img_color, str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(0,0,255), thickness=1)
    cv.rectangle(img_color,(x,y),(x+w,y+h),(0,0,255),thickness=1)

cv.imshow("detected face",img_color)
cv.waitKey(0)
cv.destroyAllWindows()