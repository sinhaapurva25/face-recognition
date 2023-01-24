import cv2 as cv
img=cv.imread(r"test.jpg",cv.IMREAD_GRAYSCALE)
r,c=img.shape
print(r,c)
hc=cv.CascadeClassifier(r'haarcascades\haarcascade_frontalface_alt.xml')
faces_detected=hc.detectMultiScale(img,scaleFactor=1.7,minNeighbors=1)
print("no of faces detected",len(faces_detected))
for (x,y,w,h) in faces_detected:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
cv.imshow("face",img)
cv.waitKey(0)
cv.destroyAllWindows()