import cv2
import numpy as np
import os

# img = cv2.imread('faces/subject10.PNG', 0)
# cv2.imwrite('faces/subject10.PNG', img)

path = 'faces/'


def get_images(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    images = []
    labels = []

    for image_path in image_paths:
        gray = cv2.imread(image_path, 0)

        subject_number = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            images.append(gray[y: y + h, x: x + w])
            labels.append(subject_number)
    return images, labels


cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.face.LBPHFaceRecognizer_create()

images, labels = get_images(path)

recognizer.train(images, np.array(labels))

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
mconf = 123
n_pr = -1

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = faceCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in face:
            number_predicted, conf = recognizer.predict(gray[y: y + h, x: x + w])
            if conf < mconf:
                mconf = conf
                n_pr = number_predicted
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

if mconf != 123:
    print("Subject is recognized as ", n_pr)
else:
    print("Subject isn't recognised")

cap.release()
out.release()
cv2.destroyAllWindows()
