import cv2
import numpy as np
import os
from PIL import Image

path = 'faces/'
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


def get_images(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    images = []
    labels = []
    j = 0
    for image_path in image_paths:
        gray = Image.open(image_path).convert('L')
        image = np.array(gray, 'uint8')
        subject_number = int(os.path.basename(image_path).split(".")[0].replace("subject", ""))

        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            cv2.imwrite('frontalfaces/subject' + str(subject_number) + '_' + str(j) + '.jpg', image[y: y + h, x: x + w])
            print(subject_number)
            labels.append(subject_number)
            j += 1
    return images, labels


recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 90)

images, labels = get_images(path)

recognizer.train(images, np.array(labels))

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

mconf = 90
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
            cv2.putText(img, str(number_predicted), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

if mconf != 90:
    print("Subject is recognized as ", n_pr)
else:
    print("Subject isn't recognised")

cap.release()
out.release()
cv2.destroyAllWindows()
