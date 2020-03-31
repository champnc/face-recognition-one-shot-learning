import numpy as np
import cv2

# from mtcnn.mtcnn import MTCNN
# detector = MTCNN()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #cv2.imwrite("frame/frame%s.jpg" % str(y), frame)
        cv2.rectangle(frame,(x-10,y-10),(x+w+20,y+h+20),(255,0,0),1)
        sub_face = frame[y-10:y+h+20, x-10:x+w+20]
        FaceFileName = "unknowfaces/face_" + str(y) + ".jpg"
        dim = (160, 160)
        resized = cv2.resize(sub_face, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(FaceFileName, resized)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    # face_locations = detector.detect_faces(im)
    # for loc in face_locations:
    #     (x, y, w, h) = loc['box']
    #     cv2.rectangle(im, (x,y), (x+w,y+h), (80,18,236), 2)
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     text = f' {len(face_locations)} faces'
    #     cv2.putText(im, 'Champ', (x, y - 20), font, 0.5, (80, 18, 236), 1)
   
    # Display the resulting frame
    cv2.imshow('frame',frame)
    #cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
