import numpy as np
import cv2
# Import ssl for ssl issues
#from ssl import SSLContext,PROTOCOL_TLSv1

# Import urlopen to open the url of the ip webcam
#from urllib.request import urlopen

#url = 'https://192.168.43.1:8080/video'
#from mtcnn.mtcnn import MTCNN

#detector = MTCNN()

#model = 'mtcnn'

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, im = cap.read()
    #gcontext = SSLContext(PROTOCOL_TLSv1)  # Only for gangstars
    #info = urlopen(url, context=gcontext).read()


    #imgNp=np.array(bytearray(info),dtype=np.uint8)
    #im=cv2.imdecode(imgNp,-1)

    # Our operations on the frame come here
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        sub_face = im[y:y+h, x:x+w]
        FaceFileName = "unknowfaces/face_" + str(y) + ".jpg"
        cv2.imwrite(FaceFileName, sub_face)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = im[y:y+h, x:x+w]
    # face_locations = detector.detect_faces(frame)

    # for loc in face_locations:
    #     (x, y, w, h) = loc['box']
    #     cv2.rectangle(frame, (x,y), (x+w,y+h), (80,18,236), 2)

    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     text = f' {len(face_locations)} faces'
    #     cv2.putText(frame, 'Champ', (x, y - 20), font, 0.5, (80, 18, 236), 1)
   
    # Display the resulting frame
    cv2.imshow('frame',im)
    #cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
