import numpy as np
import cv2
import tensorflow as tf
import os
import pickle
import facenet
import face

from mtcnn import MTCNN
detector = MTCNN()

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

with open('my_classifier.pkl','rb') as infile:
    (model, class_names) = pickle.load(infile)
    print(class_names)

sess = tf.Session()
with sess.as_default():
    modeldir = './model/20180402-114759.pb'
    facenet.load_model(modeldir)

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

face_recognition = face.Recognition('./model/20180402-114759.pb','my_classifier.pkl', min_face_size=40)

MAX_FOLDER_SIZE = 50

def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)
    print((face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]))
    return frame   

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    faces = face_recognition.identify(frame)

    new_frame = add_overlays(frame, faces)

    for face in faces:
            if face.prob < 0.8:
                ## check in images if there already unknown directory

                ##create new unknown folder
                face_bb = face.bounding_box.astype(int)

                sub_face = frame[y-10:y+h+20, face_bb[0]:x+w+20]
                
                dim = (160, 160)
                resized = cv2.resize(sub_face, dim, interpolation = cv2.INTER_AREA)

                FaceFileName = "images/unknown" + str(face_bb[0]) + ".jpg"
                cv2.imwrite(FaceFileName, resized)

    # faces = detector.detect_faces(frame)
    # for face in faces:
    #     (x, y, w, h) = face['box'] #bounding box

    #     sub_face = frame[y-10:y+h+20, x-10:x+w+20]
    #     dim = (160, 160)
    #     resized = cv2.resize(sub_face, dim, interpolation = cv2.INTER_AREA)

    #     face_ = face_recognition.identify(resized)

    #     print(face_.name)
        

    #     #draw
    #     cv2.rectangle(frame,(x-15,y-15),(x+w+25,y+h+25),(255,0,0),1)
    #     cv2.putText(frame, 'Champ', (x, y - 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (80, 18, 236), 1)



    #     FaceFileName = "unknowfaces/face_" + str(y) + ".jpg"
    #     cv2.imwrite(FaceFileName, resized)
   
    # Display the resulting frame
    cv2.imshow('frame',new_frame)
    #cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
