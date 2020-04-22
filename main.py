import numpy as np
import cv2
import tensorflow as tf
import os
import pickle
import facenet
import face
import shutil
import glob


from mtcnn import MTCNN
detector = MTCNN()

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

face_recognition = face.Recognition('./model/20180402-114759.pb','my_classifier.pkl', min_face_size=40)

MAX_FOLDER_SIZE = 50

number_of_images = 0

number_of_known_images = 0

count = 0

def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0]-10, face_bb[1]-10), (face_bb[2]+10, face_bb[3]+10),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]+45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)
    print((face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]))
    return frame

#check folder existing in /images
fcount = 0
for root, dirs, files in os.walk('./images/'):
        fcount += len(dirs)

newdir = './images/unknown'+str(fcount+1)
if not os.path.isdir(newdir):
    os.makedirs(newdir)

   

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    faces = face_recognition.identify(frame)

    #new_frame = add_overlays(frame, faces)

    


    for face in faces:
        face_bb = face.bounding_box.astype(int)
                
        cv2.rectangle(frame,(face_bb[0]-10, face_bb[1]-10), (face_bb[2]+10, face_bb[3]+10),(0, 255, 0), 2)
        if face.name is not None:
            cv2.putText(frame, face.name, (face_bb[0], face_bb[3]+45),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),thickness=2, lineType=2)

            if face.prob < 0.75:
                sub_face = frame[face_bb[1]:face_bb[3], face_bb[0]:face_bb[2]]
                    
                dim = (160, 160)
                resized = cv2.resize(sub_face, dim, interpolation = cv2.INTER_AREA)

                if count == 5:
                    FaceFileName = newdir+ "/" + str(number_of_images) + ".jpg"
                    cv2.imwrite(FaceFileName, resized)
                    number_of_images += 1
                    count = 0
                count+=1

            elif face.prob > 0.8:
                path = "./images/"+str(face.name)
                os.listdir(path)
                image_num = len(os.listdir(path))
                if  image_num <= MAX_FOLDER_SIZE:
                    sub_face = frame[face_bb[1]:face_bb[3], face_bb[0]:face_bb[2]]
                        
                    dim = (160, 160)
                    resized = cv2.resize(sub_face, dim, interpolation = cv2.INTER_AREA)

                    if count == 5:
                        number_of_known_images = image_num
                        number_of_known_images += 1
                        FaceFileName = path + "/" + str(number_of_known_images) + ".jpg"
                        cv2.imwrite(FaceFileName, resized)
                        number_of_images += 1
                        count = 0
                    count+=1

                
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
    cv2.imshow('frame', frame)
    #cv2.imshow('gray',gray)
    keyPress = cv2.waitKey(1) & 0xFF
    if keyPress == 113: #q
        shutil.rmtree(newdir)
        break
    elif keyPress == 97: #a
        cap.release()
        cv2.destroyAllWindows()
        os.system('cmd /c "python train_classifier.py && python main.py"')
    elif keyPress == 99: #c
        filelist = glob.glob(os.path.join(newdir, "*.jpg"))
        for f in filelist:
            os.remove(f)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
