import cv2
import numpy as np
import os
import argparse
import sys
import face as f

def main(args):

    #face detector
    face_recognition = f.Detection()

    #read webcam video
    cap = cv2.VideoCapture(0)

    #take name args
    face_name = args.face_name

    path = 'images'

    directory = os.path.join(path, face_name)
    print(directory)

    #check if already have --name folder
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok = 'True')

    number_of_images = 0
    MAX_NUMBER_OF_IMAGES = 50
    count = 0

    while number_of_images < MAX_NUMBER_OF_IMAGES:
        ret, frame = cap.read()

        faces = face_recognition.find_faces(frame)

        for face in faces:
            face_bb = face.bounding_box.astype(int)
                    
            cv2.rectangle(frame,(face_bb[0]-10, face_bb[1]-10), (face_bb[2]+10, face_bb[3]+10),(0, 255, 0), 2)
            
            sub_face = frame[face_bb[1]:face_bb[3], face_bb[0]:face_bb[2]] 
            dim = (160, 160)
            resized = cv2.resize(sub_face, dim, interpolation = cv2.INTER_AREA)
                    
            if count == 5:
                FaceFileName = str(path)+ "/" + str(face_name) + "/" + str(number_of_images) + ".jpg"
                cv2.imwrite(FaceFileName, resized)
                number_of_images += 1
                count = 0
            count+=1       

        cv2.imshow('add new data', frame)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--face_name', type=str,
        help='Name of the user')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
