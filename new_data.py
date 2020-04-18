import cv2
import numpy as np
import os
import argparse
import sys

def main(args):

    #face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #read webcam video
    video_capture = cv2.VideoCapture(0)

    #take name args
    name = args.name

    path = 'images'

    directory = os.path.join(path, name)
    print(directory)

    #check if already have --name folder
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok = 'True')

    number_of_images = 0
    MAX_NUMBER_OF_IMAGES = 50
    count = 0

    while number_of_images < MAX_NUMBER_OF_IMAGES:
        ret, frame = video_capture.read()

        frame = cv2.flip(frame, 1)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x-10,y-10),(x+w+20,y+h+20),(255,0,0),1)
            sub_face = frame[y-10:y+h+20, x-10:x+w+20]
            dim = (160, 160)
            resized = cv2.resize(sub_face, dim, interpolation = cv2.INTER_AREA)

            if count == 5:
                    FaceFileName = str(path)+ "/" + str(name) + "/" + str(number_of_images) + ".jpg"
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
    
    parser.add_argument('--name', type=str,
        help='Name of the user')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
