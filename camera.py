import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
from mtcnn.mtcnn import MTCNN

def extract_face(image, required_size=(224, 224)):
    faces = []
    detector = MTCNN()
    face_coords = detector.detect_faces(image)
    for face_coord in face_coords:
        x, y, w, h = face_coord['box']
        face = image[y:y+h, x:x+w]
        print(face.shape)
        if face.shape[0] > 0 and face.shape[1] >0:
            if face.shape < required_size:
                face = cv2.resize(face, required_size, 
                                        interpolation=cv2.INTER_AREA)
            else:
                face = cv2.resize(face, required_size, 
                                        interpolation=cv2.INTER_CUBIC)
            faces.append(face)
    return face_coords, faces

def drawWithMTCNN(image, face_coords, name):
    for face_coord in face_coords:
        (x,y,w,h) = face_coord['box']
        fontScale = w/500.0
        thickness = int(w/200)
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), thickness)    
        cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,255,0), thickness)

def readImage(filepath):
    image = cv2.imread(filepath, cv2.COLOR_BGR2RGB)
    return image


model = load_model('facemodel_3.h5')
pickle_in = open('dict.pickle', 'rb')
dict_name = pickle.load(pickle_in)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    face_coords, faces = extract_face(frame)
    faces = np.asarray(faces)
    if(len(faces) != 0):
        predictions = model.predict(faces/255.0)
        for pred in predictions:        
            idx = np.argmax(pred)
            label = dict_name[idx]
            print(label)
            drawWithMTCNN(frame, face_coords, label)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()