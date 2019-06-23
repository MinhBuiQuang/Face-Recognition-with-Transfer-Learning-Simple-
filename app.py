from flask import Flask, request, render_template,make_response
import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image
import io
import base64
import cv2
import pickle

app = Flask(__name__)
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

# def convertImage(imgData1):
# 	imgstr = re.search(r'base64,(.*)',imgData1).group(1)
# 	#print(imgstr)
# 	with open('output.png','wb') as output:
# 		output.write(imgstr.decode('base64'))

def loadModel():
    print('Loading model....')
    global model
    global dict_name
    pickle_in = open('dataset/dict.pickle', 'rb')
    dict_name = pickle.load(pickle_in)
    model = load_model('models/facemodel_3.h5')
    model._make_predict_function()
    print('Model loaded!')
loadModel()
@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():    
    imageByteString = request.files["picture"].read()
    nparr = np.fromstring(imageByteString, np.uint8)   
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
    face_coords, faces = extract_face(image)
    faces = np.asarray(faces)
    predictions = model.predict(faces/255.0)
    for pred in predictions:        
        idx = np.argmax(pred)
        label = dict_name[idx]
        drawWithMTCNN(image, face_coords, label)
    retval, buffer = cv2.imencode('.png', image)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text