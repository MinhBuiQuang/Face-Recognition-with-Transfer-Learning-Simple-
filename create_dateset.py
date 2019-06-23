import cv2
import numpy as np
import os
import pickle
#import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
#from keras.utils import to_categorical
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--image_path', type=str, default='./FaceData',
                    help='Path to the directory containing people data.')
parser.add_argument('--output_face', type=str, default='./faceimg',
                    help='Path to the directory to save Face_Image.')
parser.add_argument('--output_data', type=str, default='./dataset',
                    help='Path to the directory to create Pickle outputs.')
IMG_SIZE = 224

def pickle_data(filename, data):
    saved_data = open(filename, "wb")
    pickle.dump(data, saved_data)
    saved_data.close()
    
def extract_face(image, required_size=(224, 224)):
    faces = []
    detector = MTCNN()
    face_coords = detector.detect_faces(image)
    for face_coord in face_coords:
        x, y, w, h = face_coord['box']
        face = image[y:y+h, x:x+w]
        if face.shape[0] > 0 and face.shape[1] >0:
            if face.shape < required_size:
                face = cv2.resize(face, required_size, 
                                        interpolation=cv2.INTER_AREA)
            else:
                face = cv2.resize(face, required_size, 
                                        interpolation=cv2.INTER_CUBIC)
            faces.append(face)
    return faces
        
def gen_dataset(image_path): 
    print('Extracting faces from dataset')   
    count = 0
    labels_dic = {}
    image_path = 'FaceData/'
    people = [person for person in os.listdir(image_path)]
    count = 0
    for i, person in enumerate(people):
        labels_dic[i] = person
        dir = 'faceimg/{}'.format(person)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        print('Extracting {} face'.format(person))
        for image in tqdm(os.listdir(os.path.join(image_path, person))):
            img = cv2.imread(os.path.join(os.path.join(image_path, person), image), cv2.COLOR_BGR2RGB)                 
            faces = extract_face(img, (IMG_SIZE, IMG_SIZE))
            for face in faces:
                filedir = 'faceimg/{}/{}.jpeg'.format(person, count)
                count += 1
                    
def collect_dataset(output_face):    
    print('Collecting faces')  
    face_images = []
    labels = []
    labels_dic = {}
    path = output_face
    people = [person for person in os.listdir(path)]
    for i, person in enumerate(tqdm(people)):
            labels_dic[i] = person
            imagedir = os.path.join(path, person)
            for image in os.listdir(imagedir):
                img = cv2.imread(os.path.join(imagedir, image), cv2.COLOR_BGR2RGB)
                face_images.append(img)
                labels.append(person)               
    return (face_images, np.array(labels), labels_dic)


def main(args):
    #Comment this line if you've already got cutted face images.
    gen_dataset(args.image_path)
    face_images, labels, labels_dic = collect_dataset(args.output_face)
    
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    #encoded_labels = to_categorical(encoded_labels)

    
    zipped_object = list(zip(face_images, encoded_labels))
    random.shuffle(zipped_object)

    X, y = zip(*zipped_object)
    X = np.asarray(X)
    y = np.asarray(y)
    X.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    pickle_data(os.path.join(args.output_data, 'X.pickle'), X)
    pickle_data(os.path.join(args.output_data, 'y.pickle'), y)
    pickle_data(os.path.join(args.output_data, 'dict.pickle'), labels_dic)

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)