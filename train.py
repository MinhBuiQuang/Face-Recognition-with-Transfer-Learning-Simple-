import pickle
import os
import argparse
from keras import applications, optimizers, backend as k
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras_applications.imagenet_utils import _obtain_input_shape

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='./dataset',
                    help='Path to the directory containing people data.')
parser.add_argument('--output_model', type=str, default='./models/my_model.h5',
                    help='Path to the directory to create model outputs.')

def loadData(dir):
    xPath = os.path.join(dir, 'X.pickle')
    yPath = os.path.join(dir, 'y.pickle')
    pickle_in = open(xPath,"rb")
    X = pickle.load(pickle_in)
    pickle_in = open(yPath,"rb")
    y = pickle.load(pickle_in)
    return X, y

#Model
def create_model():
    vgg_model = applications.VGG16(weights='imagenet',
                                include_top=False,
                                input_shape=(224, 224, 3))
    output_tensor = vgg_model.get_layer('block5_pool').output
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(256, activation = 'relu')(output_tensor)
    output_tensor = Dense(256, activation = 'relu')(output_tensor)
    output_tensor = Dense(256, activation = 'relu')(output_tensor)
    output_tensor = Dropout(0.5)(output_tensor)
    output_tensor = Dense(4, activation = 'softmax')(output_tensor)

    from keras.models import Model
    model = Model(input = vgg_model.input, output = output_tensor)

    for layer in model.layers[:-6]:
        layer.trainable = False

    model.compile(loss='sparse_categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])    
    return model


def main(args):
    X, y = loadData(args.data_path)
    model = create_model()
    model.summary()
    model.fit(X/255.0, y, epochs=6, validation_split=0.3)
    model.save(args.output_model)
    print('Model is saved at {}'.format(args.output_model))

if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)

