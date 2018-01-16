import sys
import os
import numpy as np

current_file_location = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_location, '../../'))  # add reference of parent directory

import utils
import read_input as rp
import models.model_manipulation as mp

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras import optimizers
from keras.utils import np_utils


K.set_image_dim_ordering('th')

def age_model():
    '''
    Age prediction model (multiclass classification)
    :return: model trained to predict age of the user
    '''
    # create model

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    return model


def gender_model():
    '''
    Gender prediction model (binary classification)
    :return: model trained to build gender of the user
    '''
    # TODO convert wiki images to 32*32
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(1, 28, 28), activation='relu'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    return model


def baseline_model():
    '''
    Baseline model
    :return:
    '''
    # create model
    # TODO change the input shape for RGB
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    return model


def train_model(input_path, input_image_dataframe, attribute_to_train):
    '''
    train model on give test/train data for given list of attribute, attribute can be age/gender etc.
    :param input_path:
    :param input_image_dataframe:
    :param attribute_to_train:
    :return:
    '''
    # build the model
    if (attribute_to_train == "age"):
        model = age_model()
    else :
        model = gender_model()

    (Image_list, gender_wiki_list) = rp.read_wiki_images(input_path)
    # normalize inputs from 0-255 to 0-1
    Image_list = Image_list / 255

    print("\nTrain on given dataset")
    (image_test_dataframe, image_validate_dataframe, image_train_dataframe) = utils.partitionData(input_image_dataframe,
                                                                                                  0.2, 0.1)
    Images_train = rp.getImageFromInput(image_train_dataframe, input_path)
    Images_train = np.concatenate((Images_train, Image_list))
    Images_validate = rp.getImageFromInput(image_validate_dataframe, input_path)
    Images_test = rp.getImageFromInput(image_test_dataframe, input_path)

    label_train = rp.get_label_for_input(image_train_dataframe, attribute_to_train)
    label_train = np.concatenate((label_train, gender_wiki_list))
    label_validate = rp.get_label_for_input(image_validate_dataframe, attribute_to_train)
    label_test = rp.get_label_for_input(image_test_dataframe, attribute_to_train)


    if (attribute_to_train == "age"):
        label_train = np_utils.to_categorical(label_train)
        label_test = np_utils.to_categorical(label_test)
        label_validate = np_utils.to_categorical(label_validate)


    # TODO fix epoch to correct value
    model.fit(Images_train, label_train, validation_data=(Images_validate, label_validate), epochs=20, batch_size=200,
              verbose=2)


    # Save model and count_vect
    mp._save_keras_model("image_cnn_" + attribute_to_train + ".h5", model)

    # Final evaluation of the model
    scores = model.evaluate(Images_test, label_test, verbose=0)
    print("Error: %.2f%%" % (100 - scores[1] * 100))


def make_predictions(input_path, attribute_to_predict):
    # Load model
    classifier = _load_model(attribute_to_predict)

    # load data
    input_image_dataframe = rp.read_images(input_path, ['userid'])
    images_to_predict = rp.getImageFromInput(input_image_dataframe, input_path)

    # get prediction
    probabilities = classifier.predict(images_to_predict)
    probabilities = probabilities.flatten()
    predictions = [float(round(x)) for x in probabilities]

    return utils._transform_genders_into_string(predictions)


def _load_model(attribute_to_predict):
    model_file_name = "image_cnn_" + attribute_to_predict + ".h5"
    classifier = mp._get_keras_model(model_file_name)
    return classifier
